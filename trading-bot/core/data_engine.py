"""
core/data_engine.py
Binance Futures veri motoru.
WebSocket ile anlık fiyat akışı, REST API ile ek veriler.
Rate limiter ve otomatik yeniden bağlanma desteği.
"""

import asyncio
import json
import threading
import time
from collections import defaultdict, deque
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import requests
from binance import Client
from binance.exceptions import BinanceAPIException

from utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# Token Bucket Rate Limiter
# ============================================================

class RateLimiter:
    """
    Token bucket algoritması ile rate limiter.
    Binance'in 2400 istek/dakika limitine uyum sağlar.
    """

    def __init__(self, max_calls: int = 1200, period: float = 60.0):
        self.max_calls = max_calls      # Maksimum çağrı sayısı
        self.period = period             # Saniye cinsinden dönem
        self.calls = deque()            # Çağrı zaman damgaları
        self._lock = threading.Lock()

    def acquire(self):
        """
        Rate limiti kontrol eder. Gerekirse bekler.
        Bloklamayan kullanım için is_allowed() kullanın.
        """
        with self._lock:
            now = time.time()
            # Süresi geçmiş çağrıları temizle
            while self.calls and now - self.calls[0] > self.period:
                self.calls.popleft()

            if len(self.calls) >= self.max_calls:
                # Limit aşıldı, bekleme süresi hesapla
                sleep_time = self.period - (now - self.calls[0]) + 0.1
                logger.warning(f"Rate limit: {sleep_time:.2f}sn bekleniyor")
                time.sleep(sleep_time)
                # Tekrar temizle
                now = time.time()
                while self.calls and now - self.calls[0] > self.period:
                    self.calls.popleft()

            self.calls.append(time.time())


# ============================================================
# Fiyat Verisi Önbelleği
# ============================================================

class PriceCache:
    """Thread-safe fiyat önbelleği."""

    def __init__(self):
        self._prices: Dict[str, float] = {}
        self._lock = threading.RLock()
        self._timestamps: Dict[str, datetime] = {}

    def update(self, symbol: str, price: float):
        with self._lock:
            self._prices[symbol] = price
            self._timestamps[symbol] = datetime.utcnow()

    def get(self, symbol: str) -> Optional[float]:
        with self._lock:
            return self._prices.get(symbol)

    def get_all(self) -> Dict[str, float]:
        with self._lock:
            return dict(self._prices)

    def get_age_seconds(self, symbol: str) -> float:
        """Fiyat verisinin kaç saniye önce güncellendiğini döndürür."""
        with self._lock:
            ts = self._timestamps.get(symbol)
            if ts is None:
                return float("inf")
            return (datetime.utcnow() - ts).total_seconds()


# ============================================================
# Mum (Kline) Verisi Önbelleği
# ============================================================

class KlineCache:
    """Her parite için mum verisi önbelleği."""

    def __init__(self, max_candles: int = 300):
        self.max_candles = max_candles
        self._data: Dict[str, Dict[str, List[dict]]] = defaultdict(dict)
        self._lock = threading.RLock()

    def update(self, symbol: str, interval: str, candles: List[dict]):
        """Mum verisi günceller (son N mum)."""
        with self._lock:
            self._data[symbol][interval] = candles[-self.max_candles:]

    def get(self, symbol: str, interval: str) -> List[dict]:
        """Mum verisi döndürür."""
        with self._lock:
            return list(self._data.get(symbol, {}).get(interval, []))

    def append(self, symbol: str, interval: str, candle: dict):
        """Yeni mumu ekler veya son mumu günceller."""
        with self._lock:
            if interval not in self._data[symbol]:
                self._data[symbol][interval] = []
            candles = self._data[symbol][interval]
            # Aynı açılış zamanı varsa güncelle, yoksa ekle
            if candles and candles[-1]["open_time"] == candle["open_time"]:
                candles[-1] = candle
            else:
                candles.append(candle)
                # Maksimum boyutu koru
                if len(candles) > self.max_candles:
                    candles.pop(0)


# ============================================================
# Veri Motoru
# ============================================================

class DataEngine:
    """
    Binance Futures veri motoru.
    - REST API ile tarihsel veri ve anlık bilgiler
    - WebSocket ile gerçek zamanlı fiyat akışı
    - Rate limiter koruması
    - Otomatik yeniden bağlanma
    """

    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet

        # Binance istemcisi
        self._client: Optional[Client] = None
        self._client_lock = threading.Lock()

        # Veri önbellekleri
        self.price_cache = PriceCache()
        self.kline_cache = KlineCache()

        # Rate limiter (dakikada 1200 istek)
        self.rate_limiter = RateLimiter(max_calls=1200, period=60.0)

        # WebSocket yönetimi
        self._ws_thread: Optional[threading.Thread] = None
        self._ws_running = False
        self._subscribed_symbols: List[str] = []
        self._price_callbacks: List[Callable] = []
        self._kline_callbacks: List[Callable] = []

        # Bağlantı durumu
        self._connected = False
        self._last_message_time: Optional[float] = None

        # Bakiye önbelleği
        self._balance_cache: Optional[dict] = None
        self._balance_updated_at: Optional[float] = None
        self._balance_ttl = 30.0  # 30 saniye

        logger.info(
            f"DataEngine başlatıldı (testnet={'açık' if testnet else 'kapalı'})"
        )

    # ----------------------------------------------------------------
    # Bağlantı Yönetimi
    # ----------------------------------------------------------------

    def connect(self) -> bool:
        """Binance REST istemcisini başlatır."""
        try:
            with self._client_lock:
                self._client = Client(
                    self.api_key,
                    self.api_secret,
                    testnet=self.testnet,
                )
                # Futures için futures_change_leverage çalışıyorsa bağlantı tamam
                self._client.futures_get_position_mode()
            self._connected = True
            logger.info("Binance REST API bağlantısı kuruldu.")
            return True
        except BinanceAPIException as e:
            logger.error(f"Binance bağlantı hatası: {e}")
            self._connected = False
            return False
        except Exception as e:
            logger.error(f"Binance bağlantı hatası (genel): {e}")
            self._connected = False
            return False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def disconnect(self):
        """Bağlantıyı kapatır."""
        self._ws_running = False
        self._connected = False
        logger.info("DataEngine bağlantısı kapatıldı.")

    # ----------------------------------------------------------------
    # REST API İşlemleri
    # ----------------------------------------------------------------

    def _get_client(self) -> Client:
        """Thread-safe istemci döndürür."""
        with self._client_lock:
            if self._client is None:
                raise RuntimeError("Binance istemcisi başlatılmamış. connect() çağrın.")
            return self._client

    def get_futures_balance(self, force_refresh: bool = False) -> dict:
        """
        USDT bazlı futures bakiyesini döndürür.
        Returns: {total, available, unrealized_pnl}
        """
        now = time.time()
        if (
            not force_refresh
            and self._balance_cache
            and self._balance_updated_at
            and (now - self._balance_updated_at) < self._balance_ttl
        ):
            return self._balance_cache

        try:
            self.rate_limiter.acquire()
            client = self._get_client()
            account = client.futures_account()

            total = float(account.get("totalWalletBalance", 0))
            available = float(account.get("availableBalance", 0))
            unrealized = float(account.get("totalUnrealizedProfit", 0))

            self._balance_cache = {
                "total": total,
                "available": available,
                "unrealized_pnl": unrealized,
                "margin_balance": float(account.get("totalMarginBalance", 0)),
            }
            self._balance_updated_at = now
            return self._balance_cache

        except BinanceAPIException as e:
            logger.error(f"Bakiye sorgu hatası: {e}")
            return self._balance_cache or {"total": 0, "available": 0, "unrealized_pnl": 0}

    def get_symbol_price(self, symbol: str) -> Optional[float]:
        """Güncel fiyatı döndürür (önbellek yoksa REST ile alır)."""
        # Önce önbellekten kontrol et
        price = self.price_cache.get(symbol)
        if price and self.price_cache.get_age_seconds(symbol) < 5:
            return price

        # REST ile al
        try:
            self.rate_limiter.acquire()
            client = self._get_client()
            ticker = client.futures_symbol_ticker(symbol=symbol)
            price = float(ticker["price"])
            self.price_cache.update(symbol, price)
            return price
        except Exception as e:
            logger.error(f"{symbol} fiyat sorgu hatası: {e}")
            return None

    def get_klines(
        self, symbol: str, interval: str, limit: int = 200
    ) -> List[dict]:
        """
        Mum verilerini döndürür.
        interval: '1m', '3m', '5m', '15m', '30m', '1h', vb.
        """
        try:
            self.rate_limiter.acquire()
            client = self._get_client()

            # Binance interval formatı: '5m' -> '5m'
            raw_klines = client.futures_klines(
                symbol=symbol, interval=interval, limit=limit
            )

            candles = []
            for k in raw_klines:
                candles.append({
                    "open_time": k[0],
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                    "close_time": k[6],
                    "quote_volume": float(k[7]),
                    "trades": int(k[8]),
                    "taker_buy_volume": float(k[9]),
                })

            # Önbelleğe kaydet
            self.kline_cache.update(symbol, interval, candles)
            return candles

        except BinanceAPIException as e:
            logger.error(f"{symbol} kline sorgu hatası: {e}")
            # Önbellekten döndür
            return self.kline_cache.get(symbol, interval)

    def get_order_book(self, symbol: str, limit: int = 10) -> dict:
        """Order book (emir defteri) verisi döndürür."""
        try:
            self.rate_limiter.acquire()
            client = self._get_client()
            ob = client.futures_order_book(symbol=symbol, limit=limit)
            return {
                "bids": [(float(b[0]), float(b[1])) for b in ob["bids"]],
                "asks": [(float(a[0]), float(a[1])) for a in ob["asks"]],
            }
        except Exception as e:
            logger.error(f"{symbol} order book hatası: {e}")
            return {"bids": [], "asks": []}

    def get_funding_rate(self, symbol: str) -> float:
        """Mevcut funding rate'i döndürür (%)."""
        try:
            self.rate_limiter.acquire()
            client = self._get_client()
            info = client.futures_funding_rate(symbol=symbol, limit=1)
            if info:
                return float(info[-1]["fundingRate"]) * 100  # Yüzdeye çevir
            return 0.0
        except Exception as e:
            logger.error(f"{symbol} funding rate hatası: {e}")
            return 0.0

    def get_open_positions(self) -> List[dict]:
        """Tüm açık pozisyonları döndürür."""
        try:
            self.rate_limiter.acquire()
            client = self._get_client()
            positions = client.futures_position_information()
            return [
                {
                    "symbol": p["symbol"],
                    "side": "LONG" if float(p["positionAmt"]) > 0 else "SHORT",
                    "size": abs(float(p["positionAmt"])),
                    "entry_price": float(p["entryPrice"]),
                    "unrealized_pnl": float(p["unRealizedProfit"]),
                    "leverage": int(p["leverage"]),
                    "liquidation_price": float(p.get("liquidationPrice", 0)),
                    "margin": float(p.get("isolatedMargin", 0)),
                }
                for p in positions
                if float(p["positionAmt"]) != 0
            ]
        except Exception as e:
            logger.error(f"Açık pozisyon sorgu hatası: {e}")
            return []

    def get_spread(self, symbol: str) -> float:
        """Ask-bid spreadini yüzde olarak döndürür."""
        ob = self.get_order_book(symbol, limit=5)
        if not ob["bids"] or not ob["asks"]:
            return 999.0
        best_bid = ob["bids"][0][0]
        best_ask = ob["asks"][0][0]
        if best_bid == 0:
            return 999.0
        return ((best_ask - best_bid) / best_bid) * 100

    def get_exchange_info(self, symbol: str) -> Optional[dict]:
        """Parite bilgilerini döndürür (min lot, tick size, vb.)."""
        try:
            self.rate_limiter.acquire()
            client = self._get_client()
            info = client.futures_exchange_info()
            for s in info["symbols"]:
                if s["symbol"] == symbol:
                    filters = {f["filterType"]: f for f in s["filters"]}
                    return {
                        "symbol": symbol,
                        "status": s["status"],
                        "base_asset": s["baseAsset"],
                        "quote_asset": s["quoteAsset"],
                        "price_precision": s["pricePrecision"],
                        "quantity_precision": s["quantityPrecision"],
                        "lot_size": filters.get("LOT_SIZE", {}),
                        "price_filter": filters.get("PRICE_FILTER", {}),
                        "min_notional": filters.get("MIN_NOTIONAL", {}).get("notional", 0),
                    }
        except Exception as e:
            logger.error(f"{symbol} exchange info hatası: {e}")
        return None

    def check_connection(self) -> bool:
        """Binance sunucusuna bağlantıyı test eder."""
        try:
            self.rate_limiter.acquire()
            client = self._get_client()
            client.futures_ping()
            self._connected = True
            return True
        except Exception:
            self._connected = False
            return False

    # ----------------------------------------------------------------
    # WebSocket Yönetimi
    # ----------------------------------------------------------------

    def add_price_callback(self, callback: Callable):
        """Fiyat güncellemesi için callback ekler."""
        self._price_callbacks.append(callback)

    def add_kline_callback(self, callback: Callable):
        """Mum güncellemesi için callback ekler."""
        self._kline_callbacks.append(callback)

    def start_websocket(self, symbols: List[str], intervals: List[str] = None):
        """
        Belirtilen pariteler için WebSocket akışını başlatır.
        symbols: ['BTCUSDT', 'ETHUSDT', ...]
        intervals: ['5m', '15m'] (None ise sadece fiyat)
        """
        if intervals is None:
            intervals = ["5m", "15m"]

        self._subscribed_symbols = symbols
        self._ws_running = True

        self._ws_thread = threading.Thread(
            target=self._ws_run,
            args=(symbols, intervals),
            daemon=True,
            name="DataEngine-WebSocket",
        )
        self._ws_thread.start()
        logger.info(f"WebSocket başlatıldı: {symbols}, interval: {intervals}")

    def stop_websocket(self):
        """WebSocket akışını durdurur."""
        self._ws_running = False
        logger.info("WebSocket durduruldu.")

    def _ws_run(self, symbols: List[str], intervals: List[str]):
        """WebSocket döngüsü (ayrı thread'de çalışır)."""
        reconnect_attempts = 0
        max_attempts = 10

        while self._ws_running:
            try:
                self._ws_connect(symbols, intervals)
                reconnect_attempts = 0  # Başarılı bağlantıda sayacı sıfırla
            except Exception as e:
                reconnect_attempts += 1
                if reconnect_attempts > max_attempts:
                    logger.error(
                        f"WebSocket {max_attempts} denemeden sonra bağlanamadı. Duruluyor."
                    )
                    break

                wait = min(2 ** reconnect_attempts, 60)  # Exponential backoff, max 60sn
                logger.warning(
                    f"WebSocket bağlantı kesildi: {e}. "
                    f"{reconnect_attempts}. deneme. {wait}sn sonra yeniden bağlanılacak."
                )
                time.sleep(wait)

        logger.info("WebSocket döngüsü sonlandı.")

    def _ws_connect(self, symbols: List[str], intervals: List[str]):
        """
        Binance WebSocket combined stream'e bağlanır.
        Fiyat (miniTicker) ve mum (kline) akışlarına abone olur.
        """
        import websocket

        # Stream listesi oluştur
        streams = []
        for symbol in symbols:
            sym = symbol.lower()
            streams.append(f"{sym}@miniTicker")  # Anlık fiyat
            for interval in intervals:
                streams.append(f"{sym}@kline_{interval}")  # Mum verisi

        # Binance combined stream URL
        stream_path = "/".join(streams)
        if self.testnet:
            ws_url = f"wss://stream.binancefuture.com/stream?streams={stream_path}"
        else:
            ws_url = f"wss://fstream.binance.com/stream?streams={stream_path}"

        logger.debug(f"WebSocket URL: {ws_url[:100]}...")

        def on_message(ws, message):
            try:
                data = json.loads(message)
                self._last_message_time = time.time()
                if "data" in data:
                    self._handle_ws_message(data["data"])
            except Exception as e:
                logger.error(f"WebSocket mesaj işleme hatası: {e}")

        def on_error(ws, error):
            logger.error(f"WebSocket hatası: {error}")

        def on_close(ws, close_status_code, close_msg):
            logger.warning(f"WebSocket kapatıldı: {close_status_code} - {close_msg}")

        def on_open(ws):
            logger.info("WebSocket bağlantısı açıldı.")

        ws_app = websocket.WebSocketApp(
            ws_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open,
        )
        ws_app.run_forever(
            ping_interval=20,
            ping_timeout=10,
        )

    def _handle_ws_message(self, data: dict):
        """Gelen WebSocket mesajını işler."""
        event_type = data.get("e")

        if event_type == "24hrMiniTicker":
            # Fiyat güncellemesi
            symbol = data.get("s")
            price = float(data.get("c", 0))  # close price
            if symbol and price > 0:
                self.price_cache.update(symbol, price)
                for callback in self._price_callbacks:
                    try:
                        callback(symbol, price)
                    except Exception as e:
                        logger.error(f"Fiyat callback hatası: {e}")

        elif event_type == "kline":
            # Mum güncellemesi
            k = data.get("k", {})
            symbol = data.get("s")
            interval = k.get("i")
            candle = {
                "open_time": k.get("t"),
                "open": float(k.get("o", 0)),
                "high": float(k.get("h", 0)),
                "low": float(k.get("l", 0)),
                "close": float(k.get("c", 0)),
                "volume": float(k.get("v", 0)),
                "close_time": k.get("T"),
                "quote_volume": float(k.get("q", 0)),
                "trades": int(k.get("n", 0)),
                "taker_buy_volume": float(k.get("V", 0)),
                "is_closed": k.get("x", False),
            }
            if symbol and interval:
                self.kline_cache.append(symbol, interval, candle)
                for callback in self._kline_callbacks:
                    try:
                        callback(symbol, interval, candle)
                    except Exception as e:
                        logger.error(f"Kline callback hatası: {e}")

    # ----------------------------------------------------------------
    # Tarihsel Veri Yükleme
    # ----------------------------------------------------------------

    def preload_klines(self, symbols: List[str], intervals: List[str], limit: int = 200):
        """
        Başlangıçta tüm pariteler için tarihsel mum verisi yükler.
        Sinyal motoru hemen çalışabilsin diye.
        """
        logger.info(f"{len(symbols)} parite için tarihsel veri yükleniyor...")
        for symbol in symbols:
            for interval in intervals:
                try:
                    candles = self.get_klines(symbol, interval, limit)
                    logger.debug(
                        f"{symbol} {interval}: {len(candles)} mum yüklendi"
                    )
                    time.sleep(0.1)  # Rate limit koruması
                except Exception as e:
                    logger.error(f"{symbol} {interval} veri yükleme hatası: {e}")

        logger.info("Tarihsel veri yükleme tamamlandı.")

    # ----------------------------------------------------------------
    # WebSocket Sağlık Kontrolü
    # ----------------------------------------------------------------

    def get_ws_status(self) -> dict:
        """WebSocket durumunu döndürür."""
        if self._last_message_time is None:
            age = None
        else:
            age = time.time() - self._last_message_time

        return {
            "running": self._ws_running,
            "last_message_age_sec": round(age, 1) if age is not None else None,
            "is_stale": age is not None and age > 30,
            "subscribed_symbols": self._subscribed_symbols,
        }
