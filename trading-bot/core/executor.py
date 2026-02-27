"""
core/executor.py
İşlem açma/kapama modülü.
Market/limit order, stop-loss, take-profit, trailing stop yönetimi.
"""

import threading
import time
from datetime import datetime
from typing import Dict, List, Optional

from binance import Client
from binance.exceptions import BinanceAPIException

from config.settings import Settings
from core.data_engine import DataEngine
from core.risk_manager import RiskManager
from database.db import DatabaseManager, Trade
from notifications.telegram import TelegramNotifier
from utils.logger import get_logger

logger = get_logger(__name__)


class OrderResult:
    """İşlem sonucu."""

    def __init__(self, success: bool, order_id: str = "", message: str = "", data: dict = None):
        self.success = success
        self.order_id = order_id
        self.message = message
        self.data = data or {}

    def __bool__(self):
        return self.success


class TrailingStopState:
    """Bir pozisyon için trailing stop durumu."""

    def __init__(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        activation_pct: float,
        callback_pct: float,
    ):
        self.symbol = symbol
        self.direction = direction
        self.entry_price = entry_price
        self.activation_pct = activation_pct  # Aktifleşme için min PNL%
        self.callback_pct = callback_pct       # Geri çekilme %
        self.activated = False
        self.peak_price: Optional[float] = None  # Ulaşılan en iyi fiyat
        self.stop_price: Optional[float] = None  # Mevcut trailing stop fiyatı

    def update(self, current_price: float) -> bool:
        """
        Trailing stop'u günceller.
        Returns: True if stop should be triggered.
        """
        if self.direction == "LONG":
            # Kâr yüzdesi
            pnl_pct = (current_price - self.entry_price) / self.entry_price * 100

            # Aktifleşme kontrolü
            if not self.activated and pnl_pct >= self.activation_pct:
                self.activated = True
                self.peak_price = current_price
                self.stop_price = current_price * (1 - self.callback_pct / 100)
                logger.info(
                    f"[TRAILING] {self.symbol} LONG aktivasyon: "
                    f"PNL={pnl_pct:.2f}%, Stop={self.stop_price:.4f}"
                )

            if self.activated:
                # Yeni tepe fiyat
                if current_price > self.peak_price:
                    self.peak_price = current_price
                    self.stop_price = current_price * (1 - self.callback_pct / 100)
                    logger.debug(
                        f"[TRAILING] {self.symbol} tepe güncellendi: "
                        f"{self.peak_price:.4f}, Stop: {self.stop_price:.4f}"
                    )

                # Tetiklenme kontrolü
                if current_price <= self.stop_price:
                    logger.info(
                        f"[TRAILING] {self.symbol} LONG tetiklendi: "
                        f"Fiyat={current_price:.4f} <= Stop={self.stop_price:.4f}"
                    )
                    return True

        elif self.direction == "SHORT":
            pnl_pct = (self.entry_price - current_price) / self.entry_price * 100

            if not self.activated and pnl_pct >= self.activation_pct:
                self.activated = True
                self.peak_price = current_price
                self.stop_price = current_price * (1 + self.callback_pct / 100)
                logger.info(
                    f"[TRAILING] {self.symbol} SHORT aktivasyon: "
                    f"PNL={pnl_pct:.2f}%, Stop={self.stop_price:.4f}"
                )

            if self.activated:
                if current_price < self.peak_price:
                    self.peak_price = current_price
                    self.stop_price = current_price * (1 + self.callback_pct / 100)

                if current_price >= self.stop_price:
                    logger.info(
                        f"[TRAILING] {self.symbol} SHORT tetiklendi: "
                        f"Fiyat={current_price:.4f} >= Stop={self.stop_price:.4f}"
                    )
                    return True

        return False

    def get_current_pnl_pct(self, current_price: float) -> float:
        """Mevcut PNL yüzdesini döndürür."""
        if self.direction == "LONG":
            return (current_price - self.entry_price) / self.entry_price * 100
        else:
            return (self.entry_price - current_price) / self.entry_price * 100


class Executor:
    """
    Binance Futures işlem yürütücüsü.
    İşlem açma, kapama, stop-loss/take-profit yönetimi.
    """

    def __init__(
        self,
        settings: Settings,
        data_engine: DataEngine,
        risk_manager: RiskManager,
        db: DatabaseManager,
        telegram: TelegramNotifier,
    ):
        self.settings = settings
        self.data = data_engine
        self.risk = risk_manager
        self.db = db
        self.telegram = telegram

        # Binance istemcisi (data engine'den alınır)
        self._client: Optional[Client] = None

        # Açık pozisyon takibi (symbol -> Trade DB nesnesi)
        self._open_trades: Dict[str, Trade] = {}
        self._trades_lock = threading.RLock()

        # Trailing stop durumları
        self._trailing_states: Dict[str, TrailingStopState] = {}

        # PNL izleme thread'i
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitoring = False

        # Mevcut açık işlemleri DB'den yükle
        self._load_open_trades()

        logger.info("Executor başlatıldı.")

    def _get_client(self) -> Client:
        """Binance istemcisini döndürür."""
        return self.data._get_client()

    def _load_open_trades(self):
        """Başlangıçta DB'deki açık işlemleri belleğe yükler."""
        try:
            open_trades = self.db.get_open_trades()
            with self._trades_lock:
                for trade in open_trades:
                    self._open_trades[trade.symbol] = trade
                    # Risk kilidini de aç
                    self.risk.acquire_position_lock(trade.symbol)
                    # Trailing stop durumunu oluştur
                    sym_cfg = self.settings.get_symbol_config(trade.symbol)
                    if sym_cfg:
                        self._trailing_states[trade.symbol] = TrailingStopState(
                            symbol=trade.symbol,
                            direction=trade.side,
                            entry_price=trade.entry_price,
                            activation_pct=sym_cfg.trailing_activation_pct,
                            callback_pct=sym_cfg.trailing_callback_pct,
                        )
            logger.info(f"{len(open_trades)} açık işlem belleğe yüklendi.")
        except Exception as e:
            logger.error(f"Açık işlem yükleme hatası: {e}")

    # ----------------------------------------------------------------
    # Kaldıraç Ayarı
    # ----------------------------------------------------------------

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Belirtilen parite için kaldıraç ayarlar."""
        try:
            client = self._get_client()
            client.futures_change_leverage(symbol=symbol, leverage=leverage)
            logger.info(f"{symbol} kaldıraç {leverage}x ayarlandı.")
            return True
        except BinanceAPIException as e:
            logger.error(f"{symbol} kaldıraç ayarlama hatası: {e}")
            return False

    # ----------------------------------------------------------------
    # İşlem Açma
    # ----------------------------------------------------------------

    def open_position(
        self,
        symbol: str,
        direction: str,
        signal_data: dict = None,
        order_type: str = "MARKET",
        limit_price: float = None,
    ) -> OrderResult:
        """
        Pozisyon açar.
        direction: 'LONG' veya 'SHORT'
        order_type: 'MARKET' veya 'LIMIT'
        """
        try:
            # Konfigürasyon
            sym_cfg = self.settings.get_symbol_config(symbol)
            if not sym_cfg:
                return OrderResult(False, message=f"{symbol} konfigürasyonu bulunamadı")

            # Güncel fiyat
            entry_price = self.data.get_symbol_price(symbol)
            if not entry_price:
                return OrderResult(False, message=f"{symbol} fiyatı alınamadı")

            if order_type == "LIMIT" and limit_price:
                entry_price = limit_price

            # Kaldıraç ayarla
            self.set_leverage(symbol, sym_cfg.leverage)

            # Pozisyon büyüklüğü
            quantity, margin_usdt = self.risk.calculate_position_size(
                symbol, entry_price, sym_cfg.leverage
            )

            if quantity <= 0:
                return OrderResult(False, message="Hesaplanan miktar sıfır")

            # Binance tarafı
            side = "BUY" if direction == "LONG" else "SELL"

            # Pozisyon kilidi al
            if not self.risk.acquire_position_lock(symbol):
                return OrderResult(
                    False, message=f"{symbol} için zaten pozisyon var"
                )

            try:
                client = self._get_client()

                # Ana emir
                order_params = {
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                }

                if order_type == "MARKET":
                    order_params["type"] = "MARKET"
                    order = client.futures_create_order(**order_params)
                else:
                    order_params["type"] = "LIMIT"
                    order_params["price"] = limit_price
                    order_params["timeInForce"] = "GTC"
                    order = client.futures_create_order(**order_params)

                # Gerçek giriş fiyatı (market order dolduktan sonra)
                if order_type == "MARKET":
                    actual_entry = float(order.get("avgPrice", entry_price))
                    if actual_entry == 0:
                        actual_entry = entry_price
                else:
                    actual_entry = limit_price or entry_price

                # Stop-loss ve take-profit
                sl_price, tp_price = self.risk.calculate_sl_tp(
                    symbol, actual_entry, direction
                )

                # Stop-loss emri
                sl_side = "SELL" if direction == "LONG" else "BUY"
                try:
                    client.futures_create_order(
                        symbol=symbol,
                        side=sl_side,
                        type="STOP_MARKET",
                        stopPrice=sl_price,
                        closePosition=True,
                    )
                    logger.info(f"{symbol} Stop-loss ayarlandı: {sl_price}")
                except BinanceAPIException as e:
                    logger.warning(f"{symbol} Stop-loss ayarlama başarısız: {e}")

                # Take-profit emri
                try:
                    client.futures_create_order(
                        symbol=symbol,
                        side=sl_side,
                        type="TAKE_PROFIT_MARKET",
                        stopPrice=tp_price,
                        closePosition=True,
                    )
                    logger.info(f"{symbol} Take-profit ayarlandı: {tp_price}")
                except BinanceAPIException as e:
                    logger.warning(f"{symbol} Take-profit ayarlama başarısız: {e}")

                # Veritabanına kaydet
                trade_data = {
                    "binance_order_id": str(order.get("orderId", "")),
                    "symbol": symbol,
                    "side": direction,
                    "status": "OPEN",
                    "entry_price": actual_entry,
                    "entry_time": datetime.utcnow(),
                    "quantity": quantity,
                    "leverage": sym_cfg.leverage,
                    "stop_loss_price": sl_price,
                    "take_profit_price": tp_price,
                    "signal_data": signal_data or {},
                }
                trade = self.db.save_trade(trade_data)

                # Belleğe ekle
                with self._trades_lock:
                    self._open_trades[symbol] = trade
                    # Trailing stop durumu
                    self._trailing_states[symbol] = TrailingStopState(
                        symbol=symbol,
                        direction=direction,
                        entry_price=actual_entry,
                        activation_pct=sym_cfg.trailing_activation_pct,
                        callback_pct=sym_cfg.trailing_callback_pct,
                    )

                # Telegram bildirimi
                self.telegram.send_trade_open(
                    symbol=symbol,
                    direction=direction,
                    entry_price=actual_entry,
                    quantity=quantity,
                    leverage=sym_cfg.leverage,
                    stop_loss=sl_price,
                    take_profit=tp_price,
                )

                logger.info(
                    f"[İŞLEM AÇILDI] {symbol} {direction} | "
                    f"Fiyat: {actual_entry} | Miktar: {quantity} | "
                    f"Kaldıraç: {sym_cfg.leverage}x | "
                    f"SL: {sl_price} | TP: {tp_price}"
                )

                return OrderResult(
                    True,
                    order_id=str(order.get("orderId", "")),
                    message="İşlem başarıyla açıldı",
                    data=trade_data,
                )

            except BinanceAPIException as e:
                # Hata durumunda kilidi serbest bırak
                self.risk.release_position_lock(symbol)
                logger.error(f"{symbol} işlem açma Binance hatası: {e}")
                return OrderResult(False, message=f"Binance hatası: {e}")

            except Exception as e:
                self.risk.release_position_lock(symbol)
                logger.error(f"{symbol} işlem açma hatası: {e}")
                return OrderResult(False, message=str(e))

        except Exception as e:
            logger.error(f"{symbol} genel işlem açma hatası: {e}")
            return OrderResult(False, message=str(e))

    # ----------------------------------------------------------------
    # İşlem Kapatma
    # ----------------------------------------------------------------

    def close_position(
        self,
        symbol: str,
        reason: str = "MANUAL",
        partial_pct: float = 100.0,
    ) -> OrderResult:
        """
        Pozisyon kapatır.
        reason: 'SL', 'TP', 'TRAILING', 'MANUAL', 'DAILY_LIMIT', 'WATCHDOG'
        partial_pct: Kapatılacak pozisyon yüzdesi (100 = tam kapatma)
        """
        with self._trades_lock:
            trade = self._open_trades.get(symbol)

        if not trade:
            return OrderResult(False, message=f"{symbol} için açık işlem bulunamadı")

        try:
            client = self._get_client()

            # Kapatma miktarı
            close_qty = trade.quantity * (partial_pct / 100.0)

            # Kapatma yönü (zıttı)
            close_side = "SELL" if trade.side == "LONG" else "BUY"

            # Market emri ile kapat
            order = client.futures_create_order(
                symbol=symbol,
                side=close_side,
                type="MARKET",
                quantity=close_qty,
                reduceOnly=True,
            )

            # Açık stop-loss ve take-profit emirlerini iptal et
            try:
                client.futures_cancel_all_open_orders(symbol=symbol)
            except Exception as e:
                logger.warning(f"{symbol} open order iptal hatası: {e}")

            # Kapanış fiyatı
            exit_price = float(order.get("avgPrice", 0))
            if exit_price == 0:
                exit_price = self.data.get_symbol_price(symbol) or trade.entry_price

            # PNL hesapla
            if trade.side == "LONG":
                pnl_pct = (exit_price - trade.entry_price) / trade.entry_price * 100
                pnl_usdt = (exit_price - trade.entry_price) * close_qty
            else:
                pnl_pct = (trade.entry_price - exit_price) / trade.entry_price * 100
                pnl_usdt = (trade.entry_price - exit_price) * close_qty

            # Kaldıraç etkisi
            pnl_usdt_leveraged = pnl_usdt  # Binance zaten kaldıraçlı PNL'i hesaplar

            # Süre
            duration = datetime.utcnow() - trade.entry_time
            duration_str = str(duration).split(".")[0]

            # Veritabanı güncelle
            updated_trade = self.db.update_trade(
                trade.id,
                status="CLOSED" if partial_pct >= 100 else "OPEN",
                exit_price=exit_price,
                exit_time=datetime.utcnow(),
                close_reason=reason,
                pnl_usdt=round(pnl_usdt_leveraged, 4),
                pnl_pct=round(pnl_pct, 4),
            )

            # Günlük istatistik güncelle
            self.db.update_daily_stats(updated_trade)

            # Risk yöneticisine bildir
            self.risk.record_trade_result(pnl_usdt_leveraged)

            # Tam kapatma ise belleği temizle
            if partial_pct >= 100:
                with self._trades_lock:
                    self._open_trades.pop(symbol, None)
                    self._trailing_states.pop(symbol, None)
                self.risk.release_position_lock(symbol)

            # Telegram bildirimi
            self.telegram.send_trade_close(
                symbol=symbol,
                direction=trade.side,
                entry_price=trade.entry_price,
                exit_price=exit_price,
                pnl_usdt=pnl_usdt_leveraged,
                pnl_pct=pnl_pct,
                duration=duration_str,
                reason=reason,
            )

            logger.info(
                f"[İŞLEM KAPANDI] {symbol} {trade.side} | "
                f"Giriş: {trade.entry_price} | Çıkış: {exit_price} | "
                f"PNL: {pnl_usdt_leveraged:+.4f} USDT ({pnl_pct:+.2f}%) | "
                f"Süre: {duration_str} | Sebep: {reason}"
            )

            return OrderResult(
                True,
                order_id=str(order.get("orderId", "")),
                message=f"İşlem kapatıldı: {reason}",
                data={
                    "exit_price": exit_price,
                    "pnl_usdt": pnl_usdt_leveraged,
                    "pnl_pct": pnl_pct,
                    "reason": reason,
                },
            )

        except BinanceAPIException as e:
            logger.error(f"{symbol} işlem kapatma Binance hatası: {e}")
            return OrderResult(False, message=f"Binance hatası: {e}")
        except Exception as e:
            logger.error(f"{symbol} işlem kapatma hatası: {e}")
            return OrderResult(False, message=str(e))

    def close_all_positions(self, reason: str = "MANUAL") -> List[OrderResult]:
        """Tüm açık pozisyonları kapatır."""
        with self._trades_lock:
            symbols = list(self._open_trades.keys())

        results = []
        for symbol in symbols:
            result = self.close_position(symbol, reason=reason)
            results.append(result)
            time.sleep(0.2)  # Rate limit koruması

        logger.info(f"Tüm pozisyonlar kapatıldı ({len(results)} işlem, sebep: {reason})")
        return results

    # ----------------------------------------------------------------
    # PNL İzleme ve Trailing Stop
    # ----------------------------------------------------------------

    def start_monitoring(self):
        """Açık pozisyon izleme thread'ini başlatır."""
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="Executor-Monitor",
        )
        self._monitor_thread.start()
        logger.info("Pozisyon izleme başlatıldı.")

    def stop_monitoring(self):
        """İzleme durdurur."""
        self._monitoring = False
        logger.info("Pozisyon izleme durduruldu.")

    def _monitor_loop(self):
        """Açık pozisyonları izleyen döngü."""
        while self._monitoring:
            try:
                with self._trades_lock:
                    symbols = list(self._open_trades.keys())

                for symbol in symbols:
                    try:
                        self._check_position(symbol)
                    except Exception as e:
                        logger.error(f"{symbol} pozisyon kontrol hatası: {e}")

                time.sleep(2)  # 2 saniyede bir kontrol

            except Exception as e:
                logger.error(f"Monitor döngü hatası: {e}")
                time.sleep(5)

    def _check_position(self, symbol: str):
        """Tek bir pozisyonu kontrol eder (PNL, trailing stop)."""
        current_price = self.data.price_cache.get(symbol)
        if not current_price:
            return

        with self._trades_lock:
            trade = self._open_trades.get(symbol)
            trailing = self._trailing_states.get(symbol)

        if not trade or not trailing:
            return

        # Trailing stop kontrolü
        should_close = trailing.update(current_price)
        if should_close:
            logger.info(f"[TRAILING STOP] {symbol} trailing stop tetiklendi.")
            self.close_position(symbol, reason="TRAILING")

    def get_open_trades_info(self) -> List[dict]:
        """Açık işlemlerin anlık durumunu döndürür."""
        result = []
        with self._trades_lock:
            for symbol, trade in self._open_trades.items():
                current_price = self.data.price_cache.get(symbol) or trade.entry_price
                trailing = self._trailing_states.get(symbol)

                if trade.side == "LONG":
                    pnl_pct = (current_price - trade.entry_price) / trade.entry_price * 100
                    pnl_usdt = (current_price - trade.entry_price) * trade.quantity
                else:
                    pnl_pct = (trade.entry_price - current_price) / trade.entry_price * 100
                    pnl_usdt = (trade.entry_price - current_price) * trade.quantity

                result.append({
                    "id": trade.id,
                    "symbol": symbol,
                    "side": trade.side,
                    "entry_price": trade.entry_price,
                    "current_price": current_price,
                    "quantity": trade.quantity,
                    "leverage": trade.leverage,
                    "stop_loss": trade.stop_loss_price,
                    "take_profit": trade.take_profit_price,
                    "pnl_pct": round(pnl_pct, 4),
                    "pnl_usdt": round(pnl_usdt, 4),
                    "trailing_activated": trailing.activated if trailing else False,
                    "trailing_stop_price": trailing.stop_price if trailing else None,
                    "entry_time": trade.entry_time.isoformat() if trade.entry_time else None,
                    "duration": str(datetime.utcnow() - trade.entry_time).split(".")[0]
                    if trade.entry_time else "N/A",
                })

        return result
