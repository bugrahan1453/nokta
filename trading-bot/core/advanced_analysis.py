"""
core/advanced_analysis.py
Gelişmiş piyasa analizi modülleri.
Tüm modüller opsiyonel — enabled=False ise nötr sonuç döner, botu bloklamaz.

İçerik:
  - Order Book Imbalance & Likidite Duvarları
  - Open Interest Analizi
  - Fear & Greed Index (ücretsiz)
  - Glassnode On-Chain (opsiyonel API key)
  - CryptoQuant (opsiyonel API key)
  - Multi-Timeframe Confluence
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field

import requests

from utils.logger import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────
# Sonuç Tipleri
# ─────────────────────────────────────────────

@dataclass
class AnalysisResult:
    enabled: bool
    success: bool
    score: float          # -1.0 (çok ayı) … +1.0 (çok boğa)
    confidence: float     # 0-100
    data: dict = field(default_factory=dict)
    reason: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @classmethod
    def neutral(cls, reason: str = "Devre dışı") -> "AnalysisResult":
        return cls(enabled=False, success=False, score=0.0, confidence=0.0, reason=reason)

    @classmethod
    def error(cls, reason: str) -> "AnalysisResult":
        return cls(enabled=True, success=False, score=0.0, confidence=0.0, reason=reason)


# ─────────────────────────────────────────────
# ORDER BOOK ANALİZİ
# ─────────────────────────────────────────────

class OrderBookAnalyzer:
    """
    Emir defteri analizi:
    - Imbalance: Alış/satış hacim dengesi
    - Likidite duvarları: Büyük emirler
    - Delta akışı (taker buy vs sell)
    """

    def __init__(self, data_engine, enabled: bool = True):
        self.data = data_engine
        self.enabled = enabled

    def analyze(self, symbol: str, depth: int = 20) -> AnalysisResult:
        if not self.enabled:
            return AnalysisResult.neutral("OB analizi kapalı")
        try:
            ob = self.data.get_order_book(symbol, limit=depth)
            bids = ob.get("bids", [])
            asks = ob.get("asks", [])

            if not bids or not asks:
                return AnalysisResult.error("Order book verisi yok")

            # Toplam hacim
            bid_vol = sum(b[1] for b in bids)
            ask_vol = sum(a[1] for a in asks)
            total   = bid_vol + ask_vol

            # Imbalance skoru: +1 = tamamen alıcı, -1 = tamamen satıcı
            imbalance = (bid_vol - ask_vol) / total if total > 0 else 0.0

            # Likidite duvarları (en büyük 3 emir)
            bid_walls = sorted(bids, key=lambda x: x[1], reverse=True)[:3]
            ask_walls = sorted(asks, key=lambda x: x[1], reverse=True)[:3]

            best_bid = bids[0][0] if bids else 0
            best_ask = asks[0][0] if asks else 0
            spread_pct = (best_ask - best_bid) / best_bid * 100 if best_bid > 0 else 0

            # Yakın duvar analizi (en büyük bid/ask'ın fiyata mesafesi)
            largest_bid_price = bid_walls[0][0] if bid_walls else 0
            largest_ask_price = ask_walls[0][0] if ask_walls else 0

            # Skor: imbalance doğrudan skor olarak kullan
            score = float(np.clip(imbalance * 1.5, -1, 1)) if True else 0.0
            # numpy import olmayabilir, basit clip
            score = max(-1.0, min(1.0, imbalance * 1.5))
            confidence = min(100, abs(imbalance) * 200)

            return AnalysisResult(
                enabled=True, success=True,
                score=round(score, 4),
                confidence=round(confidence, 2),
                data={
                    "bid_volume": round(bid_vol, 4),
                    "ask_volume": round(ask_vol, 4),
                    "imbalance": round(imbalance, 4),
                    "spread_pct": round(spread_pct, 4),
                    "largest_bid": {"price": largest_bid_price, "size": bid_walls[0][1] if bid_walls else 0},
                    "largest_ask": {"price": largest_ask_price, "size": ask_walls[0][1] if ask_walls else 0},
                },
                reason=f"OB Imbalance={imbalance:+.3f} spread={spread_pct:.3f}%",
            )
        except Exception as e:
            logger.error(f"OB analiz hatası ({symbol}): {e}")
            return AnalysisResult.error(str(e))


# ─────────────────────────────────────────────
# OPEN INTEREST ANALİZİ
# ─────────────────────────────────────────────

class OpenInterestAnalyzer:
    """
    Açık Pozisyon (OI) analizi — Binance ücretsiz.
    OI artışı + fiyat artışı → güçlü trend
    OI artışı + fiyat düşüşü → kısa sıkışması riski
    OI azalışı → pozisyon kapatma (zayıflayan trend)
    """

    def __init__(self, data_engine, enabled: bool = True):
        self.data = data_engine
        self.enabled = enabled
        self._cache: Dict[str, dict] = {}
        self._cache_ttl = 300  # 5 dakika

    def get_oi_history(self, symbol: str, period: str = "5m", limit: int = 20) -> List[dict]:
        """Binance OI geçmişini çeker."""
        try:
            client = self.data._get_client()
            data = client.futures_open_interest_hist(
                symbol=symbol, period=period, limit=limit
            )
            return data
        except Exception as e:
            logger.debug(f"OI geçmiş hatası ({symbol}): {e}")
            return []

    def analyze(self, symbol: str, current_price: float) -> AnalysisResult:
        if not self.enabled:
            return AnalysisResult.neutral("OI analizi kapalı")
        try:
            # Anlık OI
            client = self.data._get_client()
            oi_raw = client.futures_open_interest(symbol=symbol)
            current_oi = float(oi_raw.get("openInterest", 0))

            # OI geçmişi
            history = self.get_oi_history(symbol, "5m", 20)
            if len(history) < 3:
                return AnalysisResult.error("OI geçmişi yetersiz")

            oi_values = [float(h.get("sumOpenInterest", 0)) for h in history]
            oi_avg    = sum(oi_values) / len(oi_values) if oi_values else 0
            oi_prev   = oi_values[-2] if len(oi_values) >= 2 else current_oi

            # OI değişim oranı
            oi_change_pct = (current_oi - oi_prev) / oi_prev * 100 if oi_prev > 0 else 0

            # Fiyat değişimi (yaklaşık — son kapanış kullan)
            candles = self.data.kline_cache.get(symbol, "5m")
            price_change = 0.0
            if len(candles) >= 2:
                price_change = (candles[-1]["close"] - candles[-2]["close"]) / candles[-2]["close"] * 100

            # Sinyal yorumu
            oi_rising  = oi_change_pct > 0.5
            oi_falling = oi_change_pct < -0.5
            price_up   = price_change > 0.1
            price_down = price_change < -0.1

            if oi_rising and price_up:
                score, reason = 0.7, "OI↑ + Fiyat↑ → güçlü LONG trend"
            elif oi_rising and price_down:
                score, reason = -0.8, "OI↑ + Fiyat↓ → SHORT baskısı / uzun sıkışması"
            elif oi_falling and price_up:
                score, reason = 0.3, "OI↓ + Fiyat↑ → short kapanışı (kısa vadeli)"
            elif oi_falling and price_down:
                score, reason = -0.5, "OI↓ + Fiyat↓ → LONG kapatma / trend zayıflıyor"
            else:
                score, reason = 0.0, f"OI değişimi nötr ({oi_change_pct:+.2f}%)"

            return AnalysisResult(
                enabled=True, success=True,
                score=score, confidence=70.0,
                data={
                    "current_oi": round(current_oi, 2),
                    "oi_change_pct": round(oi_change_pct, 4),
                    "oi_avg": round(oi_avg, 2),
                    "price_change_pct": round(price_change, 4),
                },
                reason=reason,
            )
        except Exception as e:
            logger.debug(f"OI analiz hatası ({symbol}): {e}")
            return AnalysisResult.error(str(e))


# ─────────────────────────────────────────────
# FEAR & GREED INDEX (Ücretsiz)
# ─────────────────────────────────────────────

class FearGreedIndex:
    """
    Alternative.me Fear & Greed Index — tamamen ücretsiz.
    0-24   = Extreme Fear    → satın alma fırsatı
    25-49  = Fear
    50-74  = Greed
    75-100 = Extreme Greed  → dikkatli ol, düşüş gelebilir
    """

    API_URL = "https://api.alternative.me/fng/?limit=2"

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._cache: Optional[dict] = None
        self._cached_at: Optional[float] = None
        self._cache_ttl = 3600  # 1 saat

    def fetch(self) -> Optional[dict]:
        if self._cache and self._cached_at and (time.time() - self._cached_at) < self._cache_ttl:
            return self._cache
        try:
            resp = requests.get(self.API_URL, timeout=10)
            resp.raise_for_status()
            data = resp.json().get("data", [{}])[0]
            result = {
                "value": int(data.get("value", 50)),
                "label": data.get("value_classification", "Neutral"),
                "timestamp": data.get("timestamp"),
            }
            self._cache = result
            self._cached_at = time.time()
            return result
        except Exception as e:
            logger.debug(f"Fear&Greed fetch hatası: {e}")
            return None

    def analyze(self, direction: str) -> AnalysisResult:
        if not self.enabled:
            return AnalysisResult.neutral("Fear&Greed kapalı")
        data = self.fetch()
        if not data:
            return AnalysisResult.error("Fear&Greed verisi alınamadı")

        v = data["value"]
        label = data["label"]

        # Kontrarian yaklaşım: Extreme Fear = LONG fırsatı, Extreme Greed = dikkat
        if v <= 20:    score_raw = 0.8   # Extreme Fear → al
        elif v <= 40:  score_raw = 0.3   # Fear
        elif v <= 60:  score_raw = 0.0   # Neutral
        elif v <= 80:  score_raw = -0.3  # Greed → dikkat
        else:          score_raw = -0.7  # Extreme Greed → sat

        # LONG için: düşük fear skoru iyi (kontrarian)
        # SHORT için: yüksek greed iyi
        if direction == "SHORT":
            score_raw = -score_raw

        # Extreme değerlerde işlem engeli (opsiyonel)
        blocking = v > 90 or v < 10

        return AnalysisResult(
            enabled=True, success=True,
            score=round(score_raw, 3),
            confidence=60.0,
            data={"value": v, "label": label, "blocking": blocking},
            reason=f"F&G={v} ({label}) {'⚠️EXTREME' if blocking else ''}",
        )


# ─────────────────────────────────────────────
# GLASSNODE ON-CHAIN (Opsiyonel — API Key gerekir)
# ─────────────────────────────────────────────

class GlassnodeAnalyzer:
    """
    Glassnode On-Chain Analizi (opsiyonel, ücretsiz tier mevcut).
    API key: glassnode.com → ücretsiz kayıt.
    Ücretsiz tier: SOPR, NUPL, exchange flow gibi temel metrikler.
    """

    BASE_URL = "https://api.glassnode.com/v1/metrics"

    def __init__(self, api_key: str = "", enabled: bool = False):
        self.api_key = api_key
        self.enabled = enabled and bool(api_key)
        self._cache: Dict[str, dict] = {}

    def _get(self, endpoint: str, asset: str = "BTC", resolution: str = "24h") -> Optional[list]:
        cache_key = f"{endpoint}_{asset}"
        if cache_key in self._cache:
            if time.time() - self._cache[cache_key]["t"] < 3600:
                return self._cache[cache_key]["data"]
        try:
            url = f"{self.BASE_URL}/{endpoint}"
            resp = requests.get(url, params={
                "a": asset, "api_key": self.api_key, "i": resolution
            }, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                self._cache[cache_key] = {"data": data, "t": time.time()}
                return data
        except Exception as e:
            logger.debug(f"Glassnode API hatası ({endpoint}): {e}")
        return None

    def get_sopr(self) -> Optional[float]:
        """SOPR: >1 kârlı satışlar, <1 zararlı satışlar."""
        data = self._get("indicators/sopr")
        if data and len(data) > 0:
            return float(data[-1].get("v", 1.0))
        return None

    def get_exchange_flow(self) -> Optional[dict]:
        """Borsaya giren/çıkan BTC miktarı."""
        inflow  = self._get("transactions/transfers_volume_to_exchanges_sum")
        outflow = self._get("transactions/transfers_volume_from_exchanges_sum")
        if inflow and outflow:
            return {
                "inflow":  float(inflow[-1].get("v", 0)),
                "outflow": float(outflow[-1].get("v", 0)),
            }
        return None

    def get_nupl(self) -> Optional[float]:
        """NUPL: Net Unrealized Profit/Loss (-1 ile +1)."""
        data = self._get("indicators/nupl")
        if data and len(data) > 0:
            return float(data[-1].get("v", 0))
        return None

    def analyze(self, symbol: str) -> AnalysisResult:
        if not self.enabled:
            return AnalysisResult.neutral("Glassnode kapalı (API key yok)")

        # Sadece BTC için anlamlı
        asset = symbol.replace("USDT", "").replace("BUSD", "")
        if asset not in ("BTC", "ETH"):
            return AnalysisResult.neutral(f"Glassnode {asset} desteklemiyor")

        try:
            score_sum, count = 0.0, 0
            details = {}

            # SOPR
            sopr = self.get_sopr()
            if sopr:
                details["sopr"] = sopr
                # SOPR > 1: kârlı satışlar baskılıyor → ayı sinyali
                # SOPR < 0.99: zararına satış → dip yakın
                if sopr > 1.05:   score_sum -= 0.4
                elif sopr < 0.99: score_sum += 0.4
                count += 1

            # Exchange Flow
            flow = self.get_exchange_flow()
            if flow:
                details["exchange_flow"] = flow
                net_flow = flow["inflow"] - flow["outflow"]
                # Net pozitif akış (borsaya giriş) → satış baskısı
                if net_flow > 0: score_sum -= 0.3
                else:            score_sum += 0.3
                count += 1

            # NUPL
            nupl = self.get_nupl()
            if nupl is not None:
                details["nupl"] = nupl
                # NUPL > 0.75: Euphoria → düşüş riski
                # NUPL < 0: Capitulation → dip
                if nupl > 0.75:   score_sum -= 0.5
                elif nupl < 0:    score_sum += 0.5
                elif nupl > 0.5:  score_sum -= 0.2
                count += 1

            if count == 0:
                return AnalysisResult.error("Glassnode metrikleri alınamadı")

            final_score = score_sum / count
            return AnalysisResult(
                enabled=True, success=True,
                score=round(final_score, 4),
                confidence=65.0,
                data=details,
                reason=f"On-chain skor={final_score:+.3f} ({count} metrik)",
            )
        except Exception as e:
            logger.error(f"Glassnode analiz hatası: {e}")
            return AnalysisResult.error(str(e))


# ─────────────────────────────────────────────
# CRYPTOQUANT (Opsiyonel — API Key gerekir)
# ─────────────────────────────────────────────

class CryptoQuantAnalyzer:
    """
    CryptoQuant On-Chain/Derivatives Analizi (opsiyonel).
    API key: cryptoquant.com → temel ücretsiz tier.
    """

    BASE_URL = "https://api.cryptoquant.com/v1"

    def __init__(self, api_key: str = "", enabled: bool = False):
        self.api_key = api_key
        self.enabled = enabled and bool(api_key)

    def _get(self, path: str, params: dict = None) -> Optional[dict]:
        try:
            resp = requests.get(
                f"{self.BASE_URL}/{path}",
                params=params or {},
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=15,
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.debug(f"CryptoQuant API hatası ({path}): {e}")
        return None

    def get_fund_flow_ratio(self, symbol: str = "btc") -> Optional[float]:
        """Exchange fund flow ratio — yüksekse satış baskısı."""
        data = self._get(f"btc/exchange-flows/fund-flow-ratio", {"window": "day", "limit": 1})
        if data and "result" in data:
            items = data["result"].get("data", [])
            if items:
                return float(items[-1].get("fundFlowRatio", 0))
        return None

    def analyze(self, symbol: str) -> AnalysisResult:
        if not self.enabled:
            return AnalysisResult.neutral("CryptoQuant kapalı (API key yok)")
        try:
            asset = symbol.replace("USDT", "").lower()
            score = 0.0
            details = {}

            ffr = self.get_fund_flow_ratio(asset)
            if ffr is not None:
                details["fund_flow_ratio"] = ffr
                # Yüksek FFR = borsaya para girişi = satış baskısı
                if ffr > 0.1:   score -= 0.4
                elif ffr < 0.0: score += 0.3

            return AnalysisResult(
                enabled=True, success=True,
                score=round(score, 4), confidence=60.0,
                data=details,
                reason=f"CQ FFR={ffr:.4f}" if ffr else "CQ veri yok",
            )
        except Exception as e:
            return AnalysisResult.error(str(e))


# ─────────────────────────────────────────────
# WHALE ANALİZİ (Büyük İşlemler)
# ─────────────────────────────────────────────

class WhaleAnalyzer:
    """
    Büyük işlem (whale) tespiti.
    Binance Aggr. Trades endpoint'inden büyük işlemleri filtreler.
    API key gerektirmez.
    """

    def __init__(self, data_engine, enabled: bool = True, min_usdt: float = 500_000):
        self.data = data_engine
        self.enabled = enabled
        self.min_usdt = min_usdt  # Minimum whale işlem büyüklüğü

    def analyze(self, symbol: str, lookback_minutes: int = 15) -> AnalysisResult:
        if not self.enabled:
            return AnalysisResult.neutral("Whale analizi kapalı")
        try:
            client = self.data._get_client()
            # Son N dakika için aggr. trades
            since = int((datetime.utcnow() - timedelta(minutes=lookback_minutes)).timestamp() * 1000)
            trades = client.futures_aggregate_trades(symbol=symbol, startTime=since)

            current_price = self.data.price_cache.get(symbol) or 0
            if not current_price:
                return AnalysisResult.error("Fiyat verisi yok")

            buy_vol = sell_vol = 0.0
            whale_buys = whale_sells = 0

            for t in trades:
                qty   = float(t.get("q", 0))
                price = float(t.get("p", 0))
                notional = qty * price
                is_sell   = t.get("m", False)  # maker side sell

                if notional >= self.min_usdt:
                    if is_sell:
                        sell_vol += notional; whale_sells += 1
                    else:
                        buy_vol  += notional; whale_buys  += 1

            total = buy_vol + sell_vol
            if total == 0:
                return AnalysisResult(
                    enabled=True, success=True, score=0.0, confidence=30.0,
                    data={"whale_buys": 0, "whale_sells": 0},
                    reason=f"Son {lookback_minutes}dk whale işlem yok",
                )

            delta    = (buy_vol - sell_vol) / total
            score    = round(max(-1.0, min(1.0, delta * 1.5)), 4)
            conf     = min(100, (whale_buys + whale_sells) * 20)

            return AnalysisResult(
                enabled=True, success=True, score=score, confidence=conf,
                data={
                    "whale_buy_usd":  round(buy_vol),
                    "whale_sell_usd": round(sell_vol),
                    "whale_buy_count":  whale_buys,
                    "whale_sell_count": whale_sells,
                    "delta": round(delta, 4),
                },
                reason=f"Whale: {whale_buys}alış/{whale_sells}satış delta={delta:+.3f}",
            )
        except Exception as e:
            logger.debug(f"Whale analiz hatası ({symbol}): {e}")
            return AnalysisResult.error(str(e))


# ─────────────────────────────────────────────
# GELİŞMİŞ ANALİZ YÖNETİCİSİ
# ─────────────────────────────────────────────

class AdvancedAnalysisManager:
    """
    Tüm gelişmiş analiz modüllerini yönetir.
    Her modül opsiyonel — enabled=False ise atlanır.
    """

    def __init__(self, data_engine, settings):
        cfg = settings  # settings nesnesi

        adv = getattr(cfg, "advanced", None)

        self.ob_analyzer  = OrderBookAnalyzer(data_engine,
            enabled=getattr(adv, "ob_enabled", True))
        self.oi_analyzer  = OpenInterestAnalyzer(data_engine,
            enabled=getattr(adv, "oi_enabled", True))
        self.fg_index     = FearGreedIndex(
            enabled=getattr(adv, "fear_greed_enabled", True))
        self.glassnode    = GlassnodeAnalyzer(
            api_key=getattr(adv, "glassnode_api_key", ""),
            enabled=getattr(adv, "glassnode_enabled", False))
        self.cryptoquant  = CryptoQuantAnalyzer(
            api_key=getattr(adv, "cryptoquant_api_key", ""),
            enabled=getattr(adv, "cryptoquant_enabled", False))
        self.whale        = WhaleAnalyzer(data_engine,
            enabled=getattr(adv, "whale_enabled", True),
            min_usdt=getattr(adv, "whale_min_usdt", 500_000))

        logger.info("AdvancedAnalysisManager başlatıldı.")

    def run_all(self, symbol: str, direction: str, current_price: float) -> Dict[str, AnalysisResult]:
        """Tüm modülleri çalıştırır, sonuçları döndürür."""
        return {
            "order_book":  self.ob_analyzer.analyze(symbol),
            "open_interest": self.oi_analyzer.analyze(symbol, current_price),
            "fear_greed":  self.fg_index.analyze(direction),
            "glassnode":   self.glassnode.analyze(symbol),
            "cryptoquant": self.cryptoquant.analyze(symbol),
            "whale":       self.whale.analyze(symbol),
        }

    def aggregate_score(self, results: Dict[str, AnalysisResult]) -> Tuple[float, float, str]:
        """
        Tüm modül sonuçlarını ağırlıklı ortalama ile birleştirir.
        Returns: (birleşik_skor, güven, açıklama)
        """
        # Ağırlıklar
        weights = {
            "order_book":    2.0,
            "open_interest": 2.5,
            "fear_greed":    1.0,
            "glassnode":     1.5,
            "cryptoquant":   1.5,
            "whale":         2.0,
        }

        total_w = total_score = total_conf = 0.0
        parts = []

        for key, result in results.items():
            if not result.success:
                continue
            w = weights.get(key, 1.0)
            total_score += result.score * w
            total_conf  += result.confidence * w
            total_w     += w
            parts.append(f"{key}={result.score:+.2f}")

        if total_w == 0:
            return 0.0, 0.0, "Hiç modül sonuç üretemedi"

        agg_score = total_score / total_w
        agg_conf  = total_conf / total_w

        return round(agg_score, 4), round(agg_conf, 2), " | ".join(parts)
