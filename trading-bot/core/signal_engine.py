"""
core/signal_engine.py
3 katmanlı sinyal üretme motoru.
- Katman 1: EMA trend tespiti
- Katman 2: RSI + MACD momentum
- Katman 3: Hacim analizi + Funding rate
Her katmanın onayı zorunlu (3/3 konsensüs).
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import SignalConfig, SymbolConfig
from core.data_engine import DataEngine
from utils.logger import get_logger

logger = get_logger(__name__)


class SignalDirection(Enum):
    """İşlem yönü."""
    LONG = "LONG"
    SHORT = "SHORT"
    NONE = "NONE"


class SignalStrength(Enum):
    """Sinyal gücü."""
    WEAK = "WEAK"
    MEDIUM = "MEDIUM"
    STRONG = "STRONG"


@dataclass
class LayerResult:
    """Tek katman sonucu."""
    approved: bool
    direction: SignalDirection
    reason: str
    details: dict = field(default_factory=dict)


@dataclass
class Signal:
    """Nihai sinyal."""
    symbol: str
    direction: SignalDirection
    strength: SignalStrength
    # Katman sonuçları
    layer1: LayerResult = None
    layer2: LayerResult = None
    layer3: LayerResult = None
    # Meta bilgiler
    timestamp: datetime = field(default_factory=datetime.utcnow)
    signal_price: float = 0.0
    spread_pct: float = 0.0
    funding_rate: float = 0.0
    # Onaylı mı?
    approved: bool = False
    reject_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "direction": self.direction.value,
            "strength": self.strength.value,
            "approved": self.approved,
            "reject_reason": self.reject_reason,
            "signal_price": self.signal_price,
            "spread_pct": self.spread_pct,
            "funding_rate": self.funding_rate,
            "timestamp": self.timestamp.isoformat(),
            "layer1": {
                "approved": self.layer1.approved,
                "direction": self.layer1.direction.value if self.layer1 else None,
                "reason": self.layer1.reason if self.layer1 else None,
                "details": self.layer1.details if self.layer1 else {},
            } if self.layer1 else None,
            "layer2": {
                "approved": self.layer2.approved,
                "direction": self.layer2.direction.value if self.layer2 else None,
                "reason": self.layer2.reason if self.layer2 else None,
                "details": self.layer2.details if self.layer2 else {},
            } if self.layer2 else None,
            "layer3": {
                "approved": self.layer3.approved,
                "direction": self.layer3.direction.value if self.layer3 else None,
                "reason": self.layer3.reason if self.layer3 else None,
                "details": self.layer3.details if self.layer3 else {},
            } if self.layer3 else None,
        }


# ============================================================
# Teknik Analiz Yardımcıları
# ============================================================

def _candles_to_df(candles: List[dict]) -> pd.DataFrame:
    """Mum listesini DataFrame'e dönüştürür."""
    if not candles:
        return pd.DataFrame()
    df = pd.DataFrame(candles)
    df["close"] = df["close"].astype(float)
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["volume"] = df["volume"].astype(float)
    return df


def _ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average hesaplar."""
    return series.ewm(span=period, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index hesaplar."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """MACD, sinyal ve histogram döndürür."""
    ema_fast = _ema(series, fast)
    ema_slow = _ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


# ============================================================
# Katman 1: EMA Trend Tespiti
# ============================================================

class Layer1EMA:
    """
    EMA 21 ve EMA 50 ile trend tespiti.
    - EMA21 > EMA50: Yükselen trend (LONG sinyal)
    - EMA21 < EMA50: Düşen trend (SHORT sinyal)
    - Fiyat trend yönünde pozisyonlanmış olmalı
    - 15dk trend onayı da alınır
    """

    def __init__(self, config: SignalConfig):
        self.cfg = config

    def analyze(
        self,
        signal_candles: List[dict],
        trend_candles: List[dict],
    ) -> LayerResult:
        """
        Trend analizini yapar.
        signal_candles: 5dk mumlar (sinyal zaman dilimi)
        trend_candles: 15dk mumlar (trend onay zaman dilimi)
        """
        if len(signal_candles) < self.cfg.ema_slow + 5:
            return LayerResult(
                approved=False,
                direction=SignalDirection.NONE,
                reason=f"Yetersiz veri: {len(signal_candles)} mum (min {self.cfg.ema_slow + 5} gerekli)",
            )

        # Sinyal zaman dilimi analizi
        sig_df = _candles_to_df(signal_candles)
        sig_ema_fast = _ema(sig_df["close"], self.cfg.ema_fast)
        sig_ema_slow = _ema(sig_df["close"], self.cfg.ema_slow)

        last_ema_fast = sig_ema_fast.iloc[-1]
        last_ema_slow = sig_ema_slow.iloc[-1]
        last_close = sig_df["close"].iloc[-1]

        # EMA kesişimi tespiti (son 2 mum)
        prev_ema_fast = sig_ema_fast.iloc[-2]
        prev_ema_slow = sig_ema_slow.iloc[-2]

        # Trend zaman dilimi onayı
        trend_direction = SignalDirection.NONE
        if len(trend_candles) >= self.cfg.ema_slow:
            tr_df = _candles_to_df(trend_candles)
            tr_ema_fast = _ema(tr_df["close"], self.cfg.ema_fast)
            tr_ema_slow = _ema(tr_df["close"], self.cfg.ema_slow)
            if tr_ema_fast.iloc[-1] > tr_ema_slow.iloc[-1]:
                trend_direction = SignalDirection.LONG
            elif tr_ema_fast.iloc[-1] < tr_ema_slow.iloc[-1]:
                trend_direction = SignalDirection.SHORT

        # Sinyal yönünü belirle
        if last_ema_fast > last_ema_slow:
            signal_direction = SignalDirection.LONG
            # Fiyat EMA'nın üzerinde mi?
            price_above_ema = last_close > last_ema_fast
        elif last_ema_fast < last_ema_slow:
            signal_direction = SignalDirection.SHORT
            price_above_ema = last_close < last_ema_fast
        else:
            return LayerResult(
                approved=False,
                direction=SignalDirection.NONE,
                reason="EMA'lar eşit - belirsiz trend",
            )

        # Trend onayı kontrolü
        trend_confirmed = (
            trend_direction == SignalDirection.NONE
            or trend_direction == signal_direction
        )

        # EMA ayrışma gücü (%)
        ema_separation = abs(last_ema_fast - last_ema_slow) / last_ema_slow * 100

        approved = price_above_ema and trend_confirmed

        return LayerResult(
            approved=approved,
            direction=signal_direction,
            reason=(
                f"EMA{self.cfg.ema_fast}={'%.2f' % last_ema_fast} "
                f"{'>' if last_ema_fast > last_ema_slow else '<'} "
                f"EMA{self.cfg.ema_slow}={'%.2f' % last_ema_slow}, "
                f"Trend onayı: {'✓' if trend_confirmed else '✗'}, "
                f"Fiyat EMA yönünde: {'✓' if price_above_ema else '✗'}"
            ),
            details={
                "ema_fast": round(last_ema_fast, 4),
                "ema_slow": round(last_ema_slow, 4),
                "ema_separation_pct": round(ema_separation, 4),
                "price_above_ema": price_above_ema,
                "trend_direction": trend_direction.value,
                "trend_confirmed": trend_confirmed,
            },
        )


# ============================================================
# Katman 2: RSI + MACD Momentum
# ============================================================

class Layer2Momentum:
    """
    RSI(14) ve MACD momentum analizi.
    - LONG: RSI < aşırı satım değilse (30-70) ve MACD pozitif
    - SHORT: RSI > aşırı alım değilse (30-70) ve MACD negatif
    - Divergence algılama (bonus güç)
    """

    def __init__(self, config: SignalConfig):
        self.cfg = config

    def analyze(
        self, candles: List[dict], direction: SignalDirection
    ) -> LayerResult:
        """Momentum analizini yapar."""
        if len(candles) < self.cfg.macd_slow + self.cfg.macd_signal + 5:
            return LayerResult(
                approved=False,
                direction=SignalDirection.NONE,
                reason="MACD için yetersiz veri",
            )

        df = _candles_to_df(candles)
        closes = df["close"]

        # RSI
        rsi = _rsi(closes, self.cfg.rsi_period)
        last_rsi = rsi.iloc[-1]
        prev_rsi = rsi.iloc[-2]

        # MACD
        macd_line, signal_line, histogram = _macd(
            closes,
            self.cfg.macd_fast,
            self.cfg.macd_slow,
            self.cfg.macd_signal,
        )
        last_macd = macd_line.iloc[-1]
        last_signal = signal_line.iloc[-1]
        last_hist = histogram.iloc[-1]
        prev_hist = histogram.iloc[-2]

        # MACD kesişimi tespiti
        macd_bullish_cross = prev_hist < 0 and last_hist > 0  # Negatiften pozitife
        macd_bearish_cross = prev_hist > 0 and last_hist < 0  # Pozitiften negatife

        # LONG onay koşulları
        if direction == SignalDirection.LONG:
            rsi_ok = self.cfg.rsi_oversold < last_rsi < self.cfg.rsi_overbought
            macd_ok = last_hist > 0 or macd_bullish_cross
            rsi_trending_up = last_rsi > prev_rsi
            approved = rsi_ok and macd_ok
            reject_reason = []
            if not rsi_ok:
                if last_rsi <= self.cfg.rsi_oversold:
                    reject_reason.append(f"RSI aşırı satımda ({last_rsi:.1f})")
                else:
                    reject_reason.append(f"RSI aşırı alımda ({last_rsi:.1f})")
            if not macd_ok:
                reject_reason.append(f"MACD negatif ({last_hist:.4f})")

        # SHORT onay koşulları
        elif direction == SignalDirection.SHORT:
            rsi_ok = self.cfg.rsi_oversold < last_rsi < self.cfg.rsi_overbought
            macd_ok = last_hist < 0 or macd_bearish_cross
            rsi_trending_up = last_rsi < prev_rsi
            approved = rsi_ok and macd_ok
            reject_reason = []
            if not rsi_ok:
                reject_reason.append(f"RSI uygun değil ({last_rsi:.1f})")
            if not macd_ok:
                reject_reason.append(f"MACD pozitif ({last_hist:.4f})")

        else:
            return LayerResult(
                approved=False,
                direction=SignalDirection.NONE,
                reason="Yön belirsiz - Katman 1 onayı yok",
            )

        return LayerResult(
            approved=approved,
            direction=direction,
            reason=(
                f"RSI={last_rsi:.1f} ({'✓' if rsi_ok else '✗'}), "
                f"MACD_hist={last_hist:.4f} ({'✓' if macd_ok else '✗'})"
                + (f" | Ret: {', '.join(reject_reason)}" if reject_reason else "")
            ),
            details={
                "rsi": round(last_rsi, 2),
                "rsi_previous": round(prev_rsi, 2),
                "macd_line": round(last_macd, 4),
                "macd_signal": round(last_signal, 4),
                "macd_histogram": round(last_hist, 4),
                "macd_bullish_cross": macd_bullish_cross,
                "macd_bearish_cross": macd_bearish_cross,
            },
        )


# ============================================================
# Katman 3: Hacim + Funding Rate
# ============================================================

class Layer3Volume:
    """
    Hacim analizi ve funding rate kontrolü.
    - Hacim, N mumun ortalamasının üzerinde olmalı (threshold)
    - Taker buy/sell oranı yönü desteklemeli
    - Funding rate sınır içinde olmalı
    """

    def __init__(self, config: SignalConfig, max_funding_rate: float = 0.1):
        self.cfg = config
        self.max_funding_rate = max_funding_rate

    def analyze(
        self,
        candles: List[dict],
        direction: SignalDirection,
        funding_rate: float,
    ) -> LayerResult:
        """Hacim ve funding rate analizini yapar."""
        if len(candles) < self.cfg.volume_lookback + 2:
            return LayerResult(
                approved=False,
                direction=SignalDirection.NONE,
                reason=f"Hacim analizi için yetersiz veri",
            )

        df = _candles_to_df(candles)

        # Hacim ortalaması (son N mum)
        avg_volume = df["volume"].iloc[-(self.cfg.volume_lookback + 1):-1].mean()
        last_volume = df["volume"].iloc[-1]

        volume_ratio = last_volume / avg_volume if avg_volume > 0 else 0
        volume_ok = volume_ratio >= self.cfg.volume_threshold

        # Taker buy/sell oranı (son mum)
        last_taker_buy = df["taker_buy_volume"].iloc[-1]
        taker_sell = last_volume - last_taker_buy
        taker_ratio = last_taker_buy / last_volume if last_volume > 0 else 0.5

        # LONG: Alıcı baskısı yüksek olmalı (taker_ratio > 0.5)
        # SHORT: Satıcı baskısı yüksek olmalı (taker_ratio < 0.5)
        if direction == SignalDirection.LONG:
            taker_ok = taker_ratio > 0.5
        elif direction == SignalDirection.SHORT:
            taker_ok = taker_ratio < 0.5
        else:
            taker_ok = False

        # Funding rate kontrolü
        # Pozitif funding rate: long pozisyon tutucular ödeme yapar
        # Negatif funding rate: short pozisyon tutucular ödeme yapar
        funding_abs = abs(funding_rate)
        funding_ok = funding_abs <= self.max_funding_rate

        # Funding rate yönü kontrolü (tersi yönde yüksek funding dezavantaj)
        funding_against = False
        if direction == SignalDirection.LONG and funding_rate > self.max_funding_rate * 0.5:
            funding_against = True
        elif direction == SignalDirection.SHORT and funding_rate < -self.max_funding_rate * 0.5:
            funding_against = True

        # Onay: hacim yeterliyse taker oranı yönü desteklemiyor olsa bile geç
        # Ama funding limit aşıldıysa ret
        approved = volume_ok and funding_ok
        # Taker oranı zayıf bir indikatör olarak kabul et (bonus)

        reject_reasons = []
        if not volume_ok:
            reject_reasons.append(
                f"Hacim yetersiz ({volume_ratio:.2f}x < {self.cfg.volume_threshold}x)"
            )
        if not funding_ok:
            reject_reasons.append(
                f"Funding rate yüksek ({funding_rate:.4f}% > {self.max_funding_rate}%)"
            )

        return LayerResult(
            approved=approved,
            direction=direction,
            reason=(
                f"Hacim={volume_ratio:.2f}x ({'✓' if volume_ok else '✗'}), "
                f"Taker={taker_ratio:.2f} ({'✓' if taker_ok else 'zayıf'}), "
                f"Funding={funding_rate:.4f}% ({'✓' if funding_ok else '✗'})"
                + (f" | Ret: {', '.join(reject_reasons)}" if reject_reasons else "")
            ),
            details={
                "volume_ratio": round(volume_ratio, 3),
                "volume_ok": volume_ok,
                "taker_ratio": round(taker_ratio, 3),
                "taker_ok": taker_ok,
                "funding_rate": round(funding_rate, 6),
                "funding_ok": funding_ok,
                "funding_against": funding_against,
            },
        )


# ============================================================
# Sinyal Motoru
# ============================================================

class SignalEngine:
    """
    3 katmanlı sinyal üretme motoru.
    Her parite için bağımsız sinyal üretir.
    """

    def __init__(
        self,
        data_engine: DataEngine,
        signal_config: SignalConfig,
        max_funding_rate: float = 0.1,
    ):
        self.data = data_engine
        self.cfg = signal_config
        self.max_funding_rate = max_funding_rate

        # Katmanlar
        self.layer1 = Layer1EMA(signal_config)
        self.layer2 = Layer2Momentum(signal_config)
        self.layer3 = Layer3Volume(signal_config, max_funding_rate)

        # Son sinyal önbelleği (her parite için)
        self._last_signals: Dict[str, Signal] = {}

        # Sinyal geçmişi (son 50)
        self._signal_history: List[Signal] = []

        # Timeframe string dönüşümleri
        self._interval_map = {
            1: "1m", 3: "3m", 5: "5m", 15: "15m",
            30: "30m", 60: "1h", 120: "2h", 240: "4h",
        }

        logger.info("SignalEngine başlatıldı.")

    def _get_interval(self, minutes: int) -> str:
        """Dakikayı Binance interval formatına çevirir."""
        return self._interval_map.get(minutes, f"{minutes}m")

    def analyze(self, symbol: str) -> Signal:
        """
        Belirtilen parite için sinyal analizi yapar.
        Tüm 3 katmanı çalıştırır ve konsensüs arar.
        """
        signal_interval = self._get_interval(self.cfg.signal_timeframe)
        trend_interval = self._get_interval(self.cfg.trend_timeframe)

        # Mum verisi al
        signal_candles = self.data.kline_cache.get(symbol, signal_interval)
        trend_candles = self.data.kline_cache.get(symbol, trend_interval)

        # Güncel fiyat ve funding rate
        current_price = self.data.price_cache.get(symbol) or 0.0
        funding_rate = self.data.get_funding_rate(symbol)

        # Spread kontrolü
        spread_pct = self.data.get_spread(symbol)

        # Temel sinyal nesnesi
        signal = Signal(
            symbol=symbol,
            direction=SignalDirection.NONE,
            strength=SignalStrength.WEAK,
            signal_price=current_price,
            spread_pct=spread_pct,
            funding_rate=funding_rate,
        )

        # Spread limiti kontrolü
        if spread_pct > self.cfg.max_spread_pct:
            signal.approved = False
            signal.reject_reason = (
                f"Spread çok yüksek: %{spread_pct:.3f} > %{self.cfg.max_spread_pct}"
            )
            self._cache_signal(symbol, signal)
            return signal

        # Yeterli mum verisi var mı?
        if len(signal_candles) < self.cfg.ema_slow + 10:
            signal.approved = False
            signal.reject_reason = f"Yetersiz veri: {len(signal_candles)} mum"
            self._cache_signal(symbol, signal)
            return signal

        # ---- Katman 1: EMA Trend ----
        l1 = self.layer1.analyze(signal_candles, trend_candles)
        signal.layer1 = l1

        if not l1.approved:
            signal.approved = False
            signal.reject_reason = f"Katman 1 (EMA Trend) onaylamadı: {l1.reason}"
            self._cache_signal(symbol, signal)
            return signal

        direction = l1.direction
        signal.direction = direction

        # ---- Katman 2: RSI + MACD ----
        l2 = self.layer2.analyze(signal_candles, direction)
        signal.layer2 = l2

        if not l2.approved:
            signal.approved = False
            signal.reject_reason = f"Katman 2 (Momentum) onaylamadı: {l2.reason}"
            self._cache_signal(symbol, signal)
            return signal

        # ---- Katman 3: Hacim + Funding ----
        l3 = self.layer3.analyze(signal_candles, direction, funding_rate)
        signal.layer3 = l3

        if not l3.approved:
            signal.approved = False
            signal.reject_reason = f"Katman 3 (Hacim/Funding) onaylamadı: {l3.reason}"
            self._cache_signal(symbol, signal)
            return signal

        # ---- Tüm katmanlar onayladı! ----
        signal.approved = True

        # Sinyal gücü hesapla
        signal.strength = self._calculate_strength(l1, l2, l3)

        logger.info(
            f"[SINYAL] {symbol} {direction.value} | "
            f"Güç: {signal.strength.value} | "
            f"Fiyat: {current_price} | "
            f"Spread: {spread_pct:.3f}%"
        )

        self._cache_signal(symbol, signal)
        self._signal_history.append(signal)
        if len(self._signal_history) > 50:
            self._signal_history.pop(0)

        return signal

    def _calculate_strength(
        self,
        l1: LayerResult,
        l2: LayerResult,
        l3: LayerResult,
    ) -> SignalStrength:
        """Sinyal gücünü 3 katman detaylarına göre hesaplar."""
        score = 0

        # Katman 1: EMA ayrışma gücü
        ema_sep = l1.details.get("ema_separation_pct", 0)
        if ema_sep > 0.5:
            score += 2
        elif ema_sep > 0.2:
            score += 1

        # Trend onayı
        if l1.details.get("trend_confirmed"):
            score += 1

        # Katman 2: RSI orta bölgede
        rsi = l2.details.get("rsi", 50)
        if 40 <= rsi <= 60:
            score += 2
        elif 35 <= rsi <= 65:
            score += 1

        # MACD kesişimi (çok güçlü sinyal)
        if l2.details.get("macd_bullish_cross") or l2.details.get("macd_bearish_cross"):
            score += 2

        # Katman 3: Hacim oranı
        vol_ratio = l3.details.get("volume_ratio", 1.0)
        if vol_ratio >= 2.0:
            score += 2
        elif vol_ratio >= 1.5:
            score += 1

        # Taker oranı yönü destekliyor
        if l3.details.get("taker_ok"):
            score += 1

        # Funding rate düşük
        if abs(l3.details.get("funding_rate", 0)) < 0.05:
            score += 1

        if score >= 8:
            return SignalStrength.STRONG
        elif score >= 5:
            return SignalStrength.MEDIUM
        else:
            return SignalStrength.WEAK

    def _cache_signal(self, symbol: str, signal: Signal):
        """Sinyal önbelleğini günceller."""
        self._last_signals[symbol] = signal

    def get_last_signal(self, symbol: str) -> Optional[Signal]:
        """Son sinyal sonucunu döndürür."""
        return self._last_signals.get(symbol)

    def get_signal_history(self, limit: int = 20) -> List[dict]:
        """Son sinyallerin geçmişini döndürür."""
        return [s.to_dict() for s in self._signal_history[-limit:]]

    def analyze_all(self, symbols: List[str]) -> Dict[str, Signal]:
        """Tüm pariteler için sinyal analizi yapar."""
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.analyze(symbol)
            except Exception as e:
                logger.error(f"{symbol} sinyal analiz hatası: {e}")
        return results
