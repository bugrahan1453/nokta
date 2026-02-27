"""
core/signal_engine.py
5 katmanlı sinyal üretme motoru + Ensemble.
- Katman 1: EMA trend tespiti (zorunlu)
- Katman 2: RSI + MACD momentum (zorunlu)
- Katman 3: Hacim analizi + Funding rate (zorunlu)
- Katman 4: BB, ATR, StochRSI, VWAP, Patterns, Divergence (opsiyonel filtre)
- Katman 5: ML + Sentiment + On-Chain ensemble (tam opsiyonel)
Her katmanın onayı yapılandırılabilir. Varsayılan: 3/3 zorunlu konsensüs.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any

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
    score: float = 0.0  # -1 ile +1, ensemble için


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
    layer4: Optional[LayerResult] = None
    layer5: Optional[LayerResult] = None
    # Meta bilgiler
    timestamp: datetime = field(default_factory=datetime.utcnow)
    signal_price: float = 0.0
    spread_pct: float = 0.0
    funding_rate: float = 0.0
    # Onaylı mı?
    approved: bool = False
    reject_reason: str = ""
    # Ensemble skoru
    ensemble_score: float = 0.0
    ensemble_confidence: float = 0.0
    market_regime: str = "unknown"

    def to_dict(self) -> dict:
        def layer_dict(l):
            if not l:
                return None
            return {
                "approved": l.approved,
                "direction": l.direction.value,
                "reason": l.reason,
                "details": l.details,
                "score": round(l.score, 4),
            }

        return {
            "symbol": self.symbol,
            "direction": self.direction.value,
            "strength": self.strength.value,
            "approved": self.approved,
            "reject_reason": self.reject_reason,
            "signal_price": self.signal_price,
            "spread_pct": self.spread_pct,
            "funding_rate": self.funding_rate,
            "ensemble_score": round(self.ensemble_score, 4),
            "ensemble_confidence": round(self.ensemble_confidence, 2),
            "market_regime": self.market_regime,
            "timestamp": self.timestamp.isoformat(),
            "layer1": layer_dict(self.layer1),
            "layer2": layer_dict(self.layer2),
            "layer3": layer_dict(self.layer3),
            "layer4": layer_dict(self.layer4),
            "layer5": layer_dict(self.layer5),
        }


# ============================================================
# Teknik Analiz Yardımcıları
# ============================================================

def _candles_to_df(candles: List[dict]) -> pd.DataFrame:
    """Mum listesini DataFrame'e dönüştürür."""
    if not candles:
        return pd.DataFrame()
    df = pd.DataFrame(candles)
    for col in ["close", "open", "high", "low", "volume"]:
        df[col] = df[col].astype(float)
    if "taker_buy_volume" in df.columns:
        df["taker_buy_volume"] = df["taker_buy_volume"].astype(float)
    return df


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
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
    def __init__(self, config: SignalConfig):
        self.cfg = config

    def analyze(
        self,
        signal_candles: List[dict],
        trend_candles: List[dict],
    ) -> LayerResult:
        if len(signal_candles) < self.cfg.ema_slow + 5:
            return LayerResult(
                approved=False, direction=SignalDirection.NONE,
                reason=f"Yetersiz veri: {len(signal_candles)} mum",
            )

        sig_df = _candles_to_df(signal_candles)
        sig_ema_fast = _ema(sig_df["close"], self.cfg.ema_fast)
        sig_ema_slow = _ema(sig_df["close"], self.cfg.ema_slow)

        last_ema_fast = sig_ema_fast.iloc[-1]
        last_ema_slow = sig_ema_slow.iloc[-1]
        last_close = sig_df["close"].iloc[-1]

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

        if last_ema_fast > last_ema_slow:
            signal_direction = SignalDirection.LONG
            price_ok = last_close > last_ema_fast
        elif last_ema_fast < last_ema_slow:
            signal_direction = SignalDirection.SHORT
            price_ok = last_close < last_ema_fast
        else:
            return LayerResult(
                approved=False, direction=SignalDirection.NONE,
                reason="EMA'lar eşit - belirsiz trend",
            )

        trend_confirmed = (
            trend_direction == SignalDirection.NONE
            or trend_direction == signal_direction
        )
        ema_sep = abs(last_ema_fast - last_ema_slow) / last_ema_slow * 100
        approved = price_ok and trend_confirmed

        # Score: EMA ayrışması + trend onayı
        score = (0.5 if signal_direction == SignalDirection.LONG else -0.5)
        if trend_confirmed:
            score *= 1.3
        score = float(np.clip(score, -1, 1))

        return LayerResult(
            approved=approved,
            direction=signal_direction,
            reason=(
                f"EMA{self.cfg.ema_fast}={'%.2f' % last_ema_fast} "
                f"{'>' if last_ema_fast > last_ema_slow else '<'} "
                f"EMA{self.cfg.ema_slow}={'%.2f' % last_ema_slow}, "
                f"Trend: {'✓' if trend_confirmed else '✗'}, "
                f"Fiyat: {'✓' if price_ok else '✗'}"
            ),
            details={
                "ema_fast": round(last_ema_fast, 4),
                "ema_slow": round(last_ema_slow, 4),
                "ema_separation_pct": round(ema_sep, 4),
                "price_above_ema": price_ok,
                "trend_direction": trend_direction.value,
                "trend_confirmed": trend_confirmed,
            },
            score=score,
        )


# ============================================================
# Katman 2: RSI + MACD Momentum
# ============================================================

class Layer2Momentum:
    def __init__(self, config: SignalConfig):
        self.cfg = config

    def analyze(
        self, candles: List[dict], direction: SignalDirection
    ) -> LayerResult:
        if len(candles) < self.cfg.macd_slow + self.cfg.macd_signal + 5:
            return LayerResult(
                approved=False, direction=SignalDirection.NONE,
                reason="MACD için yetersiz veri",
            )

        df = _candles_to_df(candles)
        closes = df["close"]

        rsi = _rsi(closes, self.cfg.rsi_period)
        last_rsi = rsi.iloc[-1]
        prev_rsi = rsi.iloc[-2]

        macd_line, signal_line, histogram = _macd(
            closes, self.cfg.macd_fast, self.cfg.macd_slow, self.cfg.macd_signal
        )
        last_hist = histogram.iloc[-1]
        prev_hist = histogram.iloc[-2]

        macd_bullish_cross = prev_hist < 0 and last_hist > 0
        macd_bearish_cross = prev_hist > 0 and last_hist < 0

        if direction == SignalDirection.LONG:
            rsi_ok = self.cfg.rsi_oversold < last_rsi < self.cfg.rsi_overbought
            macd_ok = last_hist > 0 or macd_bullish_cross
            reject = []
            if not rsi_ok:
                reject.append(f"RSI={last_rsi:.1f}")
            if not macd_ok:
                reject.append(f"MACD_hist={last_hist:.4f}")
        elif direction == SignalDirection.SHORT:
            rsi_ok = self.cfg.rsi_oversold < last_rsi < self.cfg.rsi_overbought
            macd_ok = last_hist < 0 or macd_bearish_cross
            reject = []
            if not rsi_ok:
                reject.append(f"RSI={last_rsi:.1f}")
            if not macd_ok:
                reject.append(f"MACD_hist={last_hist:.4f}")
        else:
            return LayerResult(
                approved=False, direction=SignalDirection.NONE,
                reason="Yön belirsiz",
            )

        approved = rsi_ok and macd_ok

        # Score: RSI yönü + MACD kesişim
        rsi_norm = (last_rsi - 50) / 50  # -1 to +1
        score = rsi_norm if direction == SignalDirection.LONG else -rsi_norm
        if macd_bullish_cross or macd_bearish_cross:
            score = np.clip(score * 1.5, -1, 1)
        score = float(score)

        return LayerResult(
            approved=approved,
            direction=direction,
            reason=(
                f"RSI={last_rsi:.1f} ({'✓' if rsi_ok else '✗'}), "
                f"MACD_hist={last_hist:.4f} ({'✓' if macd_ok else '✗'})"
                + (f" | Ret: {', '.join(reject)}" if reject else "")
            ),
            details={
                "rsi": round(last_rsi, 2),
                "rsi_previous": round(prev_rsi, 2),
                "macd_line": round(macd_line.iloc[-1], 4),
                "macd_signal": round(signal_line.iloc[-1], 4),
                "macd_histogram": round(last_hist, 4),
                "macd_bullish_cross": macd_bullish_cross,
                "macd_bearish_cross": macd_bearish_cross,
            },
            score=score,
        )


# ============================================================
# Katman 3: Hacim + Funding Rate
# ============================================================

class Layer3Volume:
    def __init__(self, config: SignalConfig, max_funding_rate: float = 0.1):
        self.cfg = config
        self.max_funding_rate = max_funding_rate

    def analyze(
        self,
        candles: List[dict],
        direction: SignalDirection,
        funding_rate: float,
    ) -> LayerResult:
        if len(candles) < self.cfg.volume_lookback + 2:
            return LayerResult(
                approved=False, direction=SignalDirection.NONE,
                reason="Hacim analizi için yetersiz veri",
            )

        df = _candles_to_df(candles)
        avg_volume = df["volume"].iloc[-(self.cfg.volume_lookback + 1):-1].mean()
        last_volume = df["volume"].iloc[-1]
        volume_ratio = last_volume / avg_volume if avg_volume > 0 else 0
        volume_ok = volume_ratio >= self.cfg.volume_threshold

        taker_ratio = 0.5
        if "taker_buy_volume" in df.columns:
            last_taker_buy = df["taker_buy_volume"].iloc[-1]
            taker_ratio = last_taker_buy / last_volume if last_volume > 0 else 0.5

        if direction == SignalDirection.LONG:
            taker_ok = taker_ratio > 0.5
        elif direction == SignalDirection.SHORT:
            taker_ok = taker_ratio < 0.5
        else:
            taker_ok = False

        funding_ok = abs(funding_rate) <= self.max_funding_rate
        approved = volume_ok and funding_ok

        reject = []
        if not volume_ok:
            reject.append(f"Hacim={volume_ratio:.2f}x < {self.cfg.volume_threshold}x")
        if not funding_ok:
            reject.append(f"Funding={funding_rate:.4f}%")

        # Score: hacim gücü + taker yönü
        score = min(1.0, (volume_ratio - 1.0) / 2.0)
        if not taker_ok:
            score *= 0.5
        if direction == SignalDirection.SHORT:
            score = -score
        score = float(np.clip(score, -1, 1))

        return LayerResult(
            approved=approved,
            direction=direction,
            reason=(
                f"Hacim={volume_ratio:.2f}x ({'✓' if volume_ok else '✗'}), "
                f"Taker={taker_ratio:.2f} ({'✓' if taker_ok else 'zayıf'}), "
                f"Funding={funding_rate:.4f}% ({'✓' if funding_ok else '✗'})"
                + (f" | Ret: {', '.join(reject)}" if reject else "")
            ),
            details={
                "volume_ratio": round(volume_ratio, 3),
                "volume_ok": volume_ok,
                "taker_ratio": round(taker_ratio, 3),
                "taker_ok": taker_ok,
                "funding_rate": round(funding_rate, 6),
                "funding_ok": funding_ok,
            },
            score=score,
        )


# ============================================================
# Katman 4: Gelişmiş İndikatörler (Opsiyonel Filtre)
# ============================================================

class Layer4Advanced:
    """
    BB, ATR, StochRSI, VWAP, Candlestick Patterns, RSI Divergence.
    Bu katman sadece filtre olarak çalışır (veto hakkı var ama
    konfigürasyona göre devre dışı bırakılabilir).
    """

    def __init__(self, enabled: bool = True, strict: bool = False):
        self.enabled = enabled
        self.strict = strict  # True: veto hakkı var; False: sadece skor etkiler

        # Göstergeleri lazy import
        self._indicators = None

    def _get_indicators(self):
        if self._indicators is None:
            try:
                import core.indicators as ind
                self._indicators = ind
            except ImportError:
                logger.warning("core.indicators import edilemedi")
        return self._indicators

    def analyze(
        self,
        candles: List[dict],
        direction: SignalDirection,
    ) -> LayerResult:
        if not self.enabled:
            return LayerResult(
                approved=True,
                direction=direction,
                reason="Katman 4 devre dışı",
                score=0.0,
            )

        ind = self._get_indicators()
        if ind is None:
            return LayerResult(
                approved=True,
                direction=direction,
                reason="Katman 4: indicators modülü yok",
                score=0.0,
            )

        if len(candles) < 50:
            return LayerResult(
                approved=True,
                direction=direction,
                reason="Katman 4: Yetersiz veri (geçildi)",
                score=0.0,
            )

        df = _candles_to_df(candles)
        score_components = []
        details = {}
        veto = False
        veto_reason = ""

        try:
            # ── Bollinger Bands ──────────────────────────────
            try:
                bb = ind.bollinger_bands(df['close'])
                bb_sig = ind.bb_signal(bb, df['close'].iloc[-1])
                if direction == SignalDirection.LONG:
                    if bb_sig == 'BUY':
                        score_components.append(0.7)
                    elif bb_sig == 'SELL':
                        score_components.append(-0.3)
                    else:
                        score_components.append(0.0)
                elif direction == SignalDirection.SHORT:
                    if bb_sig == 'SELL':
                        score_components.append(0.7)
                    elif bb_sig == 'BUY':
                        score_components.append(-0.3)
                    else:
                        score_components.append(0.0)
                details['bb_signal'] = bb_sig
                details['bb_width'] = round(bb.width.iloc[-1] if hasattr(bb.width, 'iloc') else 0, 6)
                details['bb_squeeze'] = bb.squeeze.iloc[-1] if hasattr(bb.squeeze, 'iloc') else False
            except Exception as e:
                logger.debug(f"BB hesaplama hatası: {e}")

            # ── Stochastic RSI ───────────────────────────────
            try:
                stoch = ind.stochastic_rsi(df['close'])
                stoch_sig = ind.stoch_rsi_signal(stoch)
                if direction == SignalDirection.LONG:
                    s = 0.5 if stoch_sig == 'BUY' else (-0.3 if stoch_sig == 'SELL' else 0.0)
                else:
                    s = 0.5 if stoch_sig == 'SELL' else (-0.3 if stoch_sig == 'BUY' else 0.0)
                score_components.append(s)
                details['stoch_rsi_signal'] = stoch_sig
            except Exception as e:
                logger.debug(f"StochRSI hesaplama hatası: {e}")

            # ── VWAP ────────────────────────────────────────
            try:
                vwap_val = ind.vwap(df)
                if vwap_val is not None:
                    vwap_sig = ind.vwap_signal(df['close'].iloc[-1], vwap_val, df)
                    if direction == SignalDirection.LONG:
                        s = 0.4 if vwap_sig == 'ABOVE' else -0.4
                    else:
                        s = 0.4 if vwap_sig == 'BELOW' else -0.4
                    score_components.append(s)
                    details['vwap_signal'] = vwap_sig
            except Exception as e:
                logger.debug(f"VWAP hesaplama hatası: {e}")

            # ── ATR Volatilite Filtresi ──────────────────────
            try:
                atr_val = ind.atr(df['high'], df['low'], df['close'])
                atr_ratio = atr_val.iloc[-1] / df['close'].iloc[-1] if df['close'].iloc[-1] > 0 else 0
                details['atr_ratio'] = round(atr_ratio, 6)
                # Çok yüksek volatilite (>%5 ATR) veto
                if self.strict and atr_ratio > 0.05:
                    veto = True
                    veto_reason = f"ATR çok yüksek (%{atr_ratio*100:.2f})"
            except Exception as e:
                logger.debug(f"ATR hesaplama hatası: {e}")

            # ── Candlestick Patterns ─────────────────────────
            try:
                patterns = ind.detect_candle_patterns(df)
                if patterns:
                    bullish = sum(1 for p in patterns if p.bullish)
                    bearish = sum(1 for p in patterns if not p.bullish)
                    if direction == SignalDirection.LONG:
                        s = min(0.8, bullish * 0.3 - bearish * 0.2)
                    else:
                        s = min(0.8, bearish * 0.3 - bullish * 0.2)
                    score_components.append(s)
                    details['patterns'] = [p.name for p in patterns]
            except Exception as e:
                logger.debug(f"Pattern hesaplama hatası: {e}")

            # ── RSI Divergence ───────────────────────────────
            try:
                div = ind.rsi_divergence(df['close'], df['low'], df['high'])
                if div:
                    if direction == SignalDirection.LONG and div.bullish_divergence:
                        score_components.append(0.6)
                    elif direction == SignalDirection.SHORT and div.bearish_divergence:
                        score_components.append(0.6)
                    details['divergence'] = {
                        'bullish': div.bullish_divergence,
                        'bearish': div.bearish_divergence,
                    }
            except Exception as e:
                logger.debug(f"Divergence hesaplama hatası: {e}")

        except Exception as e:
            logger.error(f"Katman 4 genel hata: {e}")

        # Skor hesapla
        if score_components:
            final_score = float(np.mean(score_components))
        else:
            final_score = 0.0

        # Veto durumu
        if veto and self.strict:
            return LayerResult(
                approved=False,
                direction=direction,
                reason=f"Katman 4 VETO: {veto_reason}",
                details=details,
                score=final_score,
            )

        # Strict modda negatif skor veto'ya dönüşebilir
        approved = True
        if self.strict and final_score < -0.3:
            approved = False
            reason = f"Katman 4: Zayıf sinyal (skor={final_score:.2f})"
        else:
            reason = f"Katman 4 geçti (skor={final_score:.2f})"

        return LayerResult(
            approved=approved,
            direction=direction,
            reason=reason,
            details=details,
            score=final_score,
        )


# ============================================================
# Katman 5: ML + Sentiment + Advanced Ensemble (Tam Opsiyonel)
# ============================================================

class Layer5Ensemble:
    """
    ML, Sentiment ve Advanced Analysis (OB, OI, Fear&Greed, On-Chain, Whale)
    sonuçlarını ensemble ile birleştirir.
    Bu katman asla veto yapmamalıdır - sadece skoru etkiler.
    """

    def __init__(
        self,
        enabled: bool = True,
        advanced_manager=None,
        ml_layer=None,
        sentiment_layer=None,
        ensemble_aggregator=None,
    ):
        self.enabled = enabled
        self.advanced = advanced_manager
        self.ml = ml_layer
        self.sentiment = sentiment_layer
        self.ensemble = ensemble_aggregator

    def analyze(
        self,
        symbol: str,
        df,
        direction: SignalDirection,
        order_book: Optional[dict] = None,
        ai_regime: Optional[str] = None,
    ) -> LayerResult:
        if not self.enabled:
            return LayerResult(
                approved=True, direction=direction,
                reason="Katman 5 devre dışı", score=0.0,
            )

        from core.advanced_analysis import AnalysisResult

        module_results: Dict[str, Any] = {}

        # Advanced Analysis (OB, OI, Fear&Greed, On-Chain, Whale)
        if self.advanced:
            try:
                adv = self.advanced.run_all(symbol, df, order_book)
                module_results.update(adv)
            except Exception as e:
                logger.error(f"Advanced analysis hatası: {e}")

        # ML Layer
        if self.ml:
            try:
                ml_result = self.ml.predict(symbol, df)
                if ml_result:
                    module_results['ml_ensemble'] = ml_result
            except Exception as e:
                logger.error(f"ML katman hatası: {e}")

        # Sentiment Layer
        if self.sentiment:
            try:
                sent_result = self.sentiment.analyze(symbol)
                if sent_result:
                    module_results['sentiment_combined'] = sent_result
            except Exception as e:
                logger.error(f"Sentiment katman hatası: {e}")

        if not module_results:
            return LayerResult(
                approved=True, direction=direction,
                reason="Katman 5: Modül sonucu yok", score=0.0,
            )

        # Ensemble aggregasyon
        if self.ensemble:
            try:
                ens_signal = self.ensemble.aggregate(
                    module_results, df=df, ai_regime=ai_regime
                )
                score = ens_signal.score
                reason = (
                    f"Katman 5 ensemble: {ens_signal.direction} "
                    f"(skor={ens_signal.score:.2f}, güven={ens_signal.confidence:.1f}%)"
                )
                details = ens_signal.to_dict()
            except Exception as e:
                logger.error(f"Ensemble hatası: {e}")
                score = 0.0
                reason = "Katman 5: Ensemble hata"
                details = {}
        else:
            # Ensemble olmadan basit ortalama
            scores = [r.score for r in module_results.values()
                      if hasattr(r, 'score') and not np.isnan(r.score)]
            score = float(np.mean(scores)) if scores else 0.0
            reason = f"Katman 5 ort. skor: {score:.2f} ({len(scores)} modül)"
            details = {'module_count': len(scores)}

        # Yön uyumsuzluğunda skoru hafifçe cezalandır
        if direction == SignalDirection.LONG and score < -0.3:
            reason += " [UZUN uyumsuzluk]"
        elif direction == SignalDirection.SHORT and score > 0.3:
            reason += " [KISA uyumsuzluk]"

        return LayerResult(
            approved=True,  # Katman 5 hiçbir zaman veto yapmaz
            direction=direction,
            reason=reason,
            details=details,
            score=float(np.clip(score, -1, 1)),
        )


# ============================================================
# Sinyal Motoru
# ============================================================

class SignalEngine:
    """
    5 katmanlı sinyal üretme motoru.
    Katman 1-3 zorunlu, 4-5 opsiyonel.
    """

    def __init__(
        self,
        data_engine: DataEngine,
        signal_config: SignalConfig,
        max_funding_rate: float = 0.1,
        # Opsiyonel bileşenler
        advanced_manager=None,
        ml_layer=None,
        sentiment_layer=None,
        ensemble_aggregator=None,
        layer4_enabled: bool = True,
        layer4_strict: bool = False,
        layer5_enabled: bool = True,
    ):
        self.data = data_engine
        self.cfg = signal_config
        self.max_funding_rate = max_funding_rate

        # Zorunlu katmanlar
        self.layer1 = Layer1EMA(signal_config)
        self.layer2 = Layer2Momentum(signal_config)
        self.layer3 = Layer3Volume(signal_config, max_funding_rate)

        # Opsiyonel katmanlar
        self.layer4 = Layer4Advanced(
            enabled=layer4_enabled,
            strict=layer4_strict,
        )
        self.layer5 = Layer5Ensemble(
            enabled=layer5_enabled,
            advanced_manager=advanced_manager,
            ml_layer=ml_layer,
            sentiment_layer=sentiment_layer,
            ensemble_aggregator=ensemble_aggregator,
        )

        # Önbellek ve geçmiş
        self._last_signals: Dict[str, Signal] = {}
        self._signal_history: List[Signal] = []

        self._interval_map = {
            1: "1m", 3: "3m", 5: "5m", 15: "15m",
            30: "30m", 60: "1h", 120: "2h", 240: "4h",
        }

        logger.info(
            f"SignalEngine başlatıldı. Katman4:{layer4_enabled}(strict={layer4_strict}), "
            f"Katman5:{layer5_enabled}"
        )

    def _get_interval(self, minutes: int) -> str:
        return self._interval_map.get(minutes, f"{minutes}m")

    def analyze(self, symbol: str, ai_regime: Optional[str] = None) -> Signal:
        """Belirtilen parite için 5 katmanlı sinyal analizi."""
        signal_interval = self._get_interval(self.cfg.signal_timeframe)
        trend_interval = self._get_interval(self.cfg.trend_timeframe)

        signal_candles = self.data.kline_cache.get(symbol, signal_interval)
        trend_candles = self.data.kline_cache.get(symbol, trend_interval)
        current_price = self.data.price_cache.get(symbol) or 0.0
        funding_rate = self.data.get_funding_rate(symbol)
        spread_pct = self.data.get_spread(symbol)

        signal = Signal(
            symbol=symbol,
            direction=SignalDirection.NONE,
            strength=SignalStrength.WEAK,
            signal_price=current_price,
            spread_pct=spread_pct,
            funding_rate=funding_rate,
            market_regime=ai_regime or 'unknown',
        )

        # Spread limiti
        if spread_pct > self.cfg.max_spread_pct:
            signal.approved = False
            signal.reject_reason = (
                f"Spread çok yüksek: %{spread_pct:.3f} > %{self.cfg.max_spread_pct}"
            )
            self._cache_signal(symbol, signal)
            return signal

        # Veri kontrolü
        if len(signal_candles) < self.cfg.ema_slow + 10:
            signal.approved = False
            signal.reject_reason = f"Yetersiz veri: {len(signal_candles)} mum"
            self._cache_signal(symbol, signal)
            return signal

        # ── Katman 1: EMA ──
        l1 = self.layer1.analyze(signal_candles, trend_candles)
        signal.layer1 = l1
        if not l1.approved:
            signal.approved = False
            signal.reject_reason = f"K1(EMA): {l1.reason}"
            self._cache_signal(symbol, signal)
            return signal

        direction = l1.direction
        signal.direction = direction

        # ── Katman 2: Momentum ──
        l2 = self.layer2.analyze(signal_candles, direction)
        signal.layer2 = l2
        if not l2.approved:
            signal.approved = False
            signal.reject_reason = f"K2(Momentum): {l2.reason}"
            self._cache_signal(symbol, signal)
            return signal

        # ── Katman 3: Hacim/Funding ──
        l3 = self.layer3.analyze(signal_candles, direction, funding_rate)
        signal.layer3 = l3
        if not l3.approved:
            signal.approved = False
            signal.reject_reason = f"K3(Hacim): {l3.reason}"
            self._cache_signal(symbol, signal)
            return signal

        # ── Katman 4: Gelişmiş İndikatörler ──
        df = _candles_to_df(signal_candles)
        l4 = self.layer4.analyze(signal_candles, direction)
        signal.layer4 = l4
        if not l4.approved:
            signal.approved = False
            signal.reject_reason = f"K4(Gelişmiş): {l4.reason}"
            self._cache_signal(symbol, signal)
            return signal

        # ── Katman 5: ML + Sentiment + Advanced ──
        order_book = None
        try:
            order_book = self.data.get_order_book(symbol)
        except Exception:
            pass

        l5 = self.layer5.analyze(symbol, df, direction, order_book, ai_regime)
        signal.layer5 = l5
        # Katman 5 asla veto yapmaz, sadece ensemble skor etkiler

        # ── Tüm zorunlu katmanlar onayladı ──
        signal.approved = True

        # Ensemble skoru hesapla
        layer_scores = {
            'layer1_ema': l1.score,
            'layer2_momentum': l2.score,
            'layer3_volume': l3.score,
            'layer4_indicators': l4.score,
        }
        if l5.score != 0:
            layer_scores['layer5_ensemble'] = l5.score

        # Ensemble skoru varsa kullan, yoksa katman ortalama
        if l5.details and 'score' in l5.details:
            signal.ensemble_score = l5.details['score']
            signal.ensemble_confidence = l5.details.get('confidence', 50.0)
        else:
            signal.ensemble_score = float(np.mean(list(layer_scores.values())))
            signal.ensemble_confidence = 50.0

        signal.strength = self._calculate_strength(l1, l2, l3, l4, l5)

        logger.info(
            f"[SINYAL] {symbol} {direction.value} | "
            f"Güç: {signal.strength.value} | "
            f"Ensemble: {signal.ensemble_score:.3f} | "
            f"Rejim: {signal.market_regime} | "
            f"Fiyat: {current_price}"
        )

        self._cache_signal(symbol, signal)
        self._signal_history.append(signal)
        if len(self._signal_history) > 100:
            self._signal_history.pop(0)

        return signal

    def _calculate_strength(
        self,
        l1: LayerResult,
        l2: LayerResult,
        l3: LayerResult,
        l4: Optional[LayerResult] = None,
        l5: Optional[LayerResult] = None,
    ) -> SignalStrength:
        score = 0

        # Katman 1 katkısı
        ema_sep = l1.details.get("ema_separation_pct", 0)
        if ema_sep > 0.5:
            score += 2
        elif ema_sep > 0.2:
            score += 1
        if l1.details.get("trend_confirmed"):
            score += 1

        # Katman 2 katkısı
        rsi = l2.details.get("rsi", 50)
        if 40 <= rsi <= 60:
            score += 2
        elif 35 <= rsi <= 65:
            score += 1
        if l2.details.get("macd_bullish_cross") or l2.details.get("macd_bearish_cross"):
            score += 2

        # Katman 3 katkısı
        vol_ratio = l3.details.get("volume_ratio", 1.0)
        if vol_ratio >= 2.0:
            score += 2
        elif vol_ratio >= 1.5:
            score += 1
        if l3.details.get("taker_ok"):
            score += 1
        if abs(l3.details.get("funding_rate", 0)) < 0.05:
            score += 1

        # Katman 4 katkısı (opsiyonel)
        if l4 and l4.score > 0.3:
            score += 2
        elif l4 and l4.score > 0:
            score += 1

        # Katman 5 katkısı (opsiyonel)
        if l5 and abs(l5.score) > 0.5:
            score += 2
        elif l5 and abs(l5.score) > 0.2:
            score += 1

        if score >= 10:
            return SignalStrength.STRONG
        elif score >= 6:
            return SignalStrength.MEDIUM
        else:
            return SignalStrength.WEAK

    def _cache_signal(self, symbol: str, signal: Signal):
        self._last_signals[symbol] = signal

    def get_last_signal(self, symbol: str) -> Optional[Signal]:
        return self._last_signals.get(symbol)

    def get_signal_history(self, limit: int = 20) -> List[dict]:
        return [s.to_dict() for s in self._signal_history[-limit:]]

    def analyze_all(
        self, symbols: List[str], ai_regime: Optional[str] = None
    ) -> Dict[str, Signal]:
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.analyze(symbol, ai_regime=ai_regime)
            except Exception as e:
                logger.error(f"{symbol} sinyal analiz hatası: {e}")
        return results
