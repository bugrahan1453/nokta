"""
ensemble.py - Ensemble / Meta-Öğrenme Katmanı
Tüm analiz modüllerinin ağırlıklı oylamasını birleştirir.
Piyasa rejimine göre dinamik ağırlık ayarlama.
Bayesian model ortalaması (opsiyonel).
"""

import os
import json
import logging
import numpy as np
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field

from core.advanced_analysis import AnalysisResult

logger = logging.getLogger(__name__)


# ── Sinyal Tipi ─────────────────────────────────────────────────────────────

@dataclass
class EnsembleSignal:
    """Ensemble çıktısı."""
    direction: str          # LONG / SHORT / NEUTRAL
    score: float            # -1 ile +1
    confidence: float       # 0-100
    strength: str           # WEAK / MEDIUM / STRONG
    should_trade: bool
    module_scores: Dict[str, float] = field(default_factory=dict)
    module_weights: Dict[str, float] = field(default_factory=dict)
    market_regime: str = 'unknown'
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'direction': self.direction,
            'score': round(self.score, 4),
            'confidence': round(self.confidence, 2),
            'strength': self.strength,
            'should_trade': self.should_trade,
            'module_scores': {k: round(v, 4) for k, v in self.module_scores.items()},
            'module_weights': {k: round(v, 4) for k, v in self.module_weights.items()},
            'market_regime': self.market_regime,
            'timestamp': self.timestamp.isoformat(),
        }


# ── Piyasa Rejimi ────────────────────────────────────────────────────────────

class RegimeDetector:
    """
    ADX, volatilite ve trend güçlüne göre piyasa rejimi tespit eder.
    Rejim → modül ağırlıklarını dinamik ayarlamak için kullanılır.
    """

    REGIMES = ['trending_up', 'trending_down', 'sideways', 'high_volatility', 'unknown']

    def detect(self, df=None, ai_regime: Optional[str] = None) -> str:
        """
        Önce AI katmanından gelen rejimi kullan (zaten analiz edilmiş).
        AI yoksa basit kural tabanlı tespit yap.
        """
        if ai_regime and ai_regime in self.REGIMES:
            return ai_regime

        if df is None or len(df) < 20:
            return 'unknown'

        try:
            import pandas as pd
            close = df['close']
            high = df['high']
            low = df['low']

            # ATR tabanlı volatilite
            tr = pd.concat([
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs()
            ], axis=1).max(axis=1)
            atr = tr.rolling(14).mean().iloc[-1]
            atr_ratio = atr / close.iloc[-1] if close.iloc[-1] > 0 else 0

            # EMA trend
            ema21 = close.ewm(span=21).mean().iloc[-1]
            ema50 = close.ewm(span=50).mean().iloc[-1]
            ema_diff = (ema21 - ema50) / ema50 if ema50 > 0 else 0

            # ADX
            plus_dm = (high.diff()).where((high.diff() > low.diff().abs()) & (high.diff() > 0), 0)
            minus_dm = (low.diff().abs()).where((low.diff().abs() > high.diff()) & (low.diff() < 0), 0)
            tr_smooth = tr.rolling(14).mean()
            if tr_smooth.iloc[-1] > 0:
                plus_di = 100 * plus_dm.rolling(14).mean().iloc[-1] / tr_smooth.iloc[-1]
                minus_di = 100 * minus_dm.rolling(14).mean().iloc[-1] / tr_smooth.iloc[-1]
                di_sum = plus_di + minus_di
                dx = 100 * abs(plus_di - minus_di) / di_sum if di_sum > 0 else 0
            else:
                dx = 0

            # Karar
            if atr_ratio > 0.05:  # %5+ ATR → yüksek volatilite
                return 'high_volatility'
            elif dx > 25 and ema_diff > 0.01:
                return 'trending_up'
            elif dx > 25 and ema_diff < -0.01:
                return 'trending_down'
            elif dx < 20:
                return 'sideways'
            else:
                return 'unknown'

        except Exception as e:
            logger.error(f"Rejim tespit hatası: {e}")
            return 'unknown'


# ── Ağırlık Tabloları ────────────────────────────────────────────────────────

# Base ağırlıklar (rejimden bağımsız)
BASE_WEIGHTS = {
    'layer1_ema': 1.5,
    'layer2_momentum': 1.5,
    'layer3_volume': 1.2,
    'layer4_indicators': 1.3,
    'order_book': 0.8,
    'open_interest': 0.9,
    'fear_greed': 0.6,
    'on_chain': 0.7,
    'whale': 0.7,
    'ml_ensemble': 1.0,
    'sentiment_combined': 0.8,
}

# Regime modifier'ları (çarpanlar)
REGIME_MODIFIERS: Dict[str, Dict[str, float]] = {
    'trending_up': {
        'layer1_ema': 1.4,
        'layer2_momentum': 1.2,
        'order_book': 0.9,
        'fear_greed': 0.7,
        'ml_ensemble': 1.1,
    },
    'trending_down': {
        'layer1_ema': 1.4,
        'layer2_momentum': 1.2,
        'order_book': 0.9,
        'fear_greed': 0.7,
        'ml_ensemble': 1.1,
    },
    'sideways': {
        'layer1_ema': 0.7,
        'layer4_indicators': 1.4,
        'order_book': 1.3,
        'sentiment_combined': 1.2,
        'fear_greed': 1.1,
    },
    'high_volatility': {
        'fear_greed': 1.4,
        'whale': 1.3,
        'open_interest': 1.2,
        'layer2_momentum': 0.8,
        'ml_ensemble': 0.7,
    },
    'unknown': {},
}


def compute_weights(regime: str, custom_weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """Rejim ve özel ağırlıklara göre modül ağırlıklarını hesaplar."""
    weights = BASE_WEIGHTS.copy()

    if custom_weights:
        for k, v in custom_weights.items():
            if k in weights:
                weights[k] = v

    modifiers = REGIME_MODIFIERS.get(regime, {})
    for module, multiplier in modifiers.items():
        if module in weights:
            weights[module] *= multiplier

    return weights


# ── Bayesian Model Ortalaması ────────────────────────────────────────────────

class BayesianAverager:
    """
    Modellerin geçmiş performansına göre posterior ağırlık günceller.
    Çok basit versiyonu: her modelin doğruluk oranını takip eder.
    """

    def __init__(self, modules: List[str], prior: float = 0.5):
        self.modules = modules
        self.successes: Dict[str, int] = {m: int(prior * 10) for m in modules}
        self.trials: Dict[str, int] = {m: 10 for m in modules}

    def update(self, module: str, correct: bool):
        if module not in self.modules:
            return
        self.trials[module] += 1
        if correct:
            self.successes[module] += 1

    def get_weights(self) -> Dict[str, float]:
        """Beta dağılımı beklentisi: (alpha) / (alpha + beta)"""
        weights = {}
        for m in self.modules:
            alpha = self.successes.get(m, 1)
            beta = self.trials.get(m, 2) - alpha
            beta = max(1, beta)
            weights[m] = alpha / (alpha + beta)
        return weights

    def incorporate_weights(self, base_weights: Dict[str, float]) -> Dict[str, float]:
        """Bayesian ağırlıkları base ağırlıklarla birleştirir."""
        bayes = self.get_weights()
        combined = {}
        for module, w in base_weights.items():
            bayes_w = bayes.get(module, 0.5)
            combined[module] = w * (0.5 + bayes_w)  # Bayesian katkısını sınırla
        return combined


# ── Ana Ensemble Sınıfı ─────────────────────────────────────────────────────

class EnsembleAggregator:
    """
    Tüm modül çıktılarını alır, ağırlıklı oylama ile birleştirir.
    Konsensüs eşiklerine göre LONG/SHORT/NEUTRAL kararı verir.
    """

    # Sinyal gücü eşikleri
    STRENGTH_THRESHOLDS = {
        'STRONG': 0.55,
        'MEDIUM': 0.35,
        'WEAK': 0.15,
    }

    # Yön için minimum skor
    DIRECTION_THRESHOLD = 0.15

    # İşlem için minimum güven
    MIN_CONFIDENCE = 40.0

    def __init__(self, settings=None):
        self.settings = settings
        self._load_config()
        self.regime_detector = RegimeDetector()
        self.bayesian = BayesianAverager(list(BASE_WEIGHTS.keys()))
        self._trade_outcomes: List[Dict] = []
        self._custom_weights: Optional[Dict[str, float]] = None

    def _load_config(self):
        self.min_confidence = float(os.getenv('ENSEMBLE_MIN_CONFIDENCE', str(self.MIN_CONFIDENCE)))
        self.direction_threshold = float(os.getenv('ENSEMBLE_DIRECTION_THRESHOLD', str(self.DIRECTION_THRESHOLD)))
        self.use_bayesian = os.getenv('ENSEMBLE_USE_BAYESIAN', 'true').lower() == 'true'

        # Özel ağırlık dosyası
        weight_file = os.getenv('ENSEMBLE_WEIGHT_FILE', '')
        if weight_file and os.path.exists(weight_file):
            try:
                with open(weight_file) as f:
                    self._custom_weights = json.load(f)
                logger.info(f"Ensemble özel ağırlıklar yüklendi: {weight_file}")
            except Exception as e:
                logger.warning(f"Özel ağırlık yükleme hatası: {e}")

    def aggregate(
        self,
        module_results: Dict[str, AnalysisResult],
        df=None,
        ai_regime: Optional[str] = None,
    ) -> EnsembleSignal:
        """
        module_results: {'layer1_ema': AnalysisResult, 'order_book': AnalysisResult, ...}
        """
        if not module_results:
            return EnsembleSignal(
                direction='NEUTRAL', score=0.0, confidence=0.0,
                strength='WEAK', should_trade=False,
                market_regime='unknown'
            )

        # Piyasa rejimi tespit et
        regime = self.regime_detector.detect(df=df, ai_regime=ai_regime)

        # Ağırlıkları hesapla
        weights = compute_weights(regime, self._custom_weights)
        if self.use_bayesian:
            weights = self.bayesian.incorporate_weights(weights)

        # Sadece sonuç olan modüller için
        valid_results = {
            k: v for k, v in module_results.items()
            if v is not None and not np.isnan(v.score)
        }

        if not valid_results:
            return EnsembleSignal(
                direction='NEUTRAL', score=0.0, confidence=0.0,
                strength='WEAK', should_trade=False,
                market_regime=regime
            )

        # Ağırlıklı skor hesapla
        total_weight = 0.0
        weighted_score = 0.0
        weighted_confidence = 0.0
        module_scores = {}
        module_weights_used = {}

        for module_name, result in valid_results.items():
            w = weights.get(module_name, 1.0)
            # Güvene göre ağırlık azalt
            conf_factor = result.confidence / 100.0
            effective_w = w * conf_factor

            weighted_score += result.score * effective_w
            weighted_confidence += result.confidence * w
            total_weight += effective_w

            module_scores[module_name] = result.score
            module_weights_used[module_name] = round(effective_w, 4)

        if total_weight == 0:
            final_score = 0.0
            final_confidence = 0.0
        else:
            final_score = weighted_score / total_weight
            final_confidence = weighted_confidence / len(valid_results)

        # Yön kararı
        if final_score > self.direction_threshold:
            direction = 'LONG'
        elif final_score < -self.direction_threshold:
            direction = 'SHORT'
        else:
            direction = 'NEUTRAL'

        # Güç seviyesi
        abs_score = abs(final_score)
        if abs_score >= self.STRENGTH_THRESHOLDS['STRONG']:
            strength = 'STRONG'
        elif abs_score >= self.STRENGTH_THRESHOLDS['MEDIUM']:
            strength = 'MEDIUM'
        elif abs_score >= self.STRENGTH_THRESHOLDS['WEAK']:
            strength = 'WEAK'
        else:
            strength = 'WEAK'

        # İşlem yapılmalı mı?
        should_trade = (
            direction != 'NEUTRAL' and
            final_confidence >= self.min_confidence and
            strength in ('MEDIUM', 'STRONG')
        )

        return EnsembleSignal(
            direction=direction,
            score=float(np.clip(final_score, -1, 1)),
            confidence=float(np.clip(final_confidence, 0, 100)),
            strength=strength,
            should_trade=should_trade,
            module_scores=module_scores,
            module_weights=module_weights_used,
            market_regime=regime,
            timestamp=datetime.now(),
        )

    def record_outcome(
        self,
        module_scores: Dict[str, float],
        direction: str,
        pnl_pct: float,
    ):
        """
        İşlem kapandığında çağrılır.
        Bayesian güncellemesi yapar: hangi modüllerin doğru tahmin ettiğini takip eder.
        """
        trade_correct = pnl_pct > 0

        for module_name, score in module_scores.items():
            module_predicted_long = score > 0
            correct = (
                (direction == 'LONG' and module_predicted_long and trade_correct) or
                (direction == 'SHORT' and not module_predicted_long and trade_correct) or
                (trade_correct)  # basit: kazanan işlemde herkesin puanı artar
            )
            self.bayesian.update(module_name, correct)

        self._trade_outcomes.append({
            'direction': direction,
            'pnl_pct': pnl_pct,
            'module_scores': module_scores,
            'timestamp': datetime.now().isoformat(),
        })

        # Son 200 işlemi tut
        if len(self._trade_outcomes) > 200:
            self._trade_outcomes.pop(0)

    def get_module_performance(self) -> Dict[str, Any]:
        """Bayesian ağırlıkları ve modül performans özetini döner."""
        bayesian_weights = self.bayesian.get_weights()
        outcomes = self._trade_outcomes

        if not outcomes:
            return {
                'bayesian_weights': bayesian_weights,
                'total_trades': 0,
                'win_rate': 0.0,
            }

        wins = sum(1 for o in outcomes if o['pnl_pct'] > 0)
        return {
            'bayesian_weights': {k: round(v, 4) for k, v in bayesian_weights.items()},
            'total_trades': len(outcomes),
            'win_rate': round(wins / len(outcomes) * 100, 2),
            'avg_pnl': round(sum(o['pnl_pct'] for o in outcomes) / len(outcomes), 4),
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            'min_confidence': self.min_confidence,
            'direction_threshold': self.direction_threshold,
            'use_bayesian': self.use_bayesian,
            'performance': self.get_module_performance(),
        }
