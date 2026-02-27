"""
ml_layer.py - Makine Öğrenmesi Katmanı (Opsiyonel)
XGBoost, LightGBM, LSTM/GRU modelleri
Kendi işlem geçmişinden online öğrenme
Tüm kütüphaneler opsiyonel - kurulu değilse devre dışı
"""

import os
import json
import pickle
import logging
import threading
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path

from core.advanced_analysis import AnalysisResult

logger = logging.getLogger(__name__)

# ── Opsiyonel kütüphaneler ──────────────────────────────────────────────────
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.info("XGBoost kurulu değil. pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.info("LightGBM kurulu değil. pip install lightgbm")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.info("PyTorch kurulu değil. pip install torch")

try:
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.info("scikit-learn kurulu değil. pip install scikit-learn")


# ── Özellik Mühendisliği ────────────────────────────────────────────────────

class FeatureEngineer:
    """
    OHLCV + indikatör verilerinden ML özellikleri üretir.
    Tüm özellikleri normalize eder.
    """

    FEATURE_NAMES = [
        # Fiyat özellikleri
        'returns_1', 'returns_3', 'returns_5', 'returns_10',
        'log_returns_1', 'hl_ratio', 'oc_ratio',
        # Volatilite
        'atr_ratio', 'bb_width', 'bb_position',
        'vol_1h', 'vol_4h', 'vol_24h',
        # Momentum
        'rsi', 'rsi_delta', 'stoch_rsi_k', 'stoch_rsi_d',
        'macd_signal', 'macd_hist', 'cci', 'williams_r',
        # Trend
        'ema_cross_21_50', 'ema_cross_50_200',
        'adx', 'adx_trend', 'supertrend_signal',
        # Hacim
        'volume_ratio', 'taker_ratio', 'vwap_deviation',
        'volume_trend_3', 'volume_trend_5',
        # İstatistiksel
        'z_score', 'skewness', 'kurtosis',
        # Saat/gün özellikleri
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    ]

    def __init__(self):
        self.scaler = RobustScaler() if SKLEARN_AVAILABLE else None
        self._fitted = False

    def extract_features(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """DataFrame'den özellik vektörü çıkarır (son mum için)."""
        try:
            if len(df) < 50:
                return None

            features = {}
            close = df['close']
            high = df['high']
            low = df['low']
            volume = df['volume']
            n = len(df)

            # Fiyat getirileri
            features['returns_1'] = close.pct_change(1).iloc[-1]
            features['returns_3'] = close.pct_change(3).iloc[-1]
            features['returns_5'] = close.pct_change(5).iloc[-1]
            features['returns_10'] = close.pct_change(10).iloc[-1]
            features['log_returns_1'] = np.log(close.iloc[-1] / close.iloc[-2]) if close.iloc[-2] > 0 else 0
            features['hl_ratio'] = (high.iloc[-1] - low.iloc[-1]) / close.iloc[-1]
            features['oc_ratio'] = (close.iloc[-1] - df['open'].iloc[-1]) / close.iloc[-1]

            # ATR
            tr = pd.concat([
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs()
            ], axis=1).max(axis=1)
            atr = tr.rolling(14).mean().iloc[-1]
            features['atr_ratio'] = atr / close.iloc[-1] if close.iloc[-1] > 0 else 0

            # Bollinger Bands
            sma20 = close.rolling(20).mean()
            std20 = close.rolling(20).std()
            bb_upper = sma20 + 2 * std20
            bb_lower = sma20 - 2 * std20
            bb_width_val = (bb_upper.iloc[-1] - bb_lower.iloc[-1]) / sma20.iloc[-1] if sma20.iloc[-1] > 0 else 0
            bb_pos = (close.iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1]) \
                if (bb_upper.iloc[-1] - bb_lower.iloc[-1]) > 0 else 0.5
            features['bb_width'] = bb_width_val
            features['bb_position'] = np.clip(bb_pos, 0, 1)

            # Volatilite (std / mean)
            features['vol_1h'] = close.pct_change().rolling(4).std().iloc[-1]  # ~1h on 15m
            features['vol_4h'] = close.pct_change().rolling(16).std().iloc[-1]
            features['vol_24h'] = close.pct_change().rolling(96).std().iloc[-1]

            # RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi_series = 100 - (100 / (1 + rs))
            rsi_val = rsi_series.iloc[-1]
            features['rsi'] = rsi_val / 100.0  # normalize 0-1
            features['rsi_delta'] = (rsi_series.diff().iloc[-1]) / 100.0

            # Stochastic RSI
            rsi_min = rsi_series.rolling(14).min()
            rsi_max = rsi_series.rolling(14).max()
            stoch_k_raw = (rsi_series - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan)
            stoch_k = stoch_k_raw.rolling(3).mean()
            stoch_d = stoch_k.rolling(3).mean()
            features['stoch_rsi_k'] = stoch_k.iloc[-1] if not pd.isna(stoch_k.iloc[-1]) else 0.5
            features['stoch_rsi_d'] = stoch_d.iloc[-1] if not pd.isna(stoch_d.iloc[-1]) else 0.5

            # MACD
            ema12 = close.ewm(span=12).mean()
            ema26 = close.ewm(span=26).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9).mean()
            macd_hist_val = macd_line.iloc[-1] - signal_line.iloc[-1]
            features['macd_signal'] = 1.0 if macd_line.iloc[-1] > signal_line.iloc[-1] else -1.0
            features['macd_hist'] = np.tanh(macd_hist_val / close.iloc[-1] * 100)

            # CCI
            tp = (high + low + close) / 3
            sma_tp = tp.rolling(20).mean()
            mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
            cci_val = (tp.iloc[-1] - sma_tp.iloc[-1]) / (0.015 * mad.iloc[-1]) if mad.iloc[-1] > 0 else 0
            features['cci'] = np.tanh(cci_val / 200)

            # Williams %R
            highest_high = high.rolling(14).max()
            lowest_low = low.rolling(14).min()
            wpr = -100 * (highest_high.iloc[-1] - close.iloc[-1]) / \
                  (highest_high.iloc[-1] - lowest_low.iloc[-1]) \
                  if (highest_high.iloc[-1] - lowest_low.iloc[-1]) > 0 else -50
            features['williams_r'] = (wpr + 50) / 50  # normalize -1 to 1

            # EMA kesişim sinyalleri
            ema21 = close.ewm(span=21).mean()
            ema50 = close.ewm(span=50).mean()
            ema200 = close.ewm(span=200).mean() if n >= 200 else ema50
            features['ema_cross_21_50'] = 1.0 if ema21.iloc[-1] > ema50.iloc[-1] else -1.0
            features['ema_cross_50_200'] = 1.0 if ema50.iloc[-1] > ema200.iloc[-1] else -1.0

            # ADX
            plus_dm = (high.diff()).where((high.diff() > low.diff().abs()) & (high.diff() > 0), 0)
            minus_dm = (low.diff().abs()).where((low.diff().abs() > high.diff()) & (low.diff() < 0), 0)
            tr_smooth = tr.rolling(14).mean()
            adx_val = 0
            if tr_smooth.iloc[-1] > 0:
                plus_di = 100 * plus_dm.rolling(14).mean().iloc[-1] / tr_smooth.iloc[-1]
                minus_di = 100 * minus_dm.rolling(14).mean().iloc[-1] / tr_smooth.iloc[-1]
                di_sum = plus_di + minus_di
                dx = 100 * abs(plus_di - minus_di) / di_sum if di_sum > 0 else 0
                adx_val = dx / 100.0
            features['adx'] = adx_val
            features['adx_trend'] = 1.0 if adx_val > 0.25 else 0.0

            # Supertrend (basit)
            atr_14 = tr.rolling(14).mean()
            upper_band = ((high + low) / 2) + 3 * atr_14
            lower_band = ((high + low) / 2) - 3 * atr_14
            st_signal = 1.0 if close.iloc[-1] > lower_band.iloc[-1] else -1.0
            features['supertrend_signal'] = st_signal

            # Hacim
            vol_ma20 = volume.rolling(20).mean()
            features['volume_ratio'] = volume.iloc[-1] / vol_ma20.iloc[-1] if vol_ma20.iloc[-1] > 0 else 1.0
            if 'taker_buy_volume' in df.columns and 'taker_sell_volume' in df.columns:
                total_taker = df['taker_buy_volume'].iloc[-1] + df['taker_sell_volume'].iloc[-1]
                features['taker_ratio'] = df['taker_buy_volume'].iloc[-1] / total_taker if total_taker > 0 else 0.5
            else:
                features['taker_ratio'] = 0.5

            # VWAP sapması
            vwap_val = (close * volume).rolling(20).sum() / volume.rolling(20).sum()
            features['vwap_deviation'] = (close.iloc[-1] - vwap_val.iloc[-1]) / vwap_val.iloc[-1] \
                if vwap_val.iloc[-1] > 0 else 0

            # Hacim trendi
            features['volume_trend_3'] = 1.0 if volume.iloc[-1] > volume.rolling(3).mean().iloc[-1] else -1.0
            features['volume_trend_5'] = 1.0 if volume.iloc[-1] > volume.rolling(5).mean().iloc[-1] else -1.0

            # Z-Score
            close_mean = close.rolling(20).mean().iloc[-1]
            close_std = close.rolling(20).std().iloc[-1]
            features['z_score'] = (close.iloc[-1] - close_mean) / close_std if close_std > 0 else 0

            # Skewness & Kurtosis
            returns_20 = close.pct_change().rolling(20)
            features['skewness'] = returns_20.apply(lambda x: pd.Series(x).skew()).iloc[-1]
            features['kurtosis'] = returns_20.apply(lambda x: pd.Series(x).kurt()).iloc[-1]

            # Zaman özellikleri (sinyaller)
            now = pd.Timestamp.now()
            hour = now.hour
            dow = now.dayofweek
            features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
            features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
            features['dow_sin'] = np.sin(2 * np.pi * dow / 7)
            features['dow_cos'] = np.cos(2 * np.pi * dow / 7)

            # NaN'ları 0 ile doldur
            feat_array = np.array([features.get(k, 0) for k in self.FEATURE_NAMES])
            feat_array = np.nan_to_num(feat_array, nan=0.0, posinf=1.0, neginf=-1.0)
            return feat_array

        except Exception as e:
            logger.error(f"Özellik çıkarma hatası: {e}")
            return None

    def fit_scaler(self, X: np.ndarray):
        if self.scaler and len(X) > 10:
            self.scaler.fit(X)
            self._fitted = True

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.scaler and self._fitted:
            return self.scaler.transform(X.reshape(1, -1)).flatten()
        return X


# ── LSTM Model ──────────────────────────────────────────────────────────────

if TORCH_AVAILABLE:
    class LSTMModel(nn.Module):
        def __init__(self, input_size: int, hidden_size: int = 64,
                     num_layers: int = 2, dropout: float = 0.2):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
            self.attention = nn.Linear(hidden_size, 1)
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, 32),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32, 3)  # LONG / NEUTRAL / SHORT
            )

        def forward(self, x):
            lstm_out, _ = self.lstm(x)  # (batch, seq, hidden)
            attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
            context = (lstm_out * attn_weights).sum(dim=1)
            return self.fc(context)


# ── XGBoost Wrapper ─────────────────────────────────────────────────────────

class XGBoostPredictor:
    """XGBoost tabanlı yön tahmini."""

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.model_path = model_path
        self._load_or_init(model_path)

    def _load_or_init(self, path: Optional[str]):
        if not XGBOOST_AVAILABLE:
            return
        if path and os.path.exists(path):
            try:
                self.model = xgb.XGBClassifier()
                self.model.load_model(path)
                logger.info(f"XGBoost modeli yüklendi: {path}")
            except Exception as e:
                logger.warning(f"XGBoost yükleme hatası: {e}")
                self._init_model()
        else:
            self._init_model()

    def _init_model(self):
        if not XGBOOST_AVAILABLE:
            return
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=42,
            n_jobs=-1
        )

    def train(self, X: np.ndarray, y: np.ndarray) -> bool:
        if not XGBOOST_AVAILABLE or len(X) < 30:
            return False
        try:
            self.model.fit(X, y)
            if self.model_path:
                Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
                self.model.save_model(self.model_path)
            return True
        except Exception as e:
            logger.error(f"XGBoost eğitim hatası: {e}")
            return False

    def predict_proba(self, x: np.ndarray) -> Optional[np.ndarray]:
        if not XGBOOST_AVAILABLE or self.model is None:
            return None
        try:
            # Modelin fit edilip edilmediğini kontrol et
            if not hasattr(self.model, 'classes_'):
                return None
            proba = self.model.predict_proba(x.reshape(1, -1))[0]
            return proba  # [SHORT_prob, NEUTRAL_prob, LONG_prob]
        except Exception as e:
            logger.error(f"XGBoost tahmin hatası: {e}")
            return None


# ── LightGBM Wrapper ────────────────────────────────────────────────────────

class LightGBMPredictor:
    """LightGBM tabanlı yön tahmini."""

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.model_path = model_path
        self._load_or_init(model_path)

    def _load_or_init(self, path: Optional[str]):
        if not LIGHTGBM_AVAILABLE:
            return
        if path and os.path.exists(path):
            try:
                self.model = lgb.Booster(model_file=path)
                logger.info(f"LightGBM modeli yüklendi: {path}")
            except Exception as e:
                logger.warning(f"LightGBM yükleme hatası: {e}")

    def train(self, X: np.ndarray, y: np.ndarray) -> bool:
        if not LIGHTGBM_AVAILABLE or len(X) < 30:
            return False
        try:
            train_data = lgb.Dataset(X, label=y)
            params = {
                'objective': 'multiclass',
                'num_class': 3,
                'metric': 'multi_logloss',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'min_child_samples': 10,
                'lambda_l1': 0.1,
                'lambda_l2': 1.0,
                'verbose': -1,
            }
            self.model = lgb.train(
                params, train_data,
                num_boost_round=200,
                valid_sets=[train_data],
                callbacks=[lgb.early_stopping(20, verbose=False)]
            )
            if self.model_path:
                Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
                self.model.save_model(self.model_path)
            return True
        except Exception as e:
            logger.error(f"LightGBM eğitim hatası: {e}")
            return False

    def predict_proba(self, x: np.ndarray) -> Optional[np.ndarray]:
        if not LIGHTGBM_AVAILABLE or self.model is None:
            return None
        try:
            proba = self.model.predict(x.reshape(1, -1))[0]
            return proba  # [SHORT_prob, NEUTRAL_prob, LONG_prob]
        except Exception as e:
            logger.error(f"LightGBM tahmin hatası: {e}")
            return None


# ── LSTM Wrapper ────────────────────────────────────────────────────────────

class LSTMPredictor:
    """LSTM/GRU tabanlı zaman serisi tahmini."""

    SEQ_LEN = 20  # 20 mum geçmişi

    def __init__(self, input_size: int, model_path: Optional[str] = None):
        self.input_size = input_size
        self.model_path = model_path
        self.model = None
        self.optimizer = None
        self._device = None

        if TORCH_AVAILABLE:
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self._load_or_init(model_path)

    def _load_or_init(self, path: Optional[str]):
        if not TORCH_AVAILABLE:
            return
        self.model = LSTMModel(self.input_size).to(self._device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        if path and os.path.exists(path):
            try:
                state = torch.load(path, map_location=self._device)
                self.model.load_state_dict(state['model'])
                self.optimizer.load_state_dict(state['optimizer'])
                logger.info(f"LSTM modeli yüklendi: {path}")
            except Exception as e:
                logger.warning(f"LSTM yükleme hatası: {e}")

    def train(self, sequences: np.ndarray, labels: np.ndarray,
              epochs: int = 20) -> bool:
        """sequences: (N, SEQ_LEN, input_size), labels: (N,) in {0,1,2}"""
        if not TORCH_AVAILABLE or self.model is None or len(sequences) < 30:
            return False
        try:
            X_t = torch.FloatTensor(sequences).to(self._device)
            y_t = torch.LongTensor(labels).to(self._device)
            criterion = nn.CrossEntropyLoss()

            self.model.train()
            for epoch in range(epochs):
                self.optimizer.zero_grad()
                out = self.model(X_t)
                loss = criterion(out, y_t)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            if self.model_path:
                Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()
                }, self.model_path)
            return True
        except Exception as e:
            logger.error(f"LSTM eğitim hatası: {e}")
            return False

    def predict_proba(self, sequence: np.ndarray) -> Optional[np.ndarray]:
        """sequence: (SEQ_LEN, input_size)"""
        if not TORCH_AVAILABLE or self.model is None:
            return None
        try:
            x = torch.FloatTensor(sequence).unsqueeze(0).to(self._device)
            self.model.eval()
            with torch.no_grad():
                logits = self.model(x)
                proba = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            return proba  # [SHORT_prob, NEUTRAL_prob, LONG_prob]
        except Exception as e:
            logger.error(f"LSTM tahmin hatası: {e}")
            return None


# ── Trade History Buffer ────────────────────────────────────────────────────

@dataclass
class TradeRecord:
    features: np.ndarray
    label: int  # 0=SHORT, 1=NEUTRAL, 2=LONG
    pnl_pct: float
    timestamp: datetime = field(default_factory=datetime.now)


class TradeBuffer:
    """İşlem geçmişini tutar, online öğrenme için eğitim seti oluşturur."""

    def __init__(self, max_size: int = 2000, min_pnl_for_label: float = 0.3):
        self.buffer: List[TradeRecord] = []
        self.max_size = max_size
        self.min_pnl_for_label = min_pnl_for_label
        self._lock = threading.Lock()

    def add(self, features: np.ndarray, direction: str, pnl_pct: float):
        """
        İşlem kapandıktan sonra çağrılır.
        label: PNL > threshold → doğru yön (2=LONG, 0=SHORT)
                PNL < -threshold → yanlış (zıt etiket)
                Arası → NEUTRAL (1)
        """
        if features is None:
            return

        if pnl_pct > self.min_pnl_for_label:
            label = 2 if direction == 'LONG' else 0  # kazanan yönü onayla
        elif pnl_pct < -self.min_pnl_for_label:
            label = 0 if direction == 'LONG' else 2  # kaybeden → zıt
        else:
            label = 1  # nötr

        record = TradeRecord(features=features.copy(), label=label, pnl_pct=pnl_pct)

        with self._lock:
            self.buffer.append(record)
            if len(self.buffer) > self.max_size:
                self.buffer.pop(0)

    def get_training_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        with self._lock:
            if len(self.buffer) < 30:
                return None, None
            X = np.array([r.features for r in self.buffer])
            y = np.array([r.label for r in self.buffer])
        return X, y

    def size(self) -> int:
        with self._lock:
            return len(self.buffer)


# ── Ana ML Katmanı ──────────────────────────────────────────────────────────

class MLLayer:
    """
    Tüm ML modellerini yöneten ana sınıf.
    Devre dışıysa veya yeterli veri yoksa AnalysisResult.neutral() döner.
    """

    # Sınıf etiketi → sinyal yönü
    LABEL_MAP = {0: -1.0, 1: 0.0, 2: 1.0}
    LABEL_NAME = {0: 'SHORT', 1: 'NEUTRAL', 2: 'LONG'}

    def __init__(self, settings=None):
        self.settings = settings
        self.enabled = self._check_enabled()
        self.feature_engineer = FeatureEngineer()
        self.trade_buffer = TradeBuffer()
        self._lock = threading.Lock()

        # Model yolları
        self.model_dir = Path('models')

        # Model nesneleri
        self.xgb_model: Optional[XGBoostPredictor] = None
        self.lgb_model: Optional[LightGBMPredictor] = None
        self.lstm_model: Optional[LSTMPredictor] = None

        # Model ağırlıkları (başarı oranına göre güncellenir)
        self.model_weights = {'xgb': 1.0, 'lgb': 1.0, 'lstm': 0.5}

        # Eğitim sayacı
        self._retrain_counter: Dict[str, int] = {}
        self._last_train: Dict[str, datetime] = {}
        self._min_train_interval = timedelta(hours=1)

        if self.enabled:
            self._init_models()
            logger.info(f"MLLayer başlatıldı. XGBoost:{XGBOOST_AVAILABLE} "
                        f"LightGBM:{LIGHTGBM_AVAILABLE} LSTM:{TORCH_AVAILABLE}")

    def _check_enabled(self) -> bool:
        if self.settings:
            return getattr(self.settings, 'ml_enabled', True)
        env_val = os.getenv('ML_ENABLED', 'true').lower()
        return env_val == 'true'

    def _init_models(self):
        n_features = len(FeatureEngineer.FEATURE_NAMES)

        if XGBOOST_AVAILABLE:
            self.xgb_model = XGBoostPredictor(
                model_path=str(self.model_dir / 'xgb_model.json')
            )

        if LIGHTGBM_AVAILABLE:
            self.lgb_model = LightGBMPredictor(
                model_path=str(self.model_dir / 'lgb_model.txt')
            )

        if TORCH_AVAILABLE:
            self.lstm_model = LSTMPredictor(
                input_size=n_features,
                model_path=str(self.model_dir / 'lstm_model.pt')
            )

    def predict(self, symbol: str, df: pd.DataFrame) -> AnalysisResult:
        """
        Ana tahmin metodu.
        Mevcut modellerin ensemble ortalamasını döner.
        """
        if not self.enabled:
            return AnalysisResult.neutral('ML katmanı devre dışı')

        if df is None or len(df) < 50:
            return AnalysisResult.neutral('Yetersiz veri')

        # Otomatik yeniden eğitim
        self._auto_retrain(symbol)

        # Özellik çıkar
        features = self.feature_engineer.extract_features(df)
        if features is None:
            return AnalysisResult.neutral('Özellik çıkarma başarısız')

        # Normalize et
        features_scaled = self.feature_engineer.transform(features)

        predictions = []
        confidences = []

        # XGBoost tahmini
        if self.xgb_model:
            xgb_proba = self.xgb_model.predict_proba(features_scaled)
            if xgb_proba is not None:
                xgb_score = self._proba_to_score(xgb_proba)
                xgb_conf = float(np.max(xgb_proba)) * 100
                predictions.append(('xgb', xgb_score, xgb_conf))

        # LightGBM tahmini
        if self.lgb_model:
            lgb_proba = self.lgb_model.predict_proba(features_scaled)
            if lgb_proba is not None:
                lgb_score = self._proba_to_score(lgb_proba)
                lgb_conf = float(np.max(lgb_proba)) * 100
                predictions.append(('lgb', lgb_score, lgb_conf))

        # LSTM tahmini (son SEQ_LEN mumun özelliklerini kullan)
        if self.lstm_model and len(df) >= LSTMPredictor.SEQ_LEN:
            sequence = self._build_sequence(df, LSTMPredictor.SEQ_LEN)
            if sequence is not None:
                lstm_proba = self.lstm_model.predict_proba(sequence)
                if lstm_proba is not None:
                    lstm_score = self._proba_to_score(lstm_proba)
                    lstm_conf = float(np.max(lstm_proba)) * 100
                    predictions.append(('lstm', lstm_score, lstm_conf))

        if not predictions:
            return AnalysisResult.neutral('Hiçbir model tahmin üretemedi')

        # Ağırlıklı ensemble
        total_weight = sum(self.model_weights.get(name, 1.0) for name, _, _ in predictions)
        weighted_score = sum(
            self.model_weights.get(name, 1.0) * score
            for name, score, _ in predictions
        ) / total_weight

        avg_confidence = np.mean([conf for _, _, conf in predictions])
        # Düşük güven → düşük etki
        adj_confidence = avg_confidence * min(1.0, self.trade_buffer.size() / 100)

        # Detay
        detail = {
            'models': [{'name': n, 'score': round(s, 3), 'confidence': round(c, 1)}
                       for n, s, c in predictions],
            'weighted_score': round(weighted_score, 3),
            'buffer_size': self.trade_buffer.size(),
        }

        return AnalysisResult(
            score=float(np.clip(weighted_score, -1, 1)),
            confidence=float(np.clip(adj_confidence, 0, 100)),
            source='ml_ensemble',
            details=detail,
            timestamp=datetime.now(),
        )

    def _proba_to_score(self, proba: np.ndarray) -> float:
        """[SHORT_p, NEUTRAL_p, LONG_p] → -1 ile +1 arası skor"""
        if len(proba) != 3:
            return 0.0
        return float(proba[2] - proba[0])

    def _build_sequence(self, df: pd.DataFrame, seq_len: int) -> Optional[np.ndarray]:
        """Son seq_len mum için özellik dizisi oluşturur."""
        sequences = []
        try:
            for i in range(seq_len, 0, -1):
                sub_df = df.iloc[:-i] if i > 0 else df
                if len(sub_df) < 50:
                    continue
                feat = self.feature_engineer.extract_features(sub_df)
                if feat is None:
                    return None
                sequences.append(self.feature_engineer.transform(feat))

            if len(sequences) != seq_len:
                return None
            return np.array(sequences)  # (seq_len, n_features)
        except Exception as e:
            logger.error(f"Sekans oluşturma hatası: {e}")
            return None

    def _auto_retrain(self, symbol: str):
        """Yeterli veri varsa ve son eğitimden beri yeterli süre geçtiyse yeniden eğit."""
        last = self._last_train.get(symbol)
        if last and (datetime.now() - last) < self._min_train_interval:
            return

        X, y = self.trade_buffer.get_training_data()
        if X is None:
            return

        # Scaler güncelle
        self.feature_engineer.fit_scaler(X)
        X_scaled = np.array([self.feature_engineer.transform(x) for x in X])

        trained = False
        if self.xgb_model:
            ok = self.xgb_model.train(X_scaled, y)
            if ok:
                trained = True
                logger.info(f"[{symbol}] XGBoost yeniden eğitildi ({len(X)} örnek)")

        if self.lgb_model:
            ok = self.lgb_model.train(X_scaled, y)
            if ok:
                trained = True
                logger.info(f"[{symbol}] LightGBM yeniden eğitildi ({len(X)} örnek)")

        if trained:
            self._last_train[symbol] = datetime.now()

    def record_trade(self, df: pd.DataFrame, direction: str, pnl_pct: float):
        """
        İşlem kapandığında çağrılır. Modeli eğitmek için veri kaydeder.
        """
        if not self.enabled:
            return
        try:
            features = self.feature_engineer.extract_features(df)
            if features is not None:
                self.trade_buffer.add(features, direction, pnl_pct)
        except Exception as e:
            logger.error(f"Trade kayıt hatası: {e}")

    def get_status(self) -> Dict[str, Any]:
        return {
            'enabled': self.enabled,
            'xgboost_available': XGBOOST_AVAILABLE,
            'lightgbm_available': LIGHTGBM_AVAILABLE,
            'lstm_available': TORCH_AVAILABLE,
            'trade_buffer_size': self.trade_buffer.size(),
            'model_weights': self.model_weights,
            'last_train': {k: v.isoformat() for k, v in self._last_train.items()},
        }
