"""
core/risk_manager.py
Risk yönetimi modülü.
Pozisyon büyüklüğü hesaplama, çift pozisyon kilidi,
günlük kayıp limiti, kaldıraç doğrulama.
Gelişmiş: ATR tabanlı dinamik SL/TP, Kelly kriteri, Chandelier Exit.
"""

import threading
import numpy as np
import pandas as pd
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

from config.settings import RiskConfig, Settings, SymbolConfig
from core.data_engine import DataEngine
from database.db import DatabaseManager
from utils.logger import get_logger

logger = get_logger(__name__)


# ── ATR Hesaplama Yardımcısı ────────────────────────────────────────────────

def _calc_atr(candles: List[dict], period: int = 14) -> Optional[float]:
    """Mum listesinden ATR hesaplar."""
    if not candles or len(candles) < period + 1:
        return None
    try:
        df = pd.DataFrame(candles)
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        close = df['close'].astype(float)
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        atr = tr.ewm(span=period, adjust=False).mean()
        return float(atr.iloc[-1])
    except Exception:
        return None


def _calc_win_rate_avg_rr(trade_history: List[dict]) -> Tuple[float, float]:
    """
    İşlem geçmişinden win rate ve ortalama kazanç/kayıp oranı hesaplar.
    Returns: (win_rate, avg_win_loss_ratio)
    """
    if not trade_history:
        return 0.5, 1.5  # Varsayılan

    wins = [t['pnl_usdt'] for t in trade_history if t.get('pnl_usdt', 0) > 0]
    losses = [abs(t['pnl_usdt']) for t in trade_history if t.get('pnl_usdt', 0) < 0]

    if not wins and not losses:
        return 0.5, 1.5

    win_rate = len(wins) / len(trade_history) if trade_history else 0.5

    avg_win = sum(wins) / len(wins) if wins else 1.0
    avg_loss = sum(losses) / len(losses) if losses else 1.0
    avg_rr = avg_win / avg_loss if avg_loss > 0 else 1.5

    return win_rate, avg_rr


class RiskCheckResult:
    """Risk kontrolü sonucu."""

    def __init__(self, approved: bool, reason: str = "", details: dict = None):
        self.approved = approved
        self.reason = reason
        self.details = details or {}

    def __bool__(self):
        return self.approved

    def __repr__(self):
        return f"RiskCheck(approved={self.approved}, reason='{self.reason}')"


class RiskManager:
    """
    İşlem öncesi tüm risk kontrollerini yapan sınıf.
    Thread-safe tasarım.
    """

    def __init__(
        self,
        settings: Settings,
        data_engine: DataEngine,
        db: DatabaseManager,
    ):
        self.settings = settings
        self.risk_cfg = settings.risk
        self.data = data_engine
        self.db = db

        # Günlük kayıp takibi
        self._daily_loss: float = 0.0
        self._daily_loss_date: Optional[date] = None
        self._daily_loss_lock = threading.Lock()

        # Pozisyon kilitleri (parite -> True)
        self._position_locks: Dict[str, bool] = {}
        self._lock = threading.RLock()

        # Durduruldu mu?
        self._stopped_by_daily_limit = False

        # Günlük istatistikleri senkronize et
        self._sync_daily_loss()

        logger.info("RiskManager başlatıldı.")

    # ----------------------------------------------------------------
    # Günlük Kayıp Yönetimi
    # ----------------------------------------------------------------

    def _sync_daily_loss(self):
        """Veritabanından bugünkü kaybı senkronize eder."""
        try:
            today = date.today()
            stats = self.db.get_today_stats()
            with self._daily_loss_lock:
                # Sadece negatif PNL (kayıp)
                self._daily_loss = abs(min(0.0, stats.get("total_pnl_usdt", 0.0)))
                self._daily_loss_date = today

            logger.debug(f"Günlük kayıp senkronize edildi: {self._daily_loss:.2f} USDT")
        except Exception as e:
            logger.error(f"Günlük kayıp senkronizasyon hatası: {e}")

    def record_trade_result(self, pnl_usdt: float):
        """
        İşlem kapandığında PNL kaydeder.
        Günlük kayıp limitini günceller.
        """
        with self._daily_loss_lock:
            today = date.today()
            # Tarih değişti mi?
            if self._daily_loss_date != today:
                self._daily_loss = 0.0
                self._daily_loss_date = today
                self._stopped_by_daily_limit = False
                logger.info("Yeni gün, günlük kayıp sıfırlandı.")

            if pnl_usdt < 0:
                self._daily_loss += abs(pnl_usdt)
                logger.info(
                    f"Günlük kayıp güncellendi: {self._daily_loss:.2f} USDT / "
                    f"{self.risk_cfg.daily_max_loss:.2f} USDT limit"
                )

                if self._daily_loss >= self.risk_cfg.daily_max_loss:
                    self._stopped_by_daily_limit = True
                    logger.warning(
                        f"GÜNLÜK KAYIP LİMİTİ AŞILDI! "
                        f"{self._daily_loss:.2f} USDT >= {self.risk_cfg.daily_max_loss:.2f} USDT. "
                        f"Bot durduruldu."
                    )

    @property
    def daily_loss(self) -> float:
        with self._daily_loss_lock:
            return self._daily_loss

    @property
    def daily_loss_remaining(self) -> float:
        """Kalan günlük kayıp marjı."""
        with self._daily_loss_lock:
            return max(0.0, self.risk_cfg.daily_max_loss - self._daily_loss)

    @property
    def daily_loss_pct(self) -> float:
        """Günlük kayıp limitinin yüzde kaçı kullanıldı."""
        if self.risk_cfg.daily_max_loss == 0:
            return 100.0
        return (self._daily_loss / self.risk_cfg.daily_max_loss) * 100

    # ----------------------------------------------------------------
    # Pozisyon Kilidi
    # ----------------------------------------------------------------

    def acquire_position_lock(self, symbol: str) -> bool:
        """
        Parite için pozisyon kilidi alır.
        Zaten kilitliyse False döner (çift pozisyon önlemi).
        """
        with self._lock:
            if self._position_locks.get(symbol, False):
                return False
            self._position_locks[symbol] = True
            return True

    def release_position_lock(self, symbol: str):
        """Parite pozisyon kilidini serbest bırakır."""
        with self._lock:
            self._position_locks[symbol] = False

    def is_symbol_locked(self, symbol: str) -> bool:
        """Paritede açık pozisyon var mı?"""
        with self._lock:
            return self._position_locks.get(symbol, False)

    def get_locked_symbols(self) -> list:
        """Kilitlenen tüm pariteleri döndürür."""
        with self._lock:
            return [sym for sym, locked in self._position_locks.items() if locked]

    # ----------------------------------------------------------------
    # Pozisyon Büyüklüğü Hesaplama
    # ----------------------------------------------------------------

    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        leverage: int,
    ) -> Tuple[float, float]:
        """
        Pozisyon büyüklüğünü hesaplar.

        Returns:
            (quantity, margin_usdt)
            quantity: İşlem miktarı (coin)
            margin_usdt: Kullanılan marj (USDT)
        """
        sym_cfg = self.settings.get_symbol_config(symbol)
        if not sym_cfg:
            raise ValueError(f"{symbol} için konfigürasyon bulunamadı.")

        # Mevcut bakiye
        balance = self.data.get_futures_balance()
        available_balance = balance.get("available", 0.0)
        total_balance = balance.get("total", 0.0)

        if total_balance <= 0:
            raise ValueError("Bakiye sıfır veya negatif.")

        # Kullanılacak marj = toplam kasa * position_size_pct / 100
        margin_usdt = total_balance * (sym_cfg.position_size_pct / 100.0)

        # Mevcut bakiyeden büyük olmamalı
        margin_usdt = min(margin_usdt, available_balance * 0.95)  # %5 güvenlik marjı

        if margin_usdt <= 0:
            raise ValueError(
                f"Yetersiz bakiye: {available_balance:.2f} USDT mevcut, "
                f"minimum {total_balance * (sym_cfg.position_size_pct / 100):.2f} USDT gerekli."
            )

        # Notional değer = marj * kaldıraç
        notional_value = margin_usdt * leverage

        # Miktarı hesapla
        if entry_price <= 0:
            raise ValueError("Giriş fiyatı sıfır veya negatif.")

        quantity = notional_value / entry_price

        # Parite hassasiyetine göre yuvarlama
        exchange_info = self.data.get_exchange_info(symbol)
        if exchange_info:
            qty_precision = exchange_info.get("quantity_precision", 3)
            quantity = round(quantity, qty_precision)

        logger.debug(
            f"{symbol} pozisyon büyüklüğü: {quantity} @ {entry_price} "
            f"(marj: {margin_usdt:.2f} USDT, kaldıraç: {leverage}x)"
        )

        return quantity, margin_usdt

    def calculate_sl_tp(
        self,
        symbol: str,
        entry_price: float,
        direction: str,
        candles: Optional[List[dict]] = None,
        atr_sl_multiplier: float = 2.0,
        atr_tp_multiplier: float = 3.0,
    ) -> Tuple[float, float]:
        """
        Stop-loss ve take-profit fiyatlarını hesaplar.
        Eğer candles verilirse ATR tabanlı dinamik SL/TP kullanır.
        Aksi halde konfigürasyondaki sabit yüzdeler kullanılır.

        Args:
            atr_sl_multiplier: ATR × bu çarpan = SL mesafesi
            atr_tp_multiplier: ATR × bu çarpan = TP mesafesi

        Returns:
            (stop_loss_price, take_profit_price)
        """
        sym_cfg = self.settings.get_symbol_config(symbol)
        if not sym_cfg:
            raise ValueError(f"{symbol} konfigürasyonu bulunamadı.")

        exchange_info = self.data.get_exchange_info(symbol)
        price_precision = 2
        if exchange_info:
            price_precision = exchange_info.get("price_precision", 2)

        # ATR tabanlı dinamik SL/TP dene
        use_atr = False
        if candles and len(candles) >= 20:
            atr_val = _calc_atr(candles, period=14)
            if atr_val and atr_val > 0:
                sl_dist = atr_val * atr_sl_multiplier
                tp_dist = atr_val * atr_tp_multiplier

                # SL çok dar değil mi? (en az %0.1 mesafe)
                min_sl_pct = 0.001
                if sl_dist / entry_price >= min_sl_pct:
                    use_atr = True

                    if direction == "LONG":
                        stop_loss = entry_price - sl_dist
                        take_profit = entry_price + tp_dist
                    elif direction == "SHORT":
                        stop_loss = entry_price + sl_dist
                        take_profit = entry_price - tp_dist
                    else:
                        raise ValueError(f"Geçersiz yön: {direction}")

                    stop_loss = round(stop_loss, price_precision)
                    take_profit = round(take_profit, price_precision)

                    sl_pct = sl_dist / entry_price * 100
                    tp_pct = tp_dist / entry_price * 100
                    logger.debug(
                        f"{symbol} ATR SL/TP: SL={stop_loss} ({sl_pct:.2f}%), "
                        f"TP={take_profit} ({tp_pct:.2f}%), ATR={atr_val:.4f}"
                    )
                    return stop_loss, take_profit

        # Sabit yüzde tabanlı SL/TP (fallback veya ATR yoksa)
        sl_pct = sym_cfg.stop_loss_pct / 100.0
        tp_pct = sym_cfg.take_profit_pct / 100.0

        if direction == "LONG":
            stop_loss = entry_price * (1 - sl_pct)
            take_profit = entry_price * (1 + tp_pct)
        elif direction == "SHORT":
            stop_loss = entry_price * (1 + sl_pct)
            take_profit = entry_price * (1 - tp_pct)
        else:
            raise ValueError(f"Geçersiz yön: {direction}")

        stop_loss = round(stop_loss, price_precision)
        take_profit = round(take_profit, price_precision)

        return stop_loss, take_profit

    def calculate_chandelier_exit(
        self,
        candles: List[dict],
        direction: str,
        multiplier: float = 3.0,
        period: int = 22,
    ) -> Optional[float]:
        """
        Chandelier Exit trailing stop fiyatı hesaplar.
        Trailing stop yönetimi için executor'a referans seviye sağlar.

        LONG: Highest High - ATR × multiplier
        SHORT: Lowest Low + ATR × multiplier

        Returns: Chandelier exit fiyatı veya None
        """
        if not candles or len(candles) < period + 1:
            return None
        try:
            df = pd.DataFrame(candles)
            high = df['high'].astype(float)
            low = df['low'].astype(float)
            close = df['close'].astype(float)

            tr = pd.concat([
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs()
            ], axis=1).max(axis=1)
            atr = tr.ewm(span=period, adjust=False).mean().iloc[-1]

            if direction == "LONG":
                highest_high = high.rolling(period).max().iloc[-1]
                return round(float(highest_high - atr * multiplier), 6)
            elif direction == "SHORT":
                lowest_low = low.rolling(period).min().iloc[-1]
                return round(float(lowest_low + atr * multiplier), 6)
            return None
        except Exception as e:
            logger.debug(f"Chandelier Exit hesaplama hatası: {e}")
            return None

    def calculate_position_size_kelly(
        self,
        symbol: str,
        entry_price: float,
        leverage: int,
        trade_history: Optional[List[dict]] = None,
        kelly_fraction: float = 0.25,
        min_pct: float = 0.5,
        max_pct: float = 5.0,
    ) -> Tuple[float, float]:
        """
        Kelly Kriteri ile pozisyon büyüklüğü hesaplar.
        Güvenlik için fraksiyonel Kelly kullanır (varsayılan: %25).

        Kelly formülü: f = (p × b - q) / b
            p = win_rate, q = 1 - p
            b = average win / average loss (ödül/risk oranı)

        Args:
            kelly_fraction: Kelly miktarının kullanılacak fraksiyonu (0.25 = %25 Kelly)
            min_pct: Minimum pozisyon büyüklüğü (kasa %)
            max_pct: Maksimum pozisyon büyüklüğü (kasa %)

        Returns: (quantity, margin_usdt)
        """
        sym_cfg = self.settings.get_symbol_config(symbol)
        if not sym_cfg:
            raise ValueError(f"{symbol} için konfigürasyon bulunamadı.")

        # Win rate ve ortalama kazanç/kayıp oranı
        if trade_history and len(trade_history) >= 10:
            win_rate, avg_rr = _calc_win_rate_avg_rr(trade_history)
        else:
            # Yeterli geçmiş yok → varsayılan veya config kullan
            return self.calculate_position_size(symbol, entry_price, leverage)

        # Kelly formülü
        p = win_rate
        q = 1 - p
        b = avg_rr

        kelly_f = (p * b - q) / b if b > 0 else 0

        # Negatif Kelly → işlem açma
        if kelly_f <= 0:
            logger.info(f"{symbol} Kelly negatif ({kelly_f:.3f}) - config kullanılıyor")
            return self.calculate_position_size(symbol, entry_price, leverage)

        # Fraksiyonel Kelly
        position_pct = kelly_f * kelly_fraction * 100  # yüzdeye çevir

        # Sınırlama
        position_pct = float(np.clip(position_pct, min_pct, max_pct))

        logger.info(
            f"{symbol} Kelly pozisyon: {position_pct:.2f}% "
            f"(win_rate={win_rate:.2f}, avg_rr={avg_rr:.2f}, kelly={kelly_f:.3f})"
        )

        # Bakiye hesaplama
        balance = self.data.get_futures_balance()
        available_balance = balance.get("available", 0.0)
        total_balance = balance.get("total", 0.0)

        if total_balance <= 0:
            raise ValueError("Bakiye sıfır veya negatif.")

        margin_usdt = total_balance * (position_pct / 100.0)
        margin_usdt = min(margin_usdt, available_balance * 0.95)

        if margin_usdt <= 0:
            raise ValueError("Yetersiz bakiye.")

        notional_value = margin_usdt * leverage
        quantity = notional_value / entry_price

        exchange_info = self.data.get_exchange_info(symbol)
        if exchange_info:
            qty_precision = exchange_info.get("quantity_precision", 3)
            quantity = round(quantity, qty_precision)

        return quantity, margin_usdt

    # ----------------------------------------------------------------
    # Risk Kontrol Kapısı
    # ----------------------------------------------------------------

    def check_pre_trade(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
    ) -> RiskCheckResult:
        """
        İşlem açılmadan önce tüm risk kontrollerini yapar.
        Tüm kontroller geçilmedikçe False döner.
        """
        # 1. Günlük limit kontrolü
        if self._stopped_by_daily_limit:
            return RiskCheckResult(
                False,
                f"Günlük kayıp limiti aşıldı: {self._daily_loss:.2f} USDT "
                f"/ {self.risk_cfg.daily_max_loss:.2f} USDT",
            )

        # 2. Çalışma saati kontrolü
        current_hour = datetime.utcnow().hour
        if not (self.risk_cfg.trading_start_hour <= current_hour < self.risk_cfg.trading_end_hour):
            return RiskCheckResult(
                False,
                f"Çalışma saati dışı: {current_hour:02d}:xx "
                f"(İzin: {self.risk_cfg.trading_start_hour:02d}:00 - "
                f"{self.risk_cfg.trading_end_hour:02d}:00)",
            )

        # 3. Çift pozisyon kilidi
        if self.is_symbol_locked(symbol):
            return RiskCheckResult(
                False,
                f"{symbol} için zaten açık pozisyon var.",
            )

        # 4. Maksimum pozisyon sayısı
        open_count = len(self.get_locked_symbols())
        if open_count >= self.risk_cfg.max_concurrent_positions:
            return RiskCheckResult(
                False,
                f"Maksimum eş zamanlı pozisyon sayısına ulaşıldı: "
                f"{open_count}/{self.risk_cfg.max_concurrent_positions}",
            )

        # 5. Bakiye kontrolü
        balance = self.data.get_futures_balance()
        available = balance.get("available", 0)
        if available < 10:  # Minimum 10 USDT
            return RiskCheckResult(
                False,
                f"Yetersiz bakiye: {available:.2f} USDT (minimum 10 USDT gerekli)",
            )

        # 6. Parite konfigürasyonu kontrolü
        sym_cfg = self.settings.get_symbol_config(symbol)
        if not sym_cfg:
            return RiskCheckResult(False, f"{symbol} için konfigürasyon yok")

        if not sym_cfg.enabled:
            return RiskCheckResult(False, f"{symbol} devre dışı bırakılmış")

        # 7. Kaldıraç doğrulama (Binance'e ayarlanmış mı?)
        try:
            open_positions = self.data.get_open_positions()
            for pos in open_positions:
                if pos["symbol"] == symbol:
                    if pos["leverage"] != sym_cfg.leverage:
                        logger.warning(
                            f"{symbol} kaldıraç uyuşmazlığı: "
                            f"Beklenen {sym_cfg.leverage}x, "
                            f"Mevcut {pos['leverage']}x"
                        )
                    break
        except Exception as e:
            logger.warning(f"Kaldıraç doğrulama hatası (geçiliyor): {e}")

        # 8. Funding rate kontrolü
        funding_rate = self.data.get_funding_rate(symbol)
        if abs(funding_rate) > self.risk_cfg.max_funding_rate_pct:
            return RiskCheckResult(
                False,
                f"Funding rate çok yüksek: {funding_rate:.4f}% "
                f"(limit: {self.risk_cfg.max_funding_rate_pct}%)",
            )

        # Tüm kontroller geçildi
        return RiskCheckResult(
            True,
            "Tüm risk kontrolleri geçildi.",
            details={
                "available_balance": available,
                "daily_loss": self._daily_loss,
                "daily_loss_remaining": self.daily_loss_remaining,
                "open_positions": open_count,
                "funding_rate": funding_rate,
            },
        )

    def warn_high_funding(self, symbol: str) -> Optional[str]:
        """Yüksek funding rate uyarısı döndürür (varsa)."""
        funding = self.data.get_funding_rate(symbol)
        threshold = self.risk_cfg.max_funding_rate_pct * 0.7  # %70 eşiğinde uyar
        if abs(funding) > threshold:
            return (
                f"{symbol} funding rate uyarısı: {funding:.4f}% "
                f"(eşik: {threshold:.4f}%)"
            )
        return None

    def get_status(self) -> dict:
        """Risk yöneticisinin güncel durumunu döndürür."""
        return {
            "stopped_by_daily_limit": self._stopped_by_daily_limit,
            "daily_loss_usdt": round(self._daily_loss, 4),
            "daily_loss_limit": self.risk_cfg.daily_max_loss,
            "daily_loss_pct": round(self.daily_loss_pct, 2),
            "daily_loss_remaining": round(self.daily_loss_remaining, 4),
            "locked_symbols": self.get_locked_symbols(),
            "open_position_count": len(self.get_locked_symbols()),
            "max_concurrent_positions": self.risk_cfg.max_concurrent_positions,
        }
