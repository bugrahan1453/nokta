"""
core/risk_manager.py
Risk yönetimi modülü.
Pozisyon büyüklüğü hesaplama, çift pozisyon kilidi,
günlük kayıp limiti, kaldıraç doğrulama.
"""

import threading
from datetime import date, datetime
from typing import Dict, Optional, Tuple

from config.settings import RiskConfig, Settings, SymbolConfig
from core.data_engine import DataEngine
from database.db import DatabaseManager
from utils.logger import get_logger

logger = get_logger(__name__)


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
    ) -> Tuple[float, float]:
        """
        Stop-loss ve take-profit fiyatlarını hesaplar.

        Returns:
            (stop_loss_price, take_profit_price)
        """
        sym_cfg = self.settings.get_symbol_config(symbol)
        if not sym_cfg:
            raise ValueError(f"{symbol} konfigürasyonu bulunamadı.")

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

        # Fiyat hassasiyetine göre yuvarlama
        exchange_info = self.data.get_exchange_info(symbol)
        if exchange_info:
            price_precision = exchange_info.get("price_precision", 2)
            stop_loss = round(stop_loss, price_precision)
            take_profit = round(take_profit, price_precision)

        return stop_loss, take_profit

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
