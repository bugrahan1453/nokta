"""
core/watchdog.py
Bağlantı ve sistem sağlık monitörü.
Her 30sn internet ve Binance erişim kontrolü, maintenance algılama,
açık pozisyon varken bağlantı kopunca güvenli mod.
"""

import threading
import time
from datetime import datetime
from enum import Enum
from typing import Optional

import requests

from core.data_engine import DataEngine
from notifications.telegram import TelegramNotifier
from utils.logger import get_logger

logger = get_logger(__name__)


class SystemStatus(Enum):
    """Sistem durumu."""
    NORMAL = "normal"
    DEGRADED = "degraded"       # Kısmi sorun
    MAINTENANCE = "maintenance"  # Binance bakımda
    OFFLINE = "offline"          # Tamamen çevrimdışı
    SAFE_MODE = "safe_mode"      # Güvenli mod (pozisyon açıkken bağlantı koptu)


class WatchdogStatus:
    """Watchdog durum raporu."""

    def __init__(self):
        self.system_status = SystemStatus.NORMAL
        self.internet_ok = True
        self.binance_ok = True
        self.ws_ok = True
        self.last_check = None
        self.consecutive_failures = 0
        self.safe_mode_reason = ""
        self.maintenance_message = ""

    def to_dict(self) -> dict:
        return {
            "system_status": self.system_status.value,
            "internet_ok": self.internet_ok,
            "binance_ok": self.binance_ok,
            "websocket_ok": self.ws_ok,
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "consecutive_failures": self.consecutive_failures,
            "safe_mode_reason": self.safe_mode_reason,
            "maintenance_message": self.maintenance_message,
        }


class Watchdog:
    """
    Sistem bağlantı ve sağlık monitörü.
    Periyodik kontroller yapar ve sorun tespitinde aksiyon alır.
    """

    def __init__(
        self,
        data_engine: DataEngine,
        telegram: TelegramNotifier,
        interval: int = 30,
        max_reconnect_attempts: int = 5,
        reconnect_delay: int = 5,
    ):
        self.data = data_engine
        self.telegram = telegram
        self.interval = interval
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay = reconnect_delay

        self.status = WatchdogStatus()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()

        # Dış referanslar (main.py tarafından atanır)
        self.executor = None  # Güvenli mod için
        self.bot_controller = None  # Bot durdurmak için

        # İnternet kontrolü için test URL'leri
        self._internet_test_urls = [
            "https://www.google.com",
            "https://cloudflare.com",
            "https://8.8.8.8",
        ]

        # Binance status API
        self._binance_status_url = "https://www.binancezh.com/api/v3/ping"
        self._binance_futures_url = "https://fapi.binance.com/fapi/v1/ping"

        logger.info(f"Watchdog başlatıldı (aralık: {interval}sn)")

    # ----------------------------------------------------------------
    # Başlatma / Durdurma
    # ----------------------------------------------------------------

    def start(self):
        """Watchdog izleme döngüsünü başlatır."""
        self._running = True
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="Watchdog",
        )
        self._thread.start()
        logger.info("Watchdog başlatıldı.")

    def stop(self):
        """Watchdog'u durdurur."""
        self._running = False
        logger.info("Watchdog durduruldu.")

    # ----------------------------------------------------------------
    # Ana Döngü
    # ----------------------------------------------------------------

    def _run(self):
        """Watchdog ana döngüsü."""
        while self._running:
            try:
                self._perform_checks()
            except Exception as e:
                logger.error(f"Watchdog kontrol hatası: {e}")

            time.sleep(self.interval)

    def _perform_checks(self):
        """Tüm bağlantı ve sağlık kontrollerini yapar."""
        with self._lock:
            self.status.last_check = datetime.utcnow()

            # 1. İnternet bağlantısı
            internet_ok = self._check_internet()
            if not internet_ok and self.status.internet_ok:
                # Yeni internet kesintisi
                logger.warning("İnternet bağlantısı kesildi!")
                self.telegram.send_alert(
                    "🔴 İnternet bağlantısı kesildi! Bot güvenli moda geçiyor."
                )
                self._enter_safe_mode("İnternet bağlantısı yok")

            elif internet_ok and not self.status.internet_ok:
                # İnternet geri geldi
                logger.info("İnternet bağlantısı geri geldi.")
                self.telegram.send_alert("🟢 İnternet bağlantısı geri geldi.")
                self._exit_safe_mode()

            self.status.internet_ok = internet_ok

            if not internet_ok:
                self.status.system_status = SystemStatus.OFFLINE
                return

            # 2. Binance erişimi
            binance_ok, maintenance = self._check_binance()

            if maintenance:
                if self.status.system_status != SystemStatus.MAINTENANCE:
                    logger.warning("Binance bakım modu tespit edildi!")
                    self.telegram.send_alert(
                        "🔧 Binance bakım modunda. Bot durduruldu."
                    )
                    self._stop_bot_safely("Binance bakım modu")
                self.status.system_status = SystemStatus.MAINTENANCE
                self.status.maintenance_message = "Binance bakım modunda"
                return

            if not binance_ok and self.status.binance_ok:
                logger.warning("Binance'e erişilemiyor!")
                self.telegram.send_alert(
                    "🔴 Binance'e erişilemiyor! Bağlantı sorunu."
                )
                self.status.consecutive_failures += 1

                if self.status.consecutive_failures >= 3:
                    self._enter_safe_mode("Binance erişim sorunu")

            elif binance_ok and not self.status.binance_ok:
                logger.info("Binance bağlantısı geri geldi.")
                self.status.consecutive_failures = 0
                if self.status.system_status == SystemStatus.SAFE_MODE:
                    self._exit_safe_mode()

            self.status.binance_ok = binance_ok

            # 3. WebSocket durumu
            ws_status = self.data.get_ws_status()
            ws_stale = ws_status.get("is_stale", False)

            if ws_stale and self.status.ws_ok:
                logger.warning("WebSocket verisi eskimiş (>30sn güncelleme yok)")
                self._restart_websocket()

            self.status.ws_ok = not ws_stale

            # 4. Genel durum güncelle
            if binance_ok and internet_ok and not ws_stale:
                self.status.system_status = SystemStatus.NORMAL
                self.status.consecutive_failures = 0
                self.status.maintenance_message = ""
            elif binance_ok and internet_ok:
                self.status.system_status = SystemStatus.DEGRADED

            logger.debug(
                f"Watchdog kontrol: İnternet={internet_ok}, "
                f"Binance={binance_ok}, WS={'OK' if not ws_stale else 'ESKIMIŞ'}"
            )

    # ----------------------------------------------------------------
    # Bağlantı Kontrolleri
    # ----------------------------------------------------------------

    def _check_internet(self) -> bool:
        """İnternet bağlantısını test eder."""
        for url in self._internet_test_urls:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code < 500:
                    return True
            except Exception:
                continue
        return False

    def _check_binance(self) -> tuple:
        """
        Binance erişimini test eder.
        Returns: (erişilebilir_mi, bakımda_mı)
        """
        # Önce ping
        try:
            response = requests.get(self._binance_futures_url, timeout=10)
            if response.status_code == 200:
                return True, False
        except Exception:
            pass

        # Alternatif: Binance ana site
        try:
            response = requests.get("https://api.binance.com/api/v3/ping", timeout=10)
            if response.status_code == 200:
                return True, False
        except Exception:
            pass

        # Bakım kontrolü
        try:
            response = requests.get(
                "https://www.binancezh.com/en/support/announcement/c-48",
                timeout=10,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            # Bakım sayfasında "maintenance" kelimesi varsa
            if response.status_code == 200 and "maintenance" in response.text.lower():
                return False, True
        except Exception:
            pass

        return False, False

    # ----------------------------------------------------------------
    # Tepki Eylemleri
    # ----------------------------------------------------------------

    def _restart_websocket(self):
        """WebSocket'i yeniden başlatır."""
        try:
            logger.info("WebSocket yeniden başlatılıyor...")
            self.data.stop_websocket()
            time.sleep(2)
            symbols = self.data._subscribed_symbols
            if symbols:
                self.data.start_websocket(symbols)
                logger.info("WebSocket yeniden başlatıldı.")
        except Exception as e:
            logger.error(f"WebSocket yeniden başlatma hatası: {e}")

    def _enter_safe_mode(self, reason: str):
        """
        Güvenli moda geçer.
        Açık pozisyon varsa kapatır veya yeni işlem açılmasını durdurur.
        """
        with self._lock:
            if self.status.system_status == SystemStatus.SAFE_MODE:
                return

            self.status.system_status = SystemStatus.SAFE_MODE
            self.status.safe_mode_reason = reason

        logger.warning(f"GÜVENLİ MOD AKTİF: {reason}")

        # Executor mevcut ise açık pozisyonları kontrol et
        if self.executor:
            open_trades = self.executor.get_open_trades_info()
            if open_trades:
                logger.warning(
                    f"{len(open_trades)} açık pozisyon var. "
                    "Bağlantı düzelene kadar izleniyor."
                )
                # Burada pozisyonları hemen kapatmak riskli olabilir,
                # bu yüzden sadece uyarı gönder
                self.telegram.send_alert(
                    f"⚠️ GÜVENLİ MOD: {reason}\n"
                    f"{len(open_trades)} açık pozisyon var!\n"
                    "Manuel kontrol önerilir."
                )
            else:
                logger.info("Güvenli modda açık pozisyon yok.")

        # Yeni işlem açılmasını durdur
        if self.bot_controller:
            self.bot_controller.pause_trading("Watchdog güvenli modu")

    def _exit_safe_mode(self):
        """Güvenli moddan çıkar."""
        with self._lock:
            if self.status.system_status != SystemStatus.SAFE_MODE:
                return

            self.status.system_status = SystemStatus.NORMAL
            self.status.safe_mode_reason = ""

        logger.info("Güvenli mod devre dışı bırakıldı, normal işleme devam.")

        if self.bot_controller:
            self.bot_controller.resume_trading()

        self.telegram.send_alert("🟢 Güvenli mod devre dışı bırakıldı. İşlemler devam ediyor.")

    def _stop_bot_safely(self, reason: str):
        """Botu güvenli şekilde durdurur."""
        logger.warning(f"Bot güvenli durdurma: {reason}")

        if self.bot_controller:
            self.bot_controller.pause_trading(reason)

        self.telegram.send_alert(
            f"🛑 Bot durduruldu: {reason}\n"
            "Sorun çözüldüğünde manuel olarak yeniden başlatın."
        )

    # ----------------------------------------------------------------
    # Durum Sorgulama
    # ----------------------------------------------------------------

    def get_status(self) -> dict:
        """Watchdog durum raporunu döndürür."""
        with self._lock:
            return self.status.to_dict()

    def is_system_healthy(self) -> bool:
        """Sistem sağlıklı mı?"""
        with self._lock:
            return self.status.system_status in (
                SystemStatus.NORMAL, SystemStatus.DEGRADED
            )

    def force_check(self):
        """Anlık kontrol yapar (test için)."""
        self._perform_checks()
        return self.get_status()
