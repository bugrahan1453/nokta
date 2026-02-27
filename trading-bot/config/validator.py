"""
config/validator.py
Konfigürasyon doğrulama modülü.
Başlatma sırasında ayarları kontrol eder ve mantıksız değerlerde hata fırlatır.
"""

from typing import List, Tuple
from config.settings import Settings, SymbolConfig


class ConfigValidationError(Exception):
    """Konfigürasyon doğrulama hatası."""
    pass


class ConfigValidator:
    """Bot konfigürasyonunu doğrular."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate(self) -> Tuple[bool, List[str], List[str]]:
        """
        Tüm konfigürasyonu doğrular.
        Returns: (başarılı_mı, hatalar, uyarılar)
        """
        self.errors = []
        self.warnings = []

        self._validate_binance()
        self._validate_trading_symbols()
        self._validate_risk()
        self._validate_signal()
        self._validate_web()
        self._validate_watchdog()
        self._validate_symbol_configs()

        # Opsiyonel bileşenler için uyarı
        self._check_optional_components()

        return len(self.errors) == 0, self.errors, self.warnings

    # ----------------------------------------------------------------
    def _validate_binance(self):
        """Binance API ayarlarını doğrula."""
        cfg = self.settings.binance

        if not cfg.api_key or cfg.api_key == "your_binance_api_key_here":
            self.errors.append(
                "BINANCE_API_KEY tanımlanmamış veya varsayılan değerde. "
                ".env dosyasını düzenleyin."
            )

        if not cfg.api_secret or cfg.api_secret == "your_binance_api_secret_here":
            self.errors.append(
                "BINANCE_API_SECRET tanımlanmamış veya varsayılan değerde."
            )

        if cfg.testnet:
            self.warnings.append(
                "Bot TESTNET modunda çalışıyor. Gerçek para kullanılmıyor."
            )

    def _validate_trading_symbols(self):
        """İşlem paritelerini doğrula."""
        if not self.settings.trading_symbols:
            self.errors.append(
                "TRADING_SYMBOLS tanımlanmamış. En az bir parite gerekli."
            )
            return

        # Futures paritesi genellikle USDT ile biter
        for symbol in self.settings.trading_symbols:
            if not symbol.endswith("USDT") and not symbol.endswith("BUSD"):
                self.warnings.append(
                    f"{symbol}: Futures bot USDT veya BUSD çifti bekleniyor."
                )

        if len(self.settings.trading_symbols) > 20:
            self.warnings.append(
                f"{len(self.settings.trading_symbols)} parite çok fazla. "
                "Performans sorunları yaşanabilir. 10 veya daha az önerilir."
            )

    def _validate_risk(self):
        """Risk yönetimi ayarlarını doğrula."""
        cfg = self.settings.risk

        if cfg.max_concurrent_positions < 1:
            self.errors.append(
                "MAX_CONCURRENT_POSITIONS en az 1 olmalıdır."
            )

        if cfg.max_concurrent_positions > 20:
            self.warnings.append(
                f"MAX_CONCURRENT_POSITIONS={cfg.max_concurrent_positions} çok yüksek. "
                "Risk artar, maksimum 10 önerilir."
            )

        if cfg.daily_max_loss <= 0:
            self.errors.append(
                "DAILY_MAX_LOSS 0'dan büyük olmalıdır."
            )

        if cfg.daily_max_loss > 10000:
            self.warnings.append(
                f"DAILY_MAX_LOSS={cfg.daily_max_loss} USDT çok yüksek. "
                "Büyük kayıp riski var."
            )

        if not (0 <= cfg.trading_start_hour <= 23):
            self.errors.append(
                "TRADING_START_HOUR 0-23 arasında olmalıdır."
            )

        if not (1 <= cfg.trading_end_hour <= 24):
            self.errors.append(
                "TRADING_END_HOUR 1-24 arasında olmalıdır."
            )

        if cfg.trading_start_hour >= cfg.trading_end_hour and cfg.trading_end_hour != 24:
            self.errors.append(
                "TRADING_START_HOUR TRADING_END_HOUR'dan küçük olmalıdır."
            )

        if cfg.max_funding_rate_pct <= 0:
            self.errors.append("MAX_FUNDING_RATE_PCT 0'dan büyük olmalıdır.")

        if cfg.max_funding_rate_pct > 1.0:
            self.warnings.append(
                f"MAX_FUNDING_RATE_PCT={cfg.max_funding_rate_pct}% çok yüksek. "
                "Yüksek funding rate'de işlem açmak maliyetli."
            )

    def _validate_signal(self):
        """Sinyal motoru ayarlarını doğrula."""
        cfg = self.settings.signal

        valid_timeframes = [1, 3, 5, 15, 30, 60, 120, 240, 480, 720, 1440]
        if cfg.signal_timeframe not in valid_timeframes:
            self.errors.append(
                f"SIGNAL_TIMEFRAME={cfg.signal_timeframe} geçersiz. "
                f"Geçerli değerler: {valid_timeframes}"
            )

        if cfg.trend_timeframe <= cfg.signal_timeframe:
            self.errors.append(
                f"TREND_TIMEFRAME ({cfg.trend_timeframe}) SIGNAL_TIMEFRAME "
                f"({cfg.signal_timeframe})'den büyük olmalıdır."
            )

        if cfg.rsi_overbought <= cfg.rsi_oversold:
            self.errors.append(
                f"RSI_OVERBOUGHT ({cfg.rsi_overbought}) RSI_OVERSOLD "
                f"({cfg.rsi_oversold})'den büyük olmalıdır."
            )

        if cfg.ema_fast >= cfg.ema_slow:
            self.errors.append(
                f"EMA_FAST ({cfg.ema_fast}) EMA_SLOW ({cfg.ema_slow})'den küçük olmalıdır."
            )

        if cfg.volume_threshold < 1.0:
            self.warnings.append(
                f"Volume threshold ({cfg.volume_threshold}) 1.0'dan küçük. "
                "Çok fazla düşük hacimli işarete izin verilebilir."
            )

        if cfg.max_spread_pct > 0.5:
            self.warnings.append(
                f"MAX_SPREAD_PCT={cfg.max_spread_pct}% çok yüksek. "
                "Yüksek spread'li pariteler kârsız işlemlere yol açabilir."
            )

    def _validate_web(self):
        """Web arayüzü ayarlarını doğrula."""
        cfg = self.settings.web

        if cfg.password in ("admin123", "change_this_password_123", ""):
            self.warnings.append(
                "WEB_PASSWORD varsayılan veya boş! Güvenli bir şifre belirleyin."
            )

        if cfg.secret_key in (
            "change-this-secret-key-in-production",
            "change_this_secret_key_very_long_random_string",
            ""
        ):
            self.warnings.append(
                "SECRET_KEY varsayılan değerde! Üretim ortamında mutlaka değiştirin."
            )

        if len(cfg.secret_key) < 32:
            self.warnings.append(
                "SECRET_KEY çok kısa. En az 32 karakter önerilir."
            )

        if not (1024 <= cfg.port <= 65535):
            self.errors.append(
                f"WEB_PORT={cfg.port} geçersiz. 1024-65535 arasında olmalıdır."
            )

        if cfg.debug:
            self.warnings.append(
                "FLASK_DEBUG=true! Üretim ortamında debug modu kapalı olmalıdır."
            )

    def _validate_watchdog(self):
        """Watchdog ayarlarını doğrula."""
        cfg = self.settings.watchdog

        if cfg.interval < 10:
            self.warnings.append(
                f"WATCHDOG_INTERVAL={cfg.interval}sn çok kısa. "
                "Rate limit sorunlarına yol açabilir. 30 önerilir."
            )

        if cfg.interval > 300:
            self.warnings.append(
                f"WATCHDOG_INTERVAL={cfg.interval}sn çok uzun. "
                "Bağlantı sorunlarını geç fark edebilirsiniz."
            )

        if cfg.max_reconnect_attempts < 1:
            self.errors.append("MAX_RECONNECT_ATTEMPTS en az 1 olmalıdır.")

        if cfg.reconnect_delay < 1:
            self.errors.append("RECONNECT_DELAY en az 1 saniye olmalıdır.")

    def _validate_symbol_configs(self):
        """Her parite konfigürasyonunu doğrula."""
        for symbol, cfg in self.settings.symbol_configs.items():
            self._validate_single_symbol(symbol, cfg)

    def _validate_single_symbol(self, symbol: str, cfg: SymbolConfig):
        """Tek bir parite konfigürasyonunu doğrula."""
        prefix = f"{symbol}"

        # Kaldıraç kontrolü
        if cfg.leverage < 1:
            self.errors.append(f"{prefix}: Kaldıraç en az 1 olmalıdır.")

        if cfg.leverage > 125:
            self.errors.append(
                f"{prefix}: Kaldıraç {cfg.leverage}x Binance maksimumunu aşıyor (125x)."
            )

        if cfg.leverage > 20:
            self.warnings.append(
                f"{prefix}: Kaldıraç {cfg.leverage}x çok yüksek risk. Dikkat!"
            )

        # Pozisyon büyüklüğü
        if cfg.position_size_pct <= 0:
            self.errors.append(
                f"{prefix}: position_size_pct 0'dan büyük olmalıdır."
            )

        if cfg.position_size_pct > 50:
            self.errors.append(
                f"{prefix}: position_size_pct %{cfg.position_size_pct} çok yüksek. "
                "Maksimum %50 önerilir."
            )

        if cfg.position_size_pct > 10:
            self.warnings.append(
                f"{prefix}: Kasa'nın %{cfg.position_size_pct}'i çok büyük bir pozisyon."
            )

        # Stop-loss
        if cfg.stop_loss_pct <= 0:
            self.errors.append(
                f"{prefix}: stop_loss_pct 0'dan büyük olmalıdır."
            )

        if cfg.stop_loss_pct > 10:
            self.warnings.append(
                f"{prefix}: Stop-loss %{cfg.stop_loss_pct} çok geniş. "
                "Büyük kayıp riski var."
            )

        # Take-profit
        if cfg.take_profit_pct <= 0:
            self.errors.append(
                f"{prefix}: take_profit_pct 0'dan büyük olmalıdır."
            )

        # Risk/ödül oranı kontrolü
        if cfg.take_profit_pct < cfg.stop_loss_pct:
            self.warnings.append(
                f"{prefix}: Take-profit (%{cfg.take_profit_pct}) stop-loss'tan "
                f"(%{cfg.stop_loss_pct}) küçük. Risk/ödül oranı kötü."
            )

        # Trailing stop
        if cfg.trailing_activation_pct <= 0:
            self.errors.append(
                f"{prefix}: trailing_activation_pct 0'dan büyük olmalıdır."
            )

        if cfg.trailing_callback_pct <= 0:
            self.errors.append(
                f"{prefix}: trailing_callback_pct 0'dan büyük olmalıdır."
            )

        if cfg.trailing_callback_pct >= cfg.trailing_activation_pct:
            self.warnings.append(
                f"{prefix}: trailing_callback_pct ({cfg.trailing_callback_pct}) "
                f"trailing_activation_pct ({cfg.trailing_activation_pct}) kadar veya büyük. "
                "Trailing stop çok erken kapanabilir."
            )

    def _check_optional_components(self):
        """Opsiyonel bileşenlerin durumunu kontrol et."""
        if not self.settings.telegram.enabled:
            self.warnings.append(
                "Telegram bildirimleri devre dışı. İşlem uyarıları alınamaz."
            )
        elif not self.settings.telegram.bot_token or \
                self.settings.telegram.bot_token == "your_telegram_bot_token_here":
            self.warnings.append(
                "TELEGRAM_BOT_TOKEN tanımlanmamış. Telegram bildirimleri çalışmaz."
            )

        if not self.settings.ai.enabled:
            self.warnings.append(
                "ANTHROPIC_API_KEY tanımlanmamış. AI katmanı devre dışı."
            )


def validate_config(settings: Settings) -> None:
    """
    Konfigürasyonu doğrular. Hata varsa ConfigValidationError fırlatır.
    Uyarıları log'a yazar.
    """
    from utils.logger import get_logger
    logger = get_logger(__name__)

    validator = ConfigValidator(settings)
    ok, errors, warnings = validator.validate()

    # Uyarıları logla
    for warning in warnings:
        logger.warning(f"[KONFİG UYARI] {warning}")

    # Hataları logla ve fırlat
    if not ok:
        for error in errors:
            logger.error(f"[KONFİG HATA] {error}")
        raise ConfigValidationError(
            f"Konfigürasyon hatası:\n" + "\n".join(f"  - {e}" for e in errors)
        )

    logger.info(
        f"Konfigürasyon doğrulandı: {len(warnings)} uyarı, {len(errors)} hata"
    )
