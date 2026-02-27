"""
config/settings.py
Bot ayarları ve konfigürasyon yönetimi.
.env dosyasından değerleri okur ve her bileşen için yapılandırılmış ayar nesneleri sağlar.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv()


def _env(key: str, default=None, cast=str):
    """Ortam değişkenini oku ve belirtilen tipe dönüştür."""
    val = os.getenv(key, default)
    if val is None:
        return None
    if cast == bool:
        return str(val).lower() in ("true", "1", "yes")
    try:
        return cast(val)
    except (ValueError, TypeError):
        return default


# ============================================================
# Binance API Ayarları
# ============================================================
@dataclass
class BinanceConfig:
    api_key: str = field(default_factory=lambda: _env("BINANCE_API_KEY", ""))
    api_secret: str = field(default_factory=lambda: _env("BINANCE_API_SECRET", ""))
    testnet: bool = field(default_factory=lambda: _env("BINANCE_TESTNET", False, bool))
    testnet_url: str = field(
        default_factory=lambda: _env(
            "BINANCE_TESTNET_URL", "https://testnet.binancefuture.com"
        )
    )

    @property
    def base_url(self) -> str:
        if self.testnet:
            return self.testnet_url
        return "https://fapi.binance.com"


# ============================================================
# Claude AI Ayarları
# ============================================================
@dataclass
class AIConfig:
    api_key: str = field(default_factory=lambda: _env("ANTHROPIC_API_KEY", ""))
    model: str = field(
        default_factory=lambda: _env("ANTHROPIC_MODEL", "claude-opus-4-6")
    )
    morning_report_time: str = field(
        default_factory=lambda: _env("AI_MORNING_REPORT_TIME", "08:00")
    )
    calendar_check_interval: int = field(
        default_factory=lambda: _env("CALENDAR_CHECK_INTERVAL", 60, int)
    )
    news_blackout_minutes: int = field(
        default_factory=lambda: _env("NEWS_BLACKOUT_MINUTES", 30, int)
    )
    enabled: bool = field(
        default_factory=lambda: bool(_env("ANTHROPIC_API_KEY", ""))
    )


# ============================================================
# Telegram Ayarları
# ============================================================
@dataclass
class TelegramConfig:
    bot_token: str = field(
        default_factory=lambda: _env("TELEGRAM_BOT_TOKEN", "")
    )
    chat_id: str = field(
        default_factory=lambda: _env("TELEGRAM_CHAT_ID", "")
    )
    enabled: bool = field(
        default_factory=lambda: _env("TELEGRAM_ENABLED", True, bool)
    )


# ============================================================
# Web Arayüzü Ayarları
# ============================================================
@dataclass
class WebConfig:
    password: str = field(
        default_factory=lambda: _env("WEB_PASSWORD", "admin123")
    )
    secret_key: str = field(
        default_factory=lambda: _env("SECRET_KEY", "change-this-secret-key-in-production")
    )
    port: int = field(default_factory=lambda: _env("WEB_PORT", 5000, int))
    host: str = field(default_factory=lambda: _env("WEB_HOST", "0.0.0.0"))
    debug: bool = field(default_factory=lambda: _env("FLASK_DEBUG", False, bool))


# ============================================================
# Parite Bazlı Ayarlar
# ============================================================
@dataclass
class SymbolConfig:
    """Her parite için ayrı konfigürasyon."""
    symbol: str = ""
    leverage: int = 10
    position_size_pct: float = 2.0      # Kasa yüzdesi
    stop_loss_pct: float = 1.5           # Stop-loss yüzdesi
    take_profit_pct: float = 3.0         # Take-profit yüzdesi
    trailing_activation_pct: float = 1.0 # Trailing stop aktifleşme PNL%
    trailing_callback_pct: float = 0.5   # Trailing geri çekilme %
    enabled: bool = True


def load_symbol_config(symbol: str) -> SymbolConfig:
    """
    Bir parite için konfigürasyon yükler.
    Önce parite özel değerlere, sonra varsayılanlara bakar.
    """
    prefix = symbol.upper()
    return SymbolConfig(
        symbol=symbol,
        leverage=_env(f"{prefix}_LEVERAGE",
                      _env("DEFAULT_LEVERAGE", 10, int), int),
        position_size_pct=_env(f"{prefix}_POSITION_SIZE_PCT",
                                _env("DEFAULT_POSITION_SIZE_PCT", 2.0, float), float),
        stop_loss_pct=_env(f"{prefix}_STOP_LOSS_PCT",
                           _env("DEFAULT_STOP_LOSS_PCT", 1.5, float), float),
        take_profit_pct=_env(f"{prefix}_TAKE_PROFIT_PCT",
                             _env("DEFAULT_TAKE_PROFIT_PCT", 3.0, float), float),
        trailing_activation_pct=_env(f"{prefix}_TRAILING_ACTIVATION_PCT",
                                      _env("DEFAULT_TRAILING_ACTIVATION_PCT", 1.0, float), float),
        trailing_callback_pct=_env(f"{prefix}_TRAILING_CALLBACK_PCT",
                                    _env("DEFAULT_TRAILING_CALLBACK_PCT", 0.5, float), float),
    )


# ============================================================
# Risk Yönetimi Ayarları
# ============================================================
@dataclass
class RiskConfig:
    max_concurrent_positions: int = field(
        default_factory=lambda: _env("MAX_CONCURRENT_POSITIONS", 3, int)
    )
    daily_max_loss: float = field(
        default_factory=lambda: _env("DAILY_MAX_LOSS", 100.0, float)
    )
    max_funding_rate_pct: float = field(
        default_factory=lambda: _env("MAX_FUNDING_RATE_PCT", 0.1, float)
    )
    trading_start_hour: int = field(
        default_factory=lambda: _env("TRADING_START_HOUR", 0, int)
    )
    trading_end_hour: int = field(
        default_factory=lambda: _env("TRADING_END_HOUR", 24, int)
    )


# ============================================================
# Sinyal Motoru Ayarları
# ============================================================
@dataclass
class SignalConfig:
    signal_timeframe: int = field(
        default_factory=lambda: _env("SIGNAL_TIMEFRAME", 5, int)
    )
    trend_timeframe: int = field(
        default_factory=lambda: _env("TREND_TIMEFRAME", 15, int)
    )
    # EMA periyotları
    ema_fast: int = 21
    ema_slow: int = 50
    # RSI ayarları
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    # MACD ayarları
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    # Hacim analizi: son N mumun hacim ortalaması ile karşılaştırma
    volume_lookback: int = 20
    volume_threshold: float = 1.5   # Ortalama hacmin kaç katı olmalı
    # Spread limiti (%)
    max_spread_pct: float = 0.1
    # Kovalamaca koruması: son N dakikada fiyat hareketi limiti (%)
    chase_protection_pct: float = 0.5
    # Tarihsel veri kaç mum çekilsin
    lookback_candles: int = 200


# ============================================================
# Watchdog Ayarları
# ============================================================
@dataclass
class WatchdogConfig:
    interval: int = field(
        default_factory=lambda: _env("WATCHDOG_INTERVAL", 30, int)
    )
    max_reconnect_attempts: int = field(
        default_factory=lambda: _env("MAX_RECONNECT_ATTEMPTS", 5, int)
    )
    reconnect_delay: int = field(
        default_factory=lambda: _env("RECONNECT_DELAY", 5, int)
    )


# ============================================================
# Veritabanı Ayarları
# ============================================================
@dataclass
class DatabaseConfig:
    path: str = field(
        default_factory=lambda: _env("DATABASE_PATH", "trading_bot.db")
    )

    @property
    def url(self) -> str:
        return f"sqlite:///{self.path}"


# ============================================================
# Log Ayarları
# ============================================================
@dataclass
class LogConfig:
    level: str = field(
        default_factory=lambda: _env("LOG_LEVEL", "INFO")
    )
    file: str = field(
        default_factory=lambda: _env("LOG_FILE", "logs/trading_bot.log")
    )
    max_size_mb: int = field(
        default_factory=lambda: _env("LOG_MAX_SIZE_MB", 50, int)
    )
    backup_count: int = field(
        default_factory=lambda: _env("LOG_BACKUP_COUNT", 5, int)
    )


# ============================================================
# Ana Konfigürasyon Sınıfı
# ============================================================
class Settings:
    """Tüm bot ayarlarını tek noktadan yöneten sınıf."""

    def __init__(self):
        self.binance = BinanceConfig()
        self.ai = AIConfig()
        self.telegram = TelegramConfig()
        self.web = WebConfig()
        self.risk = RiskConfig()
        self.signal = SignalConfig()
        self.watchdog = WatchdogConfig()
        self.database = DatabaseConfig()
        self.log = LogConfig()

        # İşlem yapılacak pariteler
        symbols_env = _env("TRADING_SYMBOLS", "BTCUSDT,ETHUSDT")
        self.trading_symbols: List[str] = [
            s.strip().upper() for s in symbols_env.split(",") if s.strip()
        ]

        # Her parite için ayrı konfigürasyon yükle
        self.symbol_configs: Dict[str, SymbolConfig] = {
            symbol: load_symbol_config(symbol)
            for symbol in self.trading_symbols
        }

    def get_symbol_config(self, symbol: str) -> Optional[SymbolConfig]:
        """Belirtilen parite için konfigürasyon döndürür."""
        return self.symbol_configs.get(symbol.upper())

    def update_symbol_config(self, symbol: str, **kwargs) -> None:
        """
        Çalışma zamanında parite konfigürasyonunu günceller.
        Web arayüzünden ayar değişikliği için kullanılır.
        """
        symbol = symbol.upper()
        if symbol not in self.symbol_configs:
            self.symbol_configs[symbol] = SymbolConfig(symbol=symbol)

        config = self.symbol_configs[symbol]
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

    def to_dict(self) -> dict:
        """Tüm ayarları sözlük olarak döndürür (hassas veriler maskelenir)."""
        return {
            "trading_symbols": self.trading_symbols,
            "max_concurrent_positions": self.risk.max_concurrent_positions,
            "daily_max_loss": self.risk.daily_max_loss,
            "trading_hours": f"{self.risk.trading_start_hour:02d}:00 - {self.risk.trading_end_hour:02d}:00",
            "signal_timeframe": self.signal.signal_timeframe,
            "trend_timeframe": self.signal.trend_timeframe,
            "testnet": self.binance.testnet,
            "telegram_enabled": self.telegram.enabled,
            "ai_enabled": self.ai.enabled,
            "symbol_configs": {
                sym: {
                    "leverage": cfg.leverage,
                    "position_size_pct": cfg.position_size_pct,
                    "stop_loss_pct": cfg.stop_loss_pct,
                    "take_profit_pct": cfg.take_profit_pct,
                    "trailing_activation_pct": cfg.trailing_activation_pct,
                    "trailing_callback_pct": cfg.trailing_callback_pct,
                    "enabled": cfg.enabled,
                }
                for sym, cfg in self.symbol_configs.items()
            },
        }


# Tek örnek (singleton) - tüm modüller bu nesneyi import eder
settings = Settings()
