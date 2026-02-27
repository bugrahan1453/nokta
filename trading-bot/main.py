"""
main.py
Binance Futures Trading Bot - Ana Başlatıcı.

Tüm bileşenleri başlatır, bot döngüsünü çalıştırır,
Flask web sunucusunu ayrı thread'de yönetir.
"""

import asyncio
import os
import signal
import sys
import threading
import time
from datetime import datetime, timedelta
from typing import Optional

# Proje kök dizinini Python yoluna ekle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from config.settings import settings
from config.validator import validate_config, ConfigValidationError
from database.db import init_db, get_db
from utils.logger import setup_logging, get_logger

from core.data_engine import DataEngine
from core.signal_engine import SignalEngine
from core.risk_manager import RiskManager
from core.executor import Executor
from core.watchdog import Watchdog
from core.ai_layer import AILayer
from core.advanced_analysis import AdvancedAnalysisManager
from core.ml_layer import MLLayer
from core.sentiment import SentimentLayer
from core.ensemble import EnsembleAggregator
from notifications.telegram import TelegramNotifier
from web.app import init_app, run_web_server

logger = get_logger(__name__)


# ============================================================
# Bot Kontrolcüsü
# ============================================================

class BotController:
    """
    Trading bot yaşam döngüsünü yöneten ana kontrolcü.
    Başlat/durdur/duraklat işlemlerini koordine eder.
    """

    def __init__(self):
        self._status = "stopped"  # stopped, running, paused
        self._pause_reason = ""
        self._lock = threading.RLock()
        self._start_time: Optional[datetime] = None
        self._signal_interval = settings.signal.signal_timeframe  # dakika

        # Bileşenler (init sırasında atanır)
        self.data_engine: Optional[DataEngine] = None
        self.signal_engine: Optional[SignalEngine] = None
        self.risk_manager: Optional[RiskManager] = None
        self.executor: Optional[Executor] = None
        self.watchdog: Optional[Watchdog] = None
        self.ai_layer: Optional[AILayer] = None
        self.telegram: Optional[TelegramNotifier] = None
        self.scheduler: Optional[BackgroundScheduler] = None
        self.ml_layer: Optional[MLLayer] = None
        self.sentiment_layer: Optional[SentimentLayer] = None
        self.ensemble_aggregator: Optional[EnsembleAggregator] = None

        # Ana sinyal döngüsü
        self._bot_thread: Optional[threading.Thread] = None
        self._running = False

    def get_status(self) -> str:
        with self._lock:
            return self._status

    def start(self):
        """Botu başlatır."""
        with self._lock:
            if self._status == "running":
                logger.warning("Bot zaten çalışıyor.")
                return
            self._status = "running"
            self._pause_reason = ""
            self._start_time = datetime.utcnow()

        if not self._running:
            self._running = True
            self._bot_thread = threading.Thread(
                target=self._run_loop,
                daemon=True,
                name="BotController",
            )
            self._bot_thread.start()

        logger.info("Bot başlatıldı.")

    def stop(self):
        """Botu tamamen durdurur."""
        with self._lock:
            self._status = "stopped"
        self._running = False
        logger.info("Bot durduruldu.")

        if self.telegram:
            self.telegram.send_bot_stopped("Manuel durdurma")

    def pause_trading(self, reason: str = ""):
        """İşlem açılmasını geçici olarak durdurur (açık pozisyonlar etkilenmez)."""
        with self._lock:
            if self._status != "stopped":
                self._status = "paused"
                self._pause_reason = reason
        logger.warning(f"Bot duraklatıldı: {reason}")

    def resume_trading(self):
        """Duraklatılmış botu devam ettirir."""
        with self._lock:
            if self._status == "paused":
                self._status = "running"
                self._pause_reason = ""
        logger.info("Bot devam ettiriliyor.")

    # ----------------------------------------------------------------
    # Ana Sinyal Döngüsü
    # ----------------------------------------------------------------

    def _run_loop(self):
        """
        Ana bot döngüsü.
        Her N dakikada bir sinyal analizi yapar ve gerekirse işlem açar.
        """
        logger.info("Bot sinyal döngüsü başlatıldı.")

        while self._running:
            try:
                current_status = self.get_status()

                if current_status == "running":
                    self._process_signals()
                elif current_status == "paused":
                    logger.debug(f"Bot duraklatıldı ({self._pause_reason}), sinyal işlenmedi.")

                # Sinyal zaman dilimine kadar bekle (dakika * 60 saniye)
                # Küçük aralıklarla kontrol ederek daha hızlı durdurulabilir
                wait_seconds = self._signal_interval * 60
                for _ in range(wait_seconds):
                    if not self._running:
                        break
                    time.sleep(1)

            except Exception as e:
                logger.error(f"Bot döngü hatası: {e}", exc_info=True)
                time.sleep(10)

        logger.info("Bot sinyal döngüsü sonlandı.")

    def _process_signals(self):
        """Tüm pariteler için sinyal analizi yapar ve uygun ise işlem açar."""
        if not self.signal_engine or not self.risk_manager or not self.executor:
            return

        for symbol in settings.trading_symbols:
            try:
                self._analyze_symbol(symbol)
            except Exception as e:
                logger.error(f"{symbol} işlem analiz hatası: {e}")

    def _analyze_symbol(self, symbol: str):
        """Tek parite için tam analiz ve işlem açma sürecini çalıştırır."""
        # 1. Risk öncesi kontrol
        current_price = self.data_engine.price_cache.get(symbol) or 0.0
        if not current_price:
            logger.debug(f"{symbol} fiyat verisi yok, atlanıyor.")
            return

        # 2. AI anomali kontrolü
        if self.ai_layer and self.ai_layer.enabled:
            # Basit pump/dump algılama
            candles = self.data_engine.kline_cache.get(
                symbol,
                self._get_interval_str(settings.signal.signal_timeframe)
            )
            if candles and len(candles) >= 2:
                prev_price = candles[-2]["close"]
                last_volume = candles[-1]["volume"]
                avg_vol = sum(c["volume"] for c in candles[-20:-1]) / 19 if len(candles) >= 20 else last_volume
                anomaly = self.ai_layer.detect_anomaly(
                    symbol, current_price, prev_price, last_volume, avg_vol
                )
                if anomaly and anomaly["severity"] == "HIGH":
                    logger.warning(f"{symbol} yüksek anomali, işlem açılmıyor: {anomaly}")
                    return

        # 3. Haber kara bölge kontrolü
        if self.ai_layer and self.ai_layer.enabled:
            in_blackout, reason = self.ai_layer.is_in_news_blackout()
            if in_blackout:
                logger.info(f"Haber kara bölgesi aktif: {reason}. İşlem açılmıyor.")
                return

        # 4. Sinyal analizi (AI rejimini geçir)
        ai_regime = None
        if self.ai_layer and self.ai_layer.enabled:
            candles_for_regime = self.data_engine.kline_cache.get(
                symbol,
                self._get_interval_str(settings.signal.signal_timeframe)
            )
            try:
                regime_result = self.ai_layer.analyze_market_regime(
                    symbol, candles_for_regime,
                    self.data_engine.get_funding_rate(symbol)
                )
                if regime_result.success:
                    ai_regime = regime_result.data.get('regime')
                    if not regime_result.data.get('should_trade', True):
                        logger.info(
                            f"{symbol} AI rejimi işlem öneriyor değil: "
                            f"{regime_result.data.get('reason')}"
                        )
                        return
            except Exception as e:
                logger.debug(f"AI rejim analiz hatası: {e}")

        signal = self.signal_engine.analyze(symbol, ai_regime=ai_regime)

        if not signal.approved:
            logger.debug(f"{symbol} sinyal ret: {signal.reject_reason}")
            return

        # 5. Risk kontrolü
        risk_check = self.risk_manager.check_pre_trade(
            symbol, signal.direction.value, current_price
        )

        if not risk_check.approved:
            logger.debug(f"{symbol} risk ret: {risk_check.reason}")
            return

        # 6. Funding rate uyarısı
        funding_warning = self.risk_manager.warn_high_funding(symbol)
        if funding_warning:
            logger.warning(funding_warning)
            if self.telegram:
                self.telegram.send_funding_rate_warning(
                    symbol,
                    self.data_engine.get_funding_rate(symbol),
                    settings.risk.max_funding_rate_pct * 0.7
                )

        # 7. İşlem aç
        logger.info(
            f"[SINYAL ONAYLANDI] {symbol} {signal.direction.value} "
            f"güç={signal.strength.value} fiyat={current_price}"
        )

        result = self.executor.open_position(
            symbol=symbol,
            direction=signal.direction.value,
            signal_data=signal.to_dict(),
        )

        if not result.success:
            logger.warning(f"{symbol} işlem açılamadı: {result.message}")

    @staticmethod
    def _get_interval_str(minutes: int) -> str:
        """Dakikayı Binance interval formatına çevirir."""
        interval_map = {1: "1m", 3: "3m", 5: "5m", 15: "15m", 30: "30m", 60: "1h"}
        return interval_map.get(minutes, f"{minutes}m")


# ============================================================
# Zamanlayıcı Görevler
# ============================================================

def setup_scheduler(
    bot_controller: BotController,
    db,
    telegram: TelegramNotifier,
    ai_layer: AILayer,
) -> BackgroundScheduler:
    """APScheduler zamanlayıcı görevlerini kurar."""
    scheduler = BackgroundScheduler(timezone="UTC")

    # Günlük rapor - her gün 23:55'te
    def send_daily_report():
        try:
            stats = db.get_today_stats()
            telegram.send_daily_report(stats)
            logger.info("Günlük rapor gönderildi.")
        except Exception as e:
            logger.error(f"Günlük rapor hatası: {e}")

    scheduler.add_job(
        send_daily_report,
        CronTrigger(hour=23, minute=55),
        id="daily_report",
        name="Günlük Rapor",
        replace_existing=True,
    )

    # Sabah AI raporu
    morning_time = settings.ai.morning_report_time.split(":")
    if len(morning_time) == 2:
        def send_morning_report():
            try:
                if not ai_layer.enabled:
                    return
                today_stats = db.get_today_stats()
                report_result = ai_layer.generate_morning_report(
                    settings.trading_symbols,
                    today_stats,
                    settings.to_dict(),
                )
                if report_result.success:
                    telegram.send_morning_report(report_result.data)
                logger.info("Sabah AI raporu gönderildi.")
            except Exception as e:
                logger.error(f"Sabah AI raporu hatası: {e}")

        scheduler.add_job(
            send_morning_report,
            CronTrigger(hour=int(morning_time[0]), minute=int(morning_time[1])),
            id="morning_report",
            name="Sabah AI Raporu",
            replace_existing=True,
        )

    # Risk manager günlük sıfırlama - her gün saat 00:01'de
    def reset_daily_tracker():
        try:
            if bot_controller.risk_manager:
                bot_controller.risk_manager._sync_daily_loss()
            logger.info("Günlük kayıp takibi sıfırlandı.")
        except Exception as e:
            logger.error(f"Günlük sıfırlama hatası: {e}")

    scheduler.add_job(
        reset_daily_tracker,
        CronTrigger(hour=0, minute=1),
        id="daily_reset",
        name="Günlük Sıfırlama",
        replace_existing=True,
    )

    # Ekonomik takvim kontrolü (her saatte)
    def check_calendar():
        try:
            if ai_layer.enabled:
                in_blackout, reason = ai_layer.is_in_news_blackout()
                if in_blackout:
                    bot_controller.pause_trading(f"Haber kara bölgesi: {reason}")
                    telegram.send_alert(f"📰 Haber kara bölgesi: {reason}\nBot duraklatıldı.")
                elif bot_controller.get_status() == "paused":
                    bot_controller.resume_trading()
        except Exception as e:
            logger.error(f"Takvim kontrol hatası: {e}")

    scheduler.add_job(
        check_calendar,
        IntervalTrigger(minutes=settings.ai.calendar_check_interval),
        id="calendar_check",
        name="Ekonomik Takvim Kontrolü",
        replace_existing=True,
    )

    scheduler.start()
    logger.info("Zamanlayıcı başlatıldı.")
    return scheduler


# ============================================================
# Başlatma
# ============================================================

def initialize() -> BotController:
    """
    Tüm bileşenleri sırayla başlatır.
    Returns: Tam yapılandırılmış BotController
    """
    # 1. Log sistemi
    setup_logging(
        level=settings.log.level,
        log_file=settings.log.file,
        max_size_mb=settings.log.max_size_mb,
        backup_count=settings.log.backup_count,
    )

    logger.info("=" * 60)
    logger.info("Binance Futures Trading Bot başlatılıyor...")
    logger.info(f"Pariteler: {settings.trading_symbols}")
    logger.info(f"Testnet: {settings.binance.testnet}")
    logger.info("=" * 60)

    # 2. Config doğrulama
    try:
        validate_config(settings)
    except ConfigValidationError as e:
        logger.critical(f"Konfigürasyon hatası:\n{e}")
        sys.exit(1)

    # 3. Veritabanı
    db = init_db(settings.database.url)

    # 4. Telegram
    telegram = TelegramNotifier(
        bot_token=settings.telegram.bot_token,
        chat_id=settings.telegram.chat_id,
        enabled=settings.telegram.enabled,
    )

    # 5. Veri Motoru
    data_engine = DataEngine(
        api_key=settings.binance.api_key,
        api_secret=settings.binance.api_secret,
        testnet=settings.binance.testnet,
    )

    if not data_engine.connect():
        logger.critical("Binance bağlantısı kurulamadı. Bot durduruluyor.")
        telegram.send_error("DataEngine", "Binance bağlantısı kurulamadı!")
        sys.exit(1)

    # 6. Tarihsel veri yükle
    intervals = [
        BotController._get_interval_str(settings.signal.signal_timeframe),
        BotController._get_interval_str(settings.signal.trend_timeframe),
    ]
    data_engine.preload_klines(settings.trading_symbols, intervals, limit=200)

    # 7. WebSocket başlat
    data_engine.start_websocket(settings.trading_symbols, intervals)

    # 8. AI katmanı
    ai_layer = AILayer(
        api_key=settings.ai.api_key,
        model=settings.ai.model,
        db=db,
    )

    # 9. Risk Manager
    risk_manager = RiskManager(settings, data_engine, db)

    # 10. Gelişmiş Analiz Modülleri (opsiyonel)
    adv_cfg = settings.advanced

    advanced_manager = AdvancedAnalysisManager(
        data_engine=data_engine,
        orderbook_enabled=adv_cfg.orderbook_enabled,
        oi_enabled=adv_cfg.oi_enabled,
        fear_greed_enabled=adv_cfg.fear_greed_enabled,
        glassnode_api_key=adv_cfg.glassnode_api_key if adv_cfg.glassnode_enabled else "",
        cryptoquant_api_key=adv_cfg.cryptoquant_api_key if adv_cfg.cryptoquant_enabled else "",
        whale_enabled=adv_cfg.whale_enabled,
        whale_min_usdt=adv_cfg.whale_min_usdt,
    )

    ml_layer = None
    if adv_cfg.ml_enabled:
        ml_layer = MLLayer(settings=adv_cfg)
        logger.info("ML katmanı başlatıldı.")

    sentiment_layer = None
    if adv_cfg.sentiment_enabled:
        sentiment_layer = SentimentLayer(settings=adv_cfg)
        logger.info("Sentiment katmanı başlatıldı.")

    ensemble_aggregator = EnsembleAggregator(settings=adv_cfg)

    # 11. Sinyal Motoru (5 katmanlı)
    signal_engine = SignalEngine(
        data_engine=data_engine,
        signal_config=settings.signal,
        max_funding_rate=settings.risk.max_funding_rate_pct,
        advanced_manager=advanced_manager if adv_cfg.layer5_enabled else None,
        ml_layer=ml_layer,
        sentiment_layer=sentiment_layer,
        ensemble_aggregator=ensemble_aggregator,
        layer4_enabled=adv_cfg.layer4_enabled,
        layer4_strict=adv_cfg.layer4_strict,
        layer5_enabled=adv_cfg.layer5_enabled,
    )

    # 12. Executor
    executor = Executor(
        settings=settings,
        data_engine=data_engine,
        risk_manager=risk_manager,
        db=db,
        telegram=telegram,
    )
    executor.start_monitoring()

    # 13. Watchdog
    watchdog = Watchdog(
        data_engine=data_engine,
        telegram=telegram,
        interval=settings.watchdog.interval,
        max_reconnect_attempts=settings.watchdog.max_reconnect_attempts,
        reconnect_delay=settings.watchdog.reconnect_delay,
    )

    # 14. Bot Kontrolcüsü
    bot_controller = BotController()
    bot_controller.data_engine = data_engine
    bot_controller.signal_engine = signal_engine
    bot_controller.risk_manager = risk_manager
    bot_controller.executor = executor
    bot_controller.watchdog = watchdog
    bot_controller.ai_layer = ai_layer
    bot_controller.telegram = telegram
    bot_controller.ml_layer = ml_layer
    bot_controller.sentiment_layer = sentiment_layer
    bot_controller.ensemble_aggregator = ensemble_aggregator

    # Watchdog'a referansları ver
    watchdog.executor = executor
    watchdog.bot_controller = bot_controller
    watchdog.start()

    # 15. Web uygulamasını yapılandır
    init_app(
        settings=settings,
        data_engine=data_engine,
        signal_engine=signal_engine,
        risk_manager=risk_manager,
        executor=executor,
        watchdog=watchdog,
        ai_layer=ai_layer,
        db=db,
        bot_controller=bot_controller,
    )

    # 16. Zamanlayıcı
    scheduler = setup_scheduler(bot_controller, db, telegram, ai_layer)
    bot_controller.scheduler = scheduler

    # 17. Başlangıç bildirimi
    telegram.send_bot_started(settings.trading_symbols, settings.binance.testnet)

    logger.info("Tüm bileşenler başarıyla başlatıldı.")
    return bot_controller


# ============================================================
# Kapatma
# ============================================================

def shutdown(bot_controller: BotController, signum=None, frame=None):
    """Botu güvenli şekilde kapatır."""
    logger.info("Kapatma sinyali alındı...")

    # Botu durdur
    if bot_controller:
        bot_controller.stop()

        # Zamanlayıcıyı durdur
        if bot_controller.scheduler and bot_controller.scheduler.running:
            bot_controller.scheduler.shutdown(wait=False)

        # WebSocket'i kapat
        if bot_controller.data_engine:
            bot_controller.data_engine.disconnect()

        # Executor izlemeyi durdur
        if bot_controller.executor:
            bot_controller.executor.stop_monitoring()

        # Watchdog'u durdur
        if bot_controller.watchdog:
            bot_controller.watchdog.stop()

        # Telegram'ı kapat
        if bot_controller.telegram:
            bot_controller.telegram.stop()

    logger.info("Bot başarıyla kapatıldı.")
    sys.exit(0)


# ============================================================
# Ana Giriş Noktası
# ============================================================

def main():
    """Ana fonksiyon - botu başlatır."""
    # Bileşenleri başlat
    bot_controller = initialize()

    # Sinyal işleyicileri
    signal.signal(signal.SIGINT, lambda s, f: shutdown(bot_controller, s, f))
    signal.signal(signal.SIGTERM, lambda s, f: shutdown(bot_controller, s, f))

    # Botu başlat
    bot_controller.start()

    # Web sunucusunu ayrı thread'de başlat
    web_thread = threading.Thread(
        target=run_web_server,
        kwargs={
            "host": settings.web.host,
            "port": settings.web.port,
            "debug": settings.web.debug,
        },
        daemon=True,
        name="WebServer",
    )
    web_thread.start()

    logger.info(f"Web arayüzü http://{settings.web.host}:{settings.web.port} adresinde")
    logger.info("Bot çalışıyor. Durdurmak için Ctrl+C kullanın.")

    # Ana thread'i canlı tut
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        shutdown(bot_controller)


if __name__ == "__main__":
    main()
