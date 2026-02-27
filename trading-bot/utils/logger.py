"""
utils/logger.py
Merkezi log yönetimi.
Tüm modüller bu modülden logger alır.
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path

# Log formatı
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: str = "INFO",
    log_file: str = "logs/trading_bot.log",
    max_size_mb: int = 50,
    backup_count: int = 5,
) -> None:
    """
    Uygulama geneli log sistemini yapılandırır.
    main.py'den bir kez çağrılmalıdır.
    """
    # Log klasörü yoksa oluştur
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Kök logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Önceki handler'ları temizle
    root_logger.handlers.clear()

    # ---- Konsol handler ----
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Renkli konsol formatı
    try:
        import colorlog
        colored_formatter = colorlog.ColoredFormatter(
            "%(log_color)s" + LOG_FORMAT,
            datefmt=DATE_FORMAT,
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        )
        console_handler.setFormatter(colored_formatter)
    except ImportError:
        console_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))

    root_logger.addHandler(console_handler)

    # ---- Dosya handler (rotating) ----
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=max_size_mb * 1024 * 1024,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)  # Dosyaya her şeyi yaz
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    root_logger.addHandler(file_handler)

    # Üçüncü parti kütüphanelerin çok fazla log üretmesini engelle
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("websocket").setLevel(logging.WARNING)
    logging.getLogger("telegram").setLevel(logging.WARNING)
    logging.getLogger("apscheduler").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    root_logger.info(f"Log sistemi başlatıldı: seviye={level}, dosya={log_file}")


def get_logger(name: str) -> logging.Logger:
    """Belirtilen isim için logger döndürür."""
    return logging.getLogger(name)


class DatabaseLogHandler(logging.Handler):
    """Log kayıtlarını veritabanına yazan handler."""

    def __init__(self, db_session_factory):
        super().__init__()
        self.db_session_factory = db_session_factory
        self.setLevel(logging.WARNING)  # Sadece uyarı ve üstü DB'ye

    def emit(self, record: logging.LogRecord):
        try:
            from database.db import ErrorLog
            session = self.db_session_factory()
            log_entry = ErrorLog(
                level=record.levelname,
                module=record.name,
                message=self.format(record),
            )
            session.add(log_entry)
            session.commit()
            session.close()
        except Exception:
            # DB log yazarken hata olursa sessizce geç (sonsuz döngü önlemi)
            pass
