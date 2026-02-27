"""
database/db.py
SQLite veritabanı katmanı - SQLAlchemy ORM ile.
Tablo modelleri ve veritabanı yardımcı işlevleri burada tanımlanır.
"""

import json
from datetime import datetime, date
from typing import List, Optional, Dict, Any

from sqlalchemy import (
    create_engine, Column, Integer, Float, String,
    Boolean, DateTime, Date, Text, JSON, ForeignKey,
    func, desc
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.pool import StaticPool

from utils.logger import get_logger

logger = get_logger(__name__)

Base = declarative_base()


# ============================================================
# Tablo Modelleri
# ============================================================

class Trade(Base):
    """İşlem (Trade) tablosu - tüm açılan/kapanan pozisyonlar."""
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    # İşlem kimliği (Binance order ID)
    binance_order_id = Column(String(64), nullable=True, index=True)
    # Parite
    symbol = Column(String(20), nullable=False, index=True)
    # Yön: LONG veya SHORT
    side = Column(String(10), nullable=False)
    # Durum: OPEN, CLOSED, CANCELLED
    status = Column(String(20), nullable=False, default="OPEN")
    # Giriş verileri
    entry_price = Column(Float, nullable=False)
    entry_time = Column(DateTime, default=datetime.utcnow, nullable=False)
    quantity = Column(Float, nullable=False)
    leverage = Column(Integer, nullable=False)
    # Çıkış verileri
    exit_price = Column(Float, nullable=True)
    exit_time = Column(DateTime, nullable=True)
    # Kapanma sebebi: SL, TP, TRAILING, MANUAL, DAILY_LIMIT, WATCHDOG
    close_reason = Column(String(50), nullable=True)
    # PNL hesabı (USDT)
    pnl_usdt = Column(Float, nullable=True)
    pnl_pct = Column(Float, nullable=True)
    # Risk parametreleri
    stop_loss_price = Column(Float, nullable=True)
    take_profit_price = Column(Float, nullable=True)
    # Sinyal verileri (JSON)
    signal_data = Column(JSON, nullable=True)
    # Notlar
    notes = Column(Text, nullable=True)
    # Oluşturma zamanı
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "binance_order_id": self.binance_order_id,
            "symbol": self.symbol,
            "side": self.side,
            "status": self.status,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "quantity": self.quantity,
            "leverage": self.leverage,
            "exit_price": self.exit_price,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "close_reason": self.close_reason,
            "pnl_usdt": self.pnl_usdt,
            "pnl_pct": self.pnl_pct,
            "stop_loss_price": self.stop_loss_price,
            "take_profit_price": self.take_profit_price,
            "signal_data": self.signal_data,
            "notes": self.notes,
        }


class DailyStats(Base):
    """Günlük istatistik tablosu."""
    __tablename__ = "daily_stats"

    id = Column(Integer, primary_key=True, autoincrement=True)
    stat_date = Column(Date, nullable=False, unique=True, index=True)
    # İşlem sayıları
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    # PNL
    total_pnl_usdt = Column(Float, default=0.0)
    best_trade_pnl = Column(Float, default=0.0)
    worst_trade_pnl = Column(Float, default=0.0)
    # Başlangıç/bitiş bakiyesi
    starting_balance = Column(Float, nullable=True)
    ending_balance = Column(Float, nullable=True)
    # Oluşturma
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return (self.winning_trades / self.total_trades) * 100

    def to_dict(self) -> dict:
        return {
            "date": self.stat_date.isoformat() if self.stat_date else None,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": round(self.win_rate, 2),
            "total_pnl_usdt": round(self.total_pnl_usdt, 4),
            "best_trade_pnl": round(self.best_trade_pnl, 4),
            "worst_trade_pnl": round(self.worst_trade_pnl, 4),
            "starting_balance": self.starting_balance,
            "ending_balance": self.ending_balance,
        }


class ErrorLog(Base):
    """Hata log tablosu."""
    __tablename__ = "error_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    level = Column(String(20), nullable=False)
    module = Column(String(100), nullable=True)
    message = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "level": self.level,
            "module": self.module,
            "message": self.message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class ConfigHistory(Base):
    """Konfigürasyon geçmişi tablosu."""
    __tablename__ = "config_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    # Değişikliği yapan (web, api, auto)
    changed_by = Column(String(50), default="web")
    # Değişen alan
    config_key = Column(String(100), nullable=False)
    old_value = Column(Text, nullable=True)
    new_value = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "changed_by": self.changed_by,
            "config_key": self.config_key,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class AIAnalysis(Base):
    """AI analiz sonuçları tablosu."""
    __tablename__ = "ai_analyses"

    id = Column(Integer, primary_key=True, autoincrement=True)
    analysis_type = Column(String(50), nullable=False)  # market_regime, anomaly, morning_report
    symbol = Column(String(20), nullable=True)
    result = Column(JSON, nullable=True)
    raw_response = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "analysis_type": self.analysis_type,
            "symbol": self.symbol,
            "result": self.result,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# ============================================================
# Veritabanı Yöneticisi
# ============================================================

class DatabaseManager:
    """SQLite veritabanı bağlantı ve sorgu yöneticisi."""

    def __init__(self, db_url: str):
        self.db_url = db_url
        self.engine = create_engine(
            db_url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            echo=False,
        )
        self.SessionLocal = sessionmaker(
            bind=self.engine,
            autocommit=False,
            autoflush=False,
        )
        self._initialize()

    def _initialize(self):
        """Tabloları oluştur (yoksa)."""
        try:
            Base.metadata.create_all(self.engine)
            logger.info("Veritabanı tabloları hazır.")
        except Exception as e:
            logger.error(f"Veritabanı başlatma hatası: {e}")
            raise

    def get_session(self) -> Session:
        """Yeni bir veritabanı oturumu döndürür."""
        return self.SessionLocal()

    # ----------------------------------------------------------------
    # İşlem (Trade) İşlemleri
    # ----------------------------------------------------------------

    def save_trade(self, trade_data: dict) -> Trade:
        """Yeni işlem kaydeder."""
        session = self.get_session()
        try:
            trade = Trade(**trade_data)
            session.add(trade)
            session.commit()
            session.refresh(trade)
            logger.info(f"İşlem kaydedildi: {trade.symbol} {trade.side} #{trade.id}")
            return trade
        except Exception as e:
            session.rollback()
            logger.error(f"İşlem kaydetme hatası: {e}")
            raise
        finally:
            session.close()

    def update_trade(self, trade_id: int, **kwargs) -> Optional[Trade]:
        """Mevcut işlemi günceller."""
        session = self.get_session()
        try:
            trade = session.query(Trade).filter(Trade.id == trade_id).first()
            if not trade:
                return None
            for key, value in kwargs.items():
                if hasattr(trade, key):
                    setattr(trade, key, value)
            trade.updated_at = datetime.utcnow()
            session.commit()
            session.refresh(trade)
            return trade
        except Exception as e:
            session.rollback()
            logger.error(f"İşlem güncelleme hatası: {e}")
            raise
        finally:
            session.close()

    def get_open_trades(self) -> List[Trade]:
        """Açık pozisyonları döndürür."""
        session = self.get_session()
        try:
            return session.query(Trade).filter(
                Trade.status == "OPEN"
            ).all()
        finally:
            session.close()

    def get_trade_by_symbol(self, symbol: str) -> Optional[Trade]:
        """Belirtilen parite için açık pozisyonu döndürür."""
        session = self.get_session()
        try:
            return session.query(Trade).filter(
                Trade.symbol == symbol,
                Trade.status == "OPEN"
            ).first()
        finally:
            session.close()

    def get_trades_history(
        self,
        limit: int = 100,
        offset: int = 0,
        symbol: Optional[str] = None,
    ) -> List[Trade]:
        """İşlem geçmişini döndürür."""
        session = self.get_session()
        try:
            query = session.query(Trade).filter(Trade.status != "OPEN")
            if symbol:
                query = query.filter(Trade.symbol == symbol)
            return query.order_by(desc(Trade.entry_time)).limit(limit).offset(offset).all()
        finally:
            session.close()

    def get_daily_pnl(self, days: int = 30) -> List[dict]:
        """Son N gün için günlük PNL döndürür."""
        session = self.get_session()
        try:
            results = session.query(
                func.date(Trade.exit_time).label("date"),
                func.sum(Trade.pnl_usdt).label("total_pnl"),
                func.count(Trade.id).label("trade_count"),
            ).filter(
                Trade.status == "CLOSED",
                Trade.exit_time.isnot(None),
            ).group_by(
                func.date(Trade.exit_time)
            ).order_by(
                func.date(Trade.exit_time)
            ).limit(days).all()

            return [
                {
                    "date": str(r.date),
                    "total_pnl": round(float(r.total_pnl or 0), 4),
                    "trade_count": r.trade_count,
                }
                for r in results
            ]
        finally:
            session.close()

    def get_overall_stats(self) -> dict:
        """Genel istatistikleri döndürür."""
        session = self.get_session()
        try:
            total = session.query(func.count(Trade.id)).filter(
                Trade.status == "CLOSED"
            ).scalar() or 0

            wins = session.query(func.count(Trade.id)).filter(
                Trade.status == "CLOSED",
                Trade.pnl_usdt > 0,
            ).scalar() or 0

            total_pnl = session.query(func.sum(Trade.pnl_usdt)).filter(
                Trade.status == "CLOSED"
            ).scalar() or 0.0

            best = session.query(func.max(Trade.pnl_usdt)).filter(
                Trade.status == "CLOSED"
            ).scalar() or 0.0

            worst = session.query(func.min(Trade.pnl_usdt)).filter(
                Trade.status == "CLOSED"
            ).scalar() or 0.0

            return {
                "total_trades": total,
                "winning_trades": wins,
                "losing_trades": total - wins,
                "win_rate": round((wins / total * 100) if total > 0 else 0, 2),
                "total_pnl_usdt": round(float(total_pnl), 4),
                "best_trade": round(float(best), 4),
                "worst_trade": round(float(worst), 4),
            }
        finally:
            session.close()

    def get_today_stats(self) -> dict:
        """Bugünkü istatistikleri döndürür."""
        session = self.get_session()
        try:
            today = date.today()
            today_start = datetime.combine(today, datetime.min.time())

            closed_today = session.query(Trade).filter(
                Trade.status == "CLOSED",
                Trade.exit_time >= today_start,
            ).all()

            total_pnl = sum(t.pnl_usdt or 0 for t in closed_today)
            wins = sum(1 for t in closed_today if (t.pnl_usdt or 0) > 0)

            return {
                "trade_count": len(closed_today),
                "winning_trades": wins,
                "losing_trades": len(closed_today) - wins,
                "win_rate": round((wins / len(closed_today) * 100) if closed_today else 0, 2),
                "total_pnl_usdt": round(total_pnl, 4),
            }
        finally:
            session.close()

    # ----------------------------------------------------------------
    # Hata Log İşlemleri
    # ----------------------------------------------------------------

    def log_error(self, level: str, module: str, message: str) -> None:
        """Hata kaydeder."""
        session = self.get_session()
        try:
            entry = ErrorLog(level=level, module=module, message=message)
            session.add(entry)
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Hata log kaydetme sorunu: {e}")
        finally:
            session.close()

    def get_error_logs(self, limit: int = 200, level: Optional[str] = None) -> List[ErrorLog]:
        """Hata loglarını döndürür."""
        session = self.get_session()
        try:
            query = session.query(ErrorLog)
            if level:
                query = query.filter(ErrorLog.level == level)
            return query.order_by(desc(ErrorLog.created_at)).limit(limit).all()
        finally:
            session.close()

    # ----------------------------------------------------------------
    # Konfigürasyon Geçmişi
    # ----------------------------------------------------------------

    def save_config_change(
        self, config_key: str, old_value: Any, new_value: Any, changed_by: str = "web"
    ) -> None:
        """Konfigürasyon değişikliğini kaydeder."""
        session = self.get_session()
        try:
            entry = ConfigHistory(
                changed_by=changed_by,
                config_key=config_key,
                old_value=json.dumps(old_value) if old_value is not None else None,
                new_value=json.dumps(new_value) if new_value is not None else None,
            )
            session.add(entry)
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Config geçmişi kaydetme hatası: {e}")
        finally:
            session.close()

    # ----------------------------------------------------------------
    # AI Analiz Sonuçları
    # ----------------------------------------------------------------

    def save_ai_analysis(
        self, analysis_type: str, result: dict, symbol: Optional[str] = None, raw: str = ""
    ) -> None:
        """AI analiz sonucunu kaydeder."""
        session = self.get_session()
        try:
            entry = AIAnalysis(
                analysis_type=analysis_type,
                symbol=symbol,
                result=result,
                raw_response=raw,
            )
            session.add(entry)
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"AI analiz kaydetme hatası: {e}")
        finally:
            session.close()

    def get_latest_ai_analysis(self, analysis_type: str) -> Optional[AIAnalysis]:
        """Belirtilen türdeki en son AI analizini döndürür."""
        session = self.get_session()
        try:
            return session.query(AIAnalysis).filter(
                AIAnalysis.analysis_type == analysis_type
            ).order_by(desc(AIAnalysis.created_at)).first()
        finally:
            session.close()

    # ----------------------------------------------------------------
    # Günlük İstatistik Güncelleme
    # ----------------------------------------------------------------

    def update_daily_stats(self, trade: Trade) -> None:
        """Kapatılan işlem sonrası günlük istatistikleri günceller."""
        if not trade.exit_time or trade.status != "CLOSED":
            return

        session = self.get_session()
        try:
            stat_date = trade.exit_time.date()
            stats = session.query(DailyStats).filter(
                DailyStats.stat_date == stat_date
            ).first()

            if not stats:
                stats = DailyStats(stat_date=stat_date)
                session.add(stats)

            stats.total_trades += 1
            pnl = trade.pnl_usdt or 0.0
            stats.total_pnl_usdt += pnl

            if pnl > 0:
                stats.winning_trades += 1
                if pnl > (stats.best_trade_pnl or 0):
                    stats.best_trade_pnl = pnl
            else:
                stats.losing_trades += 1
                if pnl < (stats.worst_trade_pnl or 0):
                    stats.worst_trade_pnl = pnl

            stats.updated_at = datetime.utcnow()
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Günlük istatistik güncelleme hatası: {e}")
        finally:
            session.close()


# ============================================================
# Singleton örneği - diğer modüller bunu import eder
# ============================================================
_db_manager: Optional[DatabaseManager] = None


def init_db(db_url: str) -> DatabaseManager:
    """Veritabanı yöneticisini başlatır."""
    global _db_manager
    _db_manager = DatabaseManager(db_url)
    return _db_manager


def get_db() -> DatabaseManager:
    """Mevcut veritabanı yöneticisini döndürür."""
    if _db_manager is None:
        raise RuntimeError("Veritabanı başlatılmamış. Önce init_db() çağrın.")
    return _db_manager
