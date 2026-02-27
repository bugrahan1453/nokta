"""
web/app.py
Flask web sunucusu - Türkçe arayüz.
Şifre korumalı giriş, dashboard, ayarlar, geçmiş, log ekranı.
"""

import json
import os
import threading
from datetime import datetime, timedelta
from functools import wraps
from typing import Optional

from flask import (
    Flask, render_template, request, redirect, url_for,
    jsonify, session, flash, Response
)
from werkzeug.security import generate_password_hash, check_password_hash

from utils.logger import get_logger

logger = get_logger(__name__)

# Flask uygulaması
app = Flask(__name__)

# ---- Bağımlılıklar (main.py tarafından atanır) ----
_settings = None
_data_engine = None
_signal_engine = None
_risk_manager = None
_executor = None
_watchdog = None
_ai_layer = None
_db = None
_bot_controller = None


def init_app(settings, data_engine, signal_engine, risk_manager,
             executor, watchdog, ai_layer, db, bot_controller):
    """Web uygulamasını başlatır ve bağımlılıkları atar."""
    global _settings, _data_engine, _signal_engine, _risk_manager
    global _executor, _watchdog, _ai_layer, _db, _bot_controller

    _settings = settings
    _data_engine = data_engine
    _signal_engine = signal_engine
    _risk_manager = risk_manager
    _executor = executor
    _watchdog = watchdog
    _ai_layer = ai_layer
    _db = db
    _bot_controller = bot_controller

    app.secret_key = settings.web.secret_key
    logger.info("Web uygulaması başlatıldı.")


# ============================================================
# Kimlik Doğrulama
# ============================================================

def login_required(f):
    """Oturum açmayı zorunlu kılan dekoratör."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login", next=request.url))
        return f(*args, **kwargs)
    return decorated


@app.route("/login", methods=["GET", "POST"])
def login():
    """Giriş sayfası."""
    error = None
    if request.method == "POST":
        password = request.form.get("password", "")
        if _settings and check_password_hash(
            generate_password_hash(_settings.web.password),
            password
        ) or (_settings and password == _settings.web.password):
            session["logged_in"] = True
            session.permanent = True
            app.permanent_session_lifetime = timedelta(hours=24)
            next_url = request.args.get("next") or url_for("dashboard")
            logger.info("Web arayüzü girişi başarılı.")
            return redirect(next_url)
        else:
            error = "Yanlış şifre!"
            logger.warning("Web arayüzü başarısız giriş denemesi.")

    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    """Çıkış."""
    session.clear()
    return redirect(url_for("login"))


# ============================================================
# Dashboard
# ============================================================

@app.route("/")
@login_required
def dashboard():
    """Ana dashboard sayfası."""
    return render_template("dashboard.html")


@app.route("/api/dashboard-data")
@login_required
def dashboard_data():
    """Dashboard için tüm verileri JSON olarak döndürür."""
    try:
        # Bakiye
        balance = {}
        if _data_engine:
            balance = _data_engine.get_futures_balance()

        # Açık pozisyonlar
        open_trades = []
        if _executor:
            open_trades = _executor.get_open_trades_info()

        # Risk durumu
        risk_status = {}
        if _risk_manager:
            risk_status = _risk_manager.get_status()

        # Watchdog durumu
        watchdog_status = {}
        if _watchdog:
            watchdog_status = _watchdog.get_status()

        # Bot durumu
        bot_status = "unknown"
        if _bot_controller:
            bot_status = _bot_controller.get_status()

        # Bugünkü istatistikler
        today_stats = {}
        if _db:
            today_stats = _db.get_today_stats()

        # Günlük PNL grafiği (son 14 gün)
        daily_pnl = []
        if _db:
            daily_pnl = _db.get_daily_pnl(days=14)

        # Anlık fiyatlar
        prices = {}
        if _data_engine and _settings:
            prices = {
                sym: _data_engine.price_cache.get(sym)
                for sym in _settings.trading_symbols
            }

        # AI durumu
        ai_status = {}
        if _ai_layer:
            ai_status = _ai_layer.get_status()

        return jsonify({
            "success": True,
            "balance": balance,
            "open_trades": open_trades,
            "risk_status": risk_status,
            "watchdog_status": watchdog_status,
            "bot_status": bot_status,
            "today_stats": today_stats,
            "daily_pnl": daily_pnl,
            "prices": prices,
            "ai_status": ai_status,
            "timestamp": datetime.utcnow().isoformat(),
        })
    except Exception as e:
        logger.error(f"Dashboard veri hatası: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ============================================================
# Bot Kontrol
# ============================================================

@app.route("/api/bot/start", methods=["POST"])
@login_required
def bot_start():
    """Botu başlatır."""
    try:
        if _bot_controller:
            _bot_controller.start()
            logger.info("Bot web arayüzünden başlatıldı.")
            return jsonify({"success": True, "message": "Bot başlatıldı."})
        return jsonify({"success": False, "message": "Bot kontrolcüsü bulunamadı."})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/bot/stop", methods=["POST"])
@login_required
def bot_stop():
    """Botu durdurur."""
    try:
        if _bot_controller:
            _bot_controller.stop()
            logger.info("Bot web arayüzünden durduruldu.")
            return jsonify({"success": True, "message": "Bot durduruldu."})
        return jsonify({"success": False, "message": "Bot kontrolcüsü bulunamadı."})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/bot/close-all", methods=["POST"])
@login_required
def close_all_positions():
    """Tüm pozisyonları kapatır."""
    try:
        if _executor:
            results = _executor.close_all_positions(reason="MANUAL")
            success_count = sum(1 for r in results if r.success)
            logger.info(f"Tüm pozisyonlar kapatıldı: {success_count}/{len(results)} başarılı")
            return jsonify({
                "success": True,
                "message": f"{success_count}/{len(results)} pozisyon kapatıldı.",
            })
        return jsonify({"success": False, "message": "Executor bulunamadı."})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/position/close/<symbol>", methods=["POST"])
@login_required
def close_position(symbol):
    """Tek pozisyon kapatır."""
    try:
        if _executor:
            result = _executor.close_position(symbol.upper(), reason="MANUAL")
            return jsonify({
                "success": result.success,
                "message": result.message,
                "data": result.data,
            })
        return jsonify({"success": False, "message": "Executor bulunamadı."})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ============================================================
# Ayarlar
# ============================================================

@app.route("/settings")
@login_required
def settings_page():
    """Ayarlar sayfası."""
    if not _settings:
        return redirect(url_for("dashboard"))
    config = _settings.to_dict()
    return render_template("settings.html", config=config)


@app.route("/api/settings", methods=["GET"])
@login_required
def get_settings():
    """Mevcut ayarları döndürür."""
    if not _settings:
        return jsonify({"success": False, "error": "Settings yüklenemedi"})
    return jsonify({"success": True, "settings": _settings.to_dict()})


@app.route("/api/settings/symbol", methods=["POST"])
@login_required
def update_symbol_settings():
    """Parite ayarlarını günceller."""
    try:
        data = request.get_json()
        symbol = data.get("symbol", "").upper()

        if not symbol or not _settings:
            return jsonify({"success": False, "error": "Geçersiz istek"})

        old_config = _settings.get_symbol_config(symbol)
        old_values = {
            "leverage": old_config.leverage if old_config else None,
            "position_size_pct": old_config.position_size_pct if old_config else None,
        }

        # Güncellenecek alanlar
        updates = {}
        allowed_fields = [
            "leverage", "position_size_pct", "stop_loss_pct",
            "take_profit_pct", "trailing_activation_pct",
            "trailing_callback_pct", "enabled",
        ]
        for field in allowed_fields:
            if field in data:
                val = data[field]
                if field in ("leverage",):
                    val = int(val)
                elif field == "enabled":
                    val = bool(val)
                else:
                    val = float(val)
                updates[field] = val

        _settings.update_symbol_config(symbol, **updates)

        # Config geçmişini kaydet
        if _db:
            _db.save_config_change(
                f"symbol_{symbol}",
                old_values,
                updates,
                "web"
            )

        logger.info(f"{symbol} ayarları güncellendi: {updates}")
        return jsonify({"success": True, "message": f"{symbol} ayarları kaydedildi."})

    except Exception as e:
        logger.error(f"Ayar güncelleme hatası: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/settings/general", methods=["POST"])
@login_required
def update_general_settings():
    """Genel bot ayarlarını günceller (runtime)."""
    try:
        data = request.get_json()

        if not _settings:
            return jsonify({"success": False, "error": "Settings yüklenemedi"})

        # Güncellenebilir genel ayarlar
        if "max_concurrent_positions" in data:
            _settings.risk.max_concurrent_positions = int(data["max_concurrent_positions"])
        if "daily_max_loss" in data:
            _settings.risk.daily_max_loss = float(data["daily_max_loss"])
        if "trading_start_hour" in data:
            _settings.risk.trading_start_hour = int(data["trading_start_hour"])
        if "trading_end_hour" in data:
            _settings.risk.trading_end_hour = int(data["trading_end_hour"])

        logger.info(f"Genel ayarlar güncellendi: {data}")
        return jsonify({"success": True, "message": "Genel ayarlar kaydedildi."})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ============================================================
# İşlem Geçmişi
# ============================================================

@app.route("/history")
@login_required
def history_page():
    """İşlem geçmişi sayfası."""
    return render_template("history.html")


@app.route("/api/trades")
@login_required
def get_trades():
    """İşlem geçmişini döndürür."""
    try:
        if not _db:
            return jsonify({"success": False, "error": "DB bağlantısı yok"})

        limit = int(request.args.get("limit", 100))
        offset = int(request.args.get("offset", 0))
        symbol = request.args.get("symbol")

        trades = _db.get_trades_history(limit=limit, offset=offset, symbol=symbol)
        overall_stats = _db.get_overall_stats()

        return jsonify({
            "success": True,
            "trades": [t.to_dict() for t in trades],
            "overall_stats": overall_stats,
            "count": len(trades),
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/trades/daily-pnl")
@login_required
def get_daily_pnl():
    """Günlük PNL verisi döndürür."""
    try:
        days = int(request.args.get("days", 30))
        if _db:
            data = _db.get_daily_pnl(days=days)
            return jsonify({"success": True, "data": data})
        return jsonify({"success": False, "error": "DB yok"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ============================================================
# Log Ekranı
# ============================================================

@app.route("/logs")
@login_required
def logs_page():
    """Log ekranı sayfası."""
    return render_template("logs.html")


@app.route("/api/logs")
@login_required
def get_logs():
    """Son logları döndürür."""
    try:
        limit = int(request.args.get("limit", 200))
        level = request.args.get("level")

        if not _db:
            return jsonify({"success": False, "error": "DB yok"})

        logs = _db.get_error_logs(limit=limit, level=level)
        return jsonify({
            "success": True,
            "logs": [log.to_dict() for log in logs],
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/logs/stream")
@login_required
def stream_logs():
    """
    Server-Sent Events ile canlı log akışı.
    Son log dosyasından okur.
    """
    log_file = _settings.log.file if _settings else "logs/trading_bot.log"

    def generate():
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                # Son 50 satırı ilk göster
                lines = f.readlines()
                for line in lines[-50:]:
                    yield f"data: {json.dumps({'line': line.rstrip()})}\n\n"

                # Yeni satırları takip et
                import time
                while True:
                    line = f.readline()
                    if line:
                        yield f"data: {json.dumps({'line': line.rstrip()})}\n\n"
                    else:
                        time.sleep(0.5)
                        yield "data: {\"ping\": true}\n\n"
        except FileNotFoundError:
            yield f"data: {json.dumps({'line': 'Log dosyası bulunamadı.'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================
# Sinyal Durumu
# ============================================================

@app.route("/api/signals")
@login_required
def get_signals():
    """Son sinyal durumlarını döndürür."""
    try:
        if not _signal_engine or not _settings:
            return jsonify({"success": False, "error": "Signal engine yok"})

        signals = {}
        for symbol in _settings.trading_symbols:
            sig = _signal_engine.get_last_signal(symbol)
            if sig:
                signals[symbol] = sig.to_dict()

        history = _signal_engine.get_signal_history(limit=10)

        return jsonify({
            "success": True,
            "signals": signals,
            "history": history,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ============================================================
# Hata Sayfaları
# ============================================================

@app.errorhandler(404)
def not_found(e):
    return render_template("dashboard.html"), 404


@app.errorhandler(500)
def server_error(e):
    logger.error(f"Sunucu hatası: {e}")
    return jsonify({"error": "Sunucu hatası"}), 500


def run_web_server(host: str = "0.0.0.0", port: int = 5000, debug: bool = False):
    """Web sunucusunu başlatır (ayrı thread'de çalıştırılır)."""
    app.run(
        host=host,
        port=port,
        debug=debug,
        use_reloader=False,
        threaded=True,
    )
