"""
notifications/telegram.py
Telegram bot bildirimleri.
İşlem açılış/kapanış, hata uyarıları, günlük raporlar.
"""

import threading
from datetime import datetime
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)


class TelegramNotifier:
    """
    Telegram Bot API ile bildirim gönderir.
    Asenkron gönderim için kuyruklama sistemi kullanır.
    """

    def __init__(self, bot_token: str, chat_id: str, enabled: bool = True):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = enabled and bool(bot_token) and bool(chat_id)

        self._bot = None
        self._queue = []
        self._queue_lock = threading.Lock()
        self._sender_thread: Optional[threading.Thread] = None
        self._running = False

        if self.enabled:
            self._init_bot()

    def _init_bot(self):
        """Telegram bot istemcisini başlatır."""
        try:
            import telegram
            self._bot = telegram.Bot(token=self.bot_token)
            # Bağlantı testi
            self._running = True
            self._sender_thread = threading.Thread(
                target=self._sender_loop,
                daemon=True,
                name="Telegram-Sender",
            )
            self._sender_thread.start()
            logger.info("Telegram botu başlatıldı.")
        except ImportError:
            logger.warning("python-telegram-bot yüklü değil. Bildirimler devre dışı.")
            self.enabled = False
        except Exception as e:
            logger.error(f"Telegram bot başlatma hatası: {e}")
            self.enabled = False

    def _sender_loop(self):
        """Kuyruktan mesaj gönderen döngü."""
        import time
        while self._running:
            try:
                with self._queue_lock:
                    if self._queue:
                        message = self._queue.pop(0)
                    else:
                        message = None

                if message:
                    self._send_raw(message)
                    time.sleep(0.5)  # Telegram rate limit: 30 mesaj/sn
                else:
                    time.sleep(0.2)

            except Exception as e:
                logger.error(f"Telegram gönderim döngü hatası: {e}")
                time.sleep(2)

    def _send_raw(self, text: str):
        """Telegram'a mesaj gönderir."""
        if not self.enabled or not self._bot:
            return
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            loop.run_until_complete(
                self._bot.send_message(
                    chat_id=self.chat_id,
                    text=text,
                    parse_mode="HTML",
                )
            )
            loop.close()
        except Exception as e:
            logger.error(f"Telegram gönderim hatası: {e}")

    def _queue_message(self, text: str):
        """Mesajı kuyruğa ekler."""
        if not self.enabled:
            logger.debug(f"Telegram devre dışı. Mesaj: {text[:50]}...")
            return
        with self._queue_lock:
            self._queue.append(text)

    # ----------------------------------------------------------------
    # Bildirim Metodları
    # ----------------------------------------------------------------

    def send_trade_open(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        quantity: float,
        leverage: int,
        stop_loss: float,
        take_profit: float,
    ):
        """İşlem açılış bildirimi."""
        emoji = "🟢" if direction == "LONG" else "🔴"
        sl_pct = abs(entry_price - stop_loss) / entry_price * 100
        tp_pct = abs(take_profit - entry_price) / entry_price * 100

        text = (
            f"{emoji} <b>İŞLEM AÇILDI</b>\n\n"
            f"📊 Parite: <code>{symbol}</code>\n"
            f"📈 Yön: <b>{direction}</b>\n"
            f"💰 Giriş: <code>{entry_price:.4f}</code>\n"
            f"📦 Miktar: <code>{quantity}</code>\n"
            f"⚡ Kaldıraç: <b>{leverage}x</b>\n"
            f"🛑 Stop-Loss: <code>{stop_loss:.4f}</code> (%{sl_pct:.2f})\n"
            f"✅ Take-Profit: <code>{take_profit:.4f}</code> (%{tp_pct:.2f})\n"
            f"🕐 Zaman: {datetime.utcnow().strftime('%H:%M:%S')} UTC"
        )
        self._queue_message(text)

    def send_trade_close(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        pnl_usdt: float,
        pnl_pct: float,
        duration: str,
        reason: str,
    ):
        """İşlem kapanış bildirimi."""
        pnl_emoji = "💚" if pnl_usdt >= 0 else "❤️"
        direction_emoji = "📈" if direction == "LONG" else "📉"

        # Kapanma sebebi Türkçe
        reason_map = {
            "SL": "🛑 Stop-Loss",
            "TP": "✅ Take-Profit",
            "TRAILING": "🔄 Trailing Stop",
            "MANUAL": "👤 Manuel",
            "DAILY_LIMIT": "⚠️ Günlük Limit",
            "WATCHDOG": "🔧 Watchdog",
        }
        reason_text = reason_map.get(reason, reason)

        text = (
            f"{pnl_emoji} <b>İŞLEM KAPANDI</b>\n\n"
            f"📊 Parite: <code>{symbol}</code>\n"
            f"{direction_emoji} Yön: <b>{direction}</b>\n"
            f"💰 Giriş: <code>{entry_price:.4f}</code>\n"
            f"💸 Çıkış: <code>{exit_price:.4f}</code>\n"
            f"📊 PNL: <b>{pnl_usdt:+.4f} USDT</b> ({pnl_pct:+.2f}%)\n"
            f"⏱ Süre: {duration}\n"
            f"🔖 Sebep: {reason_text}\n"
            f"🕐 Zaman: {datetime.utcnow().strftime('%H:%M:%S')} UTC"
        )
        self._queue_message(text)

    def send_alert(self, message: str):
        """Genel uyarı bildirimi."""
        text = f"⚠️ <b>BOT UYARISI</b>\n\n{message}\n\n🕐 {datetime.utcnow().strftime('%H:%M:%S')} UTC"
        self._queue_message(text)

    def send_error(self, module: str, error: str):
        """Hata bildirimi."""
        text = (
            f"🔴 <b>HATA</b>\n\n"
            f"📦 Modül: <code>{module}</code>\n"
            f"💬 Hata: {error[:200]}\n"
            f"🕐 {datetime.utcnow().strftime('%H:%M:%S')} UTC"
        )
        self._queue_message(text)

    def send_connection_lost(self, details: str = ""):
        """Bağlantı kopma bildirimi."""
        text = (
            f"🔌 <b>BAĞLANTI KESİLDİ</b>\n\n"
            f"{details}\n"
            f"🔄 Yeniden bağlanmaya çalışılıyor...\n"
            f"🕐 {datetime.utcnow().strftime('%H:%M:%S')} UTC"
        )
        self._queue_message(text)

    def send_daily_report(self, stats: dict):
        """Günlük performans raporu."""
        win_rate = stats.get("win_rate", 0)
        total_pnl = stats.get("total_pnl_usdt", 0)
        trade_count = stats.get("trade_count", 0)
        wins = stats.get("winning_trades", 0)
        losses = stats.get("losing_trades", 0)
        best = stats.get("best_trade", 0)
        worst = stats.get("worst_trade", 0)

        pnl_emoji = "💚" if total_pnl >= 0 else "❤️"
        date_str = datetime.utcnow().strftime("%d.%m.%Y")

        text = (
            f"📊 <b>GÜNLÜK RAPOR - {date_str}</b>\n\n"
            f"📈 Toplam İşlem: <b>{trade_count}</b>\n"
            f"✅ Kazanan: <b>{wins}</b>\n"
            f"❌ Kaybeden: <b>{losses}</b>\n"
            f"🎯 Win Rate: <b>%{win_rate:.1f}</b>\n\n"
            f"{pnl_emoji} Toplam PNL: <b>{total_pnl:+.4f} USDT</b>\n"
            f"🏆 En İyi İşlem: <code>{best:+.4f} USDT</code>\n"
            f"💔 En Kötü İşlem: <code>{worst:+.4f} USDT</code>\n\n"
            f"🕐 Rapor: {datetime.utcnow().strftime('%H:%M')} UTC"
        )
        self._queue_message(text)

    def send_daily_limit_reached(self, current_loss: float, limit: float):
        """Günlük kayıp limiti uyarısı."""
        text = (
            f"🚨 <b>GÜNLÜK KAYIP LİMİTİ AŞILDI</b>\n\n"
            f"💸 Günlük Kayıp: <b>{current_loss:.4f} USDT</b>\n"
            f"🔒 Limit: <b>{limit:.4f} USDT</b>\n"
            f"🛑 Bot durduruldu!\n\n"
            f"Yeniden başlatmak için web paneli kullanın.\n"
            f"🕐 {datetime.utcnow().strftime('%H:%M:%S')} UTC"
        )
        self._queue_message(text)

    def send_morning_report(self, report: dict):
        """Sabah AI raporu."""
        summary = report.get("summary", "Rapor yok")
        recommendations = report.get("recommendations", [])
        outlook = report.get("market_outlook", "neutral")

        outlook_emoji = {"bullish": "📈", "bearish": "📉", "neutral": "➡️"}.get(outlook, "📊")

        recs_text = ""
        for i, rec in enumerate(recommendations[:5], 1):  # Max 5 öneri
            priority_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(
                rec.get("priority", "LOW"), "⚪"
            )
            recs_text += f"{priority_emoji} {rec.get('description', '')}\n"

        text = (
            f"🌅 <b>SABAH RAPORU - {datetime.utcnow().strftime('%d.%m.%Y')}</b>\n\n"
            f"{outlook_emoji} <b>Piyasa Görünümü:</b> {outlook.title()}\n\n"
            f"📋 <b>Özet:</b>\n{summary[:300]}\n\n"
        )

        if recs_text:
            text += f"💡 <b>Öneriler:</b>\n{recs_text}"

        self._queue_message(text)

    def send_bot_started(self, symbols: list, testnet: bool):
        """Bot başlangıç bildirimi."""
        mode = "🧪 TESTNET" if testnet else "🔴 CANLI"
        text = (
            f"🤖 <b>BOT BAŞLATILDI</b>\n\n"
            f"🌐 Mod: {mode}\n"
            f"📊 Pariteler: {', '.join(symbols)}\n"
            f"🕐 {datetime.utcnow().strftime('%d.%m.%Y %H:%M:%S')} UTC"
        )
        self._queue_message(text)

    def send_bot_stopped(self, reason: str = ""):
        """Bot durdurma bildirimi."""
        text = (
            f"🛑 <b>BOT DURDURULDU</b>\n\n"
            f"{'📝 Sebep: ' + reason if reason else ''}\n"
            f"🕐 {datetime.utcnow().strftime('%H:%M:%S')} UTC"
        )
        self._queue_message(text)

    def send_funding_rate_warning(self, symbol: str, rate: float, threshold: float):
        """Yüksek funding rate uyarısı."""
        text = (
            f"⚡ <b>FUNDING RATE UYARISI</b>\n\n"
            f"📊 Parite: <code>{symbol}</code>\n"
            f"📊 Funding Rate: <b>{rate:.4f}%</b>\n"
            f"⚠️ Eşik: {threshold:.4f}%\n"
            f"💡 Bu parite için dikkatli olun!\n"
            f"🕐 {datetime.utcnow().strftime('%H:%M:%S')} UTC"
        )
        self._queue_message(text)

    def send_leverage_set(self, symbol: str, leverage: int):
        """Kaldıraç ayarı bildirimi."""
        text = f"⚡ {symbol} kaldıraç {leverage}x olarak ayarlandı."
        self._queue_message(text)

    def stop(self):
        """Bildirim sistemini durdurur."""
        self._running = False
        logger.info("Telegram bildirim sistemi durduruldu.")
