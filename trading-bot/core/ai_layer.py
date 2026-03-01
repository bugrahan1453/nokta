"""
core/ai_layer.py
Claude AI entegrasyon katmanı.
Piyasa rejimi analizi, ekonomik takvim, anomali tespiti,
sabah optimizasyon raporu.
"""

import json
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


class MarketRegime:
    """Piyasa rejimi sınıflandırması."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    UNKNOWN = "unknown"


class AIAnalysisResult:
    """AI analiz sonucu."""

    def __init__(self, success: bool, data: dict = None, raw: str = ""):
        self.success = success
        self.data = data or {}
        self.raw = raw
        self.timestamp = datetime.utcnow()

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
        }


class AILayer:
    """
    Claude AI entegrasyon katmanı.
    Piyasa analizi, haber filtresi ve optimizasyon önerileri.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-opus-4-6",
        db=None,
    ):
        self.api_key = api_key
        self.model = model
        self.db = db
        self.enabled = bool(api_key)

        # Anthropic istemcisi
        self._client = None
        if self.enabled:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=api_key)
                logger.info(f"AI katmanı başlatıldı (model: {model})")
            except ImportError:
                logger.warning("anthropic kütüphanesi yüklü değil. AI katmanı devre dışı.")
                self.enabled = False
            except Exception as e:
                logger.error(f"AI istemci başlatma hatası: {e}")
                self.enabled = False

        # Analiz önbelleği (son sonuçlar)
        self._cache: Dict[str, AIAnalysisResult] = {}
        self._cache_ttl = {
            "market_regime": 300,       # 5 dakika
            "anomaly": 60,              # 1 dakika
            "morning_report": 3600 * 12, # 12 saat
            "economic_calendar": 3600,   # 1 saat
        }

        # Haber kara listesi (bot bu saatlerde işlem yapmaz)
        self._news_blackout_periods: List[dict] = []
        self._lock = threading.RLock()

    # ----------------------------------------------------------------
    # Piyasa Rejimi Analizi
    # ----------------------------------------------------------------

    def analyze_market_regime(
        self,
        symbol: str,
        candles: List[dict],
        funding_rate: float = 0.0,
    ) -> AIAnalysisResult:
        """
        Piyasa rejimini belirler: trend / yatay / yüksek volatilite.
        Önbellek geçerli ise önbellekten döndürür.
        """
        cache_key = f"market_regime_{symbol}"

        # Önbellek kontrolü
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        if not self.enabled:
            return self._fallback_regime(candles)

        try:
            # Son 20 mum özeti hazırla
            if not candles:
                return AIAnalysisResult(False, {"regime": MarketRegime.UNKNOWN})

            recent = candles[-20:]
            prices = [c["close"] for c in recent]
            volumes = [c["volume"] for c in recent]
            highs = [c["high"] for c in recent]
            lows = [c["low"] for c in recent]

            # Volatilite hesapla
            price_changes = [
                abs(prices[i] - prices[i-1]) / prices[i-1] * 100
                for i in range(1, len(prices))
            ]
            avg_volatility = sum(price_changes) / len(price_changes) if price_changes else 0

            # Trend gücü
            price_range = prices[-1] - prices[0]
            trend_pct = price_range / prices[0] * 100 if prices[0] > 0 else 0

            prompt = f"""Sen bir kripto para futures trading uzmanısın.
Aşağıdaki piyasa verilerini analiz et ve JSON formatında yanıt ver.

Sembol: {symbol}
Son 20 mum verileri:
- Son fiyat: {prices[-1]:.4f}
- 20 mum önceki fiyat: {prices[0]:.4f}
- Fiyat değişimi: {trend_pct:+.2f}%
- Ortalama volatilite: {avg_volatility:.3f}%
- Funding rate: {funding_rate:.4f}%
- Son hacim trendi: {'artıyor' if volumes[-1] > sum(volumes[:-1])/len(volumes[:-1]) else 'azalıyor'}

Şu soruları yanıtla:
1. Piyasa rejimi nedir? (trending_up/trending_down/sideways/high_volatility)
2. Mevcut trend güçlü mü? (0-10 arası güç skoru)
3. Bot bu piyasada işlem yapmalı mı? (true/false)
4. Neden? (kısa açıklama)
5. Önerilen strateji nedir?

JSON formatı:
{{
  "regime": "trending_up|trending_down|sideways|high_volatility",
  "trend_strength": 0-10,
  "should_trade": true|false,
  "reason": "açıklama",
  "strategy": "açıklama",
  "confidence": 0-100
}}"""

            response = self._client.messages.create(
                model=self.model,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )

            raw_text = response.content[0].text
            # JSON çıkar
            result_data = self._extract_json(raw_text)

            if not result_data:
                return self._fallback_regime(candles)

            ai_result = AIAnalysisResult(True, result_data, raw_text)
            self._set_cache(cache_key, ai_result)

            # Veritabanına kaydet
            if self.db:
                self.db.save_ai_analysis(
                    "market_regime", result_data, symbol, raw_text
                )

            logger.info(
                f"AI piyasa rejimi ({symbol}): {result_data.get('regime')} "
                f"(güven: {result_data.get('confidence')}%)"
            )

            return ai_result

        except Exception as e:
            logger.error(f"AI piyasa rejimi analiz hatası: {e}")
            return self._fallback_regime(candles)

    def _fallback_regime(self, candles: List[dict]) -> AIAnalysisResult:
        """AI olmadığında basit matematiksel rejim analizi."""
        if not candles or len(candles) < 10:
            return AIAnalysisResult(True, {
                "regime": MarketRegime.UNKNOWN,
                "trend_strength": 5,
                "should_trade": True,
                "reason": "Yeterli veri yok",
                "confidence": 30,
            })

        prices = [c["close"] for c in candles[-20:]]
        price_change = (prices[-1] - prices[0]) / prices[0] * 100

        # Volatilite
        changes = [abs(prices[i] - prices[i-1]) / prices[i-1] * 100 for i in range(1, len(prices))]
        avg_vol = sum(changes) / len(changes) if changes else 0

        if avg_vol > 2.0:
            regime = MarketRegime.HIGH_VOLATILITY
            should_trade = False  # Aşırı volatilite: gerçek veto
        elif abs(price_change) > 3:
            regime = MarketRegime.TRENDING_UP if price_change > 0 else MarketRegime.TRENDING_DOWN
            should_trade = True
        else:
            regime = MarketRegime.SIDEWAYS
            should_trade = True  # Sinyal motoru zaten filtreler, burada bloklama

        return AIAnalysisResult(True, {
            "regime": regime,
            "trend_strength": min(10, int(abs(price_change))),
            "should_trade": should_trade,
            "reason": f"Fiyat değişimi: {price_change:+.2f}%, Volatilite: {avg_vol:.2f}%",
            "confidence": 50,
        })

    # ----------------------------------------------------------------
    # Anomali Tespiti
    # ----------------------------------------------------------------

    def detect_anomaly(
        self,
        symbol: str,
        current_price: float,
        previous_price: float,
        volume: float,
        avg_volume: float,
    ) -> Optional[dict]:
        """
        Ani pump/dump ve olağandışı hacim tespiti.
        Anomali varsa dict döndürür, yoksa None.
        """
        price_change_pct = abs(current_price - previous_price) / previous_price * 100
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0

        # Eşikler
        PRICE_THRESHOLD = 1.5  # %1.5 ani fiyat hareketi
        VOLUME_THRESHOLD = 3.0  # 3x ortalama hacim

        anomaly = None

        if price_change_pct > PRICE_THRESHOLD:
            direction = "PUMP" if current_price > previous_price else "DUMP"
            anomaly = {
                "type": direction,
                "symbol": symbol,
                "price_change_pct": round(price_change_pct, 4),
                "volume_ratio": round(volume_ratio, 2),
                "current_price": current_price,
                "timestamp": datetime.utcnow().isoformat(),
                "severity": "HIGH" if price_change_pct > 3.0 else "MEDIUM",
            }
            logger.warning(
                f"[ANOMALİ] {symbol} {direction}: "
                f"{price_change_pct:.2f}% fiyat hareketi, "
                f"{volume_ratio:.2f}x hacim"
            )
        elif volume_ratio > VOLUME_THRESHOLD:
            anomaly = {
                "type": "HIGH_VOLUME",
                "symbol": symbol,
                "price_change_pct": round(price_change_pct, 4),
                "volume_ratio": round(volume_ratio, 2),
                "current_price": current_price,
                "timestamp": datetime.utcnow().isoformat(),
                "severity": "LOW",
            }

        return anomaly

    # ----------------------------------------------------------------
    # Ekonomik Takvim
    # ----------------------------------------------------------------

    def check_economic_calendar(self) -> dict:
        """
        Ekonomik takvim kontrolü.
        Önemli haber saatlerinde bot işlem yapmaz.
        """
        cache_key = "economic_calendar"
        cached = self._get_cached(cache_key)
        if cached:
            return cached.data

        if not self.enabled:
            return {"in_blackout": False, "events": []}

        try:
            now = datetime.utcnow()
            prompt = f"""Sen bir ekonomik takvim uzmanısın.
Mevcut tarih ve saat (UTC): {now.strftime('%Y-%m-%d %H:%M')}

Önümüzdeki 2 saat içinde kripto piyasasını önemli ölçüde etkileyebilecek ekonomik haberler var mı?
(FOMC toplantısı, CPI verileri, FED açıklamaları, büyük ülkelerin GDP verileri, vb.)

JSON formatı:
{{
  "in_blackout": true|false,
  "reason": "açıklama",
  "events": [
    {{
      "name": "haber adı",
      "time": "HH:MM UTC",
      "impact": "HIGH|MEDIUM|LOW",
      "currency": "USD"
    }}
  ],
  "resume_at": "HH:MM UTC veya null"
}}

Not: Eğer bilgin yoksa veya emin değilsen, in_blackout=false döndür."""

            response = self._client.messages.create(
                model=self.model,
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}],
            )

            raw_text = response.content[0].text
            result = self._extract_json(raw_text)

            if result:
                ai_result = AIAnalysisResult(True, result, raw_text)
                self._set_cache(cache_key, ai_result)

                if result.get("in_blackout"):
                    logger.warning(
                        f"EKONOMİK TAKVİM KARA BÖLGE: {result.get('reason')}"
                    )

                return result

        except Exception as e:
            logger.error(f"Ekonomik takvim sorgu hatası: {e}")

        return {"in_blackout": False, "events": []}

    def is_in_news_blackout(self) -> tuple:
        """
        Bot haber kara bölgesinde mi?
        Returns: (kara_bölgede_mi, sebep)
        """
        try:
            calendar = self.check_economic_calendar()
            if calendar.get("in_blackout"):
                return True, calendar.get("reason", "Önemli ekonomik haber")
        except Exception as e:
            logger.error(f"Haber kara bölge kontrol hatası: {e}")

        return False, ""

    # ----------------------------------------------------------------
    # Sabah Optimizasyon Raporu
    # ----------------------------------------------------------------

    def generate_morning_report(
        self,
        symbols: List[str],
        daily_stats: dict,
        current_settings: dict,
    ) -> AIAnalysisResult:
        """
        Her sabah parametre optimizasyon önerisi üretir.
        """
        if not self.enabled:
            return AIAnalysisResult(False, {
                "summary": "AI katmanı devre dışı",
                "recommendations": [],
            })

        try:
            prompt = f"""Sen bir kripto futures trading botu optimizasyon uzmanısın.
Dünkü performans verilerini inceleyerek öneriler sun.

İzlenen pariteler: {', '.join(symbols)}

Dünkü istatistikler:
- Toplam işlem: {daily_stats.get('trade_count', 0)}
- Kazanan: {daily_stats.get('winning_trades', 0)}
- Kaybeden: {daily_stats.get('losing_trades', 0)}
- Win rate: %{daily_stats.get('win_rate', 0):.1f}
- Toplam PNL: {daily_stats.get('total_pnl_usdt', 0):+.2f} USDT

Mevcut ayarlar:
- Kaldıraç: {current_settings.get('default_leverage', 10)}x
- Pozisyon büyüklüğü: %{current_settings.get('position_size_pct', 2)}
- Stop-loss: %{current_settings.get('stop_loss_pct', 1.5)}
- Take-profit: %{current_settings.get('take_profit_pct', 3.0)}

Şunu sağla:
1. Dünkü performans özeti
2. Parametre optimizasyon önerileri (somut ve mantıklı)
3. Bugün hangi paritelere odaklanılmalı
4. Risk uyarıları (varsa)

JSON formatı:
{{
  "summary": "genel performans özeti",
  "performance_grade": "A|B|C|D|F",
  "recommendations": [
    {{
      "type": "parameter_change|focus_symbol|risk_warning",
      "description": "açıklama",
      "priority": "HIGH|MEDIUM|LOW"
    }}
  ],
  "focus_symbols": ["BTCUSDT"],
  "risk_warnings": ["uyarı metni"],
  "market_outlook": "bullish|bearish|neutral"
}}"""

            response = self._client.messages.create(
                model=self.model,
                max_tokens=800,
                messages=[{"role": "user", "content": prompt}],
            )

            raw_text = response.content[0].text
            result = self._extract_json(raw_text)

            if not result:
                result = {"summary": raw_text, "recommendations": []}

            ai_result = AIAnalysisResult(True, result, raw_text)
            self._set_cache("morning_report", ai_result)

            if self.db:
                self.db.save_ai_analysis("morning_report", result, None, raw_text)

            logger.info(
                f"Sabah raporu üretildi: "
                f"{len(result.get('recommendations', []))} öneri"
            )

            return ai_result

        except Exception as e:
            logger.error(f"Sabah raporu üretme hatası: {e}")
            return AIAnalysisResult(False, {
                "summary": f"Rapor üretme hatası: {e}",
                "recommendations": [],
            })

    # ----------------------------------------------------------------
    # Yardımcı Metodlar
    # ----------------------------------------------------------------

    def _extract_json(self, text: str) -> Optional[dict]:
        """Metin içinden JSON çıkarır."""
        import re
        # Kod bloğu içindeki JSON
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                pass

        # Doğrudan JSON
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass

        return None

    def _get_cached(self, key: str) -> Optional[AIAnalysisResult]:
        """Önbellekten sonuç döndürür (TTL kontrolü ile)."""
        with self._lock:
            result = self._cache.get(key)
            if not result:
                return None

            ttl = self._cache_ttl.get(key.split("_")[0], 300)
            age = (datetime.utcnow() - result.timestamp).total_seconds()

            if age > ttl:
                del self._cache[key]
                return None

            return result

    def _set_cache(self, key: str, result: AIAnalysisResult):
        """Sonucu önbelleğe ekler."""
        with self._lock:
            self._cache[key] = result

    def get_status(self) -> dict:
        """AI katmanının durumunu döndürür."""
        return {
            "enabled": self.enabled,
            "model": self.model if self.enabled else None,
            "cached_analyses": list(self._cache.keys()),
        }
