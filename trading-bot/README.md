# Binance Futures Trading Bot

Tam otomatik, 5 katmanlı sinyal sistemi + opsiyonel ML/Sentiment/On-Chain analizi ile Binance Futures trading botu. Türkçe web arayüzü ve Claude AI entegrasyonu.

---

## Özellikler

### Sinyal Motoru (5 Katman)

| Katman | Göstergeler | Zorunlu? |
|--------|------------|---------|
| Katman 1 | EMA 21/50 + çift zaman dilimi trend onayı | Evet |
| Katman 2 | RSI(14) + MACD(12,26,9) momentum | Evet |
| Katman 3 | Hacim oranı + Taker buy/sell + Funding rate | Evet |
| Katman 4 | BB, ATR, StochRSI, VWAP, Patterns, Divergence | Hayır (filtre) |
| Katman 5 | ML + Sentiment + On-Chain ensemble | Hayır (skor katkısı) |

Katman 1-3 geçilmeden işlem açılmaz. Katman 4 strict modda veto yapabilir. Katman 5 sadece skor etkiler.

### Teknik Göstergeler (core/indicators.py)
- ATR + Chandelier Exit trailing stop
- Bollinger Bands (Squeeze tespiti dahil)
- Stochastic RSI
- VWAP + Bantlar
- RSI Divergence (klasik + gizli)
- MACD Divergence
- 10+ Candlestick Pattern (Engulfing, Hammer, Shooting Star, Doji, Morning/Evening Star, vb.)
- Supertrend, ADX, CCI, Williams %R, Ichimoku
- Z-Score, Hurst Exponent, Skewness/Kurtosis

### Gelişmiş Analiz (core/advanced_analysis.py)
- Order Book Imbalance — ücretsiz, Binance
- Open Interest analizi (4 senaryo: long/short squeeze, teyit, uyumsuzluk)
- Fear & Greed Index — ücretsiz, alternative.me
- Glassnode on-chain: SOPR, Exchange Flow, NUPL — *opsiyonel, API key*
- CryptoQuant: Fund Flow Ratio — *opsiyonel, API key*
- Whale Analyzer (Binance aggr. trades) — ücretsiz

### Makine Öğrenmesi (core/ml_layer.py) — Opsiyonel
- XGBoost classifier
- LightGBM classifier
- LSTM + Attention (PyTorch)
- 38 özellik mühendisliği
- Kendi işlem geçmişinden online öğrenme
- Kütüphane kurulu değilse otomatik devre dışı

### Duygu Analizi (core/sentiment.py) — Opsiyonel
- CryptoPanic RSS — ücretsiz, feedparser
- Twitter API v2 — bearer token gerekli
- Reddit PRAW — client_id/secret gerekli
- FinBERT finansal NLP — transformers (~400MB)
- VADER fallback — ücretsiz

### Ensemble Sistemi (core/ensemble.py)
- Piyasa rejimine göre dinamik ağırlık (trending/sideways/high_volatility)
- Bayesian model ortalaması (geçmiş başarıya göre modül ağırlıklarını günceller)

### Risk Yönetimi
- ATR tabanlı dinamik SL/TP (sabit % yerine piyasa volatilitesine göre)
- Chandelier Exit trailing stop referans seviyesi
- Kelly Kriteri pozisyon boyutlandırması (geçmiş işlemlerden öğrenme)
- Günlük maksimum kayıp limiti
- Çift pozisyon kilidi
- Maksimum eş zamanlı pozisyon sayısı
- Funding rate üst limiti

### Diğer Bileşenler
- AI Katmanı: Claude API — piyasa rejimi, anomali tespiti, ekonomik takvim, sabah raporu
- Trailing Stop: Otomatik aktifleşme ve geri çekilme takibi
- Watchdog: İnternet + Binance kontrolü, WebSocket otomatik yeniden bağlanma
- Web Arayüzü: Flask + Bootstrap 5, şifre korumalı, Türkçe
- Telegram: İşlem bildirimleri, günlük rapor, hata uyarıları
- Veritabanı: SQLite + SQLAlchemy ORM

---

## Kurulum

### Gereksinimler
- Python 3.11+
- Binance Futures hesabı (API anahtarları ile)

### 1. Klonlayın

```bash
git clone <repo-url> trading-bot
cd trading-bot
```

### 2. Sanal Ortam (Önerilir)

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### 3. Temel Bağımlılıkları Yükleyin

```bash
pip install -r requirements.txt
```

> Not: requirements.txt içindeki opsiyonel paketler yorum satırı olarak belirtilmiştir. İhtiyacınıza göre ayrıca kurabilirsiniz.

### 4. .env Dosyasını Oluşturun

```bash
cp .env.example .env
nano .env
```

### 5. Zorunlu Ayarları Doldurun

```env
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
WEB_PASSWORD=guclu_sifre_123
SECRET_KEY=uzun_rastgele_string_buraya
TRADING_SYMBOLS=BTCUSDT,ETHUSDT
```

### 6. Botu Başlatın

```bash
python main.py
```

Web arayüzü: `http://localhost:5000`

---

## Opsiyonel Modüller

### ML Katmanı

```bash
pip install xgboost lightgbm scikit-learn
# LSTM için (opsiyonel, ~2GB):
pip install torch
```

`.env` dosyasında aktif edin:
```env
ML_ENABLED=true
```

En az **10 tamamlanan işlem** sonrasında model eğitim yapmaya başlar.

### Duygu Analizi

```bash
# Temel (ücretsiz):
pip install feedparser vaderSentiment

# Twitter için:
pip install tweepy

# Reddit için:
pip install praw

# FinBERT için (ilk çalıştırmada ~400MB indirir):
pip install transformers torch
```

`.env` dosyasında:
```env
SENTIMENT_ENABLED=true
TWITTER_BEARER_TOKEN=your_token       # opsiyonel
REDDIT_CLIENT_ID=your_id             # opsiyonel
REDDIT_CLIENT_SECRET=your_secret     # opsiyonel
FINBERT_ENABLED=false                # RAM gerektiriyor
```

### On-Chain Analiz

```env
GLASSNODE_API_KEY=your_key       # https://glassnode.com (ücretsiz tier mevcut)
CRYPTOQUANT_API_KEY=your_key     # https://cryptoquant.com
```

API key boş bırakılırsa ilgili modül otomatik devre dışı kalır.

### ATR Dinamik SL/TP

Varsayılan olarak aktif. `.env`:
```env
ATR_SL_ENABLED=true
ATR_SL_MULTIPLIER=2.0    # SL = ATR × 2
ATR_TP_MULTIPLIER=3.0    # TP = ATR × 3
```

### Kelly Kriteri

Geçmiş işlemlerden win rate + ortalama kazanç/kayıp oranını hesaplayarak optimal pozisyon boyutu belirler:
```env
KELLY_ENABLED=true
KELLY_FRACTION=0.25    # Güvenli: %25 Kelly
KELLY_MIN_TRADES=10    # Minimum geçmiş işlem sayısı
```

---

## .env Ayarları Referansı

### Binance API
| Değişken | Açıklama | Varsayılan |
|----------|----------|------------|
| `BINANCE_API_KEY` | API anahtarı | — |
| `BINANCE_API_SECRET` | API gizli anahtarı | — |
| `BINANCE_TESTNET` | Testnet modu | `false` |

> API anahtarında **Futures Trading** izni açık, **Para Çekme** izni **kapalı** olmalı.

### Parite Ayarları
| Değişken | Açıklama | Varsayılan |
|----------|----------|------------|
| `TRADING_SYMBOLS` | İşlem pariteleri | `BTCUSDT,ETHUSDT` |
| `DEFAULT_LEVERAGE` | Kaldıraç | `10` |
| `DEFAULT_POSITION_SIZE_PCT` | Kasa% | `2.0` |
| `DEFAULT_STOP_LOSS_PCT` | Stop-loss% | `1.5` |
| `DEFAULT_TAKE_PROFIT_PCT` | Take-profit% | `3.0` |

Her parite için özel değer: `BTCUSDT_LEVERAGE=10`, `ETHUSDT_POSITION_SIZE_PCT=1.5`, vb.

### Risk Yönetimi
| Değişken | Açıklama | Varsayılan |
|----------|----------|------------|
| `MAX_CONCURRENT_POSITIONS` | Max açık pozisyon | `3` |
| `DAILY_MAX_LOSS` | Günlük max kayıp (USDT) | `100.0` |
| `TRADING_START_HOUR` | Başlangıç saati (UTC) | `0` |
| `TRADING_END_HOUR` | Bitiş saati (UTC) | `24` |
| `MAX_FUNDING_RATE_PCT` | Max funding rate | `0.1` |

### Katman 4 (Gelişmiş İndikatörler)
| Değişken | Açıklama | Varsayılan |
|----------|----------|------------|
| `LAYER4_ENABLED` | Katman 4 aktif | `true` |
| `LAYER4_STRICT` | Negatif sinyal → işlem açma | `false` |

### Katman 5 (ML + Ensemble)
| Değişken | Açıklama | Varsayılan |
|----------|----------|------------|
| `LAYER5_ENABLED` | Katman 5 aktif | `true` |
| `ML_ENABLED` | ML modelleri | `false` |
| `SENTIMENT_ENABLED` | Duygu analizi | `false` |
| `ENSEMBLE_MIN_CONFIDENCE` | Min güven skoru | `40` |

---

## Proje Yapısı

```
trading-bot/
├── main.py                     # Ana başlatıcı, BotController
├── .env.example                # Tüm ayar şablonu
├── requirements.txt            # Bağımlılıklar
├── config/
│   ├── settings.py             # Konfigürasyon nesneleri
│   └── validator.py            # Başlangıç doğrulama
├── core/
│   ├── data_engine.py          # WebSocket + REST, rate limiter
│   ├── signal_engine.py        # 5 katmanlı sinyal motoru
│   ├── indicators.py           # Teknik göstergeler
│   ├── advanced_analysis.py    # OB, OI, Fear&Greed, On-Chain, Whale
│   ├── ml_layer.py             # XGBoost, LightGBM, LSTM
│   ├── sentiment.py            # Twitter, Reddit, FinBERT, RSS
│   ├── ensemble.py             # Ağırlıklı oylama + Bayesian
│   ├── risk_manager.py         # Risk kontrol, ATR SL, Kelly
│   ├── executor.py             # İşlem açma/kapama, trailing stop
│   ├── watchdog.py             # Bağlantı monitörü
│   └── ai_layer.py             # Claude AI entegrasyonu
├── web/
│   ├── app.py                  # Flask, 15 endpoint
│   ├── templates/              # base, login, dashboard, settings, history, logs
│   └── static/                 # CSS (dark theme), JS
├── notifications/
│   └── telegram.py             # Kuyruk tabanlı bildirimler
├── database/
│   └── db.py                   # SQLite + SQLAlchemy
└── utils/
    └── logger.py               # Renkli konsol + dönen dosya log
```

---

## Güvenlik

1. `.env` dosyasını asla git'e commit etmeyin
2. API anahtarında para çekme izni vermeyin
3. `WEB_PASSWORD` ve `SECRET_KEY` güçlü değerler kullanın
4. Üretimde `FLASK_DEBUG=false`
5. Testnet ile başlayın

### Testnet

```env
BINANCE_TESTNET=true
```

Canlıya geçmeden önce en az 1 hafta testnet'te çalıştırın.

---

## Sorun Giderme

```bash
# Log dosyasını takip edin
tail -f logs/trading_bot.log

# Config doğrulama
python -c "
from config.validator import validate_config
from config.settings import settings
ok, errors, warnings = validate_config(settings)
print('Hatalar:', errors)
print('Uyarılar:', warnings)
"

# Modül import testi
python -c "
from core.signal_engine import SignalEngine
from core.ensemble import EnsembleAggregator
from core.ml_layer import MLLayer
print('Tüm modüller OK')
"
```

**Binance bağlantı hatası:** API anahtarının Futures izni var mı kontrol edin. IP kısıtlaması varsa sunucu IP'sini ekleyin.

**ML modeli yok hatası:** En az 10 tamamlanan işlem gerekli. Bot işlem yaptıkça model otomatik eğitilir.

**Telegram bildirimleri gelmiyor:** `/start` mesajı attınız mı? Chat ID `@userinfobot` ile öğrenin.

---

## Risk Uyarısı

Bu yazılım eğitim ve araştırma amaçlı geliştirilmiştir.

Kripto para ticareti yüksek risk içerir. Tüm yatırımınızı kaybedebilirsiniz. Otomatik botlar her zaman kâr etmez. Kaldıraçlı işlemler kaybı büyütür.

Gerçek para ile kullanmadan önce testnet'te en az 1 ay test edin ve günlük loss limit'i düşük tutun.
