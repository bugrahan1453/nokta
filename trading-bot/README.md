# 🤖 Binance Futures Trading Bot

Gelişmiş 3 katmanlı sinyal sistemi, yapay zeka destekli piyasa analizi ve Türkçe web arayüzüne sahip tam otomatik Binance Futures trading botu.

---

## 📋 Özellikler

### ⚙️ Config Sistemi
- `.env` dosyasından API key ve tüm ayarlar
- Her parite için ayrı kaldıraç, pozisyon büyüklüğü, SL/TP ayarları
- Başlangıçta akıllı config doğrulama (mantıksız değerlerde uyarı/hata)
- Web arayüzünden çalışma zamanında ayar değiştirme

### 📊 Sinyal Motoru (3 Katman)
| Katman | Gösterge | Açıklama |
|--------|----------|----------|
| Katman 1 | EMA 21 / EMA 50 | Trend tespiti + 15dk onayı |
| Katman 2 | RSI(14) + MACD | Momentum + kesişim analizi |
| Katman 3 | Hacim + Funding Rate | Alıcı/satıcı baskısı |

> **3/3 konsensüs zorunlu** - tek bir katman ret ederse işlem açılmaz.

### 🛡️ Risk Yönetimi
- Kasa yüzdesi ile pozisyon büyüklüğü hesaplama
- Çift pozisyon kilidi (aynı paritede iki işlem yok)
- Günlük maksimum kayıp limiti (aşılınca bot durur)
- Çalışma saati kısıtlaması
- Maksimum eş zamanlı pozisyon sayısı

### 🔄 Trailing Stop
- Belirlenen PNL%'e ulaşınca otomatik aktifleşme
- Tepeden geri çekilme yüzdesine göre tetiklenme
- Her parite için ayrı trailing ayarları

### 🤖 AI Katmanı (Claude API)
- **Piyasa rejimi:** Trend/yatay/yüksek volatilite tespiti
- **Anomali algılama:** Ani pump/dump tespiti
- **Ekonomik takvim:** Önemli haber saatlerinde bot otomatik durur
- **Sabah raporu:** Her sabah optimizasyon önerileri

### 🔌 Watchdog
- Her 30sn internet ve Binance kontrolü
- WebSocket kopunca otomatik yeniden bağlanma
- Binance bakım modu algılama
- Açık pozisyon varken güvenli mod

### 🌐 Web Arayüzü
- Şifre korumalı giriş
- Anlık dashboard (bakiye, PNL, pozisyonlar)
- Parite bazlı ayar yönetimi
- İşlem geçmişi + CSV export
- Canlı log akışı
- PNL grafiği (Chart.js)

### 📱 Telegram Bildirimleri
- İşlem açılış/kapanış bildirimleri
- Hata ve uyarı mesajları
- Günlük performans raporu
- Sabah AI analiz raporu

---

## 🚀 Kurulum

### Gereksinimler
- Python 3.11+
- pip
- Binance Futures hesabı (API anahtarları ile)

### 1. Depoyu Klonlayın
```bash
git clone <repo-url> trading-bot
cd trading-bot
```

### 2. Sanal Ortam Oluşturun (Önerilir)
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### 3. Bağımlılıkları Yükleyin
```bash
pip install -r requirements.txt
```

### 4. .env Dosyasını Hazırlayın
```bash
cp .env.example .env
nano .env   # veya istediğiniz editörle açın
```

### 5. .env Dosyasını Doldurun

En az şu değerleri doldurmanız gerekir:
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

Web arayüzüne erişin: `http://localhost:5000`

---

## ⚙️ .env Ayarları Detaylı Açıklaması

### Binance API
| Değişken | Açıklama | Örnek |
|----------|----------|-------|
| `BINANCE_API_KEY` | Binance API key | `abc123...` |
| `BINANCE_API_SECRET` | Binance API secret | `xyz789...` |
| `BINANCE_TESTNET` | Testnet modu | `false` |

> ⚠️ API anahtarınızda **Futures Trading** izni açık olmalı. **Para çekme** izni **açmayın**.

### Claude AI (Opsiyonel)
| Değişken | Açıklama | Varsayılan |
|----------|----------|------------|
| `ANTHROPIC_API_KEY` | Claude API key | *(boş = AI kapalı)* |
| `ANTHROPIC_MODEL` | Model adı | `claude-opus-4-6` |

### Telegram (Opsiyonel)
| Değişken | Açıklama |
|----------|----------|
| `TELEGRAM_BOT_TOKEN` | @BotFather'dan alınan token |
| `TELEGRAM_CHAT_ID` | Chat/Kanal ID |
| `TELEGRAM_ENABLED` | `true` / `false` |

**Telegram Bot Kurulumu:**
1. Telegram'da @BotFather'a `/newbot` gönderin
2. Bot adını ve kullanıcı adını girin
3. Token'ı kopyalayın
4. @userinfobot'a mesaj atarak chat ID'nizi öğrenin

### Risk Ayarları
| Değişken | Açıklama | Varsayılan |
|----------|----------|------------|
| `MAX_CONCURRENT_POSITIONS` | Aynı anda max açık pozisyon | `3` |
| `DAILY_MAX_LOSS` | Günlük max kayıp (USDT) | `100.0` |
| `TRADING_START_HOUR` | Çalışma başlangıcı (UTC) | `0` |
| `TRADING_END_HOUR` | Çalışma bitişi (UTC) | `24` |

### Varsayılan Parite Ayarları
| Değişken | Açıklama | Varsayılan |
|----------|----------|------------|
| `DEFAULT_LEVERAGE` | Kaldıraç | `10` |
| `DEFAULT_POSITION_SIZE_PCT` | Kasa% | `2.0` |
| `DEFAULT_STOP_LOSS_PCT` | Stop-loss% | `1.5` |
| `DEFAULT_TAKE_PROFIT_PCT` | Take-profit% | `3.0` |
| `DEFAULT_TRAILING_ACTIVATION_PCT` | Trailing aktivasyon PNL% | `1.0` |
| `DEFAULT_TRAILING_CALLBACK_PCT` | Trailing geri çekilme% | `0.5` |

### Parite Özel Ayarlar
Her parite için `SYMBOL_` prefix ile ayar belirtebilirsiniz:
```env
BTCUSDT_LEVERAGE=10
BTCUSDT_POSITION_SIZE_PCT=2.0
BTCUSDT_STOP_LOSS_PCT=1.5
ETHUSDT_LEVERAGE=15
ETHUSDT_POSITION_SIZE_PCT=1.5
```

---

## 🏗️ Proje Yapısı

```
trading-bot/
├── main.py                  # Ana başlatıcı ve BotController
├── config/
│   ├── settings.py          # Tüm konfigürasyon nesneleri
│   └── validator.py         # Config doğrulama
├── core/
│   ├── data_engine.py       # WebSocket + REST veri motoru
│   ├── signal_engine.py     # 3 katmanlı sinyal sistemi
│   ├── risk_manager.py      # Risk yönetimi
│   ├── executor.py          # İşlem açma/kapama
│   ├── watchdog.py          # Bağlantı monitörü
│   └── ai_layer.py          # Claude AI entegrasyonu
├── web/
│   ├── app.py               # Flask web sunucusu
│   ├── templates/           # Jinja2 HTML şablonları
│   └── static/              # CSS, JS dosyaları
├── notifications/
│   └── telegram.py          # Telegram bot bildirimleri
├── database/
│   └── db.py                # SQLite + SQLAlchemy ORM
├── utils/
│   └── logger.py            # Merkezi log yönetimi
├── .env.example             # .env şablonu
├── requirements.txt         # Python bağımlılıkları
└── README.md                # Bu dosya
```

---

## 🔐 Güvenlik

### Önemli Uyarılar
1. **`.env` dosyasını asla git'e commit etmeyin** - `.gitignore`'a ekleyin
2. API anahtarında **para çekme izni vermeyin**
3. `WEB_PASSWORD` ve `SECRET_KEY` güçlü değerler kullanın
4. Üretimde `FLASK_DEBUG=false` olmalı
5. Testnet ile başlamak şiddetle önerilir

### Testnet ile Test
```env
BINANCE_TESTNET=true
```
Testnet'te para kaybetme riski yok. Canlıya geçmeden önce en az 1 hafta testnet'te çalıştırın.

---

## 📈 Kullanım

### Web Arayüzü
Bot başladıktan sonra `http://sunucu-ip:5000` adresine gidin.

**Dashboard:** Bakiye, açık pozisyonlar, anlık PNL, günlük istatistikler
**Ayarlar:** Parite bazlı konfigürasyon, anlık güncelleme
**Geçmiş:** Tüm işlem geçmişi, CSV export
**Loglar:** Canlı log akışı, hata kayıtları

### Bot Kontrol
- **Başlat:** Bot sinyal taramasına başlar
- **Durdur:** Yeni işlem açılmaz (açık pozisyonlar etkilenmez)
- **Tüm Kapat:** Tüm açık pozisyonlar market fiyatından kapatılır

### Manuel Pozisyon Kapatma
Dashboard'da her açık pozisyon için kırmızı X butonu ile manuel kapatabilirsiniz.

---

## 🐛 Sorun Giderme

### Bot başlamıyor
```bash
# Log dosyasını kontrol edin
tail -f logs/trading_bot.log

# Config doğrulama hatası için
python -c "from config.validator import validate_config; from config.settings import settings; validate_config(settings)"
```

### Binance bağlantı hatası
- API anahtarının futures izni var mı kontrol edin
- IP kısıtlaması varsa sunucu IP'sini Binance'e ekleyin
- Testnet modunda testnet API key kullanın

### WebSocket bağlantısı kopuyor
- Watchdog otomatik yeniden bağlanır
- Log dosyasında "WebSocket yeniden başlatıldı" mesajını bekleyin
- Sürekli kopma varsa internet bağlantısını kontrol edin

### Telegram bildirimleri gelmiyor
- Bot token doğru mu?
- Chat ID doğru mu? (`@userinfobot` ile öğrenin)
- Bota `/start` mesajı attınız mı?

---

## ⚠️ Risk Uyarısı

Bu yazılım **eğitim ve araştırma amaçlı** geliştirilmiştir.

Kripto para ticareti **yüksek risk** içerir:
- Tüm yatırımınızı kaybedebilirsiniz
- Otomatik botlar her zaman kâr etmez
- Geçmiş performans gelecek sonuçları garanti etmez
- Kaldıraçlı işlemler kaybı büyütür

**Gerçek para ile kullanmadan önce:**
1. Testnet'te en az 1 ay test edin
2. Küçük miktarlarla başlayın
3. Risk yönetimi ayarlarını dikkatlice yapın
4. Günlük loss limit'i düşük tutun

---

## 📜 Lisans

MIT License - Kişisel ve ticari kullanım serbesttir.

---

## 🤝 Katkı

Pull request ve issue'lar hoş karşılanır.

```bash
# Geliştirme ortamı
pip install -r requirements.txt
cp .env.example .env
# .env dosyasını testnet ile doldurun
python main.py
```
