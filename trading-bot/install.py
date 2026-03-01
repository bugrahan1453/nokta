"""
install.py - KriptoNokta Trading Bot Otomatik Kurulum
Kullanım: python install.py
Windows'ta çift tıklayarak da çalıştırabilirsiniz.
"""

import sys
import os
import subprocess
import shutil
from pathlib import Path

# ── Renk yardımcıları ────────────────────────────────────────────────────────

def clr(text, code):
    """ANSI renk (Windows 10+ destekler)."""
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleMode(
            ctypes.windll.kernel32.GetStdHandle(-11), 7
        )
    except Exception:
        pass
    return f"\033[{code}m{text}\033[0m"

def ok(msg):    print(clr(f"  [OK]  {msg}", "92"))
def err(msg):   print(clr(f"  [!!]  {msg}", "91"))
def info(msg):  print(clr(f"  [--]  {msg}", "96"))
def warn(msg):  print(clr(f"  [>>]  {msg}", "93"))
def title(msg): print(clr(f"\n{'='*55}\n  {msg}\n{'='*55}", "1;94"))
def sep():      print(clr("  " + "-"*50, "90"))


# ── Adım fonksiyonları ────────────────────────────────────────────────────────

def check_python():
    title("1. Python Sürümü Kontrolü")
    v = sys.version_info
    info(f"Python {v.major}.{v.minor}.{v.micro} bulundu")
    if v.major < 3 or (v.major == 3 and v.minor < 11):
        err("Python 3.11+ gerekli!")
        err("İndir: https://www.python.org/downloads/")
        return False
    ok(f"Python {v.major}.{v.minor} uyumlu")
    return True


def create_venv():
    title("2. Sanal Ortam (venv)")
    venv_dir = Path("venv")

    if venv_dir.exists():
        ok("venv zaten mevcut, atlanıyor")
        return True

    info("Sanal ortam oluşturuluyor...")
    result = subprocess.run(
        [sys.executable, "-m", "venv", "venv"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        err(f"venv oluşturulamadı: {result.stderr}")
        return False
    ok("Sanal ortam oluşturuldu")
    return True


def get_pip():
    """Sanal ortamdaki pip yolunu döndür."""
    if sys.platform == "win32":
        pip = Path("venv") / "Scripts" / "pip.exe"
        python = Path("venv") / "Scripts" / "python.exe"
    else:
        pip = Path("venv") / "bin" / "pip"
        python = Path("venv") / "bin" / "python"

    if not pip.exists():
        # venv yoksa mevcut pip
        return sys.executable.replace("python", "pip"), sys.executable
    return str(pip), str(python)


def pip_install(pip, packages, label=""):
    """Paket kur, hata durumunda devam et."""
    if isinstance(packages, str):
        packages = [packages]
    target = label or " ".join(packages)
    info(f"Kuruluyor: {target} ...")
    result = subprocess.run(
        [pip, "install", "--quiet", "--upgrade"] + packages,
        capture_output=True, text=True
    )
    if result.returncode != 0:
        warn(f"Kısmen başarısız: {target}")
        warn(f"  {result.stderr.strip()[:120]}")
        return False
    ok(f"Kuruldu: {target}")
    return True


def upgrade_pip(pip):
    """pip + setuptools + wheel'i güncelle (derleme hataları için şart)."""
    info("pip / setuptools / wheel güncelleniyor...")
    for pkg in ["pip", "setuptools", "wheel"]:
        subprocess.run(
            [pip, "install", "--quiet", "--upgrade", pkg],
            capture_output=True, text=True
        )
    ok("pip araçları güncellendi")


def install_requirements():
    title("3. Temel Paketler (requirements.txt)")
    pip, _ = get_pip()

    if not Path("requirements.txt").exists():
        err("requirements.txt bulunamadı!")
        return False

    # Önce pip araçlarını güncelle — derleme hataları genellikle buradan gelir
    upgrade_pip(pip)

    info("Bu işlem 2-5 dakika sürebilir, lütfen bekleyin...")
    result = subprocess.run(
        [pip, "install", "--quiet",
         "--prefer-binary",          # kaynak derlemek yerine wheel kullan
         "--no-build-isolation",     # zaten yüklü araçları kullan
         "-r", "requirements.txt"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        # Toplu kurulum başarısız → tek tek dene
        warn("Toplu kurulum başarısız, paketler tek tek deneniyor...")
        return _install_one_by_one(pip)
    ok("Tüm temel paketler kuruldu")
    return True


def _install_one_by_one(pip):
    """requirements.txt satırlarını tek tek kur, başarısızları raporla."""
    failed = []
    req_lines = Path("requirements.txt").read_text(encoding="utf-8").splitlines()

    # numpy her zaman önce kurulmalı — pandas ve pandas-ta buna bağlı
    info("Önce numpy kuruluyor (bağımlılık sırası)...")
    subprocess.run(
        [pip, "install", "--quiet", "--prefer-binary", "--upgrade", "numpy"],
        capture_output=True, text=True
    )

    for line in req_lines:
        line = line.split("#")[0].strip()  # satır içi yorumları temizle
        if not line:
            continue
        result = subprocess.run(
            [pip, "install", "--quiet", "--prefer-binary", line],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            ok(f"Kuruldu: {line}")
        else:
            err(f"BAŞARISIZ: {line}")
            # Sebebi tek satırda göster
            detail = (result.stderr or result.stdout).strip().splitlines()
            last = next((l for l in reversed(detail) if l.strip()), "")
            warn(f"  → {last[:120]}")
            failed.append(line)

    if failed:
        print()
        err(f"{len(failed)} paket kurulamadı:")
        for f in failed:
            warn(f"  • {f}")
        warn("Bu paketler eksik olabilir; bot kısıtlı çalışabilir.")
        return len(failed) < 5   # 5'ten az hata varsa devam et
    ok("Tüm paketler kuruldu")
    return True


def install_optional():
    title("4. Opsiyonel Paketler (Sentiment / Analiz)")
    pip, _ = get_pip()

    packages = [
        (["feedparser"],          "feedparser (CryptoPanic RSS)"),
        (["vaderSentiment"],      "vaderSentiment (duygu analizi)"),
        (["twscrape"],            "twscrape (Twitter ücretsiz scraper)"),
        (["textblob"],            "textblob (yedek duygu analizi)"),
    ]

    results = []
    for pkgs, label in packages:
        results.append(pip_install(pip, pkgs, label))

    sep()
    info("Opsiyonel ML paketleri (büyük, opsiyonel):")
    print()
    warn("  Makine Öğrenmesi (opsiyonel, ~500MB):")
    warn("    pip install xgboost lightgbm scikit-learn")
    warn("  FinBERT NLP (opsiyonel, ~400MB):")
    warn("    pip install transformers torch")
    return True


def check_env():
    title("5. .env Dosyası Kontrolü")
    env_path = Path(".env")

    if not env_path.exists():
        err(".env dosyası bulunamadı!")
        info("Örnek dosyadan kopyalayın: cp .env.example .env")
        return False

    content = env_path.read_text(encoding="utf-8")

    checks = [
        ("BINANCE_API_KEY",    "Binance API Key"),
        ("BINANCE_API_SECRET", "Binance API Secret"),
        ("TELEGRAM_BOT_TOKEN","Telegram Bot Token"),
        ("TELEGRAM_CHAT_ID",  "Telegram Chat ID"),
        ("ANTHROPIC_API_KEY", "Anthropic API Key"),
    ]

    all_ok = True
    for key, label in checks:
        # Değer var mı ve boş mu?
        val = ""
        for line in content.splitlines():
            if line.startswith(key + "="):
                val = line.split("=", 1)[1].strip()
                break
        if val:
            # Sadece ilk/son 4 karakteri göster
            masked = val[:4] + "****" + val[-4:] if len(val) > 8 else "****"
            ok(f"{label}: {masked}")
        else:
            warn(f"{label}: BOŞ (opsiyonel veya eksik)")
            if key in ("BINANCE_API_KEY", "BINANCE_API_SECRET"):
                all_ok = False

    return all_ok


def test_imports():
    title("6. Modül Import Testi")
    _, python = get_pip()

    test_code = """
import sys
errors = []
warnings = []

try:
    import binance
    print("  [OK]  python-binance")
except ImportError as e:
    errors.append(f"python-binance: {e}")

try:
    import flask
    print("  [OK]  flask")
except ImportError as e:
    errors.append(f"flask: {e}")

try:
    import pandas
    print("  [OK]  pandas")
except ImportError as e:
    errors.append(f"pandas: {e}")

try:
    import sqlalchemy
    print("  [OK]  sqlalchemy")
except ImportError as e:
    errors.append(f"sqlalchemy: {e}")

try:
    import anthropic
    print("  [OK]  anthropic")
except ImportError as e:
    errors.append(f"anthropic: {e}")

try:
    import telegram
    print("  [OK]  python-telegram-bot")
except ImportError as e:
    errors.append(f"telegram: {e}")

try:
    import feedparser
    print("  [OK]  feedparser")
except ImportError:
    warnings.append("feedparser (pip install feedparser)")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    print("  [OK]  vaderSentiment")
except ImportError:
    warnings.append("vaderSentiment (pip install vaderSentiment)")

try:
    import twscrape
    print("  [OK]  twscrape")
except ImportError:
    warnings.append("twscrape (pip install twscrape)")

if errors:
    print("\\n  [!!] HATALAR:")
    for e in errors:
        print(f"       {e}")
    sys.exit(1)

if warnings:
    print("\\n  [>>] UYARILAR (opsiyonel, kurulmadı):")
    for w in warnings:
        print(f"       {w}")

print("\\n  Zorunlu modüller OK")
"""

    result = subprocess.run(
        [python, "-c", test_code],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        err("Bazı zorunlu modüller eksik!")
        if result.stderr:
            print(result.stderr[:300])
        return False
    return True


def print_summary(success):
    title("Kurulum Özeti")

    if sys.platform == "win32":
        activate = r"venv\Scripts\activate"
        run_cmd  = "python main.py"
    else:
        activate = "source venv/bin/activate"
        run_cmd  = "python main.py"

    if success:
        ok("Kurulum tamamlandı!")
        print()
        info("Botu başlatmak için:")
        print()
        print(clr(f"    {activate}", "97"))
        print(clr(f"    {run_cmd}", "97"))
        print()
        info("Web arayüzü: http://localhost:5000")
        info("Web şifresi: .env dosyasındaki WEB_PASSWORD")
        print()
        warn("ÖNEMLİ: İlk kez çalıştırmadan önce .env dosyasını kontrol et!")
        warn("Testnet ile başlamak için: BINANCE_TESTNET=true")
    else:
        err("Kurulumda sorunlar var, yukarıdaki hataları gider")
        info("Yardım için hata mesajını paylaş")

    print()
    input("  Çıkmak için Enter'a bas...")


# ── Ana akış ─────────────────────────────────────────────────────────────────

def main():
    os.system("cls" if sys.platform == "win32" else "clear")
    print(clr("""
  ██╗  ██╗██████╗ ██╗██████╗ ████████╗ ██████╗
  ██║ ██╔╝██╔══██╗██║██╔══██╗╚══██╔══╝██╔═══██╗
  █████╔╝ ██████╔╝██║██████╔╝   ██║   ██║   ██║
  ██╔═██╗ ██╔══██╗██║██╔═══╝    ██║   ██║   ██║
  ██║  ██╗██║  ██║██║██║        ██║   ╚██████╔╝
  ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝        ╚═╝    ╚═════╝
       NOKTA  —  Binance Futures Trading Bot
       Otomatik Kurulum Scripti
""", "1;94"))

    # Klasör kontrolü
    if not Path("requirements.txt").exists():
        err("Bu scripti trading-bot/ klasöründen çalıştır!")
        err(f"Şu an: {Path.cwd()}")
        input("\nEnter...")
        sys.exit(1)

    steps = [
        check_python,
        create_venv,
        install_requirements,
        install_optional,
        check_env,
        test_imports,
    ]

    success = True
    for step in steps:
        if not step():
            # Zorunlu adımlar başarısız olunca dur
            if step in (check_python, install_requirements):
                err("Kritik hata, kurulum durduruluyor.")
                print_summary(False)
                sys.exit(1)
            success = False

    print_summary(success)


if __name__ == "__main__":
    main()
