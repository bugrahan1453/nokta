"""
core/indicators.py
Kapsamlı teknik analiz indikatörleri kütüphanesi.
50+ indikatör, sıfır dış bağımlılık (sadece numpy + pandas).
Her formül açıklamalı, anlaşılır yapı.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict, NamedTuple
from dataclasses import dataclass, field


# ─────────────────────────────────────────────
# TEMEL YARDIMCILAR
# ─────────────────────────────────────────────

def _ema(s: pd.Series, p: int) -> pd.Series:
    return s.ewm(span=p, adjust=False).mean()

def _sma(s: pd.Series, p: int) -> pd.Series:
    return s.rolling(p).mean()

def _std(s: pd.Series, p: int) -> pd.Series:
    return s.rolling(p).std(ddof=0)

def _rma(s: pd.Series, p: int) -> pd.Series:
    """Wilder'ın yumuşatılmış ortalaması (RSI için)."""
    return s.ewm(com=p - 1, adjust=False).mean()

def _true_range(df: pd.DataFrame) -> pd.Series:
    h, l, pc = df["high"], df["low"], df["close"].shift(1)
    return pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)


# ─────────────────────────────────────────────
# ATR — Average True Range
# ─────────────────────────────────────────────

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Ortalama Gerçek Aralık — volatilite ölçütü.
    SL/TP hesaplama ve pozisyon boyutlandırma için kullanılır.
    """
    return _rma(_true_range(df), period)


def atr_dynamic_sl(entry: float, atr_val: float, direction: str,
                   sl_mult: float = 2.0, tp_mult: float = 3.0) -> Tuple[float, float]:
    """ATR çarpanı ile dinamik Stop-Loss ve Take-Profit."""
    if direction == "LONG":
        return round(entry - atr_val * sl_mult, 8), round(entry + atr_val * tp_mult, 8)
    return round(entry + atr_val * sl_mult, 8), round(entry - atr_val * tp_mult, 8)


def chandelier_exit(df: pd.DataFrame, period: int = 22,
                    mult: float = 3.0) -> Tuple[pd.Series, pd.Series]:
    """
    Chandelier Exit — ATR tabanlı dinamik trailing stop.
    Long stop  = N-periyot yüksek - ATR * çarpan
    Short stop = N-periyot düşük  + ATR * çarpan
    """
    a = atr(df, period)
    long_stop  = df["high"].rolling(period).max() - a * mult
    short_stop = df["low"].rolling(period).min()  + a * mult
    return long_stop, short_stop


# ─────────────────────────────────────────────
# BOLLINGER BANDS
# ─────────────────────────────────────────────

@dataclass
class BBResult:
    upper: pd.Series
    middle: pd.Series
    lower: pd.Series
    bandwidth: pd.Series   # (upper-lower)/middle
    pct_b: pd.Series       # (close-lower)/(upper-lower)
    squeeze: pd.Series     # Bant tarihsel minimuma yakın mı


def bollinger_bands(df: pd.DataFrame, period: int = 20, dev: float = 2.0,
                    squeeze_lookback: int = 50) -> BBResult:
    """
    Bollinger Bantları.
    %B > 1 → aşırı alım (üst bandın dışı)
    %B < 0 → aşırı satım (alt bandın dışı)
    Squeeze → büyük hareket yaklaşıyor
    """
    c = df["close"]
    mid = _sma(c, period)
    sd  = _std(c, period)
    up  = mid + dev * sd
    lo  = mid - dev * sd
    bw  = (up - lo) / mid.replace(0, np.nan)
    pb  = (c - lo) / (up - lo).replace(0, np.nan)
    sq  = bw <= bw.rolling(squeeze_lookback).min() * 1.05
    return BBResult(up, mid, lo, bw, pb, sq)


def bb_signal(bb: BBResult, direction: str) -> Tuple[bool, str]:
    """Bollinger sinyal — son mum bazlı."""
    pb  = bb.pct_b.iloc[-1]
    sq_end = bool(bb.squeeze.iloc[-2]) and not bool(bb.squeeze.iloc[-1])
    if direction == "LONG":
        ok = 0.2 < pb < 1.0
    else:
        ok = 0.0 < pb < 0.8
    if sq_end:
        ok = True   # Squeeze çıkışı her yönde güçlü sinyal
    return ok, f"%B={pb:.2f} {'🔥Squeeze bitti' if sq_end else ''}"


# ─────────────────────────────────────────────
# STOCHASTIC RSI
# ─────────────────────────────────────────────

def stochastic_rsi(df: pd.DataFrame, rsi_p: int = 14, stoch_p: int = 14,
                   k_smooth: int = 3, d_smooth: int = 3) -> Tuple[pd.Series, pd.Series]:
    """
    Stochastic RSI — RSI'nın Stochastic'i.
    RSI'dan daha hızlı, daha duyarlı sinyaller.
    """
    c = df["close"]
    delta = c.diff()
    gain  = _rma(delta.clip(lower=0), rsi_p)
    loss  = _rma((-delta).clip(lower=0), rsi_p)
    rsi   = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    rmin  = rsi.rolling(stoch_p).min()
    rmax  = rsi.rolling(stoch_p).max()
    k = 100 * (rsi - rmin) / (rmax - rmin).replace(0, np.nan)
    K = k.rolling(k_smooth).mean()
    D = K.rolling(d_smooth).mean()
    return K, D


def stoch_rsi_signal(K: pd.Series, D: pd.Series, direction: str) -> Tuple[bool, str]:
    """StochRSI sinyal: çapraz + bölge."""
    lk, pk = K.iloc[-1], K.iloc[-2]
    ld, pd_ = D.iloc[-1], D.iloc[-2]
    bull_x = pk < pd_ and lk > ld
    bear_x = pk > pd_ and lk < ld
    if direction == "LONG":
        ok = (lk < 50 and lk > pk) or (lk < 20 and bull_x)
    else:
        ok = (lk > 50 and lk < pk) or (lk > 80 and bear_x)
    return ok, f"StochRSI K={lk:.1f} D={ld:.1f} {'↑cross' if bull_x else '↓cross' if bear_x else ''}"


# ─────────────────────────────────────────────
# VWAP
# ─────────────────────────────────────────────

def vwap(df: pd.DataFrame) -> pd.Series:
    """
    Hacim Ağırlıklı Ortalama Fiyat.
    Kurumsal referans fiyatı — gün içi trend filtresi.
    """
    tp = (df["high"] + df["low"] + df["close"]) / 3
    return (tp * df["volume"]).cumsum() / df["volume"].cumsum()


def vwap_bands(df: pd.DataFrame, dev: float = 1.5) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """VWAP + üst/alt standart sapma bantları."""
    tp   = (df["high"] + df["low"] + df["close"]) / 3
    vw   = vwap(df)
    vol  = df["volume"]
    variance = ((tp - vw) ** 2 * vol).cumsum() / vol.cumsum()
    sd = np.sqrt(variance)
    return vw + dev * sd, vw, vw - dev * sd


def vwap_signal(df: pd.DataFrame, direction: str) -> Tuple[bool, float, str]:
    vw  = vwap(df)
    lv  = vw.iloc[-1]
    lc  = df["close"].iloc[-1]
    dist = (lc - lv) / lv * 100
    ok  = (direction == "LONG" and lc > lv) or (direction == "SHORT" and lc < lv)
    return ok, dist, f"VWAP={lv:.4f} dist={dist:+.2f}%"


# ─────────────────────────────────────────────
# RSI DIVERJANS
# ─────────────────────────────────────────────

@dataclass
class DivergenceResult:
    bullish: bool
    bearish: bool
    hidden_bullish: bool
    hidden_bearish: bool
    description: str


def rsi_divergence(df: pd.DataFrame, rsi_p: int = 14, lookback: int = 40,
                   swing_window: int = 5) -> DivergenceResult:
    """
    RSI Diverjans — fiyat ve momentum zıt yönlere gittiğinde dönüş sinyali.
    Klasik: Fiyat yeni zirve/dip yapar ama RSI yapmaz → trend yorgunluğu.
    Gizli  : RSI yeni zirve/dip yapar ama fiyat yapmaz → trend devamı.
    """
    c = df["close"].iloc[-lookback:]
    delta = df["close"].diff()
    rsi_s = 100 - 100 / (1 + _rma(delta.clip(lower=0), rsi_p) /
                          _rma((-delta).clip(lower=0), rsi_p).replace(0, np.nan))
    rsi_s = rsi_s.iloc[-lookback:]

    def swings(s, kind="min", w=swing_window):
        idx = []
        for i in range(w, len(s) - w):
            window = s.iloc[i - w: i + w + 1]
            if kind == "min" and s.iloc[i] == window.min():
                idx.append(i)
            elif kind == "max" and s.iloc[i] == window.max():
                idx.append(i)
        return idx[-2:] if len(idx) >= 2 else []

    p_lows  = swings(c, "min")
    p_highs = swings(c, "max")
    r_lows  = swings(rsi_s, "min")
    r_highs = swings(rsi_s, "max")

    bull = bear = hbull = hbear = False
    msgs = []

    if len(p_lows) == 2 and len(r_lows) == 2:
        i1, i2 = p_lows
        if c.iloc[i2] < c.iloc[i1] and rsi_s.iloc[i2] > rsi_s.iloc[i1]:
            bull = True; msgs.append("Klasik Bullish Div")
        if c.iloc[i2] > c.iloc[i1] and rsi_s.iloc[i2] < rsi_s.iloc[i1]:
            hbull = True; msgs.append("Gizli Bullish Div (trend devamı)")

    if len(p_highs) == 2 and len(r_highs) == 2:
        i1, i2 = p_highs
        if c.iloc[i2] > c.iloc[i1] and rsi_s.iloc[i2] < rsi_s.iloc[i1]:
            bear = True; msgs.append("Klasik Bearish Div")
        if c.iloc[i2] < c.iloc[i1] and rsi_s.iloc[i2] > rsi_s.iloc[i1]:
            hbear = True; msgs.append("Gizli Bearish Div (trend devamı)")

    return DivergenceResult(bull, bear, hbull, hbear, ", ".join(msgs) or "Diverjans yok")


# ─────────────────────────────────────────────
# MUM FORMASYONLARI
# ─────────────────────────────────────────────

@dataclass
class CandlePattern:
    name: str
    bullish: bool
    strength: int  # 1=zayıf 2=orta 3=güçlü


def detect_candle_patterns(df: pd.DataFrame) -> List[CandlePattern]:
    """
    Otomatik mum formasyon tespiti.
    Engulfing, Hammer, Shooting Star, Doji, Marubozu,
    Piercing/Dark Cloud, Morning/Evening Star, Tweezer.
    """
    if len(df) < 3:
        return []
    patterns: List[CandlePattern] = []

    c2, c1, c0 = df.iloc[-1], df.iloc[-2], df.iloc[-3]

    def body(c):    return abs(c["close"] - c["open"])
    def candle(c):  return c["high"] - c["low"]
    def upper(c):   return c["high"] - max(c["open"], c["close"])
    def lower(c):   return min(c["open"], c["close"]) - c["low"]
    def bull(c):    return c["close"] > c["open"]
    def bear(c):    return c["close"] < c["open"]

    b2, c2len = body(c2), candle(c2)
    b1, c1len = body(c1), candle(c1)
    u2, l2 = upper(c2), lower(c2)
    u1, l1 = upper(c1), lower(c1)

    # Engulfing
    if bear(c1) and bull(c2) and c2["open"] <= c1["close"] and c2["close"] >= c1["open"]:
        patterns.append(CandlePattern("Bullish Engulfing", True, 3))
    if bull(c1) and bear(c2) and c2["open"] >= c1["close"] and c2["close"] <= c1["open"]:
        patterns.append(CandlePattern("Bearish Engulfing", False, 3))

    # Hammer / Hanging Man
    if c2len > 0 and b2 < c2len * 0.35 and l2 >= c2len * 0.6 and u2 <= b2 * 0.3:
        patterns.append(CandlePattern("Hammer", True, 2))
    # Shooting Star
    if c2len > 0 and b2 < c2len * 0.35 and u2 >= c2len * 0.6 and l2 <= b2 * 0.3:
        patterns.append(CandlePattern("Shooting Star", False, 2))

    # Doji
    if c2len > 0 and b2 <= c2len * 0.08:
        if l2 > u2 * 2: name, bul = "Dragonfly Doji", True
        elif u2 > l2 * 2: name, bul = "Gravestone Doji", False
        else: name, bul = "Doji", bull(c1)
        patterns.append(CandlePattern(name, bul, 1))

    # Marubozu
    if c2len > 0 and b2 >= c2len * 0.92:
        patterns.append(CandlePattern(
            "Bullish Marubozu" if bull(c2) else "Bearish Marubozu", bull(c2), 2))

    # Piercing Line
    if bear(c1) and bull(c2) and c2["open"] < c1["low"] and c2["close"] > (c1["open"] + c1["close"]) / 2:
        patterns.append(CandlePattern("Piercing Line", True, 2))

    # Dark Cloud Cover
    if bull(c1) and bear(c2) and c2["open"] > c1["high"] and c2["close"] < (c1["open"] + c1["close"]) / 2:
        patterns.append(CandlePattern("Dark Cloud Cover", False, 2))

    # Morning Star (3 mum)
    if bear(c0) and body(c1) < body(c0) * 0.4 and bull(c2) and c2["close"] > (c0["open"] + c0["close"]) / 2:
        patterns.append(CandlePattern("Morning Star", True, 3))

    # Evening Star (3 mum)
    if bull(c0) and body(c1) < body(c0) * 0.4 and bear(c2) and c2["close"] < (c0["open"] + c0["close"]) / 2:
        patterns.append(CandlePattern("Evening Star", False, 3))

    # Tweezer Top/Bottom
    if abs(c1["high"] - c2["high"]) / max(c2["high"], 1) < 0.001 and bear(c2):
        patterns.append(CandlePattern("Tweezer Top", False, 2))
    if abs(c1["low"] - c2["low"]) / max(c2["low"], 1) < 0.001 and bull(c2):
        patterns.append(CandlePattern("Tweezer Bottom", True, 2))

    return patterns


def candle_pattern_signal(patterns: List[CandlePattern], direction: str) -> Tuple[bool, str]:
    if not patterns:
        return True, "Formasyon yok (nötr)"
    bull_score = sum(p.strength for p in patterns if p.bullish)
    bear_score = sum(p.strength for p in patterns if not p.bullish)
    names = [p.name for p in patterns]
    if direction == "LONG":
        ok = bull_score >= bear_score
    else:
        ok = bear_score >= bull_score
    return ok, f"{', '.join(names)} | B:{bull_score} A:{bear_score}"


# ─────────────────────────────────────────────
# İSTATİSTİKSEL GÖSTERGELER
# ─────────────────────────────────────────────

def z_score(s: pd.Series, period: int = 20) -> pd.Series:
    """Z-Score: |Z|>2 aşırı değer — geri dönüş riski."""
    return (s - s.rolling(period).mean()) / s.rolling(period).std(ddof=0).replace(0, np.nan)


def hurst_exponent(s: pd.Series, max_lag: int = 20) -> float:
    """
    Hurst Katsayısı (0-1):
    H < 0.5 → Mean-reversion (geri dönüş)
    H = 0.5 → Random walk
    H > 0.5 → Trend (momentum)
    """
    lags   = range(2, max_lag)
    tau    = [np.sqrt(np.std(np.subtract(s.values[lag:], s.values[:-lag]))) for lag in lags]
    poly   = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0


def skewness(s: pd.Series, period: int = 20) -> pd.Series:
    """Çarpıklık — dağılım simetri ölçütü."""
    return s.rolling(period).skew()


def kurtosis(s: pd.Series, period: int = 20) -> pd.Series:
    """Basıklık — uç değer yoğunluğu."""
    return s.rolling(period).kurt()


# ─────────────────────────────────────────────
# EK TREND GÖSTERGELERİ
# ─────────────────────────────────────────────

def supertrend(df: pd.DataFrame, period: int = 10, mult: float = 3.0) -> Tuple[pd.Series, pd.Series]:
    """
    Supertrend — ATR tabanlı trend çizgisi.
    Returns: (supertrend_line, direction_series)  1=yükseliş -1=düşüş
    """
    a = atr(df, period)
    hl2 = (df["high"] + df["low"]) / 2
    upper_band = hl2 + mult * a
    lower_band = hl2 - mult * a

    st  = pd.Series(np.nan, index=df.index)
    dir_ = pd.Series(1, index=df.index)
    close = df["close"]

    for i in range(1, len(df)):
        prev_upper = upper_band.iloc[i - 1]
        prev_lower = lower_band.iloc[i - 1]
        prev_st    = st.iloc[i - 1] if not np.isnan(st.iloc[i - 1]) else lower_band.iloc[i]
        prev_dir   = dir_.iloc[i - 1]

        # Bantları sıkıştır
        cur_lower = lower_band.iloc[i]
        cur_upper = upper_band.iloc[i]
        if cur_lower < prev_lower or close.iloc[i - 1] < prev_lower:
            cur_lower = cur_lower
        else:
            cur_lower = prev_lower

        if cur_upper > prev_upper or close.iloc[i - 1] > prev_upper:
            cur_upper = cur_upper
        else:
            cur_upper = prev_upper

        lower_band.iloc[i] = cur_lower
        upper_band.iloc[i] = cur_upper

        if prev_dir == -1 and close.iloc[i] > prev_st:
            dir_.iloc[i] = 1
        elif prev_dir == 1 and close.iloc[i] < prev_st:
            dir_.iloc[i] = -1
        else:
            dir_.iloc[i] = prev_dir

        st.iloc[i] = cur_lower if dir_.iloc[i] == 1 else cur_upper

    return st, dir_


def adx(df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    ADX — Trend Gücü (0-100).
    ADX > 25 → güçlü trend, < 20 → yatay piyasa
    +DI, -DI → yön
    """
    tr  = _true_range(df)
    up  = df["high"].diff()
    dn  = -df["low"].diff()

    plus_dm  = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)

    atr_s   = _rma(tr, period)
    plus_di = 100 * _rma(pd.Series(plus_dm, index=df.index), period) / atr_s.replace(0, np.nan)
    minus_di= 100 * _rma(pd.Series(minus_dm, index=df.index), period) / atr_s.replace(0, np.nan)

    dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_s = _rma(dx, period)

    return adx_s, plus_di, minus_di


def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Commodity Channel Index.
    +100 → aşırı alım, -100 → aşırı satım
    """
    tp   = (df["high"] + df["low"] + df["close"]) / 3
    ma   = tp.rolling(period).mean()
    mad  = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - ma) / (0.015 * mad.replace(0, np.nan))


def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Williams %R — aşırı alım/satım osilatörü (-100 ile 0 arası)."""
    hh = df["high"].rolling(period).max()
    ll  = df["low"].rolling(period).min()
    return -100 * (hh - df["close"]) / (hh - ll).replace(0, np.nan)


def roc(s: pd.Series, period: int = 12) -> pd.Series:
    """Rate of Change — Değişim Oranı (%)."""
    return (s / s.shift(period) - 1) * 100


def dpo(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Detrended Price Oscillator — trend kaldırılmış fiyat."""
    shift = period // 2 + 1
    return df["close"] - _sma(df["close"], period).shift(shift)


def ichimoku(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Ichimoku Bulut — Japon trend + destek/direnç sistemi.
    tenkan: Dönüşüm çizgisi (9 periyot)
    kijun:  Baz çizgisi (26 periyot)
    senkou_a/b: Bulut bantları (52 periyot)
    chikou: Çeşni çizgisi (gecikmeli kapanış)
    """
    def midpoint(p):
        return (df["high"].rolling(p).max() + df["low"].rolling(p).min()) / 2

    tenkan  = midpoint(9)
    kijun   = midpoint(26)
    sa = ((tenkan + kijun) / 2).shift(26)
    sb = midpoint(52).shift(26)
    chikou = df["close"].shift(-26)

    return {
        "tenkan": tenkan, "kijun": kijun,
        "senkou_a": sa, "senkou_b": sb, "chikou": chikou
    }


# ─────────────────────────────────────────────
# KELLY KRİTERİ
# ─────────────────────────────────────────────

def kelly_criterion(win_rate: float, avg_win: float, avg_loss: float,
                    fraction: float = 0.25, max_pct: float = 5.0) -> float:
    """
    Optimal f* = (p*b - q) / b
    fraction = güvenlik katsayısı (0.25 = çeyrek Kelly önerilir)
    max_pct  = kasa yüzdesi güvenlik tavanı
    """
    if avg_loss == 0 or win_rate <= 0 or win_rate >= 1:
        return 1.0
    rr = abs(avg_win / avg_loss)
    f  = (win_rate * rr - (1 - win_rate)) / rr
    if f <= 0:
        return 0.5
    return min(f * fraction * 100, max_pct)


# ─────────────────────────────────────────────
# MULTI-TIMEFRAME MOMENTUM
# ─────────────────────────────────────────────

def multi_tf_momentum(c5: pd.Series, c15: pd.Series, c1h: pd.Series,
                      period: int = 14) -> dict:
    """
    3 zaman dilimi momentum skoru.
    1h > 15m > 5m ağırlığı — büyük TF daha önemli.
    aligned=True → tüm TF'ler aynı yönde.
    """
    def _roc(s, p): return ((s.iloc[-1] / s.iloc[-p]) - 1) * 100 if len(s) > p else 0.0
    m5, m15, m1h = _roc(c5, period), _roc(c15, period), _roc(c1h, period)
    comp = m5 * 0.2 + m15 * 0.3 + m1h * 0.5
    same_sign = (m5 > 0 and m15 > 0 and m1h > 0) or (m5 < 0 and m15 < 0 and m1h < 0)
    return {"5m": round(m5, 4), "15m": round(m15, 4), "1h": round(m1h, 4),
            "composite": round(comp, 4), "aligned": same_sign}


# ─────────────────────────────────────────────
# CROSS-ASSET KORELASYON
# ─────────────────────────────────────────────

def rolling_correlation(a: pd.Series, b: pd.Series, period: int = 20) -> pd.Series:
    """İki varlık arasındaki kayan korelasyon (-1 ile +1)."""
    return a.rolling(period).corr(b)


def btc_dominance_signal(btc_price: pd.Series, alt_price: pd.Series,
                          period: int = 20) -> Tuple[float, str]:
    """
    BTC/ALT korelasyon analizi.
    Yüksek korelasyon + BTC düşüşte = ALT için tehlike.
    """
    corr = rolling_correlation(btc_price, alt_price, period).iloc[-1]
    btc_roc = roc(btc_price, 5).iloc[-1]
    if corr > 0.7 and btc_roc < -1:
        return corr, f"⚠️ BTC ile yüksek korelasyon ({corr:.2f}), BTC düşüyor ({btc_roc:.2f}%)"
    return corr, f"BTC korelasyon={corr:.2f}, BTC ROC={btc_roc:.2f}%"
