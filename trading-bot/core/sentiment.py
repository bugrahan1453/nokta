"""
sentiment.py - Duygu Analizi Katmanı (Opsiyonel)
- Fear & Greed Index (ücretsiz - alternative.me)
- Twitter/X API v2 (opsiyonel - bearer token gerekli)
- Reddit PRAW (opsiyonel - client_id/secret gerekli)
- FinBERT (opsiyonel - transformers + torch gerekli)
- CryptoPanic RSS (ücretsiz - API key olmadan)
"""

import os
import re
import time
import json
import logging
import threading
import hashlib
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field
from collections import deque

from core.advanced_analysis import AnalysisResult

logger = logging.getLogger(__name__)

# ── Opsiyonel kütüphaneler ──────────────────────────────────────────────────
try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False
    logger.info("feedparser kurulu değil. pip install feedparser")

try:
    import praw
    PRAW_AVAILABLE = True
except ImportError:
    PRAW_AVAILABLE = False
    logger.info("PRAW kurulu değil. pip install praw")

try:
    import tweepy
    TWEEPY_AVAILABLE = True
except ImportError:
    TWEEPY_AVAILABLE = False
    logger.info("Tweepy kurulu değil. pip install tweepy")

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    FINBERT_AVAILABLE = True
except ImportError:
    FINBERT_AVAILABLE = False
    logger.info("Transformers kurulu değil. pip install transformers torch")

# TextBlob - basit duygu analizi fallback
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

# VADER - finans/sosyal medya için optimize duygu
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    logger.info("VADER kurulu değil. pip install vaderSentiment")


# ── Kelime Skoru Listesi (basit, ücretsiz fallback) ────────────────────────

CRYPTO_POSITIVE_WORDS = {
    'bullish', 'moon', 'pump', 'rally', 'surge', 'breakout', 'accumulate',
    'buy', 'long', 'support', 'bounce', 'upgrade', 'adoption', 'partnership',
    'launch', 'milestone', 'all-time-high', 'ath', 'growth', 'gain',
    'profit', 'hodl', 'accumulation', 'institutional', 'inflow', 'green',
    'recover', 'recovery', 'strong', 'bull', 'uptrend', 'positive',
    'yükseliş', 'artış', 'güçlü', 'alım', 'boğa', 'destek',
}

CRYPTO_NEGATIVE_WORDS = {
    'bearish', 'dump', 'crash', 'fall', 'drop', 'sell', 'short', 'resistance',
    'rejection', 'breakdown', 'capitulation', 'fud', 'fear', 'panic', 'ban',
    'hack', 'exploit', 'scam', 'rug', 'liquidation', 'loss', 'red',
    'bear', 'downtrend', 'negative', 'correction', 'warning', 'risk',
    'düşüş', 'kayıp', 'satış', 'ayı', 'panik', 'tehlike',
}


# ── Yardımcı Fonksiyonlar ───────────────────────────────────────────────────

def simple_sentiment_score(text: str) -> float:
    """
    Basit kelime tabanlı duygu skoru.
    -1.0 (çok negatif) ile +1.0 (çok pozitif) arasında.
    """
    if not text:
        return 0.0
    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return 0.0

    pos = sum(1 for w in words if w in CRYPTO_POSITIVE_WORDS)
    neg = sum(1 for w in words if w in CRYPTO_NEGATIVE_WORDS)
    total = pos + neg

    if total == 0:
        return 0.0
    return (pos - neg) / total


def vader_score(text: str, analyzer) -> float:
    """VADER compound skoru (-1 to +1)."""
    try:
        scores = analyzer.polarity_scores(text)
        return scores['compound']
    except Exception:
        return 0.0


def textblob_score(text: str) -> float:
    """TextBlob polarity (-1 to +1)."""
    try:
        return TextBlob(text).sentiment.polarity
    except Exception:
        return 0.0


def keyword_relevance(text: str, symbol: str) -> float:
    """Metnin sembolle ne kadar ilgili olduğu (0-1)."""
    symbol_map = {
        'BTCUSDT': ['bitcoin', 'btc', '$btc', '#bitcoin'],
        'ETHUSDT': ['ethereum', 'eth', '$eth', '#ethereum', 'ether'],
        'BNBUSDT': ['binance', 'bnb', '$bnb', '#bnb'],
        'SOLUSDT': ['solana', 'sol', '$sol', '#solana'],
        'XRPUSDT': ['xrp', 'ripple', '$xrp'],
        'ADAUSDT': ['cardano', 'ada', '$ada'],
        'DOGEUSDT': ['dogecoin', 'doge', '$doge'],
        'AVAXUSDT': ['avalanche', 'avax', '$avax'],
        'DOTUSDT': ['polkadot', 'dot', '$dot'],
        'MATICUSDT': ['polygon', 'matic', '$matic'],
    }
    keywords = symbol_map.get(symbol.upper(), [symbol.lower().replace('usdt', '')])
    text_lower = text.lower()
    matches = sum(1 for kw in keywords if kw in text_lower)
    return min(1.0, matches / max(1, len(keywords)))


# ── FinBERT Analiz ──────────────────────────────────────────────────────────

class FinBERTAnalyzer:
    """FinBERT modeli ile finansal duygu analizi."""

    MODEL_NAME = 'ProsusAI/finbert'

    def __init__(self):
        self._pipe = None
        self._loaded = False
        self._load_thread = None

        if FINBERT_AVAILABLE:
            self._load_thread = threading.Thread(target=self._load_model, daemon=True)
            self._load_thread.start()

    def _load_model(self):
        try:
            logger.info("FinBERT modeli yükleniyor (ilk çalıştırmada uzun sürebilir)...")
            self._pipe = pipeline(
                'text-classification',
                model=self.MODEL_NAME,
                device=0 if (FINBERT_AVAILABLE and
                              __import__('torch').cuda.is_available()) else -1
            )
            self._loaded = True
            logger.info("FinBERT yüklendi.")
        except Exception as e:
            logger.warning(f"FinBERT yükleme hatası: {e}")

    def analyze(self, texts: List[str]) -> Optional[float]:
        """Metinleri analiz eder, ortalama skor döner (-1 to +1)."""
        if not self._loaded or self._pipe is None:
            return None
        try:
            scores = []
            for text in texts[:20]:  # max 20 metin
                if len(text) > 512:
                    text = text[:512]
                result = self._pipe(text, truncation=True)[0]
                label = result['label'].lower()
                conf = result['score']
                if label == 'positive':
                    scores.append(conf)
                elif label == 'negative':
                    scores.append(-conf)
                else:
                    scores.append(0.0)
            return float(sum(scores) / len(scores)) if scores else None
        except Exception as e:
            logger.error(f"FinBERT analiz hatası: {e}")
            return None


# ── CryptoPanic RSS (Ücretsiz) ──────────────────────────────────────────────

class CryptoPanicAnalyzer:
    """CryptoPanic RSS feed üzerinden haber duygu analizi (ücretsiz)."""

    RSS_URL = "https://cryptopanic.com/news/rss/"
    CACHE_TTL = 900  # 15 dakika

    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._cache_time: Dict[str, datetime] = {}
        self.vader = SentimentIntensityAnalyzer() if VADER_AVAILABLE else None

    def get_news_sentiment(self, symbol: str) -> AnalysisResult:
        if not FEEDPARSER_AVAILABLE:
            return AnalysisResult.neutral('feedparser kurulu değil')

        cache_key = f"cryptopanic_{symbol}"
        if cache_key in self._cache:
            if (datetime.now() - self._cache_time[cache_key]).seconds < self.CACHE_TTL:
                return self._cache[cache_key]

        try:
            feed = feedparser.parse(self.RSS_URL)
            if not feed.entries:
                return AnalysisResult.neutral('CryptoPanic: RSS boş')

            scores = []
            for entry in feed.entries[:30]:
                title = entry.get('title', '')
                summary = entry.get('summary', '')
                text = f"{title} {summary}"

                # Sembolle ilgili mi?
                relevance = keyword_relevance(text, symbol)
                if relevance < 0.2 and 'bitcoin' not in text.lower() and 'crypto' not in text.lower():
                    continue

                # Skor hesapla
                if self.vader:
                    score = vader_score(text, self.vader)
                elif TEXTBLOB_AVAILABLE:
                    score = textblob_score(text)
                else:
                    score = simple_sentiment_score(text)

                weight = 0.5 + relevance * 0.5  # ilgililik ağırlığı
                scores.append(score * weight)

            if not scores:
                result = AnalysisResult.neutral('CryptoPanic: İlgili haber yok')
            else:
                avg_score = sum(scores) / len(scores)
                confidence = min(90, len(scores) * 5)
                result = AnalysisResult(
                    score=float(avg_score),
                    confidence=float(confidence),
                    source='cryptopanic_rss',
                    details={'news_count': len(scores), 'avg_score': round(avg_score, 3)},
                    timestamp=datetime.now(),
                )

            self._cache[cache_key] = result
            self._cache_time[cache_key] = datetime.now()
            return result

        except Exception as e:
            logger.error(f"CryptoPanic hatası: {e}")
            return AnalysisResult.error(f'CryptoPanic: {e}')


# ── Twitter/X Analizi (Opsiyonel) ───────────────────────────────────────────

class TwitterSentimentAnalyzer:
    """Twitter API v2 üzerinden kripto duygu analizi."""

    def __init__(self, bearer_token: str, finbert: Optional[FinBERTAnalyzer] = None):
        self.enabled = False
        self._client = None
        self.finbert = finbert
        self.vader = SentimentIntensityAnalyzer() if VADER_AVAILABLE else None
        self._cache: Dict[str, Any] = {}
        self._cache_time: Dict[str, datetime] = {}
        self.CACHE_TTL = 600  # 10 dakika

        if not TWEEPY_AVAILABLE:
            return
        if not bearer_token:
            return

        try:
            self._client = tweepy.Client(
                bearer_token=bearer_token,
                wait_on_rate_limit=False
            )
            self.enabled = True
            logger.info("Twitter API v2 bağlandı.")
        except Exception as e:
            logger.warning(f"Twitter bağlantı hatası: {e}")

    def get_sentiment(self, symbol: str) -> AnalysisResult:
        if not self.enabled:
            return AnalysisResult.neutral('Twitter devre dışı')

        cache_key = f"twitter_{symbol}"
        if cache_key in self._cache:
            if (datetime.now() - self._cache_time[cache_key]).seconds < self.CACHE_TTL:
                return self._cache[cache_key]

        # Arama sorgusu
        coin_name = symbol.replace('USDT', '').lower()
        query_map = {
            'btc': 'bitcoin OR $BTC -is:retweet lang:en',
            'eth': 'ethereum OR $ETH -is:retweet lang:en',
            'bnb': '$BNB OR binance coin -is:retweet lang:en',
            'sol': 'solana OR $SOL -is:retweet lang:en',
        }
        query = query_map.get(coin_name, f'${coin_name.upper()} -is:retweet lang:en')

        try:
            response = self._client.search_recent_tweets(
                query=query,
                max_results=50,
                tweet_fields=['created_at', 'public_metrics', 'lang'],
            )

            if not response.data:
                result = AnalysisResult.neutral('Twitter: Tweet bulunamadı')
                self._cache[cache_key] = result
                self._cache_time[cache_key] = datetime.now()
                return result

            texts = [tweet.text for tweet in response.data]
            scores = []

            # FinBERT varsa kullan
            if self.finbert and self.finbert._loaded:
                finbert_score = self.finbert.analyze(texts)
                if finbert_score is not None:
                    scores.append(('finbert', finbert_score, 0.6))

            # VADER fallback
            if self.vader:
                vader_scores = [vader_score(t, self.vader) for t in texts]
                avg_vader = sum(vader_scores) / len(vader_scores) if vader_scores else 0
                scores.append(('vader', avg_vader, 0.4 if scores else 1.0))

            if not scores:
                result = AnalysisResult.neutral('Twitter: Skor hesaplanamadı')
            else:
                total_weight = sum(w for _, _, w in scores)
                weighted_score = sum(s * w for _, s, w in scores) / total_weight
                confidence = min(85, len(texts) * 1.5)

                result = AnalysisResult(
                    score=float(np.clip(weighted_score, -1, 1)),
                    confidence=float(confidence),
                    source='twitter',
                    details={
                        'tweet_count': len(texts),
                        'scorers': [{'name': n, 'score': round(s, 3)} for n, s, _ in scores],
                    },
                    timestamp=datetime.now(),
                )

            self._cache[cache_key] = result
            self._cache_time[cache_key] = datetime.now()
            return result

        except tweepy.TooManyRequests:
            logger.warning("Twitter rate limit aşıldı.")
            return AnalysisResult.neutral('Twitter: Rate limit')
        except Exception as e:
            logger.error(f"Twitter analiz hatası: {e}")
            return AnalysisResult.error(f'Twitter: {e}')


# ── Reddit Analizi (Opsiyonel) ───────────────────────────────────────────────

class RedditSentimentAnalyzer:
    """Reddit PRAW üzerinden kripto duygu analizi."""

    SUBREDDITS = {
        'BTCUSDT': ['Bitcoin', 'CryptoCurrency', 'BitcoinMarkets'],
        'ETHUSDT': ['ethereum', 'CryptoCurrency', 'ethtrader'],
        'SOLUSDT': ['solana', 'CryptoCurrency'],
        'BNBUSDT': ['binance', 'CryptoCurrency'],
        'DEFAULT': ['CryptoCurrency', 'CryptoMarkets'],
    }

    def __init__(self, client_id: str, client_secret: str, user_agent: str = 'TradingBot/1.0',
                 finbert: Optional[FinBERTAnalyzer] = None):
        self.enabled = False
        self._reddit = None
        self.finbert = finbert
        self.vader = SentimentIntensityAnalyzer() if VADER_AVAILABLE else None
        self._cache: Dict[str, Any] = {}
        self._cache_time: Dict[str, datetime] = {}
        self.CACHE_TTL = 900  # 15 dakika

        if not PRAW_AVAILABLE:
            return
        if not client_id or not client_secret:
            return

        try:
            self._reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent,
                ratelimit_seconds=5,
            )
            # Bağlantıyı test et
            _ = self._reddit.user.me()
            self.enabled = True
            logger.info("Reddit API bağlandı.")
        except Exception as e:
            logger.warning(f"Reddit bağlantı hatası: {e}")

    def get_sentiment(self, symbol: str) -> AnalysisResult:
        if not self.enabled:
            return AnalysisResult.neutral('Reddit devre dışı')

        cache_key = f"reddit_{symbol}"
        if cache_key in self._cache:
            if (datetime.now() - self._cache_time[cache_key]).seconds < self.CACHE_TTL:
                return self._cache[cache_key]

        subreddits = self.SUBREDDITS.get(symbol.upper(), self.SUBREDDITS['DEFAULT'])

        try:
            all_texts = []
            for sub_name in subreddits[:2]:  # max 2 subreddit
                sub = self._reddit.subreddit(sub_name)
                posts = list(sub.hot(limit=25))
                for post in posts:
                    text = f"{post.title} {post.selftext[:200]}"
                    rel = keyword_relevance(text, symbol)
                    # Score * upvote ağırlığı
                    upvote_w = min(1.0, (post.score + 1) / 1000)
                    all_texts.append((text, rel, upvote_w))

            if not all_texts:
                result = AnalysisResult.neutral('Reddit: Post bulunamadı')
                self._cache[cache_key] = result
                self._cache_time[cache_key] = datetime.now()
                return result

            scores = []
            for text, rel, upvote_w in all_texts:
                if self.vader:
                    s = vader_score(text, self.vader)
                elif TEXTBLOB_AVAILABLE:
                    s = textblob_score(text)
                else:
                    s = simple_sentiment_score(text)
                weight = (0.3 + rel * 0.4 + upvote_w * 0.3)
                scores.append(s * weight)

            # FinBERT
            finbert_score_val = None
            if self.finbert and self.finbert._loaded:
                texts_only = [t for t, _, _ in all_texts[:15]]
                finbert_score_val = self.finbert.analyze(texts_only)

            final_score = sum(scores) / len(scores) if scores else 0.0
            if finbert_score_val is not None:
                final_score = 0.6 * finbert_score_val + 0.4 * final_score

            confidence = min(80, len(all_texts) * 2)
            result = AnalysisResult(
                score=float(np.clip(final_score, -1, 1)),
                confidence=float(confidence),
                source='reddit',
                details={
                    'post_count': len(all_texts),
                    'subreddits': subreddits[:2],
                    'finbert_used': finbert_score_val is not None,
                },
                timestamp=datetime.now(),
            )

            self._cache[cache_key] = result
            self._cache_time[cache_key] = datetime.now()
            return result

        except Exception as e:
            logger.error(f"Reddit analiz hatası: {e}")
            return AnalysisResult.error(f'Reddit: {e}')


# numpy import - yukarıdaki sınıflarda kullanılıyor
import numpy as np


# ── Ana Duygu Analizi Yöneticisi ────────────────────────────────────────────

class SentimentLayer:
    """
    Tüm duygu analizi kaynaklarını yönetir ve birleştirir.
    Devre dışı kaynaklar sessizce atlanır.
    """

    # Kaynak ağırlıkları
    SOURCE_WEIGHTS = {
        'cryptopanic': 0.3,
        'reddit': 0.25,
        'twitter': 0.25,
        'finbert': 0.2,  # tek başına kullanıldığında
    }

    def __init__(self, settings=None):
        self.settings = settings
        self.enabled = self._check_enabled()

        self._lock = threading.Lock()
        self._cache: Dict[str, AnalysisResult] = {}
        self._cache_time: Dict[str, datetime] = {}
        self.CACHE_TTL = 600  # 10 dakika

        # Bileşenler
        self.finbert: Optional[FinBERTAnalyzer] = None
        self.cryptopanic: Optional[CryptoPanicAnalyzer] = None
        self.twitter: Optional[TwitterSentimentAnalyzer] = None
        self.reddit: Optional[RedditSentimentAnalyzer] = None

        if self.enabled:
            self._init_components()

    def _check_enabled(self) -> bool:
        if self.settings:
            return getattr(self.settings, 'sentiment_enabled', True)
        return os.getenv('SENTIMENT_ENABLED', 'true').lower() == 'true'

    def _init_components(self):
        # FinBERT (opsiyonel, arka planda yükler)
        finbert_enabled = os.getenv('FINBERT_ENABLED', 'false').lower() == 'true'
        if finbert_enabled and FINBERT_AVAILABLE:
            self.finbert = FinBERTAnalyzer()

        # CryptoPanic RSS (ücretsiz, her zaman aktif)
        self.cryptopanic = CryptoPanicAnalyzer()

        # Twitter (opsiyonel)
        twitter_token = os.getenv('TWITTER_BEARER_TOKEN', '')
        if twitter_token:
            self.twitter = TwitterSentimentAnalyzer(
                bearer_token=twitter_token,
                finbert=self.finbert
            )

        # Reddit (opsiyonel)
        reddit_id = os.getenv('REDDIT_CLIENT_ID', '')
        reddit_secret = os.getenv('REDDIT_CLIENT_SECRET', '')
        if reddit_id and reddit_secret:
            self.reddit = RedditSentimentAnalyzer(
                client_id=reddit_id,
                client_secret=reddit_secret,
                finbert=self.finbert
            )

        active = ['cryptopanic']
        if self.twitter and self.twitter.enabled:
            active.append('twitter')
        if self.reddit and self.reddit.enabled:
            active.append('reddit')
        if self.finbert:
            active.append('finbert')

        logger.info(f"SentimentLayer başlatıldı. Aktif kaynaklar: {active}")

    def analyze(self, symbol: str) -> AnalysisResult:
        """Tüm kaynaklardan duygu analizi yapıp birleştirir."""
        if not self.enabled:
            return AnalysisResult.neutral('Sentiment katmanı devre dışı')

        cache_key = symbol
        with self._lock:
            if cache_key in self._cache:
                if (datetime.now() - self._cache_time[cache_key]).seconds < self.CACHE_TTL:
                    return self._cache[cache_key]

        results: List[Tuple[str, AnalysisResult]] = []

        # CryptoPanic (her zaman çalışır)
        if self.cryptopanic:
            r = self.cryptopanic.get_news_sentiment(symbol)
            if r.score != 0 or r.confidence > 0:
                results.append(('cryptopanic', r))

        # Twitter
        if self.twitter and self.twitter.enabled:
            r = self.twitter.get_sentiment(symbol)
            if not r.source.endswith('error'):
                results.append(('twitter', r))

        # Reddit
        if self.reddit and self.reddit.enabled:
            r = self.reddit.get_sentiment(symbol)
            if not r.source.endswith('error'):
                results.append(('reddit', r))

        if not results:
            return AnalysisResult.neutral('Sentiment: Kaynak bulunamadı')

        # Ağırlıklı ortalama
        total_weight = 0.0
        weighted_score = 0.0
        weighted_confidence = 0.0

        for source_name, result in results:
            w = self.SOURCE_WEIGHTS.get(source_name, 0.25)
            effective_w = w * (result.confidence / 100.0)
            weighted_score += result.score * effective_w
            weighted_confidence += result.confidence * w
            total_weight += effective_w

        if total_weight == 0:
            return AnalysisResult.neutral('Sentiment: Ağırlık hesaplanamadı')

        final_score = weighted_score / total_weight
        final_conf = weighted_confidence / len(results)

        combined = AnalysisResult(
            score=float(np.clip(final_score, -1, 1)),
            confidence=float(np.clip(final_conf, 0, 100)),
            source='sentiment_combined',
            details={
                'sources': [
                    {'name': name, 'score': round(r.score, 3), 'confidence': round(r.confidence, 1)}
                    for name, r in results
                ],
                'symbol': symbol,
            },
            timestamp=datetime.now(),
        )

        with self._lock:
            self._cache[cache_key] = combined
            self._cache_time[cache_key] = datetime.now()

        return combined

    def get_status(self) -> Dict[str, Any]:
        return {
            'enabled': self.enabled,
            'cryptopanic': self.cryptopanic is not None,
            'twitter': self.twitter.enabled if self.twitter else False,
            'reddit': self.reddit.enabled if self.reddit else False,
            'finbert': self.finbert._loaded if self.finbert else False,
            'praw_available': PRAW_AVAILABLE,
            'tweepy_available': TWEEPY_AVAILABLE,
            'finbert_available': FINBERT_AVAILABLE,
            'vader_available': VADER_AVAILABLE,
        }
