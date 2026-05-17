"""
News Crawler for Nashik Kumbh Mela 2027.
Crawls Marathi, Hindi, Gujarati, and English news sources.

Dependencies:
    pip install newspaper3k requests beautifulsoup4 langdetect lxml

Usage:
    python news_crawler.py
    python news_crawler.py --sources lokmat maha_times --max-articles 20
"""

import argparse
import hashlib
import json
import logging
import time
import urllib.robotparser
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from langdetect import detect, LangDetectException

try:
    from newspaper import Article, Config as NewspaperConfig
    NEWSPAPER_AVAILABLE = True
except ImportError:
    NEWSPAPER_AVAILABLE = False

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "knowledge_base" / "raw" / "news"

SOURCES = {
    "lokmat": {
        "name": "Lokmat", "language": "mr",
        "base_url": "https://www.lokmat.com",
        "search_url": "https://www.lokmat.com/search/?q={query}",
        "search_query": "कुंभमेळा नाशिक 2027",
        "selectors": {"article_links": ".article-title a, h2 a, h3 a", "title": "h1", "body": ".article-body p", "date": "time"},
    },
    "dainik_bhaskar": {
        "name": "Dainik Bhaskar", "language": "hi",
        "base_url": "https://www.bhaskar.com",
        "search_url": "https://www.bhaskar.com/search/?q={query}",
        "search_query": "नाशिक कुंभ मेला 2027",
        "selectors": {"article_links": "a[href*='/news/'], .story-title a", "title": "h1", "body": ".f-detail-body p", "date": ".f-detail-date"},
    },
    "maha_times": {
        "name": "Maharashtra Times", "language": "mr",
        "base_url": "https://maharashtratimes.com",
        "search_url": "https://maharashtratimes.com/searchresult.cms?query={query}",
        "search_query": "नाशिक कुंभमेळा",
        "selectors": {"article_links": "a[href*='/nashik/'], a[href*='/articleshow/'], h3 a", "title": "h1", "body": ".article_content p", "date": ".time_cptn"},
    },
    "navbharat_times": {
        "name": "Navbharat Times", "language": "hi",
        "base_url": "https://navbharattimes.indiatimes.com",
        "search_url": "https://navbharattimes.indiatimes.com/searchresult.cms?query={query}",
        "search_query": "नाशिक कुंभ",
        "selectors": {"article_links": "a[href*='/articleshow/'], h3 a", "title": "h1", "body": "article p", "date": "time"},
    },
    "the_hindu": {
        "name": "The Hindu", "language": "en",
        "base_url": "https://www.thehindu.com",
        "search_url": "https://www.thehindu.com/search/?q={query}&type=article",
        "search_query": "Nashik Kumbh Mela 2027",
        "selectors": {"article_links": ".story-card-news a, h3 a", "title": "h1", "body": ".article-body-content p", "date": "time[datetime]"},
    },
    "times_of_india": {
        "name": "Times of India", "language": "en",
        "base_url": "https://timesofindia.indiatimes.com",
        "search_url": "https://timesofindia.indiatimes.com/topic/nashik-kumbh-mela-2027",
        "search_query": "Nashik Kumbh Mela 2027",
        "selectors": {"article_links": "a[href*='/articleshow/'], figure a", "title": "h1", "body": ".artText p", "date": "time"},
    },
}

KUMBH_KEYWORDS = [
    "kumbh", "nashik", "simhastha", "godavari", "trimbakeshwar", "pilgrimage", "akhara", "mela", "2027",
    "कुंभ", "नाशिक", "सिंहस्थ", "गोदावरी", "त्र्यंबकेश्वर", "अखाड़ा", "शाही स्नान",
    "कुंभमेळा", "આখاड़ा", "કુંભ", "নাশিক",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("news_crawler")

UA = "KumbhMela2027-AIAssistant/1.0 (Educational; contact: kumbh2027@example.com)"


def url_hash(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:12]


def is_relevant(text: str) -> bool:
    tl = text.lower()
    return any(kw.lower() in tl for kw in KUMBH_KEYWORDS)


def detect_language(text: str) -> str:
    try:
        return detect(text[:500])
    except LangDetectException:
        return "unknown"


def check_robots(url: str, ua: str) -> bool:
    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(f"{base}/robots.txt")
    try:
        rp.read()
    except Exception:
        return True
    return rp.can_fetch(ua, url)


def extract_article(url: str, source_lang: str, selectors: dict, session: requests.Session) -> Optional[dict]:
    # Try newspaper3k first
    if NEWSPAPER_AVAILABLE:
        try:
            cfg = NewspaperConfig()
            cfg.fetch_images = False
            cfg.request_timeout = 15
            cfg.browser_user_agent = UA
            art = Article(url, config=cfg, language=source_lang)
            art.download()
            art.parse()
            if art.text and len(art.text) > 100:
                return {"title": art.title, "content": art.text, "publish_date": str(art.publish_date or ""), "authors": art.authors}
        except Exception:
            pass

    # BS4 fallback
    try:
        resp = session.get(url, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        title = ""
        for sel in selectors.get("title", "h1").split(","):
            el = soup.select_one(sel.strip())
            if el:
                title = el.get_text(strip=True)
                break

        body_parts = []
        for sel in selectors.get("body", "article p").split(","):
            els = soup.select(sel.strip())
            if els:
                body_parts = [e.get_text(strip=True) for e in els if e.get_text(strip=True)]
                if body_parts:
                    break
        content = "\n".join(body_parts)

        pub_date = ""
        for sel in selectors.get("date", "time").split(","):
            el = soup.select_one(sel.strip())
            if el:
                pub_date = el.get("datetime", "") or el.get_text(strip=True)
                break

        return {"title": title, "content": content, "publish_date": pub_date, "authors": []}
    except Exception as exc:
        log.warning("BS4 extract failed for %s: %s", url, exc)
        return None


def collect_urls(session: requests.Session, source_key: str, source: dict, max_urls: int) -> list[str]:
    query = source["search_query"]
    search_url = source["search_url"].format(query=requests.utils.quote(query))

    if not check_robots(search_url, UA):
        log.warning("[%s] robots.txt disallows: %s", source_key, search_url)
        return []

    try:
        resp = session.get(search_url, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        urls = []
        for a in soup.select(source["selectors"].get("article_links", "a")):
            href = a.get("href", "")
            if not href:
                continue
            full = urljoin(source["base_url"], href)
            if source["base_url"].split("//")[-1].split("/")[0] in urlparse(full).netloc:
                if full not in urls:
                    urls.append(full)
            if len(urls) >= max_urls:
                break
        log.info("[%s] Found %d candidate URLs", source_key, len(urls))
        return urls
    except Exception as exc:
        log.error("[%s] Search page failed: %s", source_key, exc)
        return []


def run(target_sources=None, max_articles_per_source=30, delay=2.0):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sources = {k: v for k, v in SOURCES.items() if target_sources is None or k in target_sources}

    session = requests.Session()
    session.headers["User-Agent"] = UA
    summary: dict[str, int] = {}

    for source_key, source in sources.items():
        log.info("=== Source: %s (%s) ===", source["name"], source["language"])
        urls = collect_urls(session, source_key, source, max_articles_per_source * 2)
        time.sleep(delay)

        saved = 0
        for url in urls:
            if saved >= max_articles_per_source:
                break

            out_path = OUTPUT_DIR / f"{source_key}_{url_hash(url)}.json"
            if out_path.exists():
                saved += 1
                continue

            if not check_robots(url, UA):
                continue

            data = extract_article(url, source["language"], source["selectors"], session)
            if not data or len(data.get("content", "")) < 100:
                time.sleep(delay)
                continue

            if not is_relevant(data.get("title", "") + " " + data.get("content", "")):
                time.sleep(delay)
                continue

            record = {
                **data,
                "url": url, "source": source["name"], "source_key": source_key,
                "language": source["language"],
                "detected_language": detect_language(data.get("content", "")),
                "crawled_at": datetime.utcnow().isoformat() + "Z",
            }
            out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            log.info("[%s] Saved: %s (%d chars)", source_key, out_path.name, len(record["content"]))
            saved += 1
            time.sleep(delay)

        summary[source_key] = saved
        log.info("[%s] Total saved: %d", source_key, saved)

    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sources", nargs="+", choices=list(SOURCES.keys()))
    p.add_argument("--max-articles", type=int, default=30)
    p.add_argument("--delay", type=float, default=2.0)
    args = p.parse_args()
    run(target_sources=args.sources, max_articles_per_source=args.max_articles, delay=args.delay)


if __name__ == "__main__":
    main()
