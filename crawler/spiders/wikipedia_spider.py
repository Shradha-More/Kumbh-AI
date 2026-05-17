"""
Wikipedia Spider for Nashik Kumbh Mela 2027 AI Assistant.
Fetches articles in 8 Indian languages and saves them as structured JSON files.

Dependencies:
    pip install requests wikipedia-api beautifulsoup4

Usage:
    python wikipedia_spider.py
    python wikipedia_spider.py --lang en hi --dry-run
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional

import requests
import wikipediaapi
from bs4 import BeautifulSoup

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "knowledge_base" / "raw" / "wikipedia"

LANGUAGES = {
    "en": "English", "hi": "Hindi", "mr": "Marathi", "gu": "Gujarati",
    "ta": "Tamil",   "te": "Telugu", "kn": "Kannada", "ml": "Malayalam",
}

TOPICS = {
    "nashik_kumbh_mela":  {"en": "Nashik-Trimbakeshwar Simhastha", "hi": "नाशिक कुंभ मेला", "mr": "नाशिक कुंभमेळा", "gu": "નાસિક કુંભ મેળો", "ta": "நாசிக் கும்பமேளா", "te": "నాసిక్ కుంభమేళా", "kn": "ನಾಸಿಕ್ ಕುಂಭಮೇಳ", "ml": "നാസിക് കുംഭമേള"},
    "simhastha":          {"en": "Simhastha", "hi": "सिंहस्थ", "mr": "सिंहस्थ", "gu": "સિંહસ્થ", "ta": "சிம்ஹஸ்த", "te": "సింహస్థ", "kn": "ಸಿಂಹಸ್ಥ", "ml": "സിംഹസ്ഥ"},
    "kumbh_mela":         {"en": "Kumbh Mela", "hi": "कुम्भ मेला", "mr": "कुंभ मेळा", "gu": "કુંભ મેળો", "ta": "கும்பமேளா", "te": "కుంభమేళా", "kn": "ಕುಂಭಮೇಳ", "ml": "കുംഭമേള"},
    "godavari_river":     {"en": "Godavari River", "hi": "गोदावरी नदी", "mr": "गोदावरी नदी", "gu": "ગોદાવરી નદી", "ta": "கோதாவரி நதி", "te": "గోదావరి నది", "kn": "ಗೋದಾವರಿ ನದಿ", "ml": "ഗോദാവരി നദി"},
    "ramkund":            {"en": "Ramkund", "hi": "रामकुंड", "mr": "रामकुंड", "gu": "રામકુંડ", "ta": "ராம்குண்ட்", "te": "రామకుండ", "kn": "ರಾಮಕುಂಡ", "ml": "രാംകുണ്ഡ്"},
    "trimbakeshwar":      {"en": "Trimbakeshwar Shiva Temple", "hi": "त्र्यंबकेश्वर", "mr": "त्र्यंबकेश्वर", "gu": "ત્ર્યંબકેશ્વર", "ta": "திரிம்பகேஸ்வர்", "te": "త్రింబకేశ్వర", "kn": "ತ್ರಿಂಬಕೇಶ್ವರ", "ml": "ത്രിംബകേശ്വർ"},
    "kalaram_temple":     {"en": "Kalaram Temple", "hi": "कालाराम मंदिर", "mr": "कालाराम मंदिर", "gu": "કાળારામ મંદિર", "ta": "காளாராம் கோயில்", "te": "కాళారామ మందిరం", "kn": "ಕಾಳಾರಾಮ ದೇವಸ್ಥಾನ", "ml": "കാളാറാം ക്ഷേത്രം"},
    "saptashrungi":       {"en": "Saptashringi", "hi": "सप्तश्रृंगी", "mr": "सप्तशृंगी", "gu": "સપ્તશ્રૃંગી", "ta": "சப்தஸ்ரிங்கி", "te": "సప్తశృంగి", "kn": "ಸಪ್ತಶೃಂಗಿ", "ml": "സപ്തശ്രൃംഗി"},
    "panchavati":         {"en": "Panchavati, Nashik", "hi": "पंचवटी, नाशिक", "mr": "पंचवटी, नाशिक", "gu": "પંચવટી, નાસિક", "ta": "பஞ்சவடி, நாசிக்", "te": "పంచవటి, నాసిక్", "kn": "ಪಂಚವಟಿ, ನಾಸಿಕ್", "ml": "പഞ്ചവടി, നാസിക്"},
    "nashik_city":        {"en": "Nashik", "hi": "नासिक", "mr": "नाशिक", "gu": "નાસિક", "ta": "நாசிக்", "te": "నాసిక్", "kn": "ನಾಸಿಕ್", "ml": "നാസിക്"},
    "juna_akhara":        {"en": "Juna Akhara", "hi": "जूना अखाड़ा", "mr": "जुना आखाडा", "gu": "જૂના અખાડા", "ta": "ஜூனா அகாரா", "te": "జూనా అఖారా", "kn": "ಜೂನಾ ಅಖಾರಾ", "ml": "ജൂനാ അഖാര"},
    "niranjani_akhara":   {"en": "Niranjani Akhara", "hi": "निरंजनी अखाड़ा", "mr": "निरंजनी आखाडा", "gu": "નિરંજની અખાડા", "ta": "நிரஞ்சனி அகாரா", "te": "నిరంజని అఖారా", "kn": "ನಿರಂಜನಿ ಅಖಾರಾ", "ml": "നിരഞ്ജനി അഖാര"},
    "mahanirvani_akhara": {"en": "Mahanirvani Akhara", "hi": "महानिर्वाणी अखाड़ा", "mr": "महानिर्वाणी आखाडा", "gu": "મહાનિર્વાણી અખાડા", "ta": "மஹாநிர்வாணி அகாரா", "te": "మహానిర్వాణి అఖారా", "kn": "ಮಹಾನಿರ್ವಾಣಿ ಅಖಾರಾ", "ml": "മഹാനിർവാണി അഖാര"},
    "shahi_snan":         {"en": "Shahi Snan", "hi": "शाही स्नान", "mr": "शाही स्नान", "gu": "શાહી સ્નાન", "ta": "ஷாஹி ஸ்நான்", "te": "శాహి స్నాన్", "kn": "ಶಾಹಿ ಸ್ನಾನ", "ml": "ഷാഹി സ്നാൻ"},
    "hindu_pilgrimage":   {"en": "Hindu pilgrimage", "hi": "हिंदू तीर्थयात्रा", "mr": "हिंदू तीर्थयात्रा", "gu": "હિન્દુ તીર્થ", "ta": "இந்து யாத்திரை", "te": "హిందూ తీర్థయాత్ర", "kn": "ಹಿಂದೂ ತೀರ್ಥಯಾತ್ರೆ", "ml": "ഹിന്ദു തീർഥാടനം"},
    "pandav_leni":        {"en": "Pandav Leni Caves", "hi": "पांडव लेणी", "mr": "पांडव लेणी", "gu": "પાંડવ લેણી", "ta": "பாண்டவ் லேணி", "te": "పాండవ్ లేణి", "kn": "ಪಾಂಡವ ಲೇಣಿ", "ml": "പാണ്ഡവ് ലേണി"},
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("wikipedia_spider")

UA = "KumbhMela2027-AIAssistant/1.0 (Educational; contact: kumbh2027@example.com)"


def sanitize_filename(text: str) -> str:
    keep = set("abcdefghijklmnopqrstuvwxyz0123456789_-")
    return "".join(c if c in keep else "_" for c in text.lower())


def fetch_via_api(wiki, title: str, lang: str, topic_key: str) -> Optional[dict]:
    try:
        page = wiki.page(title)
        if not page.exists():
            return None
        return {
            "title": page.title, "content": page.text, "summary": page.summary,
            "language": lang, "source": "wikipedia", "url": page.fullurl,
            "domain": f"{lang}.wikipedia.org", "topic_key": topic_key,
        }
    except Exception as exc:
        log.error("[%s/%s] wikipedia-api: %s", lang, topic_key, exc)
        return None


def fetch_via_requests(title: str, lang: str, topic_key: str, session: requests.Session) -> Optional[dict]:
    try:
        resp = session.get(
            f"https://{lang}.wikipedia.org/w/api.php",
            params={"action": "parse", "page": title, "prop": "text", "format": "json", "redirects": 1},
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            return None
        html = data["parse"]["text"]["*"]
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup.select(".navbox, .reflist, .mw-editsection, .noprint, table.infobox"):
            tag.decompose()
        content = soup.get_text(separator="\n", strip=True)
        return {
            "title": title, "content": content, "summary": content[:500],
            "language": lang, "source": "wikipedia",
            "url": f"https://{lang}.wikipedia.org/wiki/{title.replace(' ', '_')}",
            "domain": f"{lang}.wikipedia.org", "topic_key": topic_key,
        }
    except Exception as exc:
        log.error("[%s/%s] requests fallback: %s", lang, topic_key, exc)
        return None


def crawl(target_langs=None, target_topics=None, dry_run=False, delay=1.0):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    langs = target_langs or list(LANGUAGES.keys())
    topics = {k: v for k, v in TOPICS.items() if target_topics is None or k in target_topics}

    session = requests.Session()
    session.headers["User-Agent"] = UA

    wiki_clients = {
        lang: wikipediaapi.Wikipedia(language=lang, user_agent=UA)
        for lang in langs
    }

    fetched, skipped, failed = 0, 0, 0
    total = len(langs) * len(topics)
    done = 0

    for topic_key, lang_titles in topics.items():
        for lang in langs:
            done += 1
            title = lang_titles.get(lang)
            if not title:
                skipped += 1
                continue

            out_path = OUTPUT_DIR / f"{sanitize_filename(topic_key)}_{lang}.json"
            if out_path.exists():
                log.info("[%d/%d] Already exists: %s", done, total, out_path.name)
                skipped += 1
                continue

            if dry_run:
                log.info("[DRY-RUN] Would fetch %s/%s: '%s'", lang, topic_key, title)
                continue

            log.info("[%d/%d] Fetching %s/%s: '%s'", done, total, lang, topic_key, title)

            article = fetch_via_api(wiki_clients[lang], title, lang, topic_key)
            if not article or not article.get("content", "").strip():
                article = fetch_via_requests(title, lang, topic_key, session)

            if article and article.get("content", "").strip():
                out_path.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
                log.info("  Saved %s (%d chars)", out_path.name, len(article["content"]))
                fetched += 1
            else:
                log.warning("  No content for %s/%s", lang, topic_key)
                failed += 1

            time.sleep(delay)

    log.info("DONE — Fetched: %d | Skipped: %d | Failed: %d | Total: %d", fetched, skipped, failed, total)
    return {"fetched": fetched, "skipped": skipped, "failed": failed, "total": total}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lang", nargs="+", choices=list(LANGUAGES.keys()))
    p.add_argument("--topic", nargs="+", choices=list(TOPICS.keys()))
    p.add_argument("--delay", type=float, default=1.0)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    crawl(target_langs=args.lang, target_topics=args.topic, dry_run=args.dry_run, delay=args.delay)


if __name__ == "__main__":
    main()
