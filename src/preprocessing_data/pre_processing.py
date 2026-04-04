"""
src/preprocessing_data/pre-processing.py
==========================================
YOUR FILE — INGESTION MODULE (Part 1 of 3)

WHAT THIS FILE CONTAINS:
  Class 1 → DataDownloader  (downloads HotpotQA + Wikipedia)
  Class 2 → Preprocessor    (cleans text, unified schema)

WHERE THIS FITS IN THE PROJECT STRUCTURE:
  preprocessing_data/ = "Scripts to download or generate data and pre-process"
  This is exactly what DataDownloader and Preprocessor do.

HOW OTHER FILES USE THIS:
  from src.preprocessing_data.pre-processing import DataDownloader, Preprocessor

VIVA EXPLANATION:
  "pre-processing.py handles the first two stages of my ingestion pipeline.
   DataDownloader fetches raw data from the internet into src/data/raw/.
   Preprocessor cleans that raw data and converts both datasets into one
   unified schema, saving results to src/data/processed/."
"""

import os
import re
import json
import time
import urllib.request
import urllib.parse

# Import settings from central config
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    RAW_DIR, PROCESSED_DIR,
    HOTPOTQA_FILE, WIKIPEDIA_FILE,
    MAX_HOTPOT_RECORDS, MAX_WIKI_ARTICLES,
)

from tqdm import tqdm


# =============================================================================
# CLASS 1: DataDownloader
# =============================================================================

class DataDownloader:
    """
    Downloads raw datasets to src/data/raw/.

    SINGLE RESPONSIBILITY: Only downloads. Nothing else.

    Usage:
        downloader = DataDownloader()
        paths = downloader.download_all()
    """

    def __init__(self):
        os.makedirs(RAW_DIR,       exist_ok=True)
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        print("📁 DataDownloader ready")

    # ── PUBLIC METHODS ────────────────────────────────────────────────────────

    def download_all(self) -> dict:
        """
        Downloads both datasets.
        Returns dict: {"hotpotqa": path, "wikipedia": path}
        """
        print("\n" + "=" * 55)
        print("  DataDownloader: Fetching datasets → src/data/raw/")
        print("=" * 55)
        return {
            "hotpotqa":  self.download_hotpotqa(),
            "wikipedia": self.download_wikipedia(),
        }

    def download_hotpotqa(self) -> str:
        """Downloads HotpotQA training set from CMU servers."""
        if self._exists(HOTPOTQA_FILE, min_mb=100):
            return HOTPOTQA_FILE

        print("\n📥 Downloading HotpotQA (~540 MB)...")
        url = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json"
        try:
            urllib.request.urlretrieve(url, HOTPOTQA_FILE, self._progress)
            mb = os.path.getsize(HOTPOTQA_FILE) / 1024 / 1024
            print(f"\n   ✅ Saved ({mb:.0f} MB) → src/data/raw/hotpot_train_v1.1.json")
            return HOTPOTQA_FILE
        except Exception as e:
            print(f"\n   ❌ Failed: {e}")
            return None

    def download_wikipedia(self) -> str:
        """
        Downloads Wikipedia articles using 3 fallback methods.
        Method 1: HuggingFace datasets library
        Method 2: HuggingFace Hub API
        Method 3: Wikipedia REST API (always works)
        """
        if self._exists(WIKIPEDIA_FILE, min_lines=50):
            return WIKIPEDIA_FILE

        if os.path.exists(WIKIPEDIA_FILE):
            os.remove(WIKIPEDIA_FILE)

        print(f"\n📥 Downloading Wikipedia ({MAX_WIKI_ARTICLES} articles)...")

        for name, fn in [
            ("HuggingFace streaming",  self._hf_stream),
            ("HuggingFace Hub API",    self._hf_api),
            ("Wikipedia REST API",     self._wiki_rest),
        ]:
            print(f"   Trying {name}...")
            try:
                if fn(WIKIPEDIA_FILE):
                    n = self._count_lines(WIKIPEDIA_FILE)
                    print(f"   ✅ Saved {n} articles → src/data/raw/wikipedia_sample.jsonl")
                    return WIKIPEDIA_FILE
            except Exception as e:
                print(f"   ❌ {name} failed: {e}")

        print("   ❌ All methods failed")
        return None

    # ── PRIVATE HELPERS ───────────────────────────────────────────────────────

    def _exists(self, path, min_mb=0, min_lines=0) -> bool:
        if not os.path.exists(path):
            return False
        if min_mb and os.path.getsize(path) / 1024 / 1024 < min_mb:
            return False
        if min_lines and self._count_lines(path) < min_lines:
            return False
        size = os.path.getsize(path) / 1024 / 1024
        print(f"\n✅ Already exists ({size:.1f} MB) — skipping: {os.path.basename(path)}")
        return True

    def _count_lines(self, path) -> int:
        with open(path, encoding="utf-8") as f:
            return sum(1 for l in f if l.strip())

    def _progress(self, b, bs, ts):
        if ts > 0:
            print(f"\r   {min(b*bs/ts*100,100):.1f}% — {b*bs/1024/1024:.0f} MB",
                  end="", flush=True)

    def _write_wiki(self, f, rid, title, text, url="") -> bool:
        if not text or len(text.strip()) < 100:
            return False
        f.write(json.dumps({
            "id": str(rid), "title": title,
            "text": text.strip()[:5000], "url": url
        }, ensure_ascii=False) + "\n")
        return True

    def _hf_stream(self, path) -> bool:
        from datasets import load_dataset
        wiki  = load_dataset("wikimedia/wikipedia", "20231101.en",
                             split="train", streaming=True, trust_remote_code=True)
        saved = 0
        with open(path, "w", encoding="utf-8") as f:
            for a in wiki:
                if saved >= MAX_WIKI_ARTICLES:
                    break
                if self._write_wiki(f, a.get("id", saved),
                                    a.get("title",""), a.get("text",""),
                                    a.get("url","")):
                    saved += 1
        return saved >= MAX_WIKI_ARTICLES * 0.8

    def _hf_api(self, path) -> bool:
        base = ("https://datasets-server.huggingface.co/rows"
                "?dataset=wikimedia%2Fwikipedia&config=20231101.en"
                "&split=train&offset={o}&length=100")
        saved, offset = 0, 0
        with open(path, "w", encoding="utf-8") as f:
            while saved < MAX_WIKI_ARTICLES:
                req = urllib.request.Request(base.format(o=offset),
                      headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=30) as r:
                    rows = json.loads(r.read())["rows"]
                if not rows:
                    break
                for row in rows:
                    if saved >= MAX_WIKI_ARTICLES:
                        break
                    d = row.get("row", {})
                    if self._write_wiki(f, d.get("id", saved),
                                        d.get("title",""), d.get("text",""),
                                        d.get("url","")):
                        saved += 1
                offset += 100
                time.sleep(0.3)
        return saved >= MAX_WIKI_ARTICLES * 0.8

    def _wiki_rest(self, path) -> bool:
        titles = self._titles()[:MAX_WIKI_ARTICLES]
        api    = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"
        saved  = 0
        with open(path, "w", encoding="utf-8") as f:
            for i, t in enumerate(titles):
                try:
                    enc = urllib.parse.quote(t.replace(" ", "_"))
                    req = urllib.request.Request(api.format(enc),
                          headers={"User-Agent": "RAG-Project/1.0"})
                    with urllib.request.urlopen(req, timeout=10) as r:
                        d = json.loads(r.read())
                    if self._write_wiki(f, d.get("pageid", i),
                                        d.get("title", t),
                                        d.get("extract", ""),
                                        d.get("content_urls",{})
                                          .get("desktop",{}).get("page","")):
                        saved += 1
                    if saved % 25 == 0:
                        print(f"\r   {saved}/{len(titles)}...", end="", flush=True)
                    time.sleep(0.1)
                except Exception:
                    continue
        return saved > 50

    def _titles(self) -> list:
        return [
            "Albert Einstein","Isaac Newton","Marie Curie","Charles Darwin",
            "Nikola Tesla","Leonardo da Vinci","Stephen Hawking","Alan Turing",
            "Abraham Lincoln","Napoleon Bonaparte","Queen Victoria","Winston Churchill",
            "Mahatma Gandhi","Nelson Mandela","Thomas Edison","Benjamin Franklin",
            "George Washington","William Shakespeare","Charles Dickens","Jane Austen",
            "Sigmund Freud","Karl Marx","Plato","Aristotle","Socrates",
            "Theory of relativity","Quantum mechanics","DNA","Evolution","Big Bang",
            "Black hole","Photosynthesis","Periodic table","Electricity",
            "Nuclear fission","Internet","Artificial intelligence","Machine learning",
            "Computer","Telephone","Airplane","Automobile","Steam engine",
            "United States","United Kingdom","France","Germany","China","India",
            "Russia","Japan","Brazil","Australia","Canada","Italy","Spain",
            "New York City","London","Paris","Tokyo","Beijing","Mumbai",
            "World War I","World War II","Cold War","American Revolution",
            "French Revolution","Industrial Revolution","Renaissance",
            "Roman Empire","British Empire","Ancient Egypt","Ancient Greece",
            "The Godfather","Titanic (film)","Star Wars","The Dark Knight",
            "Pulp Fiction","Schindler's List","Forrest Gump","The Matrix",
            "FIFA World Cup","Olympic Games","NBA","NFL","Cricket","Tennis",
            "Usain Bolt","Michael Jordan","Roger Federer","Lionel Messi",
            "United Nations","NATO","European Union","World Health Organization",
            "Google","Apple Inc.","Microsoft","Amazon (company)","Tesla Inc.",
            "Harvard University","Oxford University","MIT","NASA","SpaceX","CERN",
            "The Beatles","Michael Jackson","Elvis Presley","Bob Dylan",
            "Solar System","Earth","Moon","Mars","Sun","Galaxy","Universe",
            "Climate change","Renewable energy","Nuclear power","Water","Oxygen",
            "Mathematics","Statistics","Geometry","Algebra","Calculus",
            "Computer science","Algorithm","Programming language","Database",
            "Elon Musk","Jeff Bezos","Bill Gates","Steve Jobs","Mark Zuckerberg",
            "Cancer","Heart","Brain","Immune system","Genetics","Virus","Bacteria",
            "COVID-19","Diabetes","Vaccine","Penicillin",
            "Christianity","Islam","Judaism","Hinduism","Buddhism","Philosophy",
            "Democracy","Capitalism","Communism","Stock market","Inflation",
            "Eiffel Tower","Great Wall of China","Taj Mahal","Colosseum",
            "Pyramids of Giza","Stonehenge","Machu Picchu","Statue of Liberty",
            "Amazon River","Nile","Everest","Sahara Desert","Himalayas",
            "Lion","Tiger","Elephant","Blue whale","Eagle","Dinosaur","Dolphin",
            "Romeo and Juliet","Hamlet","Pride and Prejudice","The Great Gatsby",
            "1984 (novel)","To Kill a Mockingbird","Mona Lisa","Starry Night",
            "Gravity","Speed of light","Thermodynamics","String theory","Dark matter",
            "Cryptocurrency","Bitcoin","3D printing","Virtual reality","Blockchain",
            "Pakistan","Bangladesh","Egypt","Nigeria","Kenya","Ethiopia","Iran",
            "Iraq","Saudi Arabia","Israel","Peru","Colombia","Chile","Cuba",
            "Philippines","Vietnam","Thailand","Malaysia","Indonesia","South Korea",
            "Netherlands","Belgium","Switzerland","Sweden","Norway","Poland","Greece",
            "Photon","Electron","Proton","Neutron","Magnetic field","Sound","Light",
        ]


# =============================================================================
# CLASS 2: Preprocessor
# =============================================================================

class Preprocessor:
    """
    Cleans raw data and converts to unified schema.

    SINGLE RESPONSIBILITY: Only cleans and standardises. Nothing else.

    UNIFIED SCHEMA (both datasets → this format):
      doc_id, title, text, source, question, answer,
      is_multihop, type, level, bridge_entities

    Usage:
        preprocessor = Preprocessor()
        records = preprocessor.process_all(hotpotqa_path, wikipedia_path)
    """

    def __init__(self):
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        print("🧹 Preprocessor ready")

    # ── PUBLIC METHODS ────────────────────────────────────────────────────────

    def process_all(self, hotpotqa_path: str,
                    wikipedia_path: str) -> list:
        """
        Processes both datasets → unified records list.
        Saves intermediate files to src/data/processed/.
        """
        print("\n" + "=" * 55)
        print("  Preprocessor: Cleaning → src/data/processed/")
        print("=" * 55)

        records = []

        if hotpotqa_path and os.path.exists(hotpotqa_path):
            hpqa = self.process_hotpotqa(hotpotqa_path)
            self._save(hpqa, "hotpotqa_processed.jsonl")
            records.extend(hpqa)

        if wikipedia_path and os.path.exists(wikipedia_path):
            wiki = self.process_wikipedia(wikipedia_path)
            self._save(wiki, "wikipedia_processed.jsonl")
            records.extend(wiki)

        print(f"\n  ✅ Total clean records: {len(records):,}")
        return records

    def process_hotpotqa(self, path: str) -> list:
        """
        Reads HotpotQA → unified schema.
        Only fuses SUPPORTING paragraphs (skips 8 distractor paragraphs).
        """
        print(f"\n  📖 Processing HotpotQA...")
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if MAX_HOTPOT_RECORDS:
            raw = raw[:MAX_HOTPOT_RECORDS]

        records, skipped = [], 0
        for entry in tqdm(raw, desc="  HotpotQA"):
            question  = self.clean(entry.get("question", ""))
            answer    = self.clean(entry.get("answer", ""))
            ctx_list  = entry.get("context", [])
            sup_facts = entry.get("supporting_facts", [])

            if not question or not ctx_list:
                skipped += 1
                continue

            para_map = {t: self.clean(" ".join(s)) for t, s in ctx_list}
            sup_titles = list(dict.fromkeys(t for t, _ in sup_facts)) \
                         or [t for t, _ in ctx_list]
            bridge = sup_titles[1:] if len(sup_titles) > 1 else []

            fused = " ".join(
                f"[{t}] {para_map[t]}"
                for t in sup_titles
                if t in para_map and len(para_map.get(t, "")) > 30
            )
            if len(fused) < 50:
                skipped += 1
                continue

            records.append({
                "doc_id":          entry.get("_id", ""),
                "title":           " | ".join(sup_titles),
                "text":            fused,
                "source":          "hotpotqa",
                "question":        question,
                "answer":          answer,
                "is_multihop":     True,
                "type":            entry.get("type", "bridge"),
                "level":           entry.get("level", "medium"),
                "bridge_entities": bridge,
            })

        print(f"  ✅ HotpotQA: {len(records):,} records | {skipped} skipped")
        return records

    def process_wikipedia(self, path: str) -> list:
        """Reads Wikipedia JSONL → unified schema with Wikipedia-specific cleaning."""
        print(f"\n  📖 Processing Wikipedia...")
        records, skipped = [], 0

        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in tqdm(lines, desc="  Wikipedia"):
            try:
                a = json.loads(line.strip())
            except Exception:
                skipped += 1
                continue

            text = self._clean_wiki(a.get("text", ""))
            if len(text) < 100:
                skipped += 1
                continue

            records.append({
                "doc_id":          f"wiki_{a.get('id','')}",
                "title":           self.clean(a.get("title", "")),
                "text":            text,
                "source":          "wikipedia",
                "question":        "",
                "answer":          "",
                "is_multihop":     False,
                "type":            "wiki_article",
                "level":           "",
                "bridge_entities": [],
            })

        print(f"  ✅ Wikipedia: {len(records):,} records | {skipped} skipped")
        return records

    # ── CLEANING UTILITIES ────────────────────────────────────────────────────

    def clean(self, text: str) -> str:
        """General text cleaner — removes extra whitespace and invisible chars."""
        if not text or not isinstance(text, str):
            return ""
        text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _clean_wiki(self, text: str) -> str:
        """Wikipedia-specific: removes headers, refs, template markup."""
        text = re.sub(r'==+[^=]+=+=*', '', text)   # == Section ==
        text = re.sub(r'\[\d+\]', '', text)          # [1] refs
        text = re.sub(r'thumb\|[^\n]*', '', text)    # image captions
        text = re.sub(r'\{\{[^\}]*\}\}', '', text)   # {{templates}}
        return self.clean(text)

    # ── PRIVATE HELPERS ───────────────────────────────────────────────────────

    def _save(self, records: list, filename: str):
        """Saves processed records to src/data/processed/ as JSONL."""
        path = os.path.join(PROCESSED_DIR, filename)
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  💾 Saved → src/data/processed/{filename}")


# =============================================================================
# STANDALONE RUN (for testing this file independently)
# =============================================================================

if __name__ == "__main__":
    print("Running pre-processing.py independently...")

    downloader   = DataDownloader()
    paths        = downloader.download_all()

    preprocessor = Preprocessor()
    records      = preprocessor.process_all(
        hotpotqa_path  = paths.get("hotpotqa"),
        wikipedia_path = paths.get("wikipedia"),
    )

    print(f"\n✅ Done. Total records: {len(records):,}")
    print("Next: run build_features.py")
