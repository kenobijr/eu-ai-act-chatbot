#!/usr/bin/env python3
"""scrape EU AI Act data from official web explorer."""
from src.scraper import Scraper
from pathlib import Path
import sys

raw_dir = Path("data/raw")
expected = ["annexes.json", "articles.json", "definitions.json", "recitals.json", "metadata.json"]
if all((raw_dir / f).exists() for f in expected) and "--force" not in sys.argv:
    print(f"Data exists at already in {raw_dir}/. Use --force to rebuild.")
    sys.exit(0)

print("Scraping EU AI Act (~5-10 min)...")
scraper = Scraper()
scraper.execute_scraping()
scraper.save_to_disc()
print("Done. Run build_vector_db.py next.")
