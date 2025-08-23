""" scrape EU AI Act data from official web explorer """

from src.config import SCRAPEConfig
from src.scraper import Scraper
import sys


def start_scraping():
    cfg = SCRAPEConfig()

    # prevent unintended overwrite of exitsting .jsons from former scrape
    raw_dir = cfg.output_dir
    expected = [
        "annexes.json",
        "articles.json",
        "definitions.json",
        "recitals.json",
        "metadata.json"
    ]
    if all((raw_dir / f).exists() for f in expected) and "--force" not in sys.argv:
        print(f"Data exists at already in {raw_dir}/. Use --force to rebuild.")
        sys.exit(0)

    # starts scraping; last step in scraper.execute_scraping is saving to disc
    print("Scraping EU AI Act (~5-10 min)...")
    scraper = Scraper()
    scraper.execute_scraping()
    print("Done. Run build_vector_db.py next.")


if __name__ == "__main__":
    start_scraping()
