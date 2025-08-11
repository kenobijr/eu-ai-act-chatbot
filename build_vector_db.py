#!/usr/bin/env python3
"""
- build script to create and populate the vector database.
- run this once after scraping to prepare the database for the application.
"""
from src.vector_db import DB
from pathlib import Path


def build_database():
    """ one-time database building script """
    # check if raw data exists
    raw_data_dir = Path(__file__).parent / "data" / "raw"
    required_files = ["annexes.json", "articles.json", "definitions.json", "recitals.json"]
    for file in required_files:
        if not (raw_data_dir / file).exists():
            raise FileNotFoundError(f"Missing {file}. Please run scraper.py first!")
    # start building...
    print("=" * 50)
    print("Building vector database from raw data...")
    print("=" * 50)
    # create and populate database (read_only=False for writing)
    db = DB(read_only=False)
    print("\n" + "=" * 50)
    print("Database built successfully!")
    print("=" * 50)
    # print stats
    print("\nVerifying document counts:")
    for entity in db.entities:
        count = db.collection[entity].count()
        print(f"  {entity:12} : {count:3} documents")
    print("\nVector database ready for use at: data/chroma_db/")


if __name__ == "__main__":
    build_database()
