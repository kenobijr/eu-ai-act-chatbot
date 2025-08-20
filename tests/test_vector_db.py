from chromadb import PersistentClient, Collection
from chromadb.utils import embedding_functions
from src.config import DBConfig
from src.vector_db import DB
import pytest


def test_DB_build(db_cfg):
    db = DB.build_mode(config=db_cfg)
    print("\nVerifying document counts:")
    for entity in db.entities:
        count = db.collections[entity].count()
        print(f"  {entity:12} : {count:3} documents")
    print("\nVector database ready to use at: data/chroma_db/")
