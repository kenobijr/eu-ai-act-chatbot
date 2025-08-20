"""
- build script to create and populate the vector database
- run this once after scraping to prepare the database for the application
"""


from src.vector_db import DB


def build_database():
    # start building...
    print("=" * 50)
    print("Building vector database from raw data...")
    print("=" * 50)
    # create and populate database using build_mode cls method
    db = DB.build_mode()
    print("\n" + "=" * 50)
    print("Database built successfully!")
    print("=" * 50)
    # print stats
    print("\nVerifying document counts:")
    for entity in db.entities:
        count = db.collections[entity].count()
        print(f"  {entity:12} : {count:3} documents")
    print("\nVector database ready to use at: data/chroma_db/")


if __name__ == "__main__":
    build_database()
