import chromadb
from chromadb import PersistentClient, Collection
from chromadb.utils import embedding_functions
from pathlib import Path
import json
from typing import Dict


class DB:
    def __init__(self, entities: list):
        # for each entity sep collection (table) is created in db client
        self.entities = entities
        # embedding_function via chroma_db huggingface wrapper -> attached at collection level
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
        )
        # creates persistent chroma_db instance at local repo
        self.db_client = self._setup_db()
        # creates collections for each entity without populating it
        self.collection = self._create_collections()
        # load raw json data into mem and save as objects
        self.raw_data = self._load_raw_data()
        # populate db with raw data; run all sep _populate_... methods
        for entity in self.entities:
            getattr(self, f"_populate_{entity}")()

    def _setup_db(self) -> PersistentClient:
        """
        - creates persistent db client in local repo
        - if db client already exists, it simply connects to existing db
        """
        # define save path for persistent DB & create dirs
        save_dir = Path(__file__).parent.parent / "data" / "chroma_db"
        save_dir.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(path=save_dir)

    def _create_collections(self) -> Dict[str, Collection]:
        """
        - create collection for each entity within db client if they do not exist already
        - needs specified embedding function (otherwise chroma_db dafaults are used)
        - TBD: config
        """
        collection_dict = {}
        for entity in self.entities:
            collection = self.db_client.get_or_create_collection(
                name=entity,
                embedding_function=self.embedding_function,
                configuration={
                    "hnsw": {
                        "space": "cosine"
                    }
                },
            )
            collection_dict[entity] = collection
        return collection_dict

    def _load_raw_data(self) -> Dict[str, Dict]:
        """
        - load json at specified path at each entity and save as dict at object
        - processing of entity dicts done in sep methods due to distinct structures
        """
        input_dir = Path(__file__).parent.parent / "data" / "raw"
        raw_data_dict = {}
        for entity in self.entities:
            with open(f"{input_dir}/{entity}.json", mode="r", encoding="utf-8") as f:
                raw_data_dict[entity] = json.load(f)
                print(f"Loaded {len(raw_data_dict[entity])} {entity}")
        return raw_data_dict

    def _populate_definitions(self) -> None:
        """
        - add 68 definitions to collection; title as metadata; id as unique identifier
        - add context clue "EU AI Act Definition: ...." at beginning of each document
        """
        ids, docs, meta = [], [], []
        for def_id, def_data in self.raw_data["definitions"].items():
            ids.append(def_id)
            doc = f"Definition: {def_data["title"]}\n{def_data["text_content"]}"
            docs.append(doc)
            meta.append({
                "type": "definition",
                "term": def_data["title"],
            })
        # add processed entries with upsert: adds if new, updates if exists
        self.collection["definitions"].upsert(
            ids=ids,
            documents=docs,
            metadatas=meta,
        )
        print(f"Added {len(self.raw_data["definitions"].items())} to definition collection.")

    def _populate_recitals(self) -> None:
        """ add 180 recitals to collection; no metadata available for them """
        ids, docs, meta = [], [], []
        for rec_id, rec_data in self.raw_data["recitals"].items():
            ids.append(rec_id)
            doc = f"Recital {rec_id.split('_')[1]}\n{rec_data["text_content"]}"
            docs.append(doc)
            meta.append({
                "type": "recital",
            })
        # add processed entries with upsert: adds if new, updates if exists
        self.collection["recitals"].upsert(
            ids=ids,
            documents=docs,
            metadatas=meta,
        )
        print(f"Added {len(self.raw_data["recitals"].items())} to recital collection.")

    def _populate_annexes(self) -> None:
        """
        - add 13 annexes to collection; title as metadata; id as unique identifier
        """
        ids, docs, meta = [], [], []
        for annex_id, annex_data in self.raw_data["annexes"].items():
            ids.append(annex_id)
            doc = f"Annex: {annex_data["title"]}\n{annex_data["text_content"]}"
            docs.append(doc)
            meta.append({
                "type": "annex",
                "term": annex_data["title"],
            })
        # add processed entries with upsert: adds if new, updates if exists
        self.collection["annexes"].upsert(
            ids=ids,
            documents=docs,
            metadatas=meta,
        )
        print(f"Added {len(self.raw_data["annexes"].items())} to annex collection.")

    def _populate_articles(self) -> None:
        """
        - add 112 articles as "central hub"
        """
        ids, docs, meta = [], [], []
        for art_id, art_data in self.raw_data["articles"].items():
            ids.append(art_id)
            section_title = art_data["section_title"] if art_data["section_title"] is not None else ""
            doc = f"""Article: {art_data["title"]}
            Part of {art_data["chapter_title"]} {section_title}
            Date of entry into force: {art_data["entry_date"]}
            {art_data["text_content"]}"""
            docs.append(doc)
            meta.append({
                "type": "article",
                "term": art_data["title"],
                "chapter": art_data["chapter_title"],
                "entry_date": art_data["entry_date"],
                # convert lists to json strings - chromadb does not store lists
                "related_articles": json.dumps(art_data.get("related_article_ids", [])),
                "related_recitals": json.dumps(art_data.get("related_recital_ids", [])),
                "related_annexes": json.dumps(art_data.get("related_annex_ids", []))
            })
        # add processed entries with upsert: adds if new, updates if exists
        self.collection["articles"].upsert(
            ids=ids,
            documents=docs,
            metadatas=meta,
        )
        print(f"Added {len(self.raw_data["articles"].items())} to article collection.")



def main():
    db_entities = ["annexes", "articles", "definitions", "recitals"]
    db = DB(db_entities)
    # test definitions query -> "definition_03"
    results_def = db.collection["definitions"].query(
        query_texts=["provider"],
        n_results=1,
    )
    print(results_def)
    # test recitals query -> "recital_010"
    results_rec = db.collection["recitals"].query(
        query_texts=["The fundamental right to the protection of personal data is safeguarded in particular by Regulations (EU) 2016/679[11] and (EU) 2018/1725[12]"],
        n_results=1,
    )
    print(results_rec)
    # test annex query -> "annex_03"
    results_annex = db.collection["annexes"].query(
        query_texts=["High-risk AI systems pursuant toArticle 6(2) are the AI systems listed in any"],
        n_results=1,
    )
    print(results_annex)
    # test article query -> "annex_002"
    results_article = db.collection["articles"].query(
        query_texts=["1. This Regulation applies to:(a) providers placing on the market or putting"],
        n_results=1,
    )
    print(results_article)


if __name__ == "__main__":
    main()
