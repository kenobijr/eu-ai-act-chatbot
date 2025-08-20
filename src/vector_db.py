"""
- module can be accessed in read or build mode via classmethods
- the classmethods are wrappers which pass the read_only flag further
- db is build / accessed with settings saved in DBConfig
- related scripts:
    - build_vector_db.py to execute db build
    - upload_vector_db.py to upload build db to hf datasets
"""


import chromadb
from chromadb import PersistentClient, Collection
from chromadb.utils import embedding_functions
from src.config import DBConfig
import json
from typing import Dict


class DB:

    @classmethod
    def build_mode(cls):
        """ wrapper to build the db using source .json files  """
        return cls(config=None, read_only=False)

    @classmethod
    def read_mode(cls):
        """ wrapper to access db in read only mode"""
        return cls(config=None, read_only=True)

    def __init__(self, config: DBConfig = None, read_only: bool = True):
        # ----- shared code needed in read and build mode
        self.config = config if config is not None else DBConfig()
        # collection entities derived from config
        self.entities = self.config.entities
        # connects to or creates persistent chroma_db instance
        self.db_client = self._db_create_or_connect()
        # embedding_function via chroma_db huggingface wrapper -> attached at collection level
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.config.embedding_function,
            device=self.config.embedding_device,
        )
        # ----- specific code depending on chosen mode
        # read mode: connect to existing collections
        if read_only:
            self.collections = self._get_collections()
        # build mode: create and populate collections
        else:
            self._build_db()

    def _db_create_or_connect(self) -> PersistentClient:
        """
        - connects to persistent db client in local repo
        - creates db directory if it doesn't exist (for both read/write modes)
        - chromadb automatically handles existing vs new DB
        """
        # define save path for persistent DB & create dirs
        save_dir = self.config.save_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(path=save_dir)

    def _get_collections(self) -> Dict[str, Collection]:
        """ get existing collections for read-only access """
        collection_dict = {}
        for entity in self.entities:
            try:
                collection = self.db_client.get_collection(
                    name=entity,
                    embedding_function=self.embedding_function,
                )
                collection_dict[entity] = collection
            except ValueError:
                raise ValueError(f"Collection {entity} not found. Run build_vector_db.py first!")
        return collection_dict

    def _build_db(self) -> None:
        """
        - steering all methods to build db and saving build states
        - builds on top of db client which was created already at this point
        """
        # create raw collections for each entity
        self.collections = self._create_collections()
        # load .json source files for each entity into ram
        self.raw_data = self._load_raw_data()
        # save raw data into collections for each entity by executing all _populate_... methods
        for entity in self.entities:
            getattr(self, f"_populate_{entity}")()

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
                        "space": self.config.measurement_unit
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
        raw_data_dict = {}
        data_dir = self.config.data_dir
        for entity in self.entities:
            # check if file exists
            file_path = data_dir / f"{entity}{self.config.file_extension}"
            if not file_path.exists():
                raise FileNotFoundError(f"Loading raw data failed. Missing file at {file_path}.")
            with open(file_path, mode="r", encoding="utf-8") as f:
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
            doc = f"Definition: {def_data['title']}\n{def_data['text_content']}"
            docs.append(doc)
            meta.append({
                "type": "definition",
                "term": def_data["title"],
            })
        # add processed entries with upsert: adds if new, updates if exists
        self.collections["definitions"].upsert(
            ids=ids,
            documents=docs,
            metadatas=meta,
        )
        print(f"Added {len(self.raw_data['definitions'].items())} to definition collection.")

    def _populate_recitals(self) -> None:
        """ add 180 recitals to collection; no metadata available for them """
        ids, docs, meta = [], [], []
        for rec_id, rec_data in self.raw_data["recitals"].items():
            ids.append(rec_id)
            doc = f"Recital {rec_id.split('_')[1]}\n{rec_data['text_content']}"
            docs.append(doc)
            meta.append({
                "type": "recital",
            })
        # add processed entries with upsert: adds if new, updates if exists
        self.collections["recitals"].upsert(
            ids=ids,
            documents=docs,
            metadatas=meta,
        )
        print(f"Added {len(self.raw_data['recitals'].items())} to recital collection.")

    def _populate_annexes(self) -> None:
        """
        - add 13 annexes to collection; title as metadata; id as unique identifier
        """
        ids, docs, meta = [], [], []
        for annex_id, annex_data in self.raw_data["annexes"].items():
            ids.append(annex_id)
            doc = f"Annex: {annex_data['title']}\n{annex_data['text_content']}"
            docs.append(doc)
            meta.append({
                "type": "annex",
                "term": annex_data["title"],
            })
        # add processed entries with upsert: adds if new, updates if exists
        self.collections["annexes"].upsert(
            ids=ids,
            documents=docs,
            metadatas=meta,
        )
        print(f"Added {len(self.raw_data['annexes'].items())} to annex collection.")

    def _populate_articles(self) -> None:
        """
        - add 112 articles as "central hub"
        """
        ids, docs, meta = [], [], []
        for art_id, art_data in self.raw_data["articles"].items():
            ids.append(art_id)
            sect_title = art_data["section_title"] if art_data["section_title"] is not None else ""
            doc = f"""Article: {art_data["title"]}
            Part of {art_data["chapter_title"]} {sect_title}
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
        self.collections["articles"].upsert(
            ids=ids,
            documents=docs,
            metadatas=meta,
        )
        print(f"Added {len(self.raw_data['articles'].items())} to article collection.")
