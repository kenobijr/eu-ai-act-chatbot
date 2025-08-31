"""
- db can be accessed in read or build mode via classmethods
- db is build / accessed with settings saved in DBConfig
- related scripts:
    - build_vector_db.py to execute db build
    - upload_vector_db.py to upload build db to hf datasets
"""


from chromadb import PersistentClient, Collection
from chromadb.utils import embedding_functions
from src.config import DBConfig
import json
from typing import Dict


class DB:

    @classmethod
    def build_mode(cls, config=None):
        if config is not None:
            assert isinstance(config, DBConfig), "Invalid DBConfig file"
        return cls(config=config, read_only=False)

    @classmethod
    def read_mode(cls, config=None):
        if config is not None:
            assert isinstance(config, DBConfig), "Invalid DBConfig file"
        return cls(config=config, read_only=True)

    def __init__(self, config: DBConfig = None, read_only: bool = True):
        # ----- shared code needed in read and build mode
        self.config = config if config is not None else DBConfig()
        self.entities = self.config.entities
        # connects to or creates persistent chroma_db instance
        self.db_client = self._db_create_or_connect()
        # embedding_function via chroma_db huggingface wrapper -> attached at collection level
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.config.embedding_function,
            device=self.config.embedding_device,
        )
        # ----- specific code depending on chosen mode
        if read_only:
            self.collections = self._get_collections()
        else:
            self._build_db()

    def _db_create_or_connect(self) -> PersistentClient:
        """ connects to persistent db client or creates new one """
        save_dir = self.config.save_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        return PersistentClient(path=save_dir)

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
        """ steering all methods to build db and saving build states """
        # create raw collections for each entity
        self.collections = self._create_collections()
        # load .json source files for each entity into ram
        self.raw_data = self._load_raw_data()
        # save raw data into collections for each entity
        for entity in self.entities:
            self._populate_collection(entity)

    def _create_collections(self) -> Dict[str, Collection]:
        """
        - create collection for each entity within db client if they do not exist already
        - needs specified embedding function (otherwise chroma_db dafaults are used)
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
        """ load jsons at specified path at each entity and save as dict at object """
        raw_data_dict = {}
        data_dir = self.config.data_dir
        for entity in self.entities:
            file_path = data_dir / f"{entity}{self.config.file_extension}"
            if not file_path.exists():
                raise FileNotFoundError(f"Loading raw data failed. Missing file at {file_path}.")
            with open(file_path, mode="r", encoding="utf-8") as f:
                raw_data_dict[entity] = json.load(f)
                print(f"Loaded {len(raw_data_dict[entity])} {entity}")
        return raw_data_dict

    def _populate_collection(self, entity: str) -> None:
        """ generic population method for all entity types """
        ids, docs, meta = [], [], []
        data = self.raw_data[entity]
        for item_id, item_data in data.items():
            ids.append(item_id)
            # build document with entity-specific formatting
            if entity == "definitions":
                doc = f"Definition: {item_data['title']}\n{item_data['text_content']}"
                meta_item = {"type": "definition", "term": item_data["title"]}
            elif entity == "recitals":
                doc = f"Recital {item_id.split('_')[1]}\n{item_data['text_content']}"
                meta_item = {"type": "recital"}
            elif entity == "annexes":
                doc = f"{item_data['title']}\n{item_data['text_content']}"
                meta_item = {"type": "annex", "term": item_data["title"]}
            else:  # articles
                sect_title = item_data.get("section_title", "")
                if sect_title is None:
                    sect_title = ""
                doc = f"""{item_data['title']}
            Part of {item_data['chapter_title']} {sect_title}
            Date of entry into force: {item_data['entry_date']}
            {item_data['text_content']}"""
                meta_item = {
                    "type": "article",
                    "term": item_data["title"],
                    "chapter": item_data["chapter_title"],
                    "entry_date": item_data["entry_date"],
                    "related_articles": json.dumps(item_data.get("related_article_ids", [])),
                    "related_recitals": json.dumps(item_data.get("related_recital_ids", [])),
                    "related_annexes": json.dumps(item_data.get("related_annex_ids", []))
                }
            docs.append(doc)
            meta.append(meta_item)
        self.collections[entity].upsert(ids=ids, documents=docs, metadatas=meta)
        print(f"Added {len(data)} to {entity} collection.")
