import chromadb
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from pathlib import Path


class DB:
    def __init__(self, entities: list):
        # entities for which to create collections within the local_db_client
        self.entities = entities
        # embedding_function via chroma_db huggingface wrapper
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
        )
        # creates persistent chroma_db instance to local repo & downloads related embedding model
        self.db_client = self._setup_db()
        # create collections for each entity
        self.collection = self._create_collections()

    def _setup_db(self) -> PersistentClient:
        # define save path for persistent DB & create dirs
        save_dir = Path(__file__).parent.parent / "data" / "chroma_db"
        save_dir.mkdir(parents=True, exist_ok=True)
        # create persistent db source files in local repo
        return chromadb.PersistentClient(path=save_dir)

    def _create_collections(self):
        collection_dict = {}
        # create collections if they do not exist already
        for entity in self.entities:
            # get_collection returns collection object if it exists
            try:
                collection = self.db_client.get_collection(
                    name=entity,
                    embedding_function=self.embedding_function
                )
            # if collection does not exist, ValueError is triggered -> create new!
            except ValueError:
                collection = self.db_client.create_collection(
                    name=entity,
                    embedding_function=self.embedding_function
                )
            # assign collection obj to dict - either new created or existing one
            collection_dict[entity] = collection
        return collection_dict


def main():
    db_entities = ["annexes", "articles", "definitions", "recitals"]
    db = DB(db_entities)
    db.collection["definitions"].add(
        ids=["definition_01", "definition_02", "definition_03"],
        documents=[
            "AI system means a machine-based system that is designed to operate with varying levels of autonomy and that may exhibit adaptiveness after deployment, and that, for explicit or implicit objectives, infers, from the input it receives, how to generate outputs such as predictions, content, recommendations, or decisions that can influence physical or virtual environments",
            "risk means the combination of the probability of an occurrence of harm and the severity of that harm",
            "provider means a natural or legal person, public authority, agency or other body that develops an AI system or a general-purpose AI model or that has an AI system or a general-purpose AI model developed and places it on the market or puts the AI system into service under its own name or trademark, whether for payment or free of charge"
        ],
        metadatas=[{"title": "AI system"}, {"title": "risk"}, {"title": "provider"}],
    )
    results = db.collection["definitions"].query(
        query_texts=["provider"],
        n_results=3,
    )
    print(results)


if __name__ == "__main__":
    main()
