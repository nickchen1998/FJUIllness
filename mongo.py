import contextlib
from pymongo.mongo_client import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
from env_settings import EnvSettings
from langchain_openai import OpenAIEmbeddings
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
from uuid import uuid4
from langchain_core.documents import Document


@contextlib.contextmanager
def get_mongo_database() -> Database:
    env_settings = EnvSettings()
    client = MongoClient(host=env_settings.MONGODB_ATLAS_URI)
    try:
        yield Database(client, name=env_settings.MONGODB_DATABASE)
    finally:
        client.close()


def insert_datas(datas: list):
    env_settings = EnvSettings()

    with get_mongo_database() as database:
        vector_store = MongoDBAtlasVectorSearch(
            collection=Collection(database, name="illness"),
            embedding=OpenAIEmbeddings(model="text-embedding-3-small", api_key=env_settings.OPENAI_API_KEY),
            index_name="illness_refactor_question",
            relevance_score_fn="cosine",
        )

        documents = []
        for data in datas:
            documents.append(Document(
                page_content=data.pop("refactor_question"),
                metadata=data
            ))

        uuids = [str(uuid4()) for _ in range(len(documents))]
        vector_store.add_documents(documents, uuids)