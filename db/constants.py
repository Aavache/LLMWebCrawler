# TODO: most variable here should be set in .env file
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
USER = "admin"
PASSWORD = "admin"
URI = f"http://{MILVUS_HOST}:{MILVUS_PORT}"
COLLECTION_NAME = "web_crawler_data"
INDEX_PARAM = {"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 1024}}
DB_COLS = {
    "URL": "url",
    "TEXT": "text",
    "EMBED": "embeddings",
}
