import milvus

COLLECTION_NAME = "web_crawler_data"


class VectorDBClient:
    """Vector Database client"""
    def __init__(self, embedding_size: int):
        self.milvus_client = milvus.Milvus()
        self.schema = [
            milvus.FieldSchema(name="url", dtype=milvus.DataType.STRING, is_primary=True),
            milvus.FieldSchema(name="text", dtype=milvus.DataType.STRING),
            milvus.FieldSchema(name="embeddings", dtype=milvus.DataType.FLOAT_VECTOR, dim=embedding_size),
        ]

    def insert(self, url, text, embeddings):
        """Insert an crawled entry in milvus"""
        # Create Milvus collection (if it doesn't exist)
        if COLLECTION_NAME not in self.milvus_client.list_collections():
            milvus_client.create_collection(COLLECTION_NAME, self.schema)

        # Insert data into Milvus
        entities = [
            {"url": url, "text": text, "embeddings": embeddings.tolist()},
        ]
        milvus_client.insert(COLLECTION_NAME, entities)
