from pymilvus import (Collection, CollectionSchema, DataType, FieldSchema,
                      connections, utility)

from db.constants import (COLLECTION_NAME, DB_COLS, INDEX_PARAM, MILVUS_HOST,
                          MILVUS_PORT)


class VectorDBClient:
    """Vector database client."""

    def __init__(self, embed_size: int, batch_size: int):
        # Unpack parameters
        self.batch_size = batch_size
        self.embed_size = embed_size

        self._setup_db_connection()

        self._reset_batch()

    def _setup_db_connection(self):
        """Setup the Milvus connection."""
        # connections.connect(host=MILVUS_HOST, port=MILVUS_PORT, password=PASSWORD, secure=True)
        connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)

        self.schema = [
            FieldSchema(name=DB_COLS["URL"], dtype=DataType.VARCHAR, is_primary=True, max_length=1024),
            FieldSchema(name=DB_COLS["TEXT"], dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name=DB_COLS["EMBED"], dtype=DataType.FLOAT_VECTOR, dim=self.embed_size),
        ]
        if not utility.has_collection(COLLECTION_NAME):
            col_schema = CollectionSchema(fields=self.schema)
            self.collection = Collection(name=COLLECTION_NAME, schema=col_schema)
            assert utility.has_collection(COLLECTION_NAME), " it could not be created"
            self.collection.create_index(field_name=DB_COLS["EMBED"], index_params=INDEX_PARAM)
            print("It was created successfully")
        else:
            self.collection = Collection(name=COLLECTION_NAME)

        self.collection.load()

    def _reset_batch(self):
        """Reset the batch."""
        self.batch = [[], [], []]  # url, text, embeddings

    def _submit_batch(self):
        """Submit the batch to Milvus."""
        self.collection.insert(COLLECTION_NAME, self._batch)
        self.collection.flush()
        self._reset_batch()

    def insert(self, url, text, embeddings):
        """Insert a crawled entry in milvus.

            NOTE: the method will only insert the entries in the DB
            if the batch size is reached.

        Parameters
        ----------
        url : str
            The URL of the crawled page.
        text : str
            The text of the crawled page.
        embeddings : numpy.ndarray
            The embeddings of the crawled page.
        """
        # Insert data into Milvus
        self.batch.append([url], [text], [embeddings.tolist()])

        if len(self.batch[0]) >= self.batch_size:
            self._submit_batch()

    def close(self):
        """Close the Milvus connection."""
        if len(self.batch[0]):
            self._submit_batch()
        self.milvus_client.close()
