import argparse

import pymilvus

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--host", type=str, default="localhost")
    argparser.add_argument("--port", type=int, default=19530)
    args = argparser.parse_args()

    try:
        # Connect to Milvus DB
        pymilvus.connections.connect("default", host=args.host, port=args.port)

        if not pymilvus.utility.has_collection(args.collection):
            raise Exception(f"Collection not found: {args.collection}")

        collection = pymilvus.Collection(args.collection)
        collection.load()

        print(f"Collection schema: {collection.schema}")

    except Exception as e:
        print(f"Error when listing collection: {e}")
