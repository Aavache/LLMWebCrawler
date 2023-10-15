import argparse

import pymilvus

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--collection", type=str, default="collection")
    argparser.add_argument("--host", type=str, default="localhost")
    argparser.add_argument("--port", type=int, default=19530)
    args = argparser.parse_args()

    try:
        # Connect to Milvus DB
        pymilvus.connections.connect("default", host=args.host, port=args.port)

        # List collections
        print("Here the list of collections: \n")
        print(pymilvus.list_collections(timeout=None, using="default"))
    except Exception as e:
        print(f"Error when listing collection: {e}")
