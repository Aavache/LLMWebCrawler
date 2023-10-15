import argparse

import pymilvus

from llm import MODEL_REGISTRY

SEARCH_PARAMS = {
    "metric_type": "L2",
    # "params": {"nprobe": 10},
}


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--text", type=str, required=True)
    argparser.add_argument("--collection", type=str, required=True)
    argparser.add_argument("--embed-col", type=str, required=True)
    argparser.add_argument("--topk", type=int, default=10)
    argparser.add_argument("--offset", type=int, default=0)
    argparser.add_argument("--llm-name", type=str, default="bert-base-uncased")
    argparser.add_argument("--host", type=str, default="localhost")
    argparser.add_argument("--port", type=int, default=19530)
    args = argparser.parse_args()

    # Load LLM
    assert args.llm_name in MODEL_REGISTRY, f"Unknown LLM {args.llm}"
    model = MODEL_REGISTRY[args.llm_name]()

    # Get text embedding
    embedding = model.text_to_embedding(args.text)

    try:
        # Connect to Milvus DB
        pymilvus.connections.connect("default", host=args.host, port=args.port)

        if not pymilvus.utility.has_collection(args.collection):
            raise Exception(f"Collection not found: {args.collection}")

        collection = pymilvus.Collection(args.collection)
        collection.load()

        search_param = SEARCH_PARAMS.copy()
        search_param["offset"] = args.offset

        print(f"Search results of the top-{args.topk}:")
        print(
            collection.search(
                anns_field=args.embed_col, limit=args.topk, collection=collection, data=[embedding], param=SEARCH_PARAMS
            )
        )

        collection.release()
    except Exception as e:
        print(f"Error when searching: {e}")
