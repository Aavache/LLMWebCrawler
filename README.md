# LLM-based Web Crawler

An scalable web crawler, here a list of the feature of this crawler:

* This service can crawl recursively the web storing links it's text and the corresponding text embedding.
* We use a large language model (e.g Bert) to obtain the text embeddings, i.e. a vector representation of the text present at each webiste.
* The service is scalable, we use Ray to spread across multiple workers.
* The entries are stored into a vector database. Vector databases are ideal to save and retrieve samples according to a vector representation.

By saving the representations into a vector database, you can retrieve similar pages according to how close two vectors are. This is critical for a browser to retrieve the most relevant results.

# CLI

Run the crawler with the terminal:

```sh
$ python cli_crawl.py --help

options:
  -h, --help            show this help message and exit
  -u INITIAL_URLS [INITIAL_URLS ...], --initial-urls INITIAL_URLS [INITIAL_URLS ...]
  -lm LANGUAGE_MODEL, --language-model LANGUAGE_MODEL
  -m MAX_DEPTH, --max-depth MAX_DEPTH
```

# API

Host the API with `uvicorn` and `FastAPI`.

```sh
uvicorn api_app:app --host 0.0.0.0 --port 80
```

Take a look to the example in `start_api_and_head_node.sh`. Note that the ray head nodes needs to be initialized first.

# Large Language Model

For our use case, we simply use [BERT](https://arxiv.org/abs/1810.04805) model implemented by [Huggingface](https://huggingface.co/) to extract embeddings from the web text. More precisely, we use [bert-base-uncased](https://huggingface.co/bert-base-uncased). Note that the code is agnostic and new models could be registered and added with few lines of code, take a look to `llm/best.py`.

# Saving crawled data

We use [Milvus](https://milvus.io/) as our main database administrator software. We use a vector-style database due to its inherited capability of searching and saving entries based on vector representations (embeddings).

## Milvus lite

Start your standalone Milvus server as follows, I suggest using an multiplexer software such as `tmux`:

```sh
tmux new -s milvus
milvus-server
```

Take a look under `scripts/` to see some of the basic requests to Milvus.

## Docker compose

You can also use the official `docker compose` template:

```sh
docker compose --file milvus-docker-compose.yml up -d
```

# Parallel computation

We use [Ray](https://docs.ray.io/en/latest/ray-core/examples/gentle_walkthrough.html), is great python framework to run distributed and parallel processing. Ray follows the master-worker paradigm, where a `head` node will request tasks to be executed to the connected workers.

## Start the head and the worker nodes in Ray

## Head node

1. Setup the head node

```sh
ray start --head
```

2. Connect your program to the head node

```py
import ray

# Connect to the head
ray.init("auto")
```

In case you want to stop ray node:
```sh
ray stop
```

Or checking the status:
```sh
ray status
```

## Worker node

1. Initialize the worker node

```sh
ray start
```

The worker node does not need to have the code implementation as the head node will serialize and submit the arguments and implementation to the workers.


## Future features

The current implementation is a PoC. Many improvements can be made:
* [Important] New entrypoint in the API to search similar URL given text.
* Optimize search and API.
* Adding new LLMs models and new chunking strategies with popular libraries, e.g. [LangChain](https://www.langchain.com/).
* Storing more features in the vector DB, perhaps, generate summaries.

## Contributing

All issues and PRs are welcome 🙂.

## Reference

* [Ray Documentation](https://docs.ray.io/en/latest/ray-core/examples/gentle_walkthrough.html)
* [Milvus](https://milvus.io/)
* [FastAPI](https://fastapi.tiangolo.com/)
* [Huggingface](https://huggingface.co/)
