import argparse

import ray

from task import WebCrawler


def main(args):
    # Prior requisite is to run `ray start --head` in the terminal
    # and connect to the existing Ray cluster with the following line
    ray.init(address="auto")

    # Instantiate Ray worker code
    crawler = WebCrawler.remote()

    print("Starting to crawl...")
    ray.get(
        [crawler.crawl.remote(url, 0, args.max_depth) for url in args.initial_urls]
    )  # Initiate the crawling remotely

    # Wait for all tasks to complete
    print("Done crawling.")
    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Distributed NLP Web Crawler",
        description="This program can crawl the web and store text embeddings",
    )
    parser.add_argument(
        "-u",
        "--initial-urls",
        nargs="+",
    )
    parser.add_argument("-lm", "--language-model", default="bert-base-uncased")
    parser.add_argument("-m", "--max-depth", default=2)
    args = parser.parse_args()

    main(args)
