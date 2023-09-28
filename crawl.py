import ray
from worker import WebCrawler


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='Distributed NLP Web Crawler',
                    description='This program can crawl the web and store text embeddings',)
    parser.add_argument('-u', '--initial-url', )
    parser.add_argument('-b', '--initial-url', )
    parser.add_argument('-db', '--db-url', default='http://localhost')
    parser.add_argument('-lm', '--language-model', default='bert-base-uncased')
    parser.add_argument('-m', '--max-depth', default=2)
    args = parser.parse_args()

    # Initialize Ray
    ray.init()
    
    # Instantiate Ray worker code
    crawler = WebCrawler.remote(
            args.initial_url, 
            args.max_depth
    )

    print("Starting to crawl...")
    ray.get(crawler.crawl.remote(initial_url, 0))  # Initiate the crawling remotely

    # Wait for all tasks to complete
    print("Done crawling.")
    ray.shutdown()

