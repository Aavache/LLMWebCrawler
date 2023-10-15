import ray

from task.crawler import WebCrawler


def run_crawl(crawler: WebCrawler, urls: list[str], max_depth: int):
    ray.get([crawler.crawl.remote(url, 0, max_depth) for url in urls])
