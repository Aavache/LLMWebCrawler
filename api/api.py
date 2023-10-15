import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor

import ray
from fastapi import FastAPI

from api.models import CrawlJob
from api.utils import run_crawl
from task import WebCrawler

# API APP object
app = FastAPI()

pool = ThreadPoolExecutor(max_workers=1)

# Distributed crawler
ray.init(address="auto")
crawler = WebCrawler.remote()


@app.post("/")
def crawl(request: CrawlJob):
    loop = asyncio.get_event_loop()
    try:
        loop.run_in_executor(
            None,
            functools.partial(
                run_crawl, data={"crawler": crawler, "urls": request.urls, "max_depth": request.max_depth}
            ),
        )

    except Exception as e:
        return {"message": f"Error: {str(e)}"}

    return {"message": "Crawling started."}
