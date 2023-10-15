from pydantic import BaseModel
from typing import Optional


class CrawlJob(BaseModel):
    url: list[str]
    max_depth: Optional[int] = 2