from typing import Optional

from pydantic import BaseModel


class CrawlJob(BaseModel):
    url: list[str]
    max_depth: Optional[int] = 2
