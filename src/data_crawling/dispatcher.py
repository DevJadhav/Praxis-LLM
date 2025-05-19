import re
import logging

from data_crawling.crawlers.base import BaseCrawler
from data_crawling.crawlers.custom_article import CustomArticleCrawler

logger = logging.getLogger("praxis-llm-workshop/crawler")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)



class CrawlerDispatcher:
    def __init__(self) -> None:
        self._crawlers = {}

    def register(self, domain: str, crawler: type[BaseCrawler]) -> None:
        self._crawlers[r"https://(www\.)?{}.com/*".format(re.escape(domain))] = crawler

    def get_crawler(self, url: str) -> BaseCrawler:
        for pattern, crawler in self._crawlers.items():
            if re.match(pattern, url):
                return crawler()
        else:
            logger.warning(
                f"No crawler found for {url}. Defaulting to CustomArticleCrawler."
            )

            return CustomArticleCrawler()
