import os
import shutil
import subprocess
import tempfile
import logging

from git import Repo
from core.db.documents import RepositoryDocument

from data_crawling.crawlers.base import BaseCrawler

logger = logging.getLogger("praxis-llm-workshop/crawler")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class GithubCrawler(BaseCrawler):
    model = RepositoryDocument

    def __init__(self, ignore=(".git", ".toml", ".lock", ".png")) -> None:
        super().__init__()
        self._ignore = ignore

    def extract(self, link: str, **kwargs) -> None:
        logger.info(f"Starting scrapping GitHub repository: {link}")

        repo_name = link.rstrip("/").split("/")[-1]

        local_temp = tempfile.mkdtemp()

        try:
            os.chdir(local_temp)
            subprocess.run(["git", "clone", link])

            repo_path = os.path.join(local_temp, os.listdir(local_temp)[0])

            tree = {}
            for root, dirs, files in os.walk(repo_path):
                dir = root.replace(repo_path, "").lstrip("/")
                if dir.startswith(self._ignore):
                    continue

                for file in files:
                    if file.endswith(self._ignore):
                        continue
                    file_path = os.path.join(dir, file)
                    with open(os.path.join(root, file), "r", errors="ignore") as f:
                        tree[file_path] = f.read().replace(" ", "")

            instance = self.model(
                name=repo_name, link=link, content=tree, owner_id=kwargs.get("user")
            )
            instance.save()

        except Exception:
            raise
        finally:
            shutil.rmtree(local_temp)

        logger.info(f"Finished scrapping GitHub repository: {link}")