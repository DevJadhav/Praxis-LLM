from typing import Any, Dict
import logging
import os
from fastapi import FastAPI, Request, HTTPException
import json

from core import lib
from core.db.documents import UserDocument
from data_crawling.crawlers import CustomArticleCrawler, GithubCrawler, LinkedInCrawler
from data_crawling.dispatcher import CrawlerDispatcher

logger = logging.getLogger("praxis-llm-workshop/crawler")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler_stream = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler_stream.setFormatter(formatter)
    logger.addHandler(handler_stream)

_dispatcher = CrawlerDispatcher()
_dispatcher.register("medium", CustomArticleCrawler)
_dispatcher.register("linkedin", LinkedInCrawler)
_dispatcher.register("github", GithubCrawler)


def handler(event, context=None) -> Dict[str, Any]:
    first_name, last_name = lib.split_user_full_name(event.get("user"))

    user_id = UserDocument.get_or_create(first_name=first_name, last_name=last_name)

    link = event.get("link")
    crawler = _dispatcher.get_crawler(link)

    try:
        crawler.extract(link=link, user=user_id)

        return {"statusCode": 200, "body": "Link processed successfully"}
    except Exception as e:
        logger.error(f"Error in handler: {str(e)}", exc_info=True)
        return {"statusCode": 500, "body": f"An error occurred: {str(e)}"}

app = FastAPI()

@app.post("/2015-03-31/functions/function/invocations")
async def lambda_invoke(request: Request):
    try:
        event = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")
    
    response = handler(event, None)
    return response

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    logger.info(f"Starting data crawler service on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
