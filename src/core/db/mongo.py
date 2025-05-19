from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure, ServerSelectionTimeoutError

from core.config import settings
from core.logger_utils import get_logger

logger = get_logger(__file__)


class MongoDatabaseConnector:
    """Singleton class to connect to MongoDB database."""

    _instance: MongoClient | None = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            max_retries = 5
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    # Connect to MongoDB with a 10 second timeout
                    cls._instance = MongoClient(
                        settings.MONGO_DATABASE_HOST,
                        serverSelectionTimeoutMS=10000
                    )
                    
                    # Verify the connection by accessing the admin database
                    cls._instance.admin.command('ping')
                    
                    # Try to check replica set status, but don't fail if not initialized
                    try:
                        rs_status = cls._instance.admin.command('replSetGetStatus')
                        logger.info(f"Connected to replica set: {rs_status.get('set', 'unknown')}")
                    except OperationFailure as e:
                        # Handle the case where replica set is not yet initialized
                        if "no replset config has been received" in str(e):
                            logger.warning("Connected to MongoDB, but replica set is not yet initialized")
                        else:
                            logger.warning(f"Connected to MongoDB, but replica set status check failed: {e}")
                    
                    logger.info(
                        f"Connection to database with uri: {settings.MONGO_DATABASE_HOST} successful"
                    )
                    break
                    
                except (ConnectionFailure, ServerSelectionTimeoutError) as e:
                    retry_count += 1
                    if retry_count >= max_retries:
                        logger.error(f"Couldn't connect to the database after {max_retries} attempts: {e}")
                        raise
                    else:
                        logger.warning(f"Retrying connection to MongoDB ({retry_count}/{max_retries}): {e}")
                        import time
                        time.sleep(5)  # Wait 5 seconds before retrying

        return cls._instance

    def get_database(self):
        assert self._instance, "Database connection not initialized"

        return self._instance[settings.MONGO_DATABASE_NAME]

    def close(self):
        if self._instance:
            self._instance.close()
            logger.info("Connected to database has been closed.")


connection = MongoDatabaseConnector()
