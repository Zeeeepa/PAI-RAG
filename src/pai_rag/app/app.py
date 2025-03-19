# init trace
import os
import asyncio
import threading
from fastapi import FastAPI
from contextlib import asynccontextmanager
from pai_rag.utils.format_logging import format_logging
from pai_rag.app.api.service import configure_app
from pai_rag.utils.download_models import ModelScopeDownloader
from pai_rag.utils.constants import DEFAULT_MODEL_DIR
from pai_rag.knowledgebase.rag_job_manager import job_manager
from pai_rag.core.service_daemon import startup_event
from pai_rag.core.rag_environment import service_environment
from loguru import logger

format_logging()

@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("Application starting up...")
    if service_environment.SHOULD_START_WEB:
        daemon_thread = threading.Thread(target=job_manager.execute_job, daemon=True)
        daemon_thread.start()

    asyncio.create_task(startup_event())
    yield

    service_environment.cleanup()
    logger.info("Application shutting down...")



os.environ["PAI_RAG_MODEL_DIR"] = DEFAULT_MODEL_DIR

service_environment.acquire_lock()

app = FastAPI(lifespan=lifespan)

ModelScopeDownloader().load_rag_models()

configure_app(app)
