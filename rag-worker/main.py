"""Entry point for the rag-worker service."""

import asyncio
import logging
import os

from dotenv import load_dotenv

# Load .env if present (no-op in production containers that set vars directly)
load_dotenv()

from config import Config  # noqa: E402 – must come after load_dotenv
from worker import RAGWorker  # noqa: E402

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def main() -> None:
    config = Config()

    if not config.openai_api_key:
        logger.warning(
            "OPENAI_API_KEY is not set. LLM and embedding calls will fail "
            "unless a compatible proxy is configured."
        )

    worker = RAGWorker(config)
    try:
        await worker.run()
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Shutdown signal received")
    finally:
        await worker.close()
        logger.info("rag-worker stopped")


if __name__ == "__main__":
    asyncio.run(main())
