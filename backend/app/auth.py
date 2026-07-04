import logging
import secrets

from fastapi import HTTPException, Request

from app.config import settings

logger = logging.getLogger(__name__)

_warned = False


async def require_api_key(request: Request) -> None:
    """Reject requests that do not carry the configured API key.

    When API_KEY is unset the check is disabled so local development keeps
    working; a warning is logged once so a production misconfiguration is
    visible in the logs.
    """
    global _warned
    if not settings.api_key:
        if not _warned:
            logger.warning(
                "API_KEY is not set — the API is running unauthenticated. "
                "Set API_KEY in the environment for production deployments."
            )
            _warned = True
        return
    provided = request.headers.get("x-api-key", "")
    if not secrets.compare_digest(provided, settings.api_key):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
