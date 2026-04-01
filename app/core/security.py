from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader
from app.core.config import settings
from app.core.logger import logger

# Define header for API Key
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key: str = Security(api_key_header)):
    """
    Validates the API key from request headers.
    """
    if api_key == settings.API_SECRET_KEY:
        return api_key
    
    logger.warning(f"Unauthorized access attempt with API Key: {api_key}")
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API Key",
        headers={"WWW-Authenticate": "ApiKey"},
    )
