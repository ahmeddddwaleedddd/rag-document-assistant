"""API-key auth. Missing key -> 401, wrong key -> 403, no keys configured -> 503 (fail closed)."""

import secrets

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from .. import config

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(key: str | None = Security(api_key_header)) -> str:
    if not config.API_KEYS:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Server has no API_KEYS configured")
    if not key:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing X-API-Key header",
                            headers={"WWW-Authenticate": "APIKey"})
    # compare_digest: constant time, so response timing does not leak key prefixes
    if not any(secrets.compare_digest(key, valid) for valid in config.API_KEYS):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Invalid API key")
    return key
