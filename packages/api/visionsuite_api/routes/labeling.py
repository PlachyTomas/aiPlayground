from fastapi import APIRouter, Request

from ..labelstudio import get_gateway

router = APIRouter()


@router.get("/api/labelstudio/status")
async def labelstudio_status(request: Request) -> dict:
    return get_gateway(request).status()
