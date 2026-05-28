from fastapi import APIRouter
from datetime import datetime

router = APIRouter()


@router.get("/health")
async def health_check():
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "powerprice-futures-signals",
        "signal_only": True,
        "disclaimer": "This system generates signals only. No live trading execution.",
    }
