"""
AlphaLens FastAPI backend.
Endpoints under /api: search, quote, profile, news, history, research, health.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from motor.motor_asyncio import AsyncIOMotorClient
from starlette.middleware.cors import CORSMiddleware

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

from providers import (  # noqa: E402
    get_analyst, get_financials, get_history, get_news, get_peers,
    get_profile, get_quote, get_ratios, search_ticker,
)
from agents import run_pipeline_streaming  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("alphalens")

mongo_url = os.environ["MONGO_URL"]
mongo_client = AsyncIOMotorClient(mongo_url)
db = mongo_client[os.environ["DB_NAME"]]

app = FastAPI(title="AlphaLens API")
api = APIRouter(prefix="/api")


# ---------------- public read endpoints ----------------

@api.get("/")
async def root() -> Dict[str, str]:
    return {"service": "AlphaLens", "status": "ok"}


@api.get("/search")
async def search(q: str = Query(..., min_length=1)) -> Dict[str, List[Dict[str, Any]]]:
    return {"results": await search_ticker(q.strip())}


@api.get("/quote/{ticker}")
async def quote(ticker: str) -> Dict[str, Any]:
    data = await get_quote(ticker.upper())
    if not data:
        raise HTTPException(404, "Quote not found")
    return data


@api.get("/profile/{ticker}")
async def profile(ticker: str) -> Dict[str, Any]:
    data = await get_profile(ticker.upper())
    if not data:
        raise HTTPException(404, "Profile not found")
    return data


@api.get("/news/{ticker}")
async def news(ticker: str) -> Dict[str, List[Dict[str, Any]]]:
    return {"items": await get_news(ticker.upper())}


@api.get("/history/{ticker}")
async def history(ticker: str) -> Dict[str, List[Dict[str, Any]]]:
    return {"points": await get_history(ticker.upper())}


@api.get("/research/recent/list")
async def recent_reports() -> Dict[str, List[Dict[str, Any]]]:
    docs = await db.research_reports.find({}, {"_id": 0}).sort("generated_at", -1).limit(8).to_list(8)
    keys = ("id", "ticker", "company_name", "recommendation", "confidence_score", "generated_at")
    return {"items": [{k: d.get(k) for k in keys} for d in docs]}


@api.get("/health")
async def health() -> Dict[str, Any]:
    async def _check(name: str, coro: Awaitable[Any]) -> Dict[str, Any]:
        try:
            r = await asyncio.wait_for(coro, timeout=6)
            return {"name": name, "ok": r is not None}
        except Exception as exc:
            return {"name": name, "ok": False, "error": str(exc)[:120]}

    results = await asyncio.gather(
        _check("fmp", get_quote("AAPL")),
        _check("news", get_news("AAPL")),
    )
    return {"providers": results, "time": datetime.now(timezone.utc).isoformat()}


# ---------------- SSE: research pipeline ----------------

def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


async def _fetch_market_data(ticker: str) -> Dict[str, Any]:
    """Fetch every market-data slice in parallel."""
    p, q, fin, rt, pr, an, nw, hist = await asyncio.gather(
        get_profile(ticker),
        get_quote(ticker),
        get_financials(ticker),
        get_ratios(ticker),
        get_peers(ticker),
        get_analyst(ticker),
        get_news(ticker),
        get_history(ticker),
    )
    return {
        "profile": p, "quote": q, "financials": fin, "ratios": rt,
        "peers": pr, "analyst": an, "news": nw, "history": hist,
    }


def _build_report(report_id: str, ticker: str, market: Dict[str, Any],
                  state: Dict[str, Any]) -> Dict[str, Any]:
    profile_name = (market.get("profile") or {}).get("name") or ticker
    return {
        "id": report_id,
        "ticker": ticker,
        "company_name": profile_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "market_data": market,
        "agents": {k: state.get(k) for k in
                   ("research", "financial", "valuation", "macro", "bull", "bear", "moderator")},
        "recommendation": (state.get("moderator") or {}).get("recommendation"),
        "confidence_score": (state.get("moderator") or {}).get("confidence_score"),
    }


async def _drain_pipeline(state: Dict[str, Any]) -> AsyncGenerator[str, None]:
    """Run the multi-agent pipeline and translate emitter callbacks into SSE chunks."""
    queue: asyncio.Queue = asyncio.Queue()

    async def emit(evt: str, data: Any) -> None:
        await queue.put((evt, data))

    async def runner() -> None:
        try:
            await run_pipeline_streaming(state, emit)
        except Exception as exc:
            logger.exception("pipeline error")
            await queue.put(("error", {"message": str(exc)}))
        finally:
            await queue.put(("__done__", None))

    task = asyncio.create_task(runner())
    while True:
        evt, data = await queue.get()
        if evt == "__done__":
            break
        yield _sse(evt, data)
    await task


@api.get("/research/start")
async def research_start(ticker: str = Query(..., min_length=1)) -> StreamingResponse:
    ticker = ticker.upper().strip()
    report_id = str(uuid.uuid4())

    async def gen() -> AsyncGenerator[str, None]:
        yield _sse("init", {"report_id": report_id, "ticker": ticker})
        yield _sse("step", {"label": "Fetching market data…"})

        market = await _fetch_market_data(ticker)
        if not market["profile"] and not market["quote"]:
            yield _sse("error", {"message": f"Ticker '{ticker}' not found at any data provider."})
            return

        yield _sse("market_data", market)

        state: Dict[str, Any] = {
            "ticker": ticker,
            "profile": market["profile"] or {},
            "quote": market["quote"] or {},
            "financials": market["financials"] or {},
            "ratios": market["ratios"] or {},
            "peers": market["peers"] or [],
            "analyst": market["analyst"] or {},
            "news": market["news"] or [],
        }

        async for chunk in _drain_pipeline(state):
            yield chunk

        report = _build_report(report_id, ticker, market, state)
        try:
            await db.research_reports.insert_one(report)
        except Exception as exc:
            logger.warning(f"save report failed: {exc}")

        yield _sse("complete", {"report_id": report_id})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@api.get("/research/{report_id}")
async def get_report(report_id: str) -> Dict[str, Any]:
    doc = await db.research_reports.find_one({"id": report_id}, {"_id": 0})
    if not doc:
        raise HTTPException(404, "Report not found")
    return doc


app.include_router(api)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("shutdown")
async def shutdown_db_client() -> None:
    mongo_client.close()
