"""
AlphaLens Multi-Agent Orchestration — LangGraph Edition
Supervisor-Worker pattern:
  Phase 1 (parallel): research, financial, valuation, macro
  Phase 2 (parallel): bull, bear
  Phase 3 (sequential): moderator (supervisor)

server.py calls run_pipeline_streaming(state, emit) — interface unchanged.
"""
import os
import json
import logging
import asyncio
import re
from typing import Any, Dict, List, Optional, TypedDict, Annotated
import operator

from openai import AsyncOpenAI
from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)

# ---------------------------
# GROQ CONFIG
# ---------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

client = AsyncOpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

MODEL_NAME = "llama-3.3-70b-versatile"


# ---------------------------
# LANGGRAPH STATE SCHEMA
# ---------------------------
class AlphaLensState(TypedDict):
    # --- Input market data (set before graph starts) ---
    ticker: str
    profile: Dict[str, Any]
    quote: Dict[str, Any]
    financials: Dict[str, Any]
    ratios: Dict[str, Any]
    peers: List[Any]
    analyst: Dict[str, Any]
    news: List[Any]

    # --- Agent outputs (filled by nodes) ---
    research: Dict[str, Any]
    financial: Dict[str, Any]
    valuation: Dict[str, Any]
    macro: Dict[str, Any]
    bull: Dict[str, Any]
    bear: Dict[str, Any]
    moderator: Dict[str, Any]

    # --- SSE emitter (callable, passed through state) ---
    emit: Any  # Callable — not serializable but stays in memory


# ---------------------------
# JSON EXTRACTION UTILITY
# ---------------------------
def _extract_json(text: str) -> Dict[str, Any]:
    if not text:
        return {}

    text = text.strip()

    m = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)

    start = text.find("{")
    if start == -1:
        return {}

    depth = 0
    end = -1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if end == -1:
        return {}

    try:
        return json.loads(text[start:end])
    except Exception as e:
        logger.warning(f"JSON parse failed: {e}")
        return {}


# ---------------------------
# LLM CALL (GROQ)
# ---------------------------
async def _ask(system: str, user: str, session_id: str) -> str:
    try:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.7,
            max_tokens=4000,
        )
        content = response.choices[0].message.content
        print(f"\n=== GROQ RESPONSE [{session_id}] ===\n{content}")
        return content
    except Exception as e:
        print(f"\n=== GROQ ERROR [{session_id}] === {e}")
        logger.error(f"Groq LLM error ({session_id}): {e}")
        return ""


# ============================================================
# LANGGRAPH NODES
# Each node receives the full state dict and returns a partial
# update dict — LangGraph merges it back into state.
# ============================================================

async def node_research(state: AlphaLensState) -> Dict[str, Any]:
    """Research agent node — business overview, moat, risks."""
    emit = state.get("emit")
    if emit:
        await emit("agent_start", {"agent": "research"})

    profile = state.get("profile") or {}
    peers = state.get("peers") or []
    peer_str = ", ".join(
        [p.get("ticker", "") if isinstance(p, dict) else str(p) for p in peers]
    ) or "n/a"

    system = "You are a senior equity research analyst. Respond ONLY with valid JSON."
    user = f"""
Company: {profile.get('name')} ({state.get('ticker')})
Sector: {profile.get('sector')} | Industry: {profile.get('industry')}
Description: {profile.get('description', '')[:1500]}
Peers: {peer_str}

Return JSON:
{{
  "business_summary": "",
  "moat": "",
  "moat_strength": "",
  "revenue_streams": [],
  "growth_drivers": [],
  "market_position": "",
  "key_risks": []
}}
"""
    raw = await _ask(system, user, f"research-{state.get('ticker')}")
    result = _extract_json(raw)

    if emit:
        await emit("agent_done", {"agent": "research", "data": result})

    return {"research": result}


async def node_financial(state: AlphaLensState) -> Dict[str, Any]:
    """Financial agent node — margins, growth, health."""
    emit = state.get("emit")
    if emit:
        await emit("agent_start", {"agent": "financial"})

    fin = state.get("financials") or {}
    ratios = state.get("ratios") or {}

    system = "You are a CFA-level financial analyst. Respond ONLY with valid JSON."
    user = f"""
Ticker: {state.get('ticker')}
Income: {json.dumps(fin.get('income', []))[:2000]}
Balance: {json.dumps(fin.get('balance', []))[:1500]}
Cashflow: {json.dumps(fin.get('cashflow', []))[:1500]}
Ratios: {json.dumps(ratios)[:1500]}

Return JSON:
{{
  "revenue_growth_yoy_pct": 0,
  "net_income_growth_yoy_pct": 0,
  "gross_margin_pct": 0,
  "operating_margin_pct": 0,
  "net_margin_pct": 0,
  "fcf_latest": 0,
  "roe_pct": 0,
  "debt_to_equity": 0,
  "current_ratio": 0,
  "financial_health": "",
  "key_observations": []
}}
"""
    raw = await _ask(system, user, f"fin-{state.get('ticker')}")
    result = _extract_json(raw)

    if emit:
        await emit("agent_done", {"agent": "financial", "data": result})

    return {"financial": result}


async def node_valuation(state: AlphaLensState) -> Dict[str, Any]:
    """Valuation agent node — DCF, PE, EV/EBITDA fair values."""
    emit = state.get("emit")
    if emit:
        await emit("agent_start", {"agent": "valuation"})

    ratios = state.get("ratios") or {}
    quote = state.get("quote") or {}

    system = "You are a valuation expert. Respond ONLY with valid JSON."
    user = f"""
Ticker: {state.get('ticker')}
Price: {quote.get('price')}
PE: {ratios.get('peTTM')}
PEG: {ratios.get('pegTTM')}
PB: {ratios.get('pbTTM')}
EV/EBITDA: {ratios.get('evToEbitda')}

Return JSON:
{{
  "dcf_fair_value": 0,
  "pe_based_fair_value": 0,
  "ev_ebitda_based_fair_value": 0,
  "fair_value_low": 0,
  "fair_value_mid": 0,
  "fair_value_high": 0,
  "upside_pct": 0,
  "valuation_verdict": "",
  "reasoning": ""
}}
"""
    raw = await _ask(system, user, f"val-{state.get('ticker')}")
    result = _extract_json(raw)

    if emit:
        await emit("agent_done", {"agent": "valuation", "data": result})

    return {"valuation": result}


async def node_macro(state: AlphaLensState) -> Dict[str, Any]:
    """Macro agent node — rate sensitivity, sector cycle, FX."""
    emit = state.get("emit")
    if emit:
        await emit("agent_start", {"agent": "macro"})

    profile = state.get("profile") or {}

    system = "You are a macroeconomic analyst. Respond ONLY with valid JSON."
    user = f"""
Sector: {profile.get('sector')}
Industry: {profile.get('industry')}

Return JSON:
{{
  "rate_sensitivity": "",
  "inflation_impact": "",
  "sector_cycle_phase": "",
  "fx_exposure": "",
  "macro_tailwinds": [],
  "macro_headwinds": [],
  "overall_macro_score": 0
}}
"""
    raw = await _ask(system, user, f"macro-{state.get('ticker')}")
    result = _extract_json(raw)

    if emit:
        await emit("agent_done", {"agent": "macro", "data": result})

    return {"macro": result}


async def node_bull(state: AlphaLensState) -> Dict[str, Any]:
    """Bull agent node — investment thesis and catalysts."""
    emit = state.get("emit")
    if emit:
        await emit("agent_start", {"agent": "bull"})

    system = "You are a bullish investor. Respond ONLY with valid JSON."
    user = f"""
Research: {json.dumps(state.get('research', {}))[:1200]}
Financial: {json.dumps(state.get('financial', {}))[:1200]}
Valuation: {json.dumps(state.get('valuation', {}))[:1200]}

Return JSON:
{{
  "thesis_points": [],
  "catalysts": [],
  "target_upside_pct": 0
}}
"""
    raw = await _ask(system, user, f"bull-{state.get('ticker')}")
    result = _extract_json(raw)

    if emit:
        await emit("agent_done", {"agent": "bull", "data": result})

    return {"bull": result}


async def node_bear(state: AlphaLensState) -> Dict[str, Any]:
    """Bear agent node — risks and downside thesis."""
    emit = state.get("emit")
    if emit:
        await emit("agent_start", {"agent": "bear"})

    system = "You are a bearish investor. Respond ONLY with valid JSON."
    user = f"""
Research: {json.dumps(state.get('research', {}))[:1200]}
Financial: {json.dumps(state.get('financial', {}))[:1200]}
Valuation: {json.dumps(state.get('valuation', {}))[:1200]}

Return JSON:
{{
  "thesis_points": [],
  "risks": [],
  "target_downside_pct": 0
}}
"""
    raw = await _ask(system, user, f"bear-{state.get('ticker')}")
    result = _extract_json(raw)

    if emit:
        await emit("agent_done", {"agent": "bear", "data": result})

    return {"bear": result}


async def node_moderator(state: AlphaLensState) -> Dict[str, Any]:
    """Moderator / Supervisor node — final INVEST/HOLD/PASS verdict."""
    emit = state.get("emit")
    if emit:
        await emit("agent_start", {"agent": "moderator"})

    system = "You are a CIO. Return final INVEST/HOLD/PASS decision in JSON only."
    user = f"""
Ticker: {state.get('ticker')}
Price: {state.get('quote', {}).get('price')}

Research: {json.dumps(state.get('research', {}))[:1000]}
Financial: {json.dumps(state.get('financial', {}))[:1000]}
Valuation: {json.dumps(state.get('valuation', {}))[:1000]}
Bull: {json.dumps(state.get('bull', {}))[:800]}
Bear: {json.dumps(state.get('bear', {}))[:800]}
Macro: {json.dumps(state.get('macro', {}))[:800]}

Return JSON:
{{
  "recommendation": "",
  "confidence_score": 0,
  "expected_upside_pct": 0,
  "expected_downside_pct": 0,
  "time_horizon": "",
  "executive_summary": "",
  "key_reasons": [],
  "what_would_change_view": []
}}
"""
    raw = await _ask(system, user, f"mod-{state.get('ticker')}")
    result = _extract_json(raw)

    if emit:
        await emit("agent_done", {"agent": "moderator", "data": result})

    return {"moderator": result}


# ============================================================
# PHASE WRAPPER NODES
# LangGraph doesn't natively run nodes in parallel within one
# graph step, so we create "phase" wrapper nodes that run the
# worker agents concurrently via asyncio.gather — giving us
# true parallelism inside the graph.
# ============================================================

async def node_phase1(state: AlphaLensState) -> Dict[str, Any]:
    """
    Phase 1 supervisor node — runs Research, Financial, Valuation,
    Macro agents in parallel using asyncio.gather.
    """
    results = await asyncio.gather(
        node_research(state),
        node_financial(state),
        node_valuation(state),
        node_macro(state),
    )
    # Merge all partial dicts into one update
    merged = {}
    for r in results:
        merged.update(r)
    return merged


async def node_phase2(state: AlphaLensState) -> Dict[str, Any]:
    """
    Phase 2 supervisor node — runs Bull and Bear agents in parallel.
    Phase 1 outputs (research/financial/valuation) are now in state.
    """
    results = await asyncio.gather(
        node_bull(state),
        node_bear(state),
    )
    merged = {}
    for r in results:
        merged.update(r)
    return merged


# ============================================================
# BUILD THE LANGGRAPH GRAPH
# ============================================================

def _build_graph() -> Any:
    """
    Compile the AlphaLens supervisor-worker graph.

    Graph topology:
        START → phase1 → phase2 → moderator → END

    phase1 and phase2 are "super-nodes" that internally run
    worker agents in parallel via asyncio.gather, mimicking
    LangGraph's Send() parallel dispatch pattern in a way that
    is fully compatible with FastAPI's asyncio event loop.
    """
    graph = StateGraph(AlphaLensState)

    # Register nodes
    graph.add_node("phase1", node_phase1)       # Research + Financial + Valuation + Macro
    graph.add_node("phase2", node_phase2)       # Bull + Bear
    graph.add_node("moderator", node_moderator) # Supervisor / CIO

    # Wire the edges
    graph.set_entry_point("phase1")
    graph.add_edge("phase1", "phase2")
    graph.add_edge("phase2", "moderator")
    graph.add_edge("moderator", END)

    return graph.compile()


# Compile once at import time — reused for every request
_GRAPH = _build_graph()


# ============================================================
# PUBLIC API — called by server.py (interface unchanged)
# ============================================================

async def run_pipeline_streaming(state: Dict[str, Any], emit) -> Dict[str, Any]:
    """
    Run the LangGraph supervisor-worker pipeline.

    Injects the SSE emit callable into state so every node can
    fire agent_start / agent_done events as it runs — server.py
    sees the same SSE event stream as before.

    Returns the final state dict (server.py reads state["moderator"] etc.)
    """
    # Inject emitter so nodes can fire SSE events
    state["emit"] = emit

    # Provide default empty dicts for agent output keys
    # (LangGraph requires all TypedDict keys to exist in initial state)
    for key in ("research", "financial", "valuation", "macro", "bull", "bear", "moderator"):
        state.setdefault(key, {})

    # ainvoke runs the compiled graph asynchronously
    final_state = await _GRAPH.ainvoke(state)

    # Copy results back into the original state dict
    # (server.py reads from the same dict reference it passed in)
    for key in ("research", "financial", "valuation", "macro", "bull", "bear", "moderator"):
        state[key] = final_state.get(key, {})

    return state
