"""
Multi-agent AI orchestration for AlphaLens.
Converted to Groq (OpenAI-compatible API).
"""
import os
import json
import logging
import asyncio
import re
from typing import Any, Dict
from openai import AsyncOpenAI

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
# JSON EXTRACTION
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
                {"role": "user", "content": user}
            ],
            temperature=0.7,
            max_tokens=4000
        )

        print("\n=== GROQ RESPONSE ===")
        print(response.choices[0].message.content)

        return response.choices[0].message.content

    except Exception as e:
        print(f"\n=== GROQ ERROR === {e}")
        logger.error(f"Groq LLM error ({session_id}): {e}")
        return ""


# ---------------------------
# AGENTS
# ---------------------------

async def research_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    profile = state.get("profile") or {}
    peers = state.get("peers") or []

    peer_str = ", ".join(
        [p.get("ticker", "") if isinstance(p, dict) else str(p) for p in peers]
    ) or "n/a"

    system = "You are a senior equity research analyst. Respond ONLY with valid JSON."

    user = f"""
Company: {profile.get('name')} ({state.get('ticker')})
Sector: {profile.get('sector')} | Industry: {profile.get('industry')}
Description: {profile.get('description','')[:1500]}
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
    return _extract_json(raw)


async def financial_agent(state: Dict[str, Any]) -> Dict[str, Any]:
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
    return _extract_json(raw)


async def valuation_agent(state: Dict[str, Any]) -> Dict[str, Any]:
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
    return _extract_json(raw)


async def bull_agent(state: Dict[str, Any]) -> Dict[str, Any]:
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
    return _extract_json(raw)


async def bear_agent(state: Dict[str, Any]) -> Dict[str, Any]:
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
    return _extract_json(raw)


async def macro_agent(state: Dict[str, Any]) -> Dict[str, Any]:
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
    return _extract_json(raw)


async def moderator_agent(state: Dict[str, Any]) -> Dict[str, Any]:
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
    return _extract_json(raw)


# ---------------------------
# PIPELINE
# ---------------------------
async def run_pipeline_streaming(state: Dict[str, Any], emit):
    await emit("agent_start", {"agent": "research"})
    await emit("agent_start", {"agent": "financial"})
    await emit("agent_start", {"agent": "valuation"})
    await emit("agent_start", {"agent": "macro"})

    r, f, v, m = await asyncio.gather(
        research_agent(state),
        financial_agent(state),
        valuation_agent(state),
        macro_agent(state),
    )

    state["research"] = r
    state["financial"] = f
    state["valuation"] = v
    state["macro"] = m

    await emit("agent_done", {"agent": "research", "data": r})
    await emit("agent_done", {"agent": "financial", "data": f})
    await emit("agent_done", {"agent": "valuation", "data": v})
    await emit("agent_done", {"agent": "macro", "data": m})

    await emit("agent_start", {"agent": "bull"})
    await emit("agent_start", {"agent": "bear"})

    bull, bear = await asyncio.gather(
        bull_agent(state),
        bear_agent(state),
    )

    state["bull"] = bull
    state["bear"] = bear

    await emit("agent_done", {"agent": "bull", "data": bull})
    await emit("agent_done", {"agent": "bear", "data": bear})

    await emit("agent_start", {"agent": "moderator"})

    mod = await moderator_agent(state)
    state["moderator"] = mod

    await emit("agent_done", {"agent": "moderator", "data": mod})

    return state