"""
Data Provider Chain for AlphaLens.
- In-memory TTL cache shields us from rate-limit storms.
- Primary: FMP (stable). Fallbacks: Finnhub, Alpha Vantage, Yahoo Finance, GNews.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

FMP_KEY = os.environ.get("FMP_API_KEY", "")
FINNHUB_KEY = os.environ.get("FINNHUB_API_KEY", "")
AV_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
GNEWS_KEY = os.environ.get("GNEWS_API_KEY", "")

FMP = "https://financialmodelingprep.com/stable"
FINNHUB_BASE = "https://finnhub.io/api/v1"
AV_BASE = "https://www.alphavantage.co/query"
GNEWS_BASE = "https://gnews.io/api/v4"
YAHOO_BASE = "https://query1.finance.yahoo.com/v8/finance/chart"

TIMEOUT = httpx.Timeout(15.0, connect=8.0)

# ---------------- TTL cache ----------------
_CACHE: Dict[str, tuple[float, Any]] = {}
CACHE_TTL = 600  # 10 minutes


def _cache_key(url: str, params: Optional[Dict[str, Any]]) -> str:
    return url + ("?" + json.dumps(params, sort_keys=True) if params else "")


async def _get(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    ttl: int = CACHE_TTL,
) -> Optional[Any]:
    key = _cache_key(url, params)
    now = time.time()
    cached = _CACHE.get(key)
    if cached and now - cached[0] < ttl:
        return cached[1]
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            r = await client.get(url, params=params, headers=headers or {})
            if r.status_code == 200:
                data = r.json()
                _CACHE[key] = (now, data)
                return data
            logger.warning(f"GET {url} -> {r.status_code}: {r.text[:160]}")
    except Exception as exc:
        logger.warning(f"GET {url} failed: {exc}")
    return None


# ---------------- Search ----------------
def _is_tickerish(q: str) -> bool:
    return 1 <= len(q) <= 5 and q.replace(".", "").replace("-", "").isalnum()


def _normalize_fmp_results(rows: List[Dict[str, Any]], skip: set) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for d in rows or []:
        sym = d.get("symbol")
        if not sym or sym in skip:
            continue
        out.append({
            "ticker": sym,
            "name": d.get("name"),
            "exchange": d.get("exchange") or d.get("exchangeFullName"),
            "currency": d.get("currency"),
        })
    return out


async def search_ticker(query: str) -> List[Dict[str, Any]]:
    q = (query or "").strip()
    if not q:
        return []

    seeded: List[Dict[str, Any]] = []
    if _is_tickerish(q):
        prof = await get_profile(q.upper())
        if prof and prof.get("ticker"):
            seeded.append({
                "ticker": prof["ticker"],
                "name": prof.get("name") or prof["ticker"],
                "exchange": prof.get("exchange"),
                "currency": prof.get("currency") or "USD",
            })

    fmp = await _get(f"{FMP}/search-name", {"query": q, "limit": 10, "apikey": FMP_KEY})
    if isinstance(fmp, list) and fmp:
        seen = {x["ticker"] for x in seeded}
        return (seeded + _normalize_fmp_results(fmp, seen))[:8]
    if seeded:
        return seeded

    fh = await _get(f"{FINNHUB_BASE}/search", {"q": q, "token": FINNHUB_KEY})
    if isinstance(fh, dict) and fh.get("result"):
        return [
            {"ticker": r.get("symbol"), "name": r.get("description"),
             "exchange": "", "currency": "USD"}
            for r in fh["result"][:8] if r.get("symbol")
        ]
    return []


# ---------------- Quote ----------------
async def get_quote(ticker: str) -> Optional[Dict[str, Any]]:
    data = await _get(f"{FMP}/quote", {"symbol": ticker, "apikey": FMP_KEY})
    if isinstance(data, list) and data:
        q = data[0]
        return {
            "ticker": q.get("symbol"), "name": q.get("name"),
            "price": q.get("price"), "change": q.get("change"),
            "changesPercentage": q.get("changePercentage"),
            "dayLow": q.get("dayLow"), "dayHigh": q.get("dayHigh"),
            "yearLow": q.get("yearLow"), "yearHigh": q.get("yearHigh"),
            "marketCap": q.get("marketCap"), "volume": q.get("volume"),
            "avgVolume": q.get("priceAvg50"), "exchange": q.get("exchange"),
            "open": q.get("open"), "previousClose": q.get("previousClose"),
            "source": "fmp",
        }
    fh = await _get(f"{FINNHUB_BASE}/quote", {"symbol": ticker, "token": FINNHUB_KEY})
    # Also pull finnhub metric to enrich 52w range from fallback
    metric = None
    if fh and isinstance(fh, dict) and fh.get("c"):
        fhm = await _get(f"{FINNHUB_BASE}/stock/metric",
                         {"symbol": ticker, "metric": "all", "token": FINNHUB_KEY})
        metric = (fhm or {}).get("metric") or {}
        return {
            "ticker": ticker, "name": ticker, "price": fh.get("c"),
            "change": fh.get("d"), "changesPercentage": fh.get("dp"),
            "dayLow": fh.get("l"), "dayHigh": fh.get("h"),
            "yearLow": metric.get("52WeekLow"),
            "yearHigh": metric.get("52WeekHigh"),
            "marketCap": (metric.get("marketCapitalization") or 0) * 1_000_000 if metric.get("marketCapitalization") else None,
            "volume": None, "avgVolume": None, "exchange": "",
            "open": fh.get("o"), "previousClose": fh.get("pc"),
            "source": "finnhub",
        }
    return None


# ---------------- Profile ----------------
def _safe_float(v):
    try:
        return float(v) if v not in (None, "", "None") else None
    except (TypeError, ValueError):
        return None


def _safe_int(v):
    try:
        return int(v) if v not in (None, "", "None") else None
    except (TypeError, ValueError):
        return None


async def _av_overview(ticker: str) -> Dict[str, Any]:
    data = await _get(AV_BASE, {"function": "OVERVIEW", "symbol": ticker, "apikey": AV_KEY})
    if isinstance(data, dict) and data.get("Symbol"):
        return data
    return {}


def _merge_profile(base: Dict[str, Any], av: Dict[str, Any]) -> Dict[str, Any]:
    """Fill missing fields in `base` from Alpha Vantage OVERVIEW output."""
    if not av:
        return base
    out = dict(base)
    out.setdefault("name", av.get("Name"))
    if not out.get("description"):
        out["description"] = av.get("Description") or out.get("description")
    if not out.get("sector"):
        out["sector"] = (av.get("Sector") or "").title() or out.get("sector")
    if not out.get("industry"):
        out["industry"] = (av.get("Industry") or "").title() or out.get("industry")
    if not out.get("country"):
        out["country"] = av.get("Country")
    if not out.get("website"):
        out["website"] = av.get("OfficialSite")
    if not out.get("employees"):
        out["employees"] = _safe_int(av.get("FullTimeEmployees"))
    if not out.get("beta"):
        out["beta"] = _safe_float(av.get("Beta"))
    if not out.get("marketCap"):
        out["marketCap"] = _safe_float(av.get("MarketCapitalization"))
    if not out.get("currency"):
        out["currency"] = av.get("Currency")
    if not out.get("exchange"):
        out["exchange"] = av.get("Exchange")
    return out


def _profile_complete(p: Dict[str, Any]) -> bool:
    """A 'good enough' profile has CEO, employees, beta, and description."""
    return bool(p.get("ceo") and p.get("employees") and p.get("beta") and p.get("description"))


async def get_profile(ticker: str) -> Optional[Dict[str, Any]]:
    base: Dict[str, Any] = {}

    data = await _get(f"{FMP}/profile", {"symbol": ticker, "apikey": FMP_KEY})
    if isinstance(data, list) and data:
        p = data[0]
        base = {
            "ticker": p.get("symbol"), "name": p.get("companyName"),
            "sector": p.get("sector"), "industry": p.get("industry"),
            "ceo": p.get("ceo"), "country": p.get("country"),
            "website": p.get("website"), "description": p.get("description"),
            "employees": p.get("fullTimeEmployees"), "ipoDate": p.get("ipoDate"),
            "image": p.get("image"), "marketCap": p.get("marketCap"),
            "beta": p.get("beta"), "currency": p.get("currency"),
            "exchange": p.get("exchange"), "source": "fmp",
        }
    else:
        fh = await _get(f"{FINNHUB_BASE}/stock/profile2",
                        {"symbol": ticker, "token": FINNHUB_KEY})
        if isinstance(fh, dict) and fh.get("name"):
            base = {
                "ticker": ticker, "name": fh.get("name"),
                "sector": fh.get("finnhubIndustry"),
                "industry": fh.get("finnhubIndustry"),
                "ceo": "", "country": fh.get("country"),
                "website": fh.get("weburl"), "description": "",
                "employees": None, "ipoDate": fh.get("ipo"),
                "image": fh.get("logo"),
                "marketCap": (fh.get("marketCapitalization") or 0) * 1_000_000
                              if fh.get("marketCapitalization") else None,
                "beta": None, "currency": fh.get("currency"),
                "exchange": fh.get("exchange"), "source": "finnhub",
            }

    if not base:
        return None

    # Enrich any missing core fields (Beta, employees, description, …) from AV OVERVIEW.
    if not _profile_complete(base):
        av = await _av_overview(ticker)
        base = _merge_profile(base, av)

    return base


# ---------------- Financials ----------------
_REVENUE_CONCEPTS = (
    "us-gaap_Revenues",
    "us-gaap_RevenueFromContractWithCustomerExcludingAssessedTax",
    "us-gaap_RevenueFromContractWithCustomerIncludingAssessedTax",
    "us-gaap_SalesRevenueNet",
)
_NET_INCOME_CONCEPTS = ("us-gaap_NetIncomeLoss",)
_GROSS_PROFIT_CONCEPTS = ("us-gaap_GrossProfit",)
_OPERATING_INCOME_CONCEPTS = (
    "us-gaap_OperatingIncomeLoss",
)
_TOTAL_ASSETS_CONCEPTS = ("us-gaap_Assets",)
_TOTAL_LIABILITIES_CONCEPTS = ("us-gaap_Liabilities",)
_EQUITY_CONCEPTS = ("us-gaap_StockholdersEquity",)
_OPER_CF_CONCEPTS = ("us-gaap_NetCashProvidedByUsedInOperatingActivities",)
_CAPEX_CONCEPTS = ("us-gaap_PaymentsToAcquirePropertyPlantAndEquipment",)


def _pick(concepts_block: List[Dict[str, Any]], targets: tuple) -> Optional[float]:
    if not concepts_block:
        return None
    for row in concepts_block:
        if row.get("concept") in targets:
            return row.get("value")
    return None


async def _fmp_financials(ticker: str) -> Dict[str, Any]:
    income = await _get(f"{FMP}/income-statement",
                        {"symbol": ticker, "limit": 4, "apikey": FMP_KEY})
    balance = await _get(f"{FMP}/balance-sheet-statement",
                         {"symbol": ticker, "limit": 4, "apikey": FMP_KEY})
    cashflow = await _get(f"{FMP}/cash-flow-statement",
                          {"symbol": ticker, "limit": 4, "apikey": FMP_KEY})

    def slim(rows, fields):
        return [{k: r.get(k) for k in fields} for r in (rows or [])[:4]]

    if not isinstance(income, list) or not income:
        return {}
    return {
        "income": slim(income, ["date", "revenue", "grossProfit", "operatingIncome",
                                "netIncome", "eps", "ebitda"]),
        "balance": slim(balance, ["date", "totalAssets", "totalLiabilities",
                                  "totalStockholdersEquity", "totalDebt",
                                  "cashAndCashEquivalents", "totalCurrentAssets",
                                  "totalCurrentLiabilities"]),
        "cashflow": slim(cashflow, ["date", "operatingCashFlow", "capitalExpenditure",
                                    "freeCashFlow", "netIncome"]),
    }


async def _finnhub_financials(ticker: str) -> Dict[str, Any]:
    fh = await _get(f"{FINNHUB_BASE}/stock/financials-reported",
                    {"symbol": ticker, "freq": "annual", "token": FINNHUB_KEY})
    if not isinstance(fh, dict) or not fh.get("data"):
        return {}
    income: List[Dict[str, Any]] = []
    balance: List[Dict[str, Any]] = []
    cashflow: List[Dict[str, Any]] = []
    for filing in (fh.get("data") or [])[:4]:
        report = filing.get("report") or {}
        date = (filing.get("endDate") or "")[:10]
        ic = report.get("ic") or []
        bs = report.get("bs") or []
        cf = report.get("cf") or []
        rev = _pick(ic, _REVENUE_CONCEPTS)
        ni = _pick(ic, _NET_INCOME_CONCEPTS)
        income.append({
            "date": date,
            "revenue": rev,
            "grossProfit": _pick(ic, _GROSS_PROFIT_CONCEPTS),
            "operatingIncome": _pick(ic, _OPERATING_INCOME_CONCEPTS),
            "netIncome": ni,
            "eps": None, "ebitda": None,
        })
        balance.append({
            "date": date,
            "totalAssets": _pick(bs, _TOTAL_ASSETS_CONCEPTS),
            "totalLiabilities": _pick(bs, _TOTAL_LIABILITIES_CONCEPTS),
            "totalStockholdersEquity": _pick(bs, _EQUITY_CONCEPTS),
            "totalDebt": None, "cashAndCashEquivalents": None,
            "totalCurrentAssets": None, "totalCurrentLiabilities": None,
        })
        op_cf = _pick(cf, _OPER_CF_CONCEPTS)
        capex = _pick(cf, _CAPEX_CONCEPTS)
        fcf = (op_cf - capex) if (op_cf is not None and capex is not None) else None
        cashflow.append({
            "date": date,
            "operatingCashFlow": op_cf,
            "capitalExpenditure": -capex if capex is not None else None,
            "freeCashFlow": fcf,
            "netIncome": ni,
        })
    return {"income": income, "balance": balance, "cashflow": cashflow}


async def get_financials(ticker: str) -> Dict[str, Any]:
    primary = await _fmp_financials(ticker)
    if primary:
        return primary
    return await _finnhub_financials(ticker)


# ---------------- Ratios / Valuation ----------------
async def _fmp_ratios(ticker: str) -> Dict[str, Any]:
    ratios = await _get(f"{FMP}/ratios-ttm", {"symbol": ticker, "apikey": FMP_KEY})
    metrics = await _get(f"{FMP}/key-metrics-ttm", {"symbol": ticker, "apikey": FMP_KEY})
    dcf = await _get(f"{FMP}/discounted-cash-flow", {"symbol": ticker, "apikey": FMP_KEY})

    r = (ratios or [{}])[0] if isinstance(ratios, list) else {}
    m = (metrics or [{}])[0] if isinstance(metrics, list) else {}
    d = (dcf or [{}])[0] if isinstance(dcf, list) else {}

    pe = r.get("priceToEarningsRatioTTM") or m.get("peRatioTTM")
    if pe is None:
        return {}
    return {
        "peTTM": pe,
        "pegTTM": r.get("priceToEarningsGrowthRatioTTM"),
        "pbTTM": r.get("priceToBookRatioTTM"),
        "psTTM": r.get("priceToSalesRatioTTM"),
        "evToEbitda": m.get("evToEBITDATTM"),
        "roeTTM": r.get("returnOnEquityTTM"),
        "roaTTM": r.get("returnOnAssetsTTM"),
        "debtToEquityTTM": r.get("debtToEquityRatioTTM"),
        "currentRatioTTM": r.get("currentRatioTTM") or m.get("currentRatioTTM"),
        "grossMarginTTM": r.get("grossProfitMarginTTM"),
        "operatingMarginTTM": r.get("operatingProfitMarginTTM"),
        "netMarginTTM": r.get("netProfitMarginTTM"),
        "dividendYieldTTM": r.get("dividendYieldTTM"),
        "dcfFairValue": d.get("dcf"),
        "currentPrice": d.get("Stock Price"),
        "source": "fmp",
    }


async def _finnhub_ratios(ticker: str) -> Dict[str, Any]:
    fh = await _get(f"{FINNHUB_BASE}/stock/metric",
                    {"symbol": ticker, "metric": "all", "token": FINNHUB_KEY})
    if not isinstance(fh, dict):
        return {}
    m = fh.get("metric") or {}

    def pct(v):
        return (v / 100.0) if v is not None else None

    return {
        "peTTM": m.get("peTTM") or m.get("peNormalizedAnnual"),
        "pegTTM": m.get("pegRatioTTM") or m.get("pegRatio5Y"),
        "pbTTM": m.get("pbAnnual") or m.get("pbQuarterly"),
        "psTTM": m.get("psTTM"),
        "evToEbitda": m.get("evEbitdaTTM") or m.get("currentEv/freeCashFlowTTM"),
        "roeTTM": pct(m.get("roeTTM") or m.get("roeRfy")),
        "roaTTM": pct(m.get("roaTTM") or m.get("roaRfy")),
        "debtToEquityTTM": m.get("totalDebt/totalEquityAnnual"),
        "currentRatioTTM": m.get("currentRatioAnnual") or m.get("currentRatioQuarterly"),
        "grossMarginTTM": pct(m.get("grossMarginTTM") or m.get("grossMarginAnnual")),
        "operatingMarginTTM": pct(m.get("operatingMarginTTM") or m.get("operatingMarginAnnual")),
        "netMarginTTM": pct(m.get("netProfitMarginTTM") or m.get("netProfitMarginAnnual")),
        "dividendYieldTTM": pct(m.get("currentDividendYieldTTM")),
        "dcfFairValue": None,
        "currentPrice": None,
        "source": "finnhub",
    }


async def get_ratios(ticker: str) -> Dict[str, Any]:
    primary = await _fmp_ratios(ticker)
    if primary:
        return primary
    return await _finnhub_ratios(ticker)


# ---------------- Peers ----------------
async def get_peers(ticker: str) -> List[Dict[str, Any]]:
    data = await _get(f"{FMP}/stock-peers", {"symbol": ticker, "apikey": FMP_KEY})
    if isinstance(data, list) and data:
        return [
            {"ticker": p.get("symbol"), "name": p.get("companyName"),
             "price": p.get("price"), "marketCap": p.get("mktCap")}
            for p in data[:6]
        ]
    fh = await _get(f"{FINNHUB_BASE}/stock/peers",
                    {"symbol": ticker, "token": FINNHUB_KEY})
    if isinstance(fh, list) and fh:
        return [{"ticker": p, "name": p, "price": None, "marketCap": None}
                for p in fh if p and p != ticker][:6]
    return []


# ---------------- Analyst ratings ----------------
async def get_analyst(ticker: str) -> Optional[Dict[str, Any]]:
    out: Dict[str, Any] = {}

    pt = await _get(f"{FMP}/price-target-consensus",
                    {"symbol": ticker, "apikey": FMP_KEY})
    if isinstance(pt, list) and pt:
        p = pt[0]
        out.update({
            "targetHigh": p.get("targetHigh"),
            "targetLow": p.get("targetLow"),
            "targetConsensus": p.get("targetConsensus"),
            "targetMedian": p.get("targetMedian"),
        })

    rec = await _get(f"{FMP}/grades-consensus",
                     {"symbol": ticker, "apikey": FMP_KEY})
    if isinstance(rec, list) and rec:
        r0 = rec[0]
        out.update({k: r0.get(k) for k in
                    ("strongBuy", "buy", "hold", "sell", "strongSell", "consensus")})
    else:
        fh = await _get(f"{FINNHUB_BASE}/stock/recommendation",
                        {"symbol": ticker, "token": FINNHUB_KEY})
        if isinstance(fh, list) and fh:
            r0 = fh[0]
            out.update({k: r0.get(k) for k in
                        ("strongBuy", "buy", "hold", "sell", "strongSell")})

    return out or None


# ---------------- News ----------------
async def get_news(ticker: str, name: Optional[str] = None) -> List[Dict[str, Any]]:
    q = name or ticker
    g = await _get(f"{GNEWS_BASE}/search",
                   {"q": q, "lang": "en", "max": 10, "token": GNEWS_KEY})
    items: List[Dict[str, Any]] = []
    if isinstance(g, dict) and g.get("articles"):
        for n in g["articles"][:10]:
            items.append({
                "title": n.get("title"),
                "url": n.get("url"),
                "site": (n.get("source") or {}).get("name"),
                "publishedAt": n.get("publishedAt"),
                "image": n.get("image"),
                "snippet": n.get("description") or "",
                "source": "gnews",
            })
        return items
    fh = await _get(f"{FINNHUB_BASE}/company-news", {
        "symbol": ticker, "from": "2024-01-01", "to": "2030-12-31",
        "token": FINNHUB_KEY,
    })
    if isinstance(fh, list):
        for n in fh[:10]:
            items.append({
                "title": n.get("headline"), "url": n.get("url"),
                "site": n.get("source"), "publishedAt": n.get("datetime"),
                "image": n.get("image"), "snippet": n.get("summary") or "",
                "source": "finnhub",
            })
    return items


# ---------------- Historical price ----------------
async def _fmp_history(ticker: str) -> List[Dict[str, Any]]:
    data = await _get(f"{FMP}/historical-price-eod/light",
                      {"symbol": ticker, "apikey": FMP_KEY})
    if isinstance(data, list) and data:
        rows = list(reversed(data[:180]))
        return [{"date": r.get("date"), "close": r.get("price")} for r in rows]
    return []


async def _yahoo_history(ticker: str) -> List[Dict[str, Any]]:
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
    data = await _get(f"{YAHOO_BASE}/{ticker}",
                      {"interval": "1d", "range": "6mo"}, headers=headers)
    try:
        result = ((data or {}).get("chart") or {}).get("result") or []
        if not result:
            return []
        r0 = result[0]
        ts = r0.get("timestamp") or []
        closes = (((r0.get("indicators") or {}).get("quote") or [{}])[0]).get("close") or []
        out: List[Dict[str, Any]] = []
        for t, c in zip(ts, closes):
            if c is None:
                continue
            out.append({
                "date": time.strftime("%Y-%m-%d", time.gmtime(t)),
                "close": float(c),
            })
        return out[-180:]
    except Exception as exc:
        logger.warning(f"yahoo parse failed for {ticker}: {exc}")
        return []


async def get_history(ticker: str) -> List[Dict[str, Any]]:
    points = await _fmp_history(ticker)
    if points:
        return points
    return await _yahoo_history(ticker)
