import json
import math
import os
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional

import streamlit as st
import yfinance as yf
from duckduckgo_search import DDGS
from openai import OpenAI

try:
    import google.generativeai as genai
except ImportError:  # pragma: no cover - optional dependency
    genai = None


st.set_page_config(
    page_title="Mobile-First AI Investment Dashboard",
    layout="centered",
    page_icon="📈",
)


MOBILE_CSS = """
<style>
    .stApp {
        background-color: #0f1116;
        color: #f3f4f6;
        font-family: "Inter", "Noto Sans JP", sans-serif;
    }
    section.main > div {
        padding-left: 12px;
        padding-right: 12px;
    }
    .header-card {
        position: sticky;
        top: 0;
        z-index: 900;
        background: #111827;
        padding: 18px 16px;
        border-radius: 18px;
        border: 1px solid #1f2937;
        box-shadow: 0 15px 30px rgba(0,0,0,0.25);
        margin-bottom: 18px;
    }
    .header-symbol {
        font-size: 0.9rem;
        letter-spacing: 0.08em;
        color: #9ca3af;
        text-transform: uppercase;
    }
    .header-price {
        font-size: 2.2rem;
        font-weight: 600;
        color: #f9fafb;
    }
    .price-change {
        font-size: 1rem;
        margin-top: 4px;
    }
    .price-change.positive { color: #10b981; }
    .price-change.negative { color: #f87171; }
    .score-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
        margin-top: 16px;
    }
    .score-card {
        background: #0b1220;
        border-radius: 14px;
        padding: 12px;
        border: 1px solid #1f2937;
    }
    .score-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #9ca3af;
    }
    .score-value {
        font-size: 1.4rem;
        font-weight: 600;
        margin-top: 4px;
    }
    .conclusion-card {
        background: #111827;
        border-radius: 18px;
        padding: 20px 18px;
        border: 1px solid #1f2937;
        margin-bottom: 18px;
    }
    .action-pill {
        display: inline-flex;
        align-items: center;
        padding: 6px 13px;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .action-Buy { background: rgba(16,185,129,0.15); color: #10b981; }
    .action-Sell { background: rgba(248,113,113,0.15); color: #f87171; }
    .action-Hold { background: rgba(251,191,36,0.15); color: #fbbf24; }
    .bullet-list li {
        margin-bottom: 4px;
    }
    .news-item {
        padding: 10px 0;
        border-bottom: 1px solid #1f2937;
    }
    .news-item:last-child { border-bottom: none; }
    .news-title {
        font-weight: 600;
        color: #d1d5db;
    }
    .news-meta {
        font-size: 0.8rem;
        color: #9ca3af;
    }
    .tabs-container [data-baseweb="tab-list"] button {
        background: transparent;
        border: none;
        color: #9ca3af;
        font-weight: 600;
    }
    .tabs-container [aria-selected="true"] {
        color: #f3f4f6 !important;
        border-bottom: 2px solid #3b82f6;
    }
    .metric-stack {
        display: flex;
        flex-direction: column;
        gap: 8px;
    }
    .metric-row {
        display: flex;
        justify-content: space-between;
        font-size: 0.95rem;
        padding: 8px 0;
        border-bottom: 1px solid #1f2937;
    }
    .metric-row:last-child { border-bottom: none; }
    .disclaimer {
        font-size: 0.78rem;
        color: #6b7280;
        margin-top: 24px;
    }
      .api-status-panel {
          background: #111827;
          border-radius: 14px;
          padding: 16px;
          border: 1px solid #1f2937;
          margin-top: 18px;
      }
      .api-status-row {
          display: flex;
          justify-content: space-between;
          font-size: 0.9rem;
          padding: 6px 0;
          border-bottom: 1px solid #1f2937;
      }
      .api-status-row:last-child { border-bottom: none; }
      .api-status-value {
          font-weight: 600;
      }
      .api-status-value.active { color: #10b981; }
      .api-status-value.inactive { color: #f87171; }
    @media (min-width: 768px) {
        .header-card, .conclusion-card {
            margin-left: auto;
            margin-right: auto;
            max-width: 520px;
        }
        .score-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }
</style>
"""

st.markdown(MOBILE_CSS, unsafe_allow_html=True)


AI_SYSTEM_PROMPT = (
    "You are an equity strategist who writes concise Japanese summaries for busy executives. "
    "Use the provided market snapshot to compare quantitative signals with analyst consensus. "
    "Output JSON exactly with the requested schema."
)

GOOGLE_API_KEY_ENV_ORDER = [
    "GOOGLE_API_KEY",
    "GOOGLE_GENAI_API_KEY",
    "GENAI_API_KEY",
    "GEMINI_API_KEY",
]

DEFAULT_GEMINI_MODEL = (
    os.getenv("GOOGLE_GENAI_MODEL")
    or os.getenv("GEMINI_MODEL")
    or os.getenv("GEMINI_MODEL_NAME")
    or "gemini-2.5-flash-lite"
)

OPENAI_DEFAULT_MODEL = "gpt-4o-mini"


CHROME_PASSWORD_MANAGER_SCRIPT = """
<script>
(function attachChromePasswordHints() {
    const doc = window.parent?.document || window.document;
    const targets = [
        { selector: 'input[aria-label*="Gemini モデルID"]', name: 'gemini_model_id', autocomplete: 'username' },
        { selector: 'input[aria-label*="Google AI Studio API Key"]', name: 'google_ai_studio_api_key', autocomplete: 'current-password' },
        { selector: 'input[aria-label*="OpenAI API Key"]', name: 'openai_api_key', autocomplete: 'off' },
        { selector: 'input[aria-label*="ティッカーシンボル"]', name: 'ticker_symbol', autocomplete: 'off' },
    ];
    let attempts = 0;
    const maxAttempts = 20;
    const delay = 400;

    function applyAttributes() {
        let pending = false;
        targets.forEach((cfg) => {
            const input = doc.querySelector(cfg.selector);
            if (!input) {
                pending = true;
                return;
            }
            input.setAttribute('name', cfg.name);
            input.setAttribute('id', cfg.name);
            input.setAttribute('autocomplete', cfg.autocomplete);
            input.setAttribute('data-managed-by', 'chrome-password-manager');
        });
        if (pending && attempts < maxAttempts) {
            attempts += 1;
            setTimeout(applyAttributes, delay);
        }
    }

    if (doc.readyState === 'complete') {
        applyAttributes();
    } else {
        doc.addEventListener('readystatechange', () => {
            if (doc.readyState === 'complete') {
                applyAttributes();
            }
        });
    }
})();
</script>
"""


def enable_chrome_password_manager_support():
    """Inject autocomplete/name attributes so Chrome can store API keys & model IDs."""
    st.markdown(CHROME_PASSWORD_MANAGER_SCRIPT, unsafe_allow_html=True)


def build_api_status_snapshot(
    openai_key: str,
    google_key: str,
    model_name: str,
    applied_at: Optional[str] = None,
) -> Dict:
    """Summarize the current API設定 status for UI display."""
    return {
        "openai_ready": bool((openai_key or "").strip()),
        "google_ready": bool((google_key or "").strip()),
        "model_name": (model_name or "").strip(),
        "last_applied": applied_at or "未適用",
    }


def render_api_status_panel(status: Optional[Dict]):
    snapshot = status or {}
    openai_ready = snapshot.get("openai_ready", False)
    google_ready = snapshot.get("google_ready", False)
    model_name = snapshot.get("model_name") or "未指定"
    last_applied = snapshot.get("last_applied") or "未適用"

    st.markdown("### 🔑 API設定ステータス")
    status_html = f"""
    <div class="api-status-panel">
        <div class="api-status-row">
            <span>OpenAI API Key</span>
            <span class="api-status-value {'active' if openai_ready else 'inactive'}">
                {'設定済み' if openai_ready else '未設定'}
            </span>
        </div>
        <div class="api-status-row">
            <span>Google AI Studio API Key</span>
            <span class="api-status-value {'active' if google_ready else 'inactive'}">
                {'設定済み' if google_ready else '未設定'}
            </span>
        </div>
        <div class="api-status-row">
            <span>Gemini モデルID</span>
            <span class="api-status-value">{model_name or '未指定'}</span>
        </div>
        <div class="api-status-row">
            <span>最終適用</span>
            <span class="api-status-value">{last_applied}</span>
        </div>
    </div>
    """
    st.markdown(status_html, unsafe_allow_html=True)


def build_ai_user_prompt(payload: Dict) -> str:
    return (
        "マーケットデータ:\n"
        f"{json.dumps(payload, ensure_ascii=False)}\n\n"
        "出力フォーマット(JSON):\n"
        "{"
        '"verdict_short":"",'
        '"action":"Buy | Sell | Hold",'
        '"score":0,'
        '"bullet_points":["","", ""],'
        '"scenario":{"bullish_case":"","bearish_case":"","competitive_edge":""},'
        '"analysis_comment":""'
        "}"
    )


def resolve_google_api_key_from_env() -> str:
    for env_name in GOOGLE_API_KEY_ENV_ORDER:
        value = os.getenv(env_name)
        if value:
            return value
    return ""


def normalize_ticker_input(raw_symbol: str) -> Dict[str, str]:
    """Convert user input like '6501' to a resolvable yfinance symbol such as '6501.T'."""
    raw_symbol = (raw_symbol or "").strip()
    normalized = raw_symbol.upper().replace("Ｔ", "T").strip()
    normalized = re.sub(r"^(?:TYO|JPX|JP|TSE):", "", normalized)
    normalized = normalized.replace(" ", "")

    conversion_note = ""
    query_symbol = normalized
    display_symbol = normalized or raw_symbol

    if normalized.isdigit() and 4 <= len(normalized) <= 5:
        query_symbol = f"{normalized}.T"
        display_symbol = raw_symbol or query_symbol
        conversion_note = f"国内証券コード {raw_symbol or normalized} を {query_symbol} として取得しました。"

    return {
        "input_symbol": raw_symbol,
        "query_symbol": query_symbol,
        "display_symbol": display_symbol or query_symbol,
        "conversion_note": conversion_note,
    }


def format_currency(value: Optional[float], currency: str = "USD") -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    currency = (currency or "USD").upper()
    symbol_map = {
        "USD": "$",
        "JPY": "¥",
        "EUR": "€",
    }
    symbol = symbol_map.get(currency, "")
    decimals = 0 if currency == "JPY" else 2
    return f"{symbol}{value:,.{decimals}f}"


def format_percent(value: Optional[float]) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{value:.2f}%"


def safe_fast_info_get(fast_info, key: str):
    """yfinance fast_info sometimes raises KeyError when a field is missing."""
    if not fast_info:
        return None
    if isinstance(fast_info, dict):
        return fast_info.get(key)
    getter = getattr(fast_info, "get", None)
    if callable(getter):
        try:
            return getter(key)
        except KeyError:
            return None
        except Exception:
            pass
    try:
        return getattr(fast_info, key)
    except (AttributeError, KeyError):
        return None
    except Exception:
        return None


@st.cache_data(ttl=900, show_spinner=False)
def fetch_ticker_snapshot(symbol: str) -> Dict:
    symbol = symbol.upper().strip()
    if not symbol:
        return {"error": "ティッカーが指定されていません。"}
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}
        fast_info = getattr(ticker, "fast_info", {}) or {}
        hist = ticker.history(period="5d", interval="1d")
    except Exception as exc:  # pragma: no cover - network
        return {"error": f"データ取得に失敗しました: {exc}"}

    price = safe_fast_info_get(fast_info, "last_price") or info.get("currentPrice")
    if price is None and not hist.empty:
        price = float(hist["Close"].iloc[-1])

    prev_close = safe_fast_info_get(fast_info, "previous_close") or info.get("previousClose")
    if prev_close is None and not hist.empty and len(hist) > 1:
        prev_close = float(hist["Close"].iloc[-2])

    day_change = day_change_pct = None
    if price is not None and prev_close:
        day_change = price - prev_close
        day_change_pct = (day_change / prev_close) * 100 if prev_close else None

    currency = (
        safe_fast_info_get(fast_info, "currency")
        or info.get("currency")
        or info.get("financialCurrency")
        or "USD"
    )

    target_mean_price = info.get("targetMeanPrice")
    target_gap_pct = None
    if price and target_mean_price:
        target_gap_pct = ((target_mean_price - price) / price) * 100

    inst_pct = info.get("institutionPercent")
    if inst_pct is not None and inst_pct <= 1:
        inst_pct = inst_pct * 100

    key_metrics = {
        "trailingPE": info.get("trailingPE"),
        "forwardPE": info.get("forwardPE"),
        "pegRatio": info.get("pegRatio"),
        "priceToBook": info.get("priceToBook"),
        "trailingEps": info.get("trailingEps"),
        "dividendYield": info.get("dividendYield", 0) * 100 if info.get("dividendYield") else None,
        "beta": info.get("beta"),
        "marketCap": info.get("marketCap"),
    }

    analyst_snapshot = {
        "recommendation_key": info.get("recommendationKey"),
        "recommendation_mean": info.get("recommendationMean"),
        "opinion_count": info.get("numberOfAnalystOpinions"),
        "target_mean_price": target_mean_price,
        "target_gap_pct": target_gap_pct,
        "institutional_ownership_pct": inst_pct,
    }

    ts = safe_fast_info_get(fast_info, "last_price_time")
    if isinstance(ts, (int, float)):
        market_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    else:
        market_time = datetime.now(timezone.utc).isoformat()

    return {
        "error": None,
        "symbol": symbol,
        "company_name": info.get("longName") or info.get("shortName") or symbol,
        "price": price,
        "previous_close": prev_close,
        "day_change": day_change,
        "day_change_pct": day_change_pct,
        "currency": currency,
        "market_time": market_time,
        "info": info,
        "analyst": analyst_snapshot,
        "key_metrics": key_metrics,
    }


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_news(query: str, max_results: int = 5) -> List[Dict]:
    if not query:
        return []
    try:
        with DDGS() as ddgs:
            results = list(
                ddgs.news(
                    keywords=f"{query} stock",
                    region="us-en",
                    safesearch="Off",
                    max_results=max_results,
                )
            )
    except Exception:  # pragma: no cover - network
        return []

    news_items = []
    for item in results[:max_results]:
        news_items.append(
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "snippet": item.get("body") or item.get("snippet"),
                "published": item.get("date"),
                "source": item.get("source"),
            }
        )
    return news_items


def build_analysis_payload(snapshot: Dict, news_items: List[Dict]) -> Dict:
    symbol_for_payload = snapshot.get("resolved_symbol") or snapshot.get("symbol")
    return {
        "symbol": symbol_for_payload,
        "company_name": snapshot["company_name"],
        "currency": snapshot["currency"],
        "price": snapshot["price"],
        "day_change_pct": snapshot["day_change_pct"],
        "analyst": snapshot["analyst"],
        "metrics": snapshot["key_metrics"],
        "news": news_items,
        "timestamp": snapshot["market_time"],
    }


def heuristic_analysis(snapshot: Dict) -> Dict:
    analyst = snapshot["analyst"]
    target_gap = analyst.get("target_gap_pct") or 0
    day_move = snapshot.get("day_change_pct") or 0
    reco = (analyst.get("recommendation_key") or "").lower()

    score = 55
    score += max(min(target_gap * 0.6, 20), -20)
    score -= max(min(abs(day_move) * 0.3, 10), 0) * (1 if day_move < 0 else -0.5)

    sentiment_bonus = {
        "strong_buy": 12,
        "buy": 8,
        "hold": 0,
        "sell": -10,
        "strong_sell": -15,
    }.get(reco, 0)
    score += sentiment_bonus
    score = max(0, min(100, round(score)))

    if score >= 66:
        action = "Buy"
        verdict = "押し目買い好機"
    elif score <= 40:
        action = "Sell"
        verdict = "リスク回避を優先"
    else:
        action = "Hold"
        verdict = "中立：様子見"

    bullets = []
    if analyst.get("target_mean_price") and snapshot.get("price"):
        gap = analyst["target_gap_pct"]
        bullets.append(f"目標株価まで {format_percent(gap)} の余地")
    if reco:
        bullets.append(f"アナリスト評価: {reco.upper()}")
    if snapshot["key_metrics"].get("trailingPE"):
        bullets.append(f"PER {snapshot['key_metrics']['trailingPE']:.1f}倍で取引中")
    while len(bullets) < 3:
        bullets.append("市場ボラティリティに備えて分散を維持")

    scenario = {
        "bullish_case": "外部AIキー未設定のため、シンプル指標で強気シナリオを推定しています。",
        "bearish_case": "短期テクニカルの振れに注意しつつファンダ指標の確認が必要です。",
        "competitive_edge": "目標株価と機関投資家動向を主要な拠り所としています。",
    }

    return {
        "verdict_short": verdict,
        "action": action,
        "score": score,
        "bullet_points": bullets[:3],
        "scenario": scenario,
        "analysis_comment": "外部AIレスポンスを取得できなかったため統計ベースの暫定コメントです。",
        "source": "heuristic",
    }


def _sanitize_ai_response(parsed: Dict) -> Dict:
    parsed = parsed or {}
    parsed["bullet_points"] = (parsed.get("bullet_points") or [])[:3]
    return parsed


def parse_ai_json_payload(message: Optional[str]) -> Optional[Dict]:
    """Accept AI responses with code fences or extra text and extract JSON."""
    if not message:
        return None
    cleaned = message.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    brace_start = cleaned.find("{")
    brace_end = cleaned.rfind("}")
    if brace_start != -1 and brace_end != -1:
        cleaned = cleaned[brace_start : brace_end + 1]
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def request_openai_analysis(api_key: Optional[str], payload: Dict) -> Optional[Dict]:
    api_key_clean = (api_key or "").strip()
    if not api_key_clean:
        return None
    try:
        client = OpenAI(api_key=api_key_clean)
        response = client.chat.completions.create(
            model=OPENAI_DEFAULT_MODEL,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": AI_SYSTEM_PROMPT},
                {"role": "user", "content": build_ai_user_prompt(payload)},
            ],
        )
        message = response.choices[0].message.content if response.choices else None
        parsed = parse_ai_json_payload(message)
        if not parsed:
            return None
        parsed["source"] = "openai"
        return _sanitize_ai_response(parsed)
    except Exception as e:  # pragma: no cover - API failure
        st.error(f"OpenAI API呼び出しエラー: {str(e)}")
        return None


def request_gemini_analysis(
    api_key: Optional[str],
    payload: Dict,
    model_name: Optional[str],
) -> Optional[Dict]:
    if genai is None:
        return None
    api_key_clean = (api_key or "").strip()
    if not api_key_clean:
        return None
    model_id = (model_name or DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL
    try:
        genai.configure(api_key=api_key_clean)
        model = genai.GenerativeModel(
            model_name=model_id,
            system_instruction=AI_SYSTEM_PROMPT,
        )
        response = model.generate_content(
            contents=build_ai_user_prompt(payload),
            generation_config={
                "temperature": 0.2,
                "response_mime_type": "application/json",
            },
        )
        message = getattr(response, "text", None)
        if not message and getattr(response, "candidates", None):
            first_candidate = response.candidates[0]
            content = getattr(first_candidate, "content", None)
            parts = getattr(content, "parts", None)
            if parts:
                first_part = parts[0]
                message = getattr(first_part, "text", None) or getattr(first_part, "content", None)
        if not message:
            return None
        parsed = parse_ai_json_payload(message)
        if not parsed:
            return None
        parsed["source"] = "gemini"
        return _sanitize_ai_response(parsed)
    except Exception as e:  # pragma: no cover - API failure
        st.error(f"Gemini API呼び出しエラー: {str(e)}")
        return None


def generate_ai_analysis(
    openai_api_key: Optional[str],
    google_api_key: Optional[str],
    snapshot: Dict,
    news_items: List[Dict],
    google_model_name: Optional[str],
) -> Dict:
    payload = build_analysis_payload(snapshot, news_items)
    fallback = heuristic_analysis(snapshot)

    google_key_clean = (google_api_key or "").strip()
    if google_key_clean:
        google_response = request_gemini_analysis(google_key_clean, payload, google_model_name)
        if google_response:
            return google_response

    openai_key_clean = (openai_api_key or "").strip()
    if openai_key_clean:
        openai_response = request_openai_analysis(openai_key_clean, payload)
        if openai_response:
            return openai_response

    return fallback


def render_header(snapshot: Dict, analysis: Dict):
    price = snapshot.get("price")
    day_pct = snapshot.get("day_change_pct")
    day_abs = snapshot.get("day_change")
    currency = snapshot.get("currency", "USD")
    symbol_label = snapshot.get("display_symbol") or snapshot.get("symbol")

    day_class = "positive" if (day_pct or 0) >= 0 else "negative"
    day_text = (
        f"{format_currency(day_abs, currency)} ({format_percent(day_pct)})"
        if day_abs is not None
        else "—"
    )

    analyst = snapshot["analyst"]
    reco_key = analyst.get("recommendation_key")
    reco_mean = analyst.get("recommendation_mean")
    if reco_mean:
        converted = round(6 - float(reco_mean), 1)  # 1 (best) -> 5
        reco_text = f"{converted}/5"
    else:
        reco_text = "N/A"
    analyst_label = reco_key.upper() if reco_key else "N/A"

    header_html = f"""
    <div class="header-card">
        <div class="header-symbol">{symbol_label} · {snapshot['company_name']}</div>
        <div class="header-price">{format_currency(price, currency)}</div>
        <div class="price-change {day_class}">{day_text}</div>
        <div class="score-grid">
            <div class="score-card">
                <div class="score-label">AI 投資スコア</div>
                <div class="score-value">{analysis.get('score', 0)}/100</div>
            </div>
            <div class="score-card">
                <div class="score-label">アナリスト推奨</div>
                <div class="score-value">{analyst_label}<br/><span style="font-size:0.85rem;color:#9ca3af;">{reco_text}</span></div>
            </div>
        </div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)


def render_conclusion(analysis: Dict):
    action = analysis.get("action", "Hold")
    verdict = analysis.get("verdict_short", "情報不足")
    bullets = analysis.get("bullet_points", [])
    bullet_html = "".join(f"<li>{point}</li>" for point in bullets)
    conclusion_html = f"""
    <div class="conclusion-card">
        <div class="action-pill action-{action}">{action}</div>
        <h2 style="margin:12px 0 6px 0;">{verdict}</h2>
        <ul class="bullet-list">{bullet_html}</ul>
    </div>
    """
    st.markdown(conclusion_html, unsafe_allow_html=True)


def describe_analysis_source(analysis: Dict) -> str:
    source = (analysis or {}).get("source")
    mapping = {
        "gemini": "Gemini (Google AI Studio)",
        "openai": "OpenAI GPT",
        "heuristic": "ヒューリスティック（API未使用）",
    }
    return mapping.get(source, "ヒューリスティック（API未使用）")


def render_tabs(analysis: Dict, snapshot: Dict, news_items: List[Dict]):
    tabs = st.tabs(["シナリオ", "プロの評価", "データ / ニュース"])

    scenario = analysis.get("scenario", {})
    with tabs[0]:
        st.markdown("**Bullシナリオ**")
        st.write(scenario.get("bullish_case", "情報不足"))
        st.markdown("**Bearシナリオ**")
        st.write(scenario.get("bearish_case", "情報不足"))
        st.markdown("**競合優位性 / Moat**")
        st.write(scenario.get("competitive_edge", "情報不足"))

    analyst = snapshot["analyst"]
    with tabs[1]:
        st.markdown("**アナリストコンセンサス**")
        st.markdown(
            f"- 推奨: `{analyst.get('recommendation_key') or 'N/A'}`\n"
            f"- アナリスト数: {analyst.get('opinion_count') or '—'}名"
        )

        st.markdown("**目標株価ギャップ**")
        st.write(
            f"平均: {format_currency(analyst.get('target_mean_price'), snapshot['currency'])} "
            f"({format_percent(analyst.get('target_gap_pct'))})"
        )

        inst = analyst.get("institutional_ownership_pct")
        st.markdown(
            "**機関投資家保有比率**: "
            f"{format_percent(inst) if inst is not None else 'データなし'}"
        )
        st.markdown("**AIコメント vs プロ**")
        st.write(analysis.get("analysis_comment", "—"))

    metrics = snapshot["key_metrics"]
    with tabs[2]:
        st.markdown("**主要指標**")
        metric_lines = []
        pairs = [
            ("PER (TTM)", metrics.get("trailingPE")),
            ("PER (Forward)", metrics.get("forwardPE")),
            ("PEG", metrics.get("pegRatio")),
            ("PBR", metrics.get("priceToBook")),
            ("EPS (TTM)", metrics.get("trailingEps")),
            ("配当利回り", metrics.get("dividendYield")),
            ("Beta", metrics.get("beta")),
            ("時価総額", metrics.get("marketCap")),
        ]
        for label, value in pairs:
            if value is None:
                formatted = "—"
            elif "利回り" in label:
                formatted = format_percent(value)
            elif label == "時価総額":
                formatted = f"${value/1_000_000_000:,.1f}B"
            else:
                formatted = f"{value:,.2f}" if isinstance(value, (float, int)) else value
            metric_lines.append(f"- {label}: {formatted}")
        st.markdown("\n".join(metric_lines))

        st.markdown("**関連ニュース**")
        if not news_items:
            st.write("最新ニュースの取得に失敗しました。")
        for news in news_items:
            st.markdown(
                f'<div class="news-item"><a class="news-title" href="{news["url"]}" target="_blank">{news["title"]}</a>'
                f'<div class="news-meta">{news.get("source") or ""} · {news.get("published") or ""}</div>'
                f'<div class="news-body">{news.get("snippet") or ""}</div></div>',
                unsafe_allow_html=True,
            )


def main():
    st.title("📱 Mobile AI Investment Dashboard")
    st.caption("忙しいビジネスマン向けの即断支援ツール（学習目的のみ）")

    google_model_input = st.text_input(
        "Gemini モデルID",
        value=DEFAULT_GEMINI_MODEL,
        help="APIキーで有効なモデルID（例: gemini-1.5-flash）を指定。Chrome パスワード管理でIDとして保存されます。",
    )
    google_api_key_default = resolve_google_api_key_from_env()
    google_api_key = st.text_input(
        "Google AI Studio API Key（Gemini / 任意）",
        type="password",
        value=google_api_key_default,
        help="環境変数から自動入力されるほか、Chrome のパスワードマネージャーにパスワードとして保存できます。",
    )
    google_model_name = (google_model_input or "").strip() or DEFAULT_GEMINI_MODEL
    
    openai_api_key_default = os.getenv("OPENAI_API_KEY", "")
    openai_api_key = st.text_input(
        "OpenAI API Key（任意・ローカルで保持）",
        type="password",
        value=openai_api_key_default,
        help="APIキーはブラウザ内のみで使用され、Chrome のパスワードマネージャーに保存して自動入力できます。",
    )
    
    ticker_input = st.text_input("ティッカーシンボル", value="AAPL")

    if "effective_openai_api_key" not in st.session_state:
        st.session_state["effective_openai_api_key"] = openai_api_key_default.strip()
    if "effective_google_api_key" not in st.session_state:
        st.session_state["effective_google_api_key"] = google_api_key_default.strip()
    if "effective_gemini_model" not in st.session_state:
        st.session_state["effective_gemini_model"] = google_model_name
    if "api_status_snapshot" not in st.session_state:
        st.session_state["api_status_snapshot"] = build_api_status_snapshot(
            st.session_state["effective_openai_api_key"],
            st.session_state["effective_google_api_key"],
            st.session_state["effective_gemini_model"],
        )

    enable_chrome_password_manager_support()
    st.markdown("#### 🔄 APIキーの適用")
    st.caption("入力したキーをアプリに反映し、画面下部のステータス表示を更新します。")
    apply_api_keys = st.button(
        "APIキーを適用して画面下部を更新",
        type="primary",
        use_container_width=True,
        help="AI 分析で使用するキーを確定し、ステータスを最新化します。",
    )
    if apply_api_keys:
        applied_openai = openai_api_key.strip()
        applied_google = google_api_key.strip()
        st.session_state["effective_openai_api_key"] = applied_openai
        st.session_state["effective_google_api_key"] = applied_google
        st.session_state["effective_gemini_model"] = google_model_name
        applied_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        st.session_state["api_status_snapshot"] = build_api_status_snapshot(
            applied_openai,
            applied_google,
            google_model_name,
            applied_timestamp,
        )
        if applied_google:
            st.success(f"Google APIキーを適用しました（長さ: {len(applied_google)}文字）。次の分析で使用されます。")
        elif applied_openai:
            st.success(f"OpenAI APIキーを適用しました（長さ: {len(applied_openai)}文字）。次の分析で使用されます。")
        else:
            st.warning("APIキーが入力されていません。ヒューリスティック分析が使用されます。")

    effective_openai_key = st.session_state.get("effective_openai_api_key", "").strip()
    effective_google_key = st.session_state.get("effective_google_api_key", "").strip()
    effective_gemini_model = st.session_state.get("effective_gemini_model", google_model_name)

    if not ticker_input:
        st.info("分析したいティッカーを入力してください。")
        return

    normalized = normalize_ticker_input(ticker_input)
    query_symbol = normalized.get("query_symbol")
    if not query_symbol:
        st.error("ティッカーの形式を確認してください。")
        return

    with st.spinner("マーケットデータを取得中..."):
        snapshot = fetch_ticker_snapshot(query_symbol)

    if snapshot.get("error"):
        st.error(snapshot["error"])
        return

    snapshot["display_symbol"] = normalized.get("display_symbol") or snapshot.get("symbol")
    snapshot["input_symbol"] = normalized.get("input_symbol")
    snapshot["resolved_symbol"] = snapshot.get("symbol")
    if normalized.get("conversion_note"):
        st.caption(normalized["conversion_note"])

    news_items = fetch_news(snapshot["company_name"])
    
    # APIキーの状態を確認
    if effective_google_key:
        st.info(f"🔑 Google APIキーが設定されています（モデル: {effective_gemini_model}）。Gemini APIを使用して分析します。")
    elif effective_openai_key:
        st.info("🔑 OpenAI APIキーが設定されています。OpenAI APIを使用して分析します。")
    else:
        st.warning("⚠️ APIキーが設定されていません。ヒューリスティック分析を使用します。")
    
    with st.spinner("AIが分析中..."):
        analysis = generate_ai_analysis(
            effective_openai_key,
            effective_google_key,
            snapshot,
            news_items,
            effective_gemini_model,
        )

    render_header(snapshot, analysis)
    st.caption(f"AIエンジン出力: {describe_analysis_source(analysis)}")
    st.markdown("### ✅ 結論エリア")
    render_conclusion(analysis)
    st.markdown("### 📊 詳細エリア")
    render_tabs(analysis, snapshot, news_items)
    render_api_status_panel(st.session_state.get("api_status_snapshot"))

    st.markdown(
        "<p class='disclaimer'>* 本アプリは教育目的の情報提供ツールです。投資判断はご自身の責任で行ってください。</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
