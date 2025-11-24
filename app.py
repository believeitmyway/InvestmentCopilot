import json
import logging
import math
import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import streamlit as st
import yfinance as yf
try:
    from ddgs import DDGS
except ImportError:
    # 後方互換性のため、古いパッケージ名も試行
    from duckduckgo_search import DDGS
from openai import OpenAI
import plotly.graph_objects as go
import plotly.express as px

try:
    import requests
    from bs4 import BeautifulSoup
    SCRAPING_AVAILABLE = True
except ImportError:
    SCRAPING_AVAILABLE = False

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


def load_prompt_file(filename: str, default: str = "") -> str:
    """プロンプトファイルを読み込む"""
    prompt_dir = os.path.join(os.path.dirname(__file__), "prompts")
    filepath = os.path.join(prompt_dir, filename)
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"警告: プロンプトファイルが見つかりません: {filepath}。デフォルト値を使用します。")
        return default
    except Exception as e:
        print(f"警告: プロンプトファイルの読み込みエラー ({filepath}): {e}。デフォルト値を使用します。")
        return default


def load_system_prompt() -> str:
    """システムプロンプトを読み込む"""
    default = (
        "You are an equity strategist who writes concise Japanese summaries for busy executives. "
        "Use the provided market snapshot to compare quantitative signals with analyst consensus. "
        "Output JSON exactly with the requested schema."
    )
    return load_prompt_file("system_prompt.txt", default)


def load_user_prompt_template() -> str:
    """ユーザープロンプトテンプレートを読み込む"""
    default = (
        "マーケットデータ:\n"
        "{market_data}\n\n"
        "出力フォーマット(JSON):\n"
        "{\n"
        '"verdict_short":"",\n'
        '"action":"Buy | Sell | Hold",\n'
        '"score":0,\n'
        '"bullet_points":["","", ""],\n'
        '"scenario":{"bullish_case":"","bearish_case":"","competitive_edge":""},\n'
        '"analysis_comment":""\n'
        "}"
    )
    return load_prompt_file("user_prompt_template.txt", default)


def load_news_search_config() -> Dict:
    """ニュース検索設定ファイルを読み込む"""
    config_dir = os.path.join(os.path.dirname(__file__), "config")
    filepath = os.path.join(config_dir, "news_search_config.json")
    
    default_config = {
        "search": {
            "max_results": 15,
            "min_required_results": 5,
            "max_retries": 3,
            "retry_delay_seconds": 2,
            "multipliers": {
                "initial_japanese": 8,
                "fallback_japanese": 4,
                "english": 5
            },
            "min_candidates": {
                "initial_japanese": 50,
                "fallback_japanese": 30,
                "english": 30
            },
            "timeout": 30,
            "article_fetch_timeout": 15
        },
        "keywords": {
            "japanese_search_templates": [
                "{company_name} 決算 業績",
                "{company_name} 決算発表",
                "{company_name} 業績発表",
                "{company_name} IR 投資家向け説明会",
                "{company_name} 株主総会",
                "{company_name} M&A 買収 合併",
                "{company_name} 大型投資 戦略発表",
                "{company_name} 株価 ニュース",
                "{company_name} 株 最新",
                "{company_name} 企業 ニュース",
                "{company_name} 最新ニュース"
            ],
            "japanese_symbol_templates": [
                "{symbol} 株価",
                "{symbol} ニュース",
                "{symbol} 決算",
                "{symbol} 業績"
            ],
            "japanese_combined_templates": [
                "{symbol} {company_name}",
                "{company_name} {symbol}"
            ],
            "english_search_templates": [
                "{query} earnings results",
                "{query} quarterly results",
                "{query} financial results",
                "{query} acquisition merger",
                "{query} strategic announcement",
                "{query} stock news",
                "{query} stock"
            ]
        },
        "scoring": {
            "focus_score": {
                "company_name_in_title": 10,
                "company_name_in_snippet": 5,
                "company_name_count_multiplier": 2,
                "company_name_count_max": 10,
                "symbol_in_title": 8,
                "symbol_in_snippet": 4,
                "symbol_count_multiplier": 2,
                "symbol_count_max": 8,
                "query_in_title": 6,
                "query_in_snippet": 3,
                "deep_analysis_bonus": 2
            },
            "importance_score": {
                "keyword_score": 2
            }
        },
        "keywords_for_scoring": {
            "shallow_article": {
                "japanese": [
                    "ランキング", "トップ", "上位", "ベスト", "ワースト",
                    "市場動向", "相場概況", "市況", "マーケットサマリー",
                    "株価ランキング", "上昇ランキング", "下落ランキング",
                    "注目銘柄", "人気銘柄", "急騰銘柄", "急落銘柄",
                    "日経平均", "TOPIX", "ダウ平均", "ナスダック",
                    "市場総括", "相場総括", "市況レポート",
                    "複数銘柄", "多数銘柄", "各銘柄", "各社"
                ],
                "english": [
                    "ranking", "top", "best", "worst", "list",
                    "market overview", "market summary", "market wrap",
                    "stock ranking", "gainers", "losers", "most active",
                    "market movers", "market recap", "daily wrap",
                    "multiple stocks", "several stocks", "various stocks"
                ]
            },
            "important": {
                "japanese": [
                    "決算", "業績", "業績発表", "決算発表", "決算説明会",
                    "ir", "投資家向け説明会", "株主総会",
                    "m&a", "買収", "合併", "統合", "提携",
                    "大型投資", "戦略発表", "経営方針", "中期経営計画",
                    "上場", "ipo", "増資", "減資", "配当",
                    "不祥事", "コンプライアンス", "リコール"
                ],
                "english": [
                    "earnings", "quarterly", "annual", "results", "financial results",
                    "acquisition", "merger", "m&a", "partnership",
                    "ipo", "dividend", "buyback", "strategic",
                    "recall", "scandal", "compliance"
                ]
            },
            "deep_analysis": {
                "japanese": [
                    "戦略", "経営方針", "中期経営計画", "事業戦略",
                    "業績分析", "財務分析", "投資判断", "投資評価",
                    "競争力", "競合分析", "市場シェア", "事業展開",
                    "IR説明会", "決算説明会", "投資家説明会"
                ],
                "english": [
                    "strategy", "business plan", "financial analysis",
                    "investment thesis", "competitive", "market share",
                    "earnings call", "investor day", "analyst meeting"
                ]
            }
        },
        "filtering": {
            "date_threshold_days": 365,
            "shallow_article": {
                "min_stock_codes": 3
            },
            "focus_score": {
                "min_focus_score": 0,
                "min_importance_score_when_focus_zero": 4
            },
            "fallback_sufficient_threshold_multiplier": 2
        }
    }
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            loaded_config = json.load(f)
            # デフォルト設定と深いマージ（ファイルにない項目はデフォルトを使用）
            def deep_merge(default: Dict, loaded: Dict) -> Dict:
                """深いマージを行う（ネストされた辞書もマージ）"""
                result = default.copy()
                for key, value in loaded.items():
                    if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                        result[key] = deep_merge(result[key], value)
                    else:
                        result[key] = value
                return result
            return deep_merge(default_config, loaded_config)
    except FileNotFoundError:
        print(f"警告: 設定ファイルが見つかりません: {filepath}。デフォルト値を使用します。")
        return default_config
    except json.JSONDecodeError as e:
        print(f"警告: 設定ファイルのJSON解析エラー ({filepath}): {e}。デフォルト値を使用します。")
        return default_config
    except Exception as e:
        print(f"警告: 設定ファイルの読み込みエラー ({filepath}): {e}。デフォルト値を使用します。")
        return default_config


AI_SYSTEM_PROMPT = load_system_prompt()
USER_PROMPT_TEMPLATE = load_user_prompt_template()
NEWS_SEARCH_CONFIG = load_news_search_config()

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
    """ユーザープロンプトを構築する"""
    market_data_json = json.dumps(payload, ensure_ascii=False)
    
    # ニュース検索結果をテキストにまとめる
    news_items = payload.get("news", [])
    news_text = ""
    if news_items:
        for n in news_items:
            title = n.get("title", "")
            snippet = n.get("snippet") or n.get("body", "")
            # タイトルと本文（snippet）を結合
            news_text += f"- Title: {title}\n  Snippet: {snippet}\n"
    else:
        news_text = "（最新ニュース情報は取得できませんでした）"
    
    # プロンプトの {market_data} と {news_context} に流し込む
    return USER_PROMPT_TEMPLATE.format(
        market_data=market_data_json,
        news_context=news_text
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
        try:
            last_close = hist["Close"].iloc[-1]
            if not (isinstance(last_close, float) and math.isnan(last_close)):
                price = float(last_close)
        except (IndexError, KeyError, ValueError, TypeError):
            pass

    prev_close = safe_fast_info_get(fast_info, "previous_close") or info.get("previousClose")
    if prev_close is None and not hist.empty and len(hist) > 1:
        try:
            prev_close_value = hist["Close"].iloc[-2]
            if not (isinstance(prev_close_value, float) and math.isnan(prev_close_value)):
                prev_close = float(prev_close_value)
        except (IndexError, KeyError, ValueError, TypeError):
            pass

    day_change = day_change_pct = None
    if price is not None and prev_close is not None and prev_close != 0:
        day_change = price - prev_close
        day_change_pct = (day_change / prev_close) * 100

    currency = (
        safe_fast_info_get(fast_info, "currency")
        or info.get("currency")
        or info.get("financialCurrency")
        or "USD"
    )

    target_mean_price = info.get("targetMeanPrice")
    target_gap_pct = None
    if price is not None and price != 0 and target_mean_price is not None:
        target_gap_pct = ((target_mean_price - price) / price) * 100

    inst_pct = info.get("institutionPercent")
    # 機関投資家保有比率: 0-1の範囲の小数（例：0.75 = 75%）の場合は100を掛けてパーセンテージに変換
    if inst_pct is not None:
        if 0 <= inst_pct <= 1:
            inst_pct = inst_pct * 100
        # 既にパーセンテージ形式（1より大きい）の場合はそのまま使用
        elif inst_pct < 0:
            inst_pct = None  # 負の値は無効

    # 配当利回り: yfinanceは既にパーセンテージ形式で返す（例：0.95 = 0.95%）
    dividend_yield_raw = info.get("dividendYield")
    dividend_yield_pct = None
    if dividend_yield_raw is not None:
        try:
            dividend_yield_float = float(dividend_yield_raw)
            # 負の値は無効
            if dividend_yield_float < 0:
                dividend_yield_pct = None
            # 100を超える場合は異常値の可能性があるため、無視
            elif dividend_yield_float > 100:
                dividend_yield_pct = None  # 異常値として無視
            else:
                # そのまま使用（既にパーセンテージ形式）
                dividend_yield_pct = dividend_yield_float
        except (ValueError, TypeError):
            dividend_yield_pct = None

    def safe_get_metric(key: str):
        """安全に指標を取得し、NaNや無効な値をNoneに変換"""
        value = info.get(key)
        if value is None:
            return None
        if isinstance(value, float) and math.isnan(value):
            return None
        if isinstance(value, (int, float)) and math.isinf(value):
            # 無限大は無効
            return None
        # 負の値が有効な指標（EPS、Betaなど）
        if key in ["trailingEps", "beta"]:
            return value
        # その他の指標で負の値は無効
        if isinstance(value, (int, float)) and value < 0:
            return None
        return value

    key_metrics = {
        "trailingPE": safe_get_metric("trailingPE"),
        "forwardPE": safe_get_metric("forwardPE"),
        "pegRatio": safe_get_metric("pegRatio"),
        "priceToBook": safe_get_metric("priceToBook"),
        "trailingEps": safe_get_metric("trailingEps"),  # EPSは負の値も有効
        "dividendYield": dividend_yield_pct,
        "beta": safe_get_metric("beta"),
        "marketCap": safe_get_metric("marketCap"),
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

    # 日本語名を取得（日本株の場合）
    company_name = info.get("longName") or info.get("shortName") or symbol
    symbol_clean = symbol.replace(".T", "").strip()
    if symbol_clean.isdigit():
        # 日本株の場合、日本語名を優先的に使用
        japanese_name = get_japanese_company_name_cached(symbol, info)
        if japanese_name:
            company_name = japanese_name

    return {
        "error": None,
        "symbol": symbol,
        "company_name": company_name,
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


@st.cache_data(ttl=300, show_spinner=False)
def fetch_stock_history(symbol: str, period: str = "1mo") -> Optional[Dict]:
    """株価の時系列データを取得する"""
    symbol = symbol.upper().strip()
    if not symbol:
        return {"error": "ティッカーが指定されていません。"}
    
    try:
        ticker = yf.Ticker(symbol)
        # period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
        # interval: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
        hist = ticker.history(period=period)
        
        if hist.empty:
            return {"error": "データが取得できませんでした。"}
        
        # データを辞書形式に変換
        data = {
            "dates": hist.index.tolist(),
            "open": hist["Open"].tolist(),
            "high": hist["High"].tolist(),
            "low": hist["Low"].tolist(),
            "close": hist["Close"].tolist(),
            "volume": hist["Volume"].tolist(),
        }
        
        return {"error": None, "data": data, "symbol": symbol}
    except Exception as exc:
        return {"error": f"データ取得に失敗しました: {exc}"}


def create_stock_chart(history_data: Dict, symbol: str, currency: str = "USD") -> go.Figure:
    """株価の時系列グラフを作成する（Plotly）"""
    from plotly.subplots import make_subplots
    
    if history_data.get("error") or not history_data.get("data"):
        fig = go.Figure()
        fig.add_annotation(
            text="データが取得できませんでした",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        return fig
    
    data = history_data["data"]
    dates = data["dates"]
    closes = data["close"]
    volumes = data["volume"]
    
    # サブプロットを作成（価格チャートと出来高チャート）
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
        subplot_titles=("株価", "出来高"),
    )
    
    # ローソク足
    fig.add_trace(
        go.Candlestick(
            x=dates,
            open=data["open"],
            high=data["high"],
            low=data["low"],
            close=data["close"],
            name="価格",
            increasing_line_color="#10b981",
            decreasing_line_color="#f87171",
        ),
        row=1, col=1
    )
    
    # 出来高
    colors = ["#10b981" if closes[i] >= data["open"][i] else "#f87171" 
              for i in range(len(dates))]
    fig.add_trace(
        go.Bar(
            x=dates,
            y=volumes,
            name="出来高",
            marker_color=colors,
            opacity=0.6,
        ),
        row=2, col=1
    )
    
    # レイアウト設定
    currency_symbol = "¥" if currency == "JPY" else "$"
    fig.update_layout(
        title=f"{symbol} 株価チャート",
        xaxis_title="日付",
        yaxis_title=f"価格 ({currency_symbol})",
        yaxis2_title="出来高",
        height=600,
        template="plotly_dark",
        hovermode="x unified",
        showlegend=False,
        xaxis_rangeslider_visible=False,
    )
    
    # グラフの背景色をダークテーマに合わせる
    fig.update_layout(
        plot_bgcolor="#0f1116",
        paper_bgcolor="#0f1116",
        font_color="#f3f4f6",
    )
    
    return fig


def get_yahoo_finance_url(symbol: str) -> str:
    """Yahoo FinanceのURLを生成"""
    symbol_clean = symbol.replace(".T", "")
    if symbol_clean.isdigit():
        # 日本株の場合
        return f"https://finance.yahoo.co.jp/quote/{symbol_clean}.T"
    else:
        # 海外株の場合
        return f"https://finance.yahoo.com/quote/{symbol}"


def parse_news_date(date_str: Optional[str]) -> Optional[datetime]:
    """ニュースの日付文字列をパースしてdatetimeオブジェクトに変換"""
    if not date_str:
        return None
    
    # 様々な日付形式に対応
    date_formats = [
        "%Y-%m-%dT%H:%M:%S%z",  # ISO形式（タイムゾーン付き）
        "%Y-%m-%dT%H:%M:%S",     # ISO形式（タイムゾーンなし）
        "%Y-%m-%d %H:%M:%S",     # 標準形式
        "%Y-%m-%d",              # 日付のみ
        "%d %b %Y",              # "01 Jan 2024"
        "%d %B %Y",              # "01 January 2024"
        "%Y年%m月%d日",           # 日本語形式
    ]
    
    for fmt in date_formats:
        try:
            return datetime.strptime(date_str, fmt)
        except (ValueError, TypeError):
            continue
    
    # 相対的な日付表現（例："2時間前"、"3日前"）の処理
    if isinstance(date_str, str):
        date_str_lower = date_str.lower()
        now = datetime.now(timezone.utc)
        
        # "時間前"、"日前"などの相対表現を処理
        import re
        hours_match = re.search(r'(\d+)\s*時間前', date_str_lower)
        if hours_match:
            hours = int(hours_match.group(1))
            return now - timedelta(hours=hours)
        
        days_match = re.search(r'(\d+)\s*日前', date_str_lower)
        if days_match:
            days = int(days_match.group(1))
            return now - timedelta(days=days)
    
    return None


def filter_recent_news(news_items: List[Dict], days_threshold: int = 30) -> List[Dict]:
    """指定日数以内のニュースのみをフィルタリング"""
    if not news_items:
        return []
    
    threshold_date = datetime.now(timezone.utc) - timedelta(days=days_threshold)
    filtered = []
    
    for item in news_items:
        date_str = item.get("published")
        if not date_str:
            # 日付情報がない場合は含める（最新の可能性がある）
            filtered.append(item)
            continue
        
        parsed_date = parse_news_date(date_str)
        if parsed_date:
            # タイムゾーン情報がない場合はUTCと仮定
            if parsed_date.tzinfo is None:
                parsed_date = parsed_date.replace(tzinfo=timezone.utc)
            
            if parsed_date >= threshold_date:
                filtered.append(item)
        else:
            # パースできない場合は含める（最新の可能性がある）
            filtered.append(item)
    
    return filtered


def is_shallow_article(item: Dict, company_name: Optional[str] = None, symbol: Optional[str] = None) -> bool:
    """ランキングや市場動向のような薄い記事かを判定"""
    config = NEWS_SEARCH_CONFIG.get("keywords_for_scoring", {}).get("shallow_article", {})
    shallow_keywords_ja = config.get("japanese", [])
    shallow_keywords_en = config.get("english", [])
    
    filtering_config = NEWS_SEARCH_CONFIG.get("filtering", {}).get("shallow_article", {})
    min_stock_codes = filtering_config.get("min_stock_codes", 3)
    
    title = (item.get("title") or "").lower()
    snippet = (item.get("snippet") or "").lower()
    text = f"{title} {snippet}"
    
    # タイトルにランキングや市場動向のキーワードが含まれているか
    for keyword in shallow_keywords_ja + shallow_keywords_en:
        if keyword in text:
            # ただし、対象銘柄名がタイトルに含まれている場合は除外（対象銘柄に焦点を当てたランキング記事の可能性）
            if company_name and company_name.lower() in title:
                continue
            if symbol and symbol.replace(".T", "").strip().lower() in title:
                continue
            return True
    
    # 複数の銘柄コードが含まれている場合（設定値以上）は薄い記事の可能性が高い
    if symbol:
        symbol_clean = symbol.replace(".T", "").strip()
        if symbol_clean.isdigit():
            # 4桁の数字（銘柄コード）が設定値以上含まれているか
            stock_codes = re.findall(r'\b\d{4}\b', text)
            if len(stock_codes) >= min_stock_codes:
                # 対象銘柄が含まれていても、他の銘柄が多く含まれている場合は薄い記事
                if symbol_clean not in stock_codes:
                    return True
                # 対象銘柄が含まれていても、設定値以上の銘柄が含まれている場合は市場動向記事の可能性が高い
                if len(set(stock_codes)) >= min_stock_codes:
                    return True
    
    return False


def calculate_focus_score(item: Dict, company_name: Optional[str] = None, symbol: Optional[str] = None, query: Optional[str] = None) -> int:
    """対象銘柄への焦点度をスコア化（高いほど対象銘柄に焦点を当てている）"""
    scoring_config = NEWS_SEARCH_CONFIG.get("scoring", {}).get("focus_score", {})
    keywords_config = NEWS_SEARCH_CONFIG.get("keywords_for_scoring", {}).get("deep_analysis", {})
    deep_analysis_keywords_ja = keywords_config.get("japanese", [])
    deep_analysis_keywords_en = keywords_config.get("english", [])
    
    title = (item.get("title") or "").lower()
    snippet = (item.get("snippet") or "").lower()
    text = f"{title} {snippet}"
    
    score = 0
    
    # 対象銘柄名がタイトルに含まれている場合は高スコア
    if company_name:
        company_name_lower = company_name.lower()
        if company_name_lower in title:
            score += scoring_config.get("company_name_in_title", 10)
        if company_name_lower in snippet:
            score += scoring_config.get("company_name_in_snippet", 5)
        
        # 対象銘柄名の出現回数をカウント
        count = text.count(company_name_lower)
        multiplier = scoring_config.get("company_name_count_multiplier", 2)
        max_score = scoring_config.get("company_name_count_max", 10)
        score += min(count * multiplier, max_score)
    
    # ティッカーシンボルが含まれている場合もスコア加算
    if symbol:
        symbol_clean = symbol.replace(".T", "").strip().lower()
        if symbol_clean in title:
            score += scoring_config.get("symbol_in_title", 8)
        if symbol_clean in snippet:
            score += scoring_config.get("symbol_in_snippet", 4)
        
        # ティッカーシンボルの出現回数をカウント
        count = text.count(symbol_clean)
        multiplier = scoring_config.get("symbol_count_multiplier", 2)
        max_score = scoring_config.get("symbol_count_max", 8)
        score += min(count * multiplier, max_score)
    
    # クエリ（英語の社名など）が含まれている場合もスコア加算
    if query:
        query_lower = query.lower()
        if query_lower in title:
            score += scoring_config.get("query_in_title", 6)
        if query_lower in snippet:
            score += scoring_config.get("query_in_snippet", 3)
    
    # 深い分析を示すキーワードが含まれている場合はボーナス
    bonus = scoring_config.get("deep_analysis_bonus", 2)
    for keyword in deep_analysis_keywords_ja + deep_analysis_keywords_en:
        if keyword in text:
            score += bonus
    
    return score


def calculate_news_importance_score(item: Dict) -> int:
    """ニュースの重要度スコアを計算（重要キーワードが含まれているか）"""
    keywords_config = NEWS_SEARCH_CONFIG.get("keywords_for_scoring", {}).get("important", {})
    important_keywords_ja = keywords_config.get("japanese", [])
    important_keywords_en = keywords_config.get("english", [])
    
    scoring_config = NEWS_SEARCH_CONFIG.get("scoring", {}).get("importance_score", {})
    keyword_score = scoring_config.get("keyword_score", 2)
    
    title = (item.get("title") or "").lower()
    snippet = (item.get("snippet") or "").lower()
    text = f"{title} {snippet}"
    
    score = 0
    for keyword in important_keywords_ja + important_keywords_en:
        if keyword in text:
            score += keyword_score  # 重要キーワードが見つかったらスコアを加算
    
    return score


def sort_news_by_importance_and_date(news_items: List[Dict], reverse: bool = True, company_name: Optional[str] = None, symbol: Optional[str] = None, query: Optional[str] = None) -> List[Dict]:
    """ニュースを重要度、焦点度、日付でソート（重要度と焦点度が高い順、同じなら新しい順）"""
    def get_sort_key(item: Dict) -> tuple:
        # 重要度スコア（高い方が優先）
        importance_score = calculate_news_importance_score(item)
        
        # 焦点度スコア（対象銘柄に焦点を当てているほど高い）
        focus_score = calculate_focus_score(item, company_name, symbol, query)
        
        # 日付
        date_str = item.get("published")
        if not date_str:
            # 日付がない場合は最も古い日付として扱う
            parsed_date = datetime(1970, 1, 1, tzinfo=timezone.utc)
        else:
            parsed_date = parse_news_date(date_str)
            if parsed_date:
                if parsed_date.tzinfo is None:
                    parsed_date = parsed_date.replace(tzinfo=timezone.utc)
            else:
                parsed_date = datetime(1970, 1, 1, tzinfo=timezone.utc)
        
        # 重要度スコア（降順）、焦点度スコア（降順）、日付（降順）でソート
        # reverse=Trueの場合、(-importance_score, -focus_score, -parsed_date.timestamp()) でソート
        # reverse=Falseの場合、その逆
        if reverse:
            return (-importance_score, -focus_score, -parsed_date.timestamp())
        else:
            return (importance_score, focus_score, parsed_date.timestamp())
    
    return sorted(news_items, key=get_sort_key)


def sort_news_by_date(news_items: List[Dict], reverse: bool = True) -> List[Dict]:
    """ニュースを日付でソート（デフォルトは新しい順）"""
    def get_sort_key(item: Dict) -> datetime:
        date_str = item.get("published")
        if not date_str:
            # 日付がない場合は最も古い日付として扱う
            return datetime(1970, 1, 1, tzinfo=timezone.utc)
        
        parsed_date = parse_news_date(date_str)
        if parsed_date:
            if parsed_date.tzinfo is None:
                parsed_date = parsed_date.replace(tzinfo=timezone.utc)
            return parsed_date
        
        return datetime(1970, 1, 1, tzinfo=timezone.utc)
    
    return sorted(news_items, key=get_sort_key, reverse=reverse)


def is_japanese_text(text: str) -> bool:
    """テキストが日本語を含むかどうかを判定"""
    if not text:
        return False
    # ひらがな、カタカナ、漢字、全角英数字の範囲をチェック
    japanese_pattern = re.compile(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF\uFF00-\uFFEF]')
    return bool(japanese_pattern.search(text))


def get_japanese_name_from_yfinance(symbol: str) -> Optional[str]:
    """yfinanceのinfoから日本語名を取得"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}
        
        # longNameまたはshortNameが日本語の場合、それを返す
        for key in ["longName", "shortName", "name"]:
            value = info.get(key)
            if value and isinstance(value, str) and is_japanese_text(value):
                return value.strip()
    except Exception as e:
        logging.debug(f"yfinanceから日本語名取得失敗 ({symbol}): {e}")
    return None


def get_japanese_name_from_yahoo_finance_jp(symbol: str) -> Optional[str]:
    """Yahoo Finance Japanから日本語名をスクレイピング"""
    if not SCRAPING_AVAILABLE:
        return None
    
    symbol_clean = symbol.replace(".T", "").strip()
    if not symbol_clean.isdigit():
        return None
    
    try:
        # Yahoo Finance JapanのURL
        url = f"https://finance.yahoo.co.jp/quote/{symbol_clean}.T"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, "lxml")
        
        # 複数のセレクタを試行
        selectors = [
            'h1[data-test="company-name"]',
            'h1.company-name',
            'h1',
            '[data-test="company-name"]',
            '.company-name',
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                text = element.get_text(strip=True)
                if text and is_japanese_text(text):
                    return text
        
        # titleタグから取得を試行
        title = soup.find("title")
        if title:
            title_text = title.get_text(strip=True)
            # タイトルから「(6501.T)」のような部分を除去
            title_text = re.sub(r'\s*\([0-9]+\.T\)\s*', '', title_text)
            if title_text and is_japanese_text(title_text):
                return title_text
                
    except Exception as e:
        logging.debug(f"Yahoo Finance Japanスクレイピング失敗 ({symbol}): {e}")
    
    return None


def get_japanese_company_name(symbol: str, yfinance_info: Optional[Dict] = None) -> Optional[str]:
    """ティッカーシンボルから日本語の社名を取得する（複数の方法を試行）"""
    if not symbol:
        return None
    
    symbol_clean = symbol.replace(".T", "").strip()
    if not symbol_clean.isdigit():
        return None
    
    # 1. yfinanceのinfoから取得（引数で渡された場合）
    if yfinance_info:
        for key in ["longName", "shortName", "name"]:
            value = yfinance_info.get(key)
            if value and isinstance(value, str) and is_japanese_text(value):
                return value.strip()
    
    # 2. yfinanceのAPIから直接取得
    japanese_name = get_japanese_name_from_yfinance(symbol)
    if japanese_name:
        return japanese_name
    
    # 3. Yahoo Finance Japanからスクレイピング
    japanese_name = get_japanese_name_from_yahoo_finance_jp(symbol)
    if japanese_name:
        return japanese_name
    
    # 4. フォールバック: 主要な日本株のティッカーシンボルと日本語社名のマッピング
    japanese_names = {
        "6501": "日立製作所",
        "6502": "東芝",
        "6503": "三菱電機",
        "6758": "ソニーグループ",
        "7203": "トヨタ自動車",
        "6752": "パナソニック",
        "9984": "ソフトバンクグループ",
        "9434": "ソフトバンク",
        "9983": "ファーストリテイリング",
        "8031": "三井物産",
        "8058": "三菱商事",
        "8001": "伊藤忠商事",
        "8002": "丸紅",
        "8306": "三菱UFJフィナンシャル・グループ",
        "8316": "三井住友フィナンシャルグループ",
        "8411": "みずほフィナンシャルグループ",
        "4063": "信越化学工業",
        "4061": "デンカ",
        "3401": "帝人",
        "3402": "東レ",
        "3405": "クラレ",
        "3407": "旭化成",
        "4911": "資生堂",
        "4912": "ライオン",
        "4452": "花王",
        "4453": "資生堂",
        "6098": "リクルートホールディングス",
        "6099": "エルピーダメモリ",
        "6178": "日本郵政",
        "6179": "日本郵政",
        "8801": "三井不動産",
        "8802": "三菱地所",
        "2914": "日本たばこ産業",
        "2501": "サッポロホールディングス",
        "2502": "アサヒグループホールディングス",
        "2503": "キリンホールディングス",
        "2531": "宝ホールディングス",
        "2801": "キッコーマン",
        "2802": "味の素",
        "2871": "ニチレイ",
        "3101": "東洋紡",
        "3103": "ユニ・チャーム",
        "3105": "日清紡ホールディングス",
        "3401": "帝人",
        "3402": "東レ",
        "3405": "クラレ",
        "3407": "旭化成",
        "4005": "住友化学",
        "4004": "昭和電工",
        "4003": "コスモエネルギーホールディングス",
        "4061": "デンカ",
        "4063": "信越化学工業",
        "4183": "三井化学",
        "4188": "三菱ケミカルホールディングス",
        "4208": "宇部興産",
        "4272": "日本化薬",
        "4452": "花王",
        "4453": "資生堂",
        "4502": "武田薬品工業",
        "4503": "アステラス製薬",
        "4506": "大日本住友製薬",
        "4507": "塩野義製薬",
        "4519": "中外製薬",
        "4523": "エーザイ",
        "4527": "ロート製薬",
        "4528": "小野薬品工業",
        "4543": "テルモ",
        "4568": "第一三共",
        "4578": "大塚ホールディングス",
        "4612": "日本ペイントホールディングス",
        "4661": "オリエンタルランド",
        "4684": "オムロン",
        "4689": "ヤフー",
        "4704": "トレンドマイクロ",
        "4751": "サイバーエージェント",
        "4755": "楽天グループ",
        "4901": "富士フイルムホールディングス",
        "4911": "資生堂",
        "5019": "出光興産",
        "5020": "ENEOSホールディングス",
        "5101": "横浜ゴム",
        "5108": "ブリヂストン",
        "5201": "AGC",
        "5214": "日本電気硝子",
        "5232": "住友大阪セメント",
        "5233": "太平洋セメント",
        "5301": "東海カーボン",
        "5332": "TOTO",
        "5333": "日本ガイシ",
        "5401": "日本製鉄",
        "5406": "神戸製鋼所",
        "5411": "JFEホールディングス",
        "5541": "大平洋金属",
        "5631": "日本製鋼所",
        "5703": "日本軽金属ホールディングス",
        "5711": "三菱マテリアル",
        "5713": "住友金属鉱山",
        "5714": "DOWAホールディングス",
        "5801": "古河電気工業",
        "5802": "住友電気工業",
        "5803": "フジクラ",
        "6098": "リクルートホールディングス",
        "6178": "日本郵政",
        "6301": "コマツ",
        "6302": "住友重機械工業",
        "6305": "日立建機",
        "6326": "クボタ",
        "6361": "荏原製作所",
        "6367": "ダイキン工業",
        "6471": "日本精工",
        "6472": "NTN",
        "6473": "ジェイテクト",
        "6501": "日立製作所",
        "6502": "東芝",
        "6503": "三菱電機",
        "6504": "富士電機",
        "6506": "安川電機",
        "6594": "日本電産",
        "6701": "日本電気",
        "6702": "富士通",
        "6723": "ルネサスエレクトロニクス",
        "6724": "セイコーエプソン",
        "6752": "パナソニック",
        "6758": "ソニーグループ",
        "6770": "アルプスアルパイン",
        "6841": "横河電機",
        "6857": "アドバンテスト",
        "6861": "キーエンス",
        "6902": "デンソー",
        "6954": "ファナック",
        "6971": "京セラ",
        "6976": "太陽誘電",
        "6981": "村田製作所",
        "7011": "三菱重工業",
        "7012": "川崎重工業",
        "7013": "IHI",
        "7201": "日産自動車",
        "7202": "いすゞ自動車",
        "7203": "トヨタ自動車",
        "7205": "日野自動車",
        "7261": "マツダ",
        "7267": "ホンダ",
        "7269": "スズキ",
        "7270": "SUBARU",
        "7272": "ヤマハ発動機",
        "7731": "ニコン",
        "7732": "トプコン",
        "7733": "オリンパス",
        "7741": "HOYA",
        "7751": "キヤノン",
        "7832": "バンダイナムコホールディングス",
        "7911": "凸版印刷",
        "7912": "大日本印刷",
        "8001": "伊藤忠商事",
        "8002": "丸紅",
        "8015": "豊田通商",
        "8031": "三井物産",
        "8058": "三菱商事",
        "8060": "野村ホールディングス",
        "8306": "三菱UFJフィナンシャル・グループ",
        "8316": "三井住友フィナンシャルグループ",
        "8354": "ふくおかフィナンシャルグループ",
        "8355": "静岡銀行",
        "8411": "みずほフィナンシャルグループ",
        "8601": "大和証券グループ本社",
        "8604": "野村ホールディングス",
        "8628": "松井証券",
        "8630": "SOMPOホールディングス",
        "8725": "MS&ADインシュアランスグループホールディングス",
        "8750": "第一生命ホールディングス",
        "8766": "東京海上ホールディングス",
        "8801": "三井不動産",
        "8802": "三菱地所",
        "8830": "住友不動産",
        "9001": "東武鉄道",
        "9005": "東急",
        "9007": "小田急電鉄",
        "9008": "京王電鉄",
        "9009": "京成電鉄",
        "9020": "東日本旅客鉄道",
        "9021": "西日本旅客鉄道",
        "9022": "東海旅客鉄道",
        "9104": "商船三井",
        "9107": "川崎汽船",
        "9202": "ANAホールディングス",
        "9301": "三菱倉庫",
        "9432": "日本電信電話",
        "9433": "KDDI",
        "9434": "ソフトバンク",
        "9501": "東京電力ホールディングス",
        "9502": "中部電力",
        "9503": "関西電力",
        "9531": "東京ガス",
        "9532": "大阪ガス",
        "9602": "東宝",
        "9681": "東京ドーム",
        "9684": "スクウェア・エニックス・ホールディングス",
        "9697": "カプコン",
        "9706": "日本空港ビルデング",
        "9719": "SCSK",
        "9735": "セコム",
        "9766": "コナミホールディングス",
        "9983": "ファーストリテイリング",
        "9984": "ソフトバンクグループ",
    }
    
    return japanese_names.get(symbol_clean)


@st.cache_data(ttl=86400, show_spinner=False)  # 24時間キャッシュ
def get_japanese_company_name_cached(symbol: str, yfinance_info: Optional[Dict] = None) -> Optional[str]:
    """キャッシュ付きの日本語社名取得関数"""
    return get_japanese_company_name(symbol, yfinance_info)


@st.cache_data(ttl=3600, show_spinner=False)  # 1時間キャッシュ
def fetch_article_content(url: str, timeout: int = 10) -> Optional[str]:
    """ニュース記事のURLから記事の全文を取得する（キャッシュ付き）"""
    if not SCRAPING_AVAILABLE or not url:
        return None
    
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, "lxml")
        
        # 一般的なニュース記事の本文セレクタを試行
        # 日本語ニュースサイト向けのセレクタ
        article_selectors = [
            'article',
            '.article-body',
            '.article-content',
            '.article-text',
            '.news-body',
            '.news-content',
            '.content-body',
            '#article-body',
            '#article-content',
            '#main-content',
            'main article',
            '[role="article"]',
            '.post-content',
            '.entry-content',
            'div.article',
            'div.content',
        ]
        
        article_text = None
        for selector in article_selectors:
            article_elem = soup.select_one(selector)
            if article_elem:
                # スクリプトやスタイルタグを除去
                for script in article_elem(["script", "style", "nav", "header", "footer", "aside", "advertisement"]):
                    script.decompose()
                
                # テキストを取得
                text = article_elem.get_text(separator="\n", strip=True)
                if text and len(text) > 100:  # 最低100文字以上あることを確認
                    article_text = text
                    break
        
        # セレクタで見つからない場合、pタグを集めて本文として使用
        if not article_text:
            paragraphs = soup.find_all("p")
            if paragraphs:
                text_parts = []
                for p in paragraphs:
                    text = p.get_text(strip=True)
                    if text and len(text) > 20:  # 短すぎる段落は除外
                        text_parts.append(text)
                if text_parts:
                    article_text = "\n".join(text_parts)
        
        # 取得したテキストをクリーンアップ
        if article_text:
            # 余分な空白を削除
            lines = [line.strip() for line in article_text.split("\n") if line.strip()]
            article_text = "\n".join(lines)
            
            # 最低200文字以上あることを確認（snippetより長いことを保証）
            if len(article_text) > 200:
                return article_text
        
        return None
    except Exception as e:
        logging.debug(f"記事全文取得失敗 ({url}): {e}")
        return None


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_news(query: str, symbol: Optional[str] = None, max_results: int = 15, yfinance_info: Optional[Dict] = None) -> List[Dict]:
    """日本語の最新ニュースを確実に取得する関数（最低件数が得られるまで再試行）"""
    if not query:
        return []
    
    # 設定ファイルからパラメータを読み込む
    config = NEWS_SEARCH_CONFIG.get("search", {})
    keywords_config = NEWS_SEARCH_CONFIG.get("keywords", {})
    filtering_config = NEWS_SEARCH_CONFIG.get("filtering", {})
    
    # 検索パラメータ
    default_max_results = config.get("max_results", 15)
    max_results = max_results if max_results != 15 else default_max_results
    min_required_results = config.get("min_required_results", 5)
    max_retries = config.get("max_retries", 3)
    retry_delay_seconds = config.get("retry_delay_seconds", 2)
    multipliers = config.get("multipliers", {})
    min_candidates = config.get("min_candidates", {})
    timeout = config.get("timeout", 30)
    article_fetch_timeout = config.get("article_fetch_timeout", 15)
    
    # キーワードテンプレート
    japanese_search_templates = keywords_config.get("japanese_search_templates", [])
    japanese_symbol_templates = keywords_config.get("japanese_symbol_templates", [])
    japanese_combined_templates = keywords_config.get("japanese_combined_templates", [])
    english_search_templates = keywords_config.get("english_search_templates", [])
    
    # フィルタリングパラメータ
    date_threshold_days = filtering_config.get("date_threshold_days", 365)
    focus_filter_config = filtering_config.get("focus_score", {})
    min_importance_score_when_focus_zero = focus_filter_config.get("min_importance_score_when_focus_zero", 4)
    fallback_sufficient_threshold_multiplier = filtering_config.get("fallback_sufficient_threshold_multiplier", 2)
    
    # 日本株かどうかを判定（.Tで終わる、または4桁の数字）
    is_japanese_stock = False
    symbol_clean = None
    if symbol:
        symbol_upper = symbol.upper().strip()
        if symbol_upper.endswith(".T") or (symbol_upper.isdigit() and 4 <= len(symbol_upper) <= 5):
            is_japanese_stock = True
            symbol_clean = symbol.replace(".T", "").strip()
    
    # 日本語の社名を取得（キャッシュ付き、yfinance_infoを渡す）
    japanese_company_name = None
    if is_japanese_stock and symbol_clean:
        japanese_company_name = get_japanese_company_name_cached(symbol, yfinance_info)
    
    news_items = []
    seen_urls = set()  # 重複チェック用
    errors = []  # エラーログ用
    
    # 最低件数が得られるまで再試行する
    for retry_attempt in range(max_retries):
        if retry_attempt > 0:
            # 再試行前に少し待機（APIレート制限を避けるため）
            time.sleep(retry_delay_seconds)
            logging.info(f"ニュース取得の再試行 {retry_attempt}/{max_retries - 1}（現在の件数: {len(news_items)}）")
        
        # 日本株の場合は日本語のニュースを優先的に取得
        if is_japanese_stock:
            # 検索キーワードのリストを構築
            search_keywords = []
            
            # 日本語の社名がある場合は、それを優先的に使用
            if japanese_company_name:
                for template in japanese_search_templates:
                    search_keywords.append(template.format(company_name=japanese_company_name))
            
            # ティッカーシンボルでの検索も追加
            if symbol_clean and symbol_clean.isdigit():
                for template in japanese_symbol_templates:
                    search_keywords.append(template.format(symbol=symbol_clean))
                # 日本語の社名がある場合は、ティッカーシンボルと組み合わせた検索も追加
                if japanese_company_name:
                    for template in japanese_combined_templates:
                        search_keywords.append(template.format(symbol=symbol_clean, company_name=japanese_company_name))
            
            # 英語の社名も検索に含める（日本語ニュースが見つからない場合のフォールバック）
            for template in japanese_search_templates:
                search_keywords.append(template.format(company_name=query))
            
            # 複数の検索を試行
            initial_multiplier = multipliers.get("initial_japanese", 8)
            initial_min_candidates = min_candidates.get("initial_japanese", 50)
            for idx, keywords in enumerate(search_keywords):
                # 既に十分な件数が得られている場合はスキップ
                if len(news_items) >= min_required_results * 2:
                    break
                
                # レート制限を避けるため、検索の間に少し待機（最初の検索以外）
                if idx > 0:
                    time.sleep(1)
                    
                try:
                    # timeoutパラメータはddgsのバージョンによってはサポートされていない可能性がある
                    try:
                        ddgs_context = DDGS(timeout=timeout)
                    except (TypeError, ValueError):
                        # timeoutパラメータがサポートされていない場合はデフォルトを使用
                        ddgs_context = DDGS()
                    
                    with ddgs_context as ddgs:
                        japanese_results = list(
                            ddgs.news(
                                keywords=keywords,
                                region="jp-ja",
                                safesearch="Off",
                                max_results=max(max_results * initial_multiplier, initial_min_candidates),
                            )
                        )
                        for item in japanese_results:
                            url = item.get("url", "")
                            title = item.get("title", "")
                            if url and url not in seen_urls and title:
                                seen_urls.add(url)
                                news_items.append(
                                    {
                                        "title": title,
                                        "url": url,
                                        "snippet": item.get("body") or item.get("snippet") or "",
                                        "published": item.get("date"),
                                        "source": item.get("source") or "",
                                        "language": "ja",
                                    }
                                )
                except Exception as e:
                    error_str = str(e)
                    # レート制限エラーの場合は特別な処理
                    if "202" in error_str or "ratelimit" in error_str.lower() or "rate limit" in error_str.lower():
                        error_msg = f"検索キーワード '{keywords}' でレート制限エラーが発生しました。しばらく待ってから再試行してください。"
                        errors.append(error_msg)
                        logging.warning(error_msg)
                        # レート制限の場合は少し長めに待機
                        if retry_attempt < max_retries - 1:
                            time.sleep(retry_delay_seconds * 2)
                    else:
                        error_msg = f"検索キーワード '{keywords}' でエラー: {error_str}"
                        errors.append(error_msg)
                        logging.warning(error_msg)
                    continue
            
            # 既に十分な件数が得られている場合はフォールバック検索をスキップ
            if len(news_items) >= min_required_results * 2:
                pass  # 次の処理に進む
            # 日本語ニュースが少ない場合、より広範囲な検索を試行
            elif len(news_items) < min_required_results:
                fallback_queries = []
                if japanese_company_name:
                    fallback_queries.append(japanese_company_name)
                if symbol_clean and symbol_clean.isdigit():
                    fallback_queries.append(symbol_clean)
                fallback_queries.append(query)
                
                fallback_multiplier = multipliers.get("fallback_japanese", 4)
                fallback_min_candidates = min_candidates.get("fallback_japanese", 30)
                for idx, fallback_query in enumerate(fallback_queries):
                    # 既に十分な件数が得られている場合はスキップ
                    if len(news_items) >= min_required_results * 2:
                        break
                    
                    # レート制限を避けるため、検索の間に少し待機（最初の検索以外）
                    if idx > 0:
                        time.sleep(1)
                        
                    try:
                        try:
                            ddgs_context = DDGS(timeout=timeout)
                        except (TypeError, ValueError):
                            ddgs_context = DDGS()
                        
                        with ddgs_context as ddgs:
                            # よりシンプルな検索クエリで再試行
                            fallback_results = list(
                                ddgs.news(
                                    keywords=fallback_query,
                                    region="jp-ja",
                                    safesearch="Off",
                                    max_results=max(max_results * fallback_multiplier, fallback_min_candidates),
                                )
                            )
                            for item in fallback_results:
                                url = item.get("url", "")
                                title = item.get("title", "")
                                if url and url not in seen_urls and title:
                                    seen_urls.add(url)
                                    news_items.append(
                                        {
                                            "title": title,
                                            "url": url,
                                            "snippet": item.get("body") or item.get("snippet") or "",
                                            "published": item.get("date"),
                                            "source": item.get("source") or "",
                                            "language": "ja",
                                        }
                                    )
                            # 十分なニュースが取得できた場合はループを抜ける
                            if len(news_items) >= max_results * fallback_sufficient_threshold_multiplier:
                                break
                    except Exception as e:
                        error_str = str(e)
                        # レート制限エラーの場合は特別な処理
                        if "202" in error_str or "ratelimit" in error_str.lower() or "rate limit" in error_str.lower():
                            error_msg = f"フォールバック検索（'{fallback_query}'）でレート制限エラーが発生しました。しばらく待ってから再試行してください。"
                            errors.append(error_msg)
                            logging.warning(error_msg)
                            # レート制限の場合は少し長めに待機
                            if retry_attempt < max_retries - 1:
                                time.sleep(retry_delay_seconds * 2)
                        else:
                            error_msg = f"フォールバック検索（'{fallback_query}'）でエラー: {error_str}"
                            errors.append(error_msg)
                            logging.warning(error_msg)
                        continue
        
        # 日本株でない場合、または日本語ニュースが少ない場合は英語のニュースも取得
        if not is_japanese_stock or len(news_items) < min_required_results:
            # 重要度の高いニュースを優先的に取得するための検索キーワード
            english_keywords = []
            for template in english_search_templates:
                english_keywords.append(template.format(query=query))
            
            english_multiplier = multipliers.get("english", 5)
            english_min_candidates = min_candidates.get("english", 30)
            for idx, keywords in enumerate(english_keywords):
                # 既に十分な件数が得られている場合はスキップ
                if len(news_items) >= min_required_results * 2:
                    break
                
                # レート制限を避けるため、検索の間に少し待機（最初の検索以外）
                if idx > 0:
                    time.sleep(1)
                    
                try:
                    try:
                        ddgs_context = DDGS(timeout=timeout)
                    except (TypeError, ValueError):
                        ddgs_context = DDGS()
                    
                    with ddgs_context as ddgs:
                        english_results = list(
                            ddgs.news(
                                keywords=keywords,
                                region="us-en",
                                safesearch="Off",
                                max_results=max(max_results * english_multiplier, english_min_candidates),
                            )
                        )
                        for item in english_results:
                            url = item.get("url", "")
                            title = item.get("title", "")
                            if url and url not in seen_urls and title:
                                seen_urls.add(url)
                                news_items.append(
                                    {
                                        "title": title,
                                        "url": url,
                                        "snippet": item.get("body") or item.get("snippet") or "",
                                        "published": item.get("date"),
                                        "source": item.get("source") or "",
                                        "language": "en",
                                    }
                                )
                except Exception as e:
                    error_str = str(e)
                    # レート制限エラーの場合は特別な処理
                    if "202" in error_str or "ratelimit" in error_str.lower() or "rate limit" in error_str.lower():
                        error_msg = f"検索キーワード '{keywords}' でレート制限エラーが発生しました。しばらく待ってから再試行してください。"
                        errors.append(error_msg)
                        logging.warning(error_msg)
                        # レート制限の場合は少し長めに待機
                        if retry_attempt < max_retries - 1:
                            time.sleep(retry_delay_seconds * 2)
                    else:
                        error_msg = f"検索キーワード '{keywords}' でエラー: {error_str}"
                        errors.append(error_msg)
                        logging.warning(error_msg)
                    continue
        
        # 最低件数に達した場合はループを抜ける
        if len(news_items) >= min_required_results:
            break
    
    # 再試行後の最終的な件数をログに記録
    if len(news_items) < min_required_results:
        logging.warning(f"最低件数（{min_required_results}件）に達しませんでした。取得件数: {len(news_items)}件")
    else:
        logging.info(f"ニュース取得成功: {len(news_items)}件（目標: {min_required_results}件以上）")
    
    # エラーが発生した場合はログに記録
    if errors and len(news_items) == 0:
        logging.error(f"ニュース取得に失敗しました。エラー数: {len(errors)}")
        for err in errors[:3]:  # 最初の3つのエラーのみ表示
            logging.error(err)
    
    # 最新のニュースのみをフィルタリング（設定ファイルの日数以内）
    news_items = filter_recent_news(news_items, days_threshold=date_threshold_days)
    
    # 薄い記事（ランキングや市場動向記事）を除外
    filtered_news_items = []
    shallow_count = 0
    for item in news_items:
        if is_shallow_article(item, japanese_company_name, symbol):
            shallow_count += 1
            continue
        filtered_news_items.append(item)
    
    news_items = filtered_news_items
    
    # フィルタリング結果をログに記録
    if shallow_count > 0:
        logging.info(f"薄い記事を {shallow_count} 件除外しました。")
    
    # 重要度、焦点度、日付でソート（重要度と焦点度が高い順、同じなら新しい順）
    news_items = sort_news_by_importance_and_date(
        news_items, 
        reverse=True,
        company_name=japanese_company_name or query,
        symbol=symbol,
        query=query
    )
    
    # 焦点度が低い記事を除外（焦点度スコアが0の記事は除外）
    # ただし、重要度が高い記事（決算発表など）は例外として含める
    focus_filtered_items = []
    low_focus_count = 0
    for item in news_items:
        focus_score = calculate_focus_score(item, japanese_company_name or query, symbol, query)
        importance_score = calculate_news_importance_score(item)
        
        # 焦点度が0かつ重要度も低い場合は除外（設定ファイルの閾値を使用）
        if focus_score == 0 and importance_score < min_importance_score_when_focus_zero:
            low_focus_count += 1
            continue
        focus_filtered_items.append(item)
    
    news_items = focus_filtered_items
    
    # フィルタリング結果をログに記録
    if low_focus_count > 0:
        logging.info(f"焦点度の低い記事を {low_focus_count} 件除外しました。")
    
    # max_resultsまでに制限（ただし、重要度と焦点度の高いニュースは優先的に含める）
    news_items = news_items[:max_results]
    
    # 各ニュースアイテムに対して記事の全文を取得（snippetが途中で切れている可能性があるため）
    # 全文取得に失敗した場合は、元のsnippetを使用
    for news_item in news_items:
        url = news_item.get("url", "")
        original_snippet = news_item.get("snippet", "")
        
        # 記事の全文を取得
        if url and original_snippet:
            full_content = fetch_article_content(url, timeout=article_fetch_timeout)
            if full_content:
                # 全文が取得できた場合は、snippetを全文で置き換え
                news_item["snippet"] = full_content
                news_item["full_content_fetched"] = True
            else:
                # 全文が取得できなかった場合は、元のsnippetを使用
                news_item["full_content_fetched"] = False
    
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
            temperature=0.0,
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
    tabs = st.tabs(["シナリオ", "プロの評価", "データ / ニュース", "rawデータ"])

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
        # 株価グラフセクション
        st.markdown("**📈 株価チャート**")
        symbol = snapshot.get("symbol") or snapshot.get("resolved_symbol")
        currency = snapshot.get("currency", "USD")
        
        # 期間選択
        period_options = {
            "1日": "1d",
            "5日": "5d",
            "1週間": "1wk",
            "1ヶ月": "1mo",
            "3ヶ月": "3mo",
            "6ヶ月": "6mo",
            "1年": "1y",
            "2年": "2y",
            "5年": "5y",
        }
        selected_period_label = st.selectbox(
            "期間を選択",
            options=list(period_options.keys()),
            index=3,  # デフォルトは1ヶ月
            key="stock_chart_period"
        )
        selected_period = period_options[selected_period_label]
        
        if symbol:
            with st.spinner("株価データを取得中..."):
                history_data = fetch_stock_history(symbol, period=selected_period)
            
            if history_data.get("error"):
                st.error(f"株価データの取得に失敗しました: {history_data['error']}")
            else:
                # グラフを表示
                fig = create_stock_chart(history_data, symbol, currency)
                st.plotly_chart(fig, use_container_width=True)
                
                # データ提供元へのリンク
                yahoo_url = get_yahoo_finance_url(symbol)
                st.markdown(
                    f'<div style="text-align: center; margin-top: 10px;">'
                    f'<a href="{yahoo_url}" target="_blank" style="color: #3b82f6; text-decoration: none;">'
                    f'📊 Yahoo Financeで詳細を見る</a></div>',
                    unsafe_allow_html=True
                )
                st.caption("グラフをクリックして拡大表示できます。データ提供元: Yahoo Finance")
        else:
            st.warning("シンボル情報が取得できませんでした。")
        
        st.divider()
        
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
            st.warning("最新ニュースの取得に失敗しました。ネットワーク接続や検索サービスの状態を確認してください。")
        else:
            st.caption(f"📰 {len(news_items)} 件のニュースを取得しました")
        for news in news_items:
            st.markdown(
                f'<div class="news-item"><a class="news-title" href="{news["url"]}" target="_blank">{news["title"]}</a>'
                f'<div class="news-meta">{news.get("source") or ""} · {news.get("published") or ""}</div>'
                f'<div class="news-body">{news.get("snippet") or ""}</div></div>',
                unsafe_allow_html=True,
            )

    with tabs[3]:
        st.markdown("**📋 rawデータ**")
        st.caption("取得したデータをそのまま表示します。デバッグやプロンプト改良の参考にしてください。")
        
        # 分析用ペイロードを構築
        payload = build_analysis_payload(snapshot, news_items)
        
        st.markdown("#### 1. 株価・経営指標データ（snapshot）")
        st.json(snapshot)
        
        st.markdown("#### 2. ニュースデータ（news_items）")
        st.json(news_items)
        
        st.markdown("#### 3. AI分析用ペイロード（payload）")
        st.json(payload)
        
        st.markdown("#### 4. AI分析結果（analysis）")
        st.json(analysis)


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
    
    ticker_input = st.text_input("ティッカーシンボル", value="6501")

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

    with st.spinner("最新ニュースを取得中..."):
        # snapshotからinfoを取得して日本語名取得に活用
        yfinance_info = snapshot.get("info", {})
        news_items = fetch_news(snapshot["company_name"], symbol=snapshot.get("symbol"), yfinance_info=yfinance_info)
    
    # ニュース取得結果のフィードバック
    if not news_items:
        st.warning("⚠️ 最新ニュースの取得に失敗しました。ネットワーク接続や検索サービスの状態を確認してください。")
    elif len(news_items) < 3:
        st.info(f"ℹ️ ニュースを {len(news_items)} 件取得しました（目標: 5件）。")
    
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
