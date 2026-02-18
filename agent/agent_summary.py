"""종목별 뉴스/펀더멘털/투자의견을 수집하고,
Qwen 계열 LLM(기본: 로컬 Ollama의 qwen2.5:3b)으로 요약 JSON을 생성하는 엔드투엔드 스크립트.

전제:
- Qwen 모델이 OpenAI 호환 엔드포인트로 배포되어 있다고 가정.
- 기본값은 로컬 Ollama 서버의 qwen2.5:3b를 사용하며,
  아래 환경 변수로 다른 엔드포인트/모델로 변경 가능:
    - LLM_BASE_URL : 예) "http://localhost:11434/v1" (Ollama 기본) 또는 프록시 URL
    - LLM_API_KEY  : 필요 시 API Key, Ollama 기본값은 "ollama"
    - QWEN3_MODEL  : 모델 이름 (기본값: "qwen2.5:3b")

필요 패키지:
    pip install openai finvizfinance duckduckgo-search requests beautifulsoup4

사용 예시 (터미널에서):
    python agent_summary.py AAPL kr

터미널에서 인자를 주지 않고 실행하면, 티커/회사명을 직접 입력받아 사용한다.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, asdict
from datetime import date
from typing import Any, Dict, List, Optional

import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from finvizfinance.quote import finvizfinance
from openai import OpenAI


@dataclass
class NewsItem:
    title: str
    source: Optional[str]
    date: Optional[str]
    link: Optional[str]


_QWEN3_MODEL_DEFAULT = "qwen2.5:3b"
_QWEN3_MODEL = os.getenv("QWEN3_MODEL", _QWEN3_MODEL_DEFAULT)
_LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
_LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")

_client: Optional[OpenAI] = None


# 이 스크립트에서 허용하는 티커 목록
ALLOWED_TICKERS = {"AAPL", "TSLA", "NVDA", "PLTR", "AMZN", "GOOGL", "AMD", "META"}


def _get_client() -> OpenAI:
    """전역 OpenAI(호환) 클라이언트를 Lazy-init 한다.

    Qwen 계열 모델을 OpenAI 호환 서버(예: Ollama, vLLM 등)로 띄워 두었다는
    가정 하에, base_url, api_key를 환경 변수에서 읽어 사용한다.
    """

    global _client
    if _client is None:
        _client = OpenAI(api_key=_LLM_API_KEY, base_url=_LLM_BASE_URL)
    return _client


# ===================== 데이터 수집 함수 =====================


def get_finviz_fundamentals(ticker: str) -> Dict[str, Any]:
    """finvizfinance를 이용해 기본 펀더멘털/의견/내부자 정보를 가져온다."""

    quote = finvizfinance(ticker)
    fundamentals = quote.ticker_fundament()

    recommendation = fundamentals.get("Recom")
    insider_own = fundamentals.get("Insider Own")

    inside_df = None
    latest_insider_trade: Dict[str, Any] | None = None
    try:
        inside_df = quote.ticker_inside_trader()
        if inside_df is not None and hasattr(inside_df, "empty") and not inside_df.empty:
            try:
                inside_df = inside_df.sort_values(by="Date", ascending=False)
            except Exception:
                pass
            latest_insider_trade = inside_df.iloc[0].to_dict()
    except Exception:
        inside_df = None
        latest_insider_trade = None

    return {
        "fundamentals": fundamentals,
        "recommendation": recommendation,
        "insider_own": insider_own,
        "insider_trades": inside_df,
        "latest_insider_trade": latest_insider_trade,
    }


def get_latest_news_ddg(query: str, max_results: int = 5) -> List[NewsItem]:
    """DuckDuckGo 뉴스 검색으로 실시간 뉴스 헤드라인 조회."""

    news_items: List[NewsItem] = []
    with DDGS() as ddgs:
        for item in ddgs.news(query, max_results=max_results, region="us-en"):
            news_items.append(
                NewsItem(
                    title=item.get("title", ""),
                    source=item.get("source"),
                    date=item.get("date"),
                    link=item.get("url"),
                )
            )
    return news_items


def get_broker_opinion_from_einfomax(ticker: str, company_name_ko: str) -> Dict[str, Any]:
    """Finviz 종목 페이지의 Recom 점수로부터 매수/매도 의견을 추정한다.

    Finviz 종목 페이지 예시:
        https://finviz.com/quote.ashx?t=TSLA&p=d

    위 URL에서 티커 부분(t=TSLA)에 입력 받은 ticker를 대문자로 넣어 요청하고,
    스냅샷 테이블 내의 "Recom" 값을 읽어서 다음 기준으로 의견을 만든다.

        <= 1.5  -> Strong Buy
        <= 2.5  -> Buy
        <= 3.5  -> Hold
        <= 4.5  -> Sell
        >  4.5  -> Strong Sell
    """

    base: Dict[str, Any] = {
        "source": "Finviz - Analyst Recommendation (Recom)",
        "company_name": company_name_ko,
        "opinion": None,
        "as_of": date.today().isoformat(),
    }

    url = f"https://finviz.com/quote.ashx?t={ticker.upper()}&p=d"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        result = dict(base)
        result["error"] = f"request_failed: {e}"
        return result

    try:
        soup = BeautifulSoup(resp.text, "html.parser")
        # snapshot-table2 안에서 "Recom" 셀을 찾고, 그 다음 셀의 값을 사용
        recom_value_str: Optional[str] = None
        for td in soup.select("table.snapshot-table2 td"):
            if td.get_text(strip=True) == "Recom":
                val_td = td.find_next("td")
                if val_td is not None:
                    recom_value_str = val_td.get_text(strip=True)
                break

        if not recom_value_str:
            result = dict(base)
            result["error"] = "recom_not_found"
            return result

        try:
            recom_val = float(recom_value_str)
        except ValueError:
            result = dict(base)
            result["error"] = f"invalid_recom_value: {recom_value_str}"
            return result

        # Recom 점수 → 의견 매핑
        if recom_val <= 1.5:
            verdict = "Strong Buy"
        elif recom_val <= 2.5:
            verdict = "Buy"
        elif recom_val <= 3.5:
            verdict = "Hold"
        elif recom_val <= 4.5:
            verdict = "Sell"
        else:
            verdict = "Strong Sell"

        result = dict(base)
        result.update({
            "opinion": verdict,
            "detail": {
                "recom_raw": recom_val,
                "url": url,
            },
        })
        return result

    except Exception as e:
        result = dict(base)
        result["error"] = f"parse_failed: {e}"
        return result


def build_raw_payload(
    ticker: str,
    company_name_en: str,
    company_name_ko: str,
    finviz_data: Dict[str, Any],
    news_items: List[NewsItem],
    broker_opinion: Dict[str, Any],
    news_fetch_error: Optional[str],
) -> Dict[str, Any]:
    """수집된 결과들을 LLM 입력용 원시 JSON 구조로 정리한다."""

    return {
        "ticker": ticker,
        "company_name_en": company_name_en,
        "company_name_ko": company_name_ko,
        "finviz": {
            "recommendation": finviz_data.get("recommendation"),
            "insider_own": finviz_data.get("insider_own"),
            "latest_insider_trade": finviz_data.get("latest_insider_trade"),
            "fundamentals": finviz_data.get("fundamentals"),
        },
        "news_items": [asdict(n) for n in news_items],
        "news_fetch_error": news_fetch_error,
        "broker_opinion_raw": broker_opinion,
    }


def summarize_with_qwen3(
    raw_payload: Dict[str, Any],
    *,
    model: str | None = None,
    temperature: float = 0.2,
    timeout: float | None = 60.0,
    output_language: str = "kr",
) -> Dict[str, Any]:
    """Qwen 계열 LLM(기본: qwen2.5:3b)을 이용해 금융 요약 JSON을 생성한다.

    Parameters
    ----------
    raw_payload : Dict[str, Any]
        build_raw_payload()로 만든 입력 또는 그와 동등한 구조.
    model : str | None, default None
        사용할 모델 이름. None이면 QWEN3_MODEL 환경 변수 또는 기본값 사용.
    temperature : float, default 0.2
        창의성/무작위성 정도.
    timeout : float | None
        LLM 호출 타임아웃(초). 지원하는 서버에서만 적용됨.

    Returns
    -------
    Dict[str, Any]
        LLM이 생성한 JSON 파싱 결과. 파싱 실패 시 fallback 구조 반환.
    """

    client = _get_client()
    model_name = model or _QWEN3_MODEL

    # LLM에 보여 줄 원시 데이터(JSON 문자열)
    raw_json_str = json.dumps(raw_payload, ensure_ascii=False, indent=2)

    lang = output_language.lower()

    if lang == "en":
        system_prompt = (
            "You are a financial trading assistant. "
            "From raw fundamentals, news, and broker opinions for a single stock, "
            "you must extract ONLY the information useful for trading decisions. "
            "Always answer ONLY with JSON, no explanations, no markdown. "
            "Use English for any natural-language text values, but keep keys in English."
        )

        user_prompt = (
            "Analyze the following raw JSON for a single stock and produce a concise JSON "
            "object that is directly usable by a trading agent.\n"
            "Strictly follow the schema below (keys and types). Keys are in English, values in English too.\n"
            "{\n"
            "  \"summary\": {\n"
            "    \"news\": string,\n"
            "    \"broker\": string,\n"
            "    \"overall\": string\n"
            "  },\n"
            "  \"sentiment_score\": number,\n"
            "  \"impact_score\": number,\n"
            "  \"signal_type\": string,\n"
            "  \"trading_implication\": string,\n"
            "  \"reliability\": string\n"
            "}\n\n"
            "Field descriptions (do NOT include these as comments in the JSON):\n"
            "- summary.news: Major news summary (English, keep company/person names exactly as in raw data).\n"
            "- summary.broker: One-sentence broker/analyst opinion summary based ONLY on broker_opinion_raw from its source (for example, Finviz Analyst Recommendation / Recom). Do not mix news content. Example: \"According to Finviz - Analyst Recommendation (Recom), as of 2025-01-01 it is a Strong Buy opinion.\"\n"
            "- summary.overall: Overall summary combining BOTH the news summary and the broker opinion. It must describe the situation from a balanced, neutral perspective (do not exaggerate positivity or negativity beyond what the data supports), and should not simply restate either the news or broker sentence.\n"
            "- sentiment_score: -1.0 (very negative) ~ +1.0 (very positive).\n"
            "- impact_score: 0 (no impact) ~ 10 (very large market impact).\n"
            "- signal_type: one of MOMENTUM (bullish trend), RISK (negative/risk-off), NEUTRAL (mixed).\n"
            "- trading_implication: Concrete trading advice (1-2 sentences, English). It should be written after considering ALL of the above fields (summary, sentiment_score, impact_score, signal_type, reliability), and must not introduce any new facts beyond what is already reflected in those fields. Do NOT repeat or paraphrase the summary.news or summary.overall sentences; assume the reader has already seen them and focus only on what to do (hold, buy, trim, avoid, etc.) and under what conditions.\n"
            "- reliability: one of HIGH (facts/filings), MEDIUM (major news), LOW (rumor/speculation).\n\n"
            "Rules:\n"
            "- Output ONLY the JSON object above, no extra text or markdown.\n"
            "- Keep company_name_en / company_name_ko exactly as in the raw JSON when you mention the company. Do not change spelling.\n"
            "- For the news summary, only use facts that appear in the raw news headlines/bodies. Do NOT invent new people/events.\n"
            "- If raw_payload.news_fetch_error is not null or news_items is empty, do NOT invent news. Instead, set summary.news to a short message that news could not be fetched or there were no significant recent news, and base the overall summary mainly on fundamentals and the broker opinion.\n"
            "- If news contains terms like 'sell', 'sold', 'reduced stake', 'cut stake', 'trimmed', '매도', '감축', '지분 축소', then sentiment_score must NOT be positive; keep it <= 0 (neutral to negative).\n"
            "- The broker field must always be exactly one sentence of the form: \"According to <source>, as of <date> it is a <verdict> opinion.\"\n"
            "  Use the verdict value from broker_opinion_raw.opinion (for example, Strong Buy, Buy, Hold, Sell, Strong Sell). If it is missing, clearly state that there is no clear consensus recommendation.\n"
            "- If signals from news, fundamentals, and broker opinion conflict, avoid overconfident positive/negative wording; prefer a more neutral tone.\n"
            "- sentiment_score and impact_score must be floating-point numbers.\n\n"
            "Here is the raw JSON data to analyze (do NOT just copy it, you must summarize/structure it according to the schema):\n"
            f"{raw_json_str}\n"
        )
    else:
        system_prompt = (
            "You are a financial trading assistant. "
            "From raw fundamentals, news, and broker opinions for a single stock, "
            "you must extract ONLY the information useful for trading decisions. "
            "Always answer ONLY with JSON, no explanations, no markdown. "
            "Use Korean for any natural-language text values, but keep keys in English."
        )

        # LLM이 만들어야 할 최종 출력 스키마를 명시
        # (유저가 첨부한 설계 요건에 맞춘 구조)
        user_prompt = (
            "아래의 단일 종목(raw 데이터 JSON)을 분석해서, 매매 의사결정에 필요한 핵심 정보만 "
            "추출/요약한 JSON 객체를 만들어 주세요.\n"
            "항상 아래 스키마(키/타입/의미)를 정확히 따르세요. 키 이름은 영어, 값은 한국어로 작성합니다.\n"
            "{\n"
            "  \"summary\": {\n"
            "    \"news\": string,\n"
            "    \"broker\": string,\n"
            "    \"overall\": string\n"
            "  },\n"
            "  \"sentiment_score\": number,\n"
            "  \"impact_score\": number,\n"
            "  \"signal_type\": string,\n"
            "  \"trading_implication\": string,\n"
            "  \"reliability\": string\n"
            "}\n\n"
            "필드 설명(참고용, JSON 안에 주석으로 넣지 마세요):\n"
            "- summary.news: 주요 뉴스 요약 (한국어, 회사명/인물명은 raw 데이터 그대로 사용).\n"
            "- summary.broker: 증권사/애널리스트 투자의견 요약. 반드시 raw JSON의 broker_opinion_raw(예: Finviz Analyst Recommendation / Recom) 정보만 참고해서 작성하고, news 내용은 섞지 마세요. 예: \"Finviz - Analyst Recommendation (Recom)에 따르면 2025-01-01 기준 Strong Buy 의견입니다.\"\n"
            "- summary.overall: 전체 종합 요약. news 요약과 broker 요약을 모두 참고하되, 특정 방향(강한 매수·매도)으로 과장하지 말고, 데이터에서 드러나는 범위 안에서 균형 잡힌(중립적인) 관점으로 현재 상황을 설명하세요. news나 broker 문장을 그대로 복사하지 말고, 두 정보를 종합해서 한 단계 위에서 요약합니다.\n"
            "- sentiment_score: -1.0 (매우 부정) ~ +1.0 (매우 긍정).\n"
            "- impact_score: 0 (영향 없음) ~ 10 (시장 크게 흔듦).\n"
            "- signal_type: MOMENTUM(추세 호재), RISK(위험/악재), NEUTRAL(중립) 중 하나.\n"
            "- trading_implication: 트레이딩 에이전트가 참고할 구체적인 조언 (한국어 한두 문장). 위의 summary, sentiment_score, impact_score, signal_type, reliability 등을 모두 고려한 뒤, 그 결과를 바탕으로 한 매매/포지션 운용 조언을 작성해야 하며, 새로운 팩트나 이벤트를 추가로 만들어내서는 안 됩니다. 이 필드는 summary.news나 summary.overall의 내용을 다시 요약/복사하지 말고, 이미 그 내용을 읽었다고 가정한 상태에서 \"어떻게 행동할지(매수/추가매수/부분매도/관망/회피 등)\"에 대해서만 간결하게 조언해야 합니다.\n"
            "- reliability: HIGH(공시/팩트), MEDIUM(메이저 뉴스), LOW(루머/추정) 중 하나.\n\n"
            "주의사항:\n"
            "- 반드시 위 JSON 객체만 출력하고, 설명 문장이나 마크다운을 붙이지 마세요.\n"
            "-  news 요약과 broker 필드는 각각 독립적으로 작성하고, 서로 섞이지 않도록 주의하세요.\n"
            "- 회사명은 raw JSON의 company_name_en / company_name_ko를 그대로 사용하고, 철자를 바꾸거나 다른 이름으로 바꾸지 마세요.\n"
            "- news 요약은 raw JSON의 뉴스 헤드라인/본문에 실제로 언급된 내용만 사용하고, 새로운 사실/인물/이벤트를 지어내지 마세요.\n"
            "- raw_payload.news_fetch_error 값이 null이 아니거나 news_items가 비어 있으면, 뉴스 내용을 지어내지 말고, 요약할 뉴스가 없거나 뉴스 수집에 실패했다는 식의 짧은 문장으로 summary.news를 작성하세요. 이 경우 overall 요약은 주로 펀더멘털과 broker 의견을 기반으로 하되, \"최근 뉴스 정보는 부족/부재하다\"는 점을 간단히 언급할 수 있습니다.\n"
            "- 뉴스에 \"매도\", \"감축\", \"지분 축소\", \"sold\", \"sell\", \"reduced stake\", \"cut stake\", \"trimmed\" 등의 표현이 주로 등장하면 sentiment_score를 양수(>0)로 두지 말고, 0 이하(중립~부정)로 설정하세요.\n"
            "- broker 필드는 항상 한 문장으로, \"<source>에 따르면 <date> 기준 <opinion> 의견입니다.\" 형태로 작성하고,\n"
            "  opinion 값이 \"Strong Buy\", \"Buy\", \"Hold\", \"Sell\", \"Strong Sell\" 등일 때는 그대로 한국어 문장 안에 포함해서 사용하세요(예: \"Strong Buy 의견입니다\"). opinion이 비어 있으면 뚜렷한 컨센서스가 없다고 짧게 언급하세요.\n"
            "- 뉴스/펀더멘털/투자의견 사이에 긍정과 부정이 섞여서 방향성이 애매하면, 과도하게 긍정/부정으로 단정하지 말고 중립에 가깝게 요약하세요. 이때 summary.overall은 특히 양 측면(호재/리스크)을 모두 언급하는 균형 잡힌 톤을 유지해야 합니다.\n"
            "- sentiment_score와 impact_score는 실수(float)로 작성하세요.\n\n"
            "다음은 분석해야 할 raw 데이터(JSON)입니다. 이 데이터를 그대로 복사하지 말고, 위 스키마에 맞춰 요약/구조화만 하세요.\n"
            f"{raw_json_str}\n"
        )

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            timeout=timeout,
            # 가능한 경우 JSON 모드 사용 (서버가 지원하지 않으면 무시될 수 있음)
            response_format={"type": "json_object"},  # type: ignore[arg-type]
        )

        content = response.choices[0].message.content  # type: ignore[index]
        if not content:
            raise ValueError("LLM 응답이 비어 있습니다.")

        # 혹시 문자열 앞뒤에 공백/코드블록이 섞여 있어도 최대한 JSON만 추출
        content_str = str(content).strip()

        # ```json ... ``` 형태 제거
        if content_str.startswith("```"):
            content_str = content_str.lstrip("`")
            if content_str.lower().startswith("json"):
                content_str = content_str[4:]
            # 끝쪽 ``` 제거
            if "```" in content_str:
                content_str = content_str.split("```", 1)[0]
            content_str = content_str.strip()

        return json.loads(content_str)

    except Exception as e:  # 파싱 실패나 LLM 오류에 대한 안전장치
        return {
            "error": "llm_summarization_failed",
            "message": str(e),
            "raw_payload_sample": {
                "ticker": raw_payload.get("ticker"),
                "company_name_en": raw_payload.get("company_name_en"),
                "company_name_ko": raw_payload.get("company_name_ko"),
            },
        }


__all__ = [
    "build_raw_payload",
    "summarize_with_qwen3",
]


def main(argv: Optional[List[str]] = None) -> None:
    """티커(회사명 코드)와 언어를 기준으로 데이터를 수집하고 LLM 요약을 출력한다.

    우선순위 및 사용 예:
    1) 커맨드라인 인자
       - python agent_summary.py AAPL           -> 회사명: AAPL, 언어: kr
       - python agent_summary.py AAPL kr        -> 회사명: AAPL, 언어: kr
       - python agent_summary.py TSLA en        -> 회사명: TSLA, 언어: en
       - 필요하면 아래와 같이 영어/한글 이름까지 줄 수도 있음(선택 사항)
         python agent_summary.py TSLA "Tesla" "테슬라" kr
    2) 인자가 없으면 터미널에서 인터랙티브 입력으로 반드시 값을 받는다.
    """

    args = list(argv) if argv is not None else sys.argv[1:]

    if len(args) >= 1:
 
        ticker = args[0].upper()

        def _is_lang_token(token: str) -> bool:
            t = token.lower()
            return t in {"ko", "kr", "korean", "en", "eng", "english"}

        if len(args) == 1:
            # 티커만 주어진 경우: 회사명은 티커, 언어는 기본값(kr)
            company_en = ticker
            company_ko = ticker
            output_lang = "kr"
        elif len(args) == 2:
            # 두 번째 인자가 언어 토큰이면 언어로 인식, 아니면 영어 이름으로 인식
            if _is_lang_token(args[1]):
                company_en = ticker
                company_ko = ticker
                output_lang = args[1].lower()
            else:
                company_en = args[1]
                company_ko = ticker
                output_lang = "kr"
        elif len(args) == 3:
            # 세 번째 인자가 언어 토큰이면 [ticker, company_en, lang]
            if _is_lang_token(args[2]):
                company_en = args[1]
                company_ko = ticker
                output_lang = args[2].lower()
            else:
                # [ticker, company_en, company_ko]
                company_en = args[1]
                company_ko = args[2]
                output_lang = "kr"
        else:
            # [ticker, company_en, company_ko, lang]
            company_en = args[1]
            company_ko = args[2]
            output_lang = args[3].lower()
    else:
        print("[입력] 회사 정보를 입력하세요")

        # 티커는 반드시 입력되도록 반복
        ticker = ""
        while not ticker:
            ticker = input("Company(AAPL/TSLA/NVDA/PLTR/AMZN/GOOGL/AMD/META): ").strip().upper()
            if not ticker:
                print("티커는 필수 입력입니다. 다시 입력해 주세요.")
                continue
            if ticker not in ALLOWED_TICKERS:
                print("허용된 티커가 아닙니다. AAPL, TSLA, NVDA, PLTR, AMZN, GOOGL, AMD, META 중에서만 선택해 주세요.")
                ticker = ""
                continue
        # 인터랙티브 모드에서는 회사명을 따로 받지 않고 티커를 이름으로 사용
        company_en = ticker
        company_ko = ticker

        # 출력 언어 선택
        output_lang = ""
        while output_lang not in {"kr", "en"}:
            choice = input("language(kr/eng): ").strip().lower()
            if choice in {"ko", "kr", "korean"}:
                # 내부 표현은 kr로 통일하되, ko도 호환 입력으로 허용
                output_lang = "kr"
            elif choice in {"en", "eng", "english"}:
                output_lang = "en"
            else:
                print("입력 값이 올바르지 않습니다. kr 또는 en 중 하나를 입력해 주세요.")

    # 커맨드라인 인자로 받은 출력 언어도 kr/en 표기로 정규화
    if "output_lang" in locals():
        if output_lang in {"ko", "kr", "korean"}:
            output_lang = "kr"
        elif output_lang in {"en", "eng", "english"}:
            output_lang = "en"

    # 회사명이 비어 있으면 티커를 이름으로 재사용
    if not company_en:
        company_en = ticker
    if not company_ko:
        company_ko = ticker

    print(f"=== 종목: {company_en} ({ticker}) / {company_ko} | 출력 언어: {output_lang} ===")

    # 1) finviz 기본 정보
    print("\n[1] finviz 펀더멘털/의견/내부자 정보 수집")
    try:
        finviz_data = get_finviz_fundamentals(ticker)
        print(f"- Recommendation (Recom): {finviz_data.get('recommendation')}")
        print(f"- Insider Own: {finviz_data.get('insider_own')}")
    except Exception as e:
        print(f"finviz 데이터 조회 실패: {e}")
        finviz_data = {
            "fundamentals": None,
            "recommendation": None,
            "insider_own": None,
            "insider_trades": None,
            "latest_insider_trade": None,
        }

    # 2) DuckDuckGo 뉴스
    print("\n[2] DuckDuckGo 실시간 뉴스 수집")
    news_fetch_error: Optional[str] = None
    try:
        news_query = f"{company_en} stock news"
        news_items = get_latest_news_ddg(news_query, max_results=5)
        for idx, item in enumerate(news_items, start=1):
            print(f"[{idx}] {item.title}")
    except Exception as e:
        print(f"뉴스 검색 실패: {e}")
        news_items = []
        news_fetch_error = str(e)

    # 3) Finviz 애널리스트 Recom(종합 의견)
    print("\n[3] Finviz 애널리스트 Recom(종합 의견) 수집")
    try:
        broker_opinion = get_broker_opinion_from_einfomax(ticker, company_ko)
        print(f"- 종목명: {broker_opinion.get('company_name')}")
        print(f"- 투자의견: {broker_opinion.get('opinion')}")
        print(f"- 기준일: {broker_opinion.get('as_of')}")
    except Exception as e:
        print(f"투자의견 크롤링 실패: {e}")
        broker_opinion = {
            "source": "Finviz - Analyst Recommendation (Recom)",
            "company_name": company_ko,
            "opinion": None,
            "as_of": None,
        }

    # 4) LLM 요약
    print("\n[4] Qwen 요약(JSON) 생성")
    raw_payload = build_raw_payload(
        ticker=ticker,
        company_name_en=company_en,
        company_name_ko=company_ko,
        finviz_data=finviz_data,
        news_items=news_items,
        broker_opinion=broker_opinion,
        news_fetch_error=news_fetch_error,
    )

    summary = summarize_with_qwen3(raw_payload, output_language=output_lang)
    print("\n=== 요약 JSON ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단되었습니다.")
        sys.exit(1)
