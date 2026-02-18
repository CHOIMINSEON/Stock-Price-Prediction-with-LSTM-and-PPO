"""종목별 뉴스/펀더멘털/투자의견(연합인포맥스) 수집 스크립트.

필요 패키지 (가상환경에서 설치 권장):

	pip install finvizfinance duckduckgo-search selenium webdriver-manager beautifulsoup4 lxml

주의사항:
- https://globalmonitor.einfomax.co.kr 서비스는 저작권 및 이용약관이 있습니다.
  실제 자동 크롤링/상용 사용 전 반드시 약관을 확인하고 팀 내 정책에 맞게 사용하세요.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import re

from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from finvizfinance.quote import finvizfinance
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager


# ===================== 설정 =====================

# 예시: 애플
TICKER = "AAPL"  # finviz, 뉴스 검색용 티커
COMPANY_NAME_EN = "Apple"
COMPANY_NAME_KO = "애플"


@dataclass
class NewsItem:
	title: str
	source: str | None
	date: str | None
	link: str | None


def get_finviz_fundamentals(ticker: str) -> Dict[str, Any]:
	"""finvizfinance를 이용해 기본 펀더멘털/의견/내부자 정보를 가져온다.

	반환 예시:
		{
			"fundamentals": {...},
			"recommendation": "Buy",
			"insider_own": "0.10%",
		}
	"""

	quote = finvizfinance(ticker)
	fundamentals = quote.ticker_fundament()

	# finviz 표에서 자주 쓰는 항목 샘플만 추려서 사용
	recommendation = fundamentals.get("Recom")
	insider_own = fundamentals.get("Insider Own")

	# 내부자 거래 내역 전체 (DataFrame) 및 그 중 가장 최근 1건
	inside_df = None
	latest_insider_trade: Dict[str, Any] | None = None
	try:
		inside_df = quote.ticker_inside_trader()
		if inside_df is not None and hasattr(inside_df, "empty") and not inside_df.empty:
			# Date 컬럼 기준으로 최근순(내림차순) 정렬
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
	"""DuckDuckGo 뉴스 검색으로 실시간 뉴스 헤드라인 조회.

	DDGS는 별도 API Key가 필요 없다.
	"""

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


def _init_selenium_driver() -> webdriver.Chrome:
	"""Chrome WebDriver 초기화 (webdriver-manager 사용)."""

	options = webdriver.ChromeOptions()
	# 필요 시 헤드리스 모드 사용
	# options.add_argument("--headless=new")
	options.add_argument("--no-sandbox")
	options.add_argument("--disable-dev-shm-usage")

	service = Service(ChromeDriverManager().install())
	driver = webdriver.Chrome(service=service, options=options)
	driver.set_window_size(1400, 900)
	return driver


def get_broker_opinion_from_einfomax(company_name_ko: str) -> Dict[str, Any]:
	"""연합인포맥스 Global Monitor 페이지에서 투자의견 관련 정보 크롤링.

	- 검색창에 한글 종목명(예: "애플")을 입력해 해당 종목 페이지를 연 뒤,
	  HTML 내에서 "투자의견" 텍스트가 있는 셀을 찾아 그 옆 값을 읽는다.
	- 페이지 구조가 변경될 수 있으므로, 동작하지 않을 경우
	  검색 input / 버튼 선택자와 파싱 로직을 조정해야 한다.
	"""

	url = "https://globalmonitor.einfomax.co.kr/mr_usa_hts.html#/04/01"
	driver = _init_selenium_driver()

	try:
		driver.get(url)

		wait = WebDriverWait(driver, 20)
		wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))

		# ---- 종목 검색 ----
		# 정확한 selector는 사이트 구조를 보고 조정 필요.
		# placeholder에 "종목"이 들어간 input을 우선 시도.
		search_input = None
		try:
			search_input = driver.find_element(By.CSS_SELECTOR, "input[placeholder*='종목']")
		except Exception:
			# fallback: 상단 검색창 후보 (텍스트 input 중 첫 번째)
			inputs = driver.find_elements(By.CSS_SELECTOR, "input[type='text']")
			if inputs:
				search_input = inputs[0]

		if search_input is None:
			raise RuntimeError("검색 입력창을 찾지 못했습니다. selector를 수정하세요.")

		search_input.clear()
		search_input.send_keys(company_name_ko)
		search_input.send_keys(Keys.RETURN)

		# 종목 로딩 및 "투자의견" 텍스트가 페이지에 나타날 때까지 대기
		wait.until(
			EC.text_to_be_present_in_element((By.TAG_NAME, "body"), "투자의견")
		)

		html = driver.page_source
		soup = BeautifulSoup(html, "lxml")

		# 기본 정보 테이블에서 "투자의견" 셀을 찾아 바로 오른쪽 값을 추출
		opinion_text = None
		opinion_cell = soup.find(
			lambda tag: tag.name in {"td", "th", "span"}
			and tag.get_text(strip=True) == "투자의견"
		)
		if opinion_cell is not None:
			value_cell = opinion_cell.find_next("td")
			if value_cell is not None:
				opinion_text = value_cell.get_text(strip=True)

		# 투자의견 분포 기준일 텍스트 (예: "데이터기준일 : 2025-12-12")
		# 실제 DOM 상으로는 div.pull-right.box-unit.ng-binding 안에 들어 있으므로,
		# 해당 요소들을 Selenium으로 직접 순회하며 날짜를 추출한다.
		as_of_date = None
		try:
			date_elems = driver.find_elements(By.CSS_SELECTOR, "div.pull-right.box-unit.ng-binding")
			for el in date_elems:
				text = el.text.strip()
				if "기준일" in text:
					m = re.search(r"([0-9]{4}[-./][0-9]{2}[-./][0-9]{2})", text)
					if m:
						as_of_date = m.group(1)
						break
		except Exception:
			as_of_date = None

		return {
			"source": "연합인포맥스 Global Monitor",
			"company_name": company_name_ko,
			"opinion": opinion_text,
			"as_of": as_of_date,
		}

	finally:
		driver.quit()


def main() -> None:
	ticker = TICKER
	company_en = COMPANY_NAME_EN
	company_ko = COMPANY_NAME_KO

	print(f"=== 종목: {company_en} ({ticker}) / {company_ko} ===")

	# 1) finviz 기본 정보
	print("\n[1] finviz 펀더멘털/의견/내부자 정보")
	try:
		finviz_data = get_finviz_fundamentals(ticker)
		print(f"- Recommendation (Recom): {finviz_data['recommendation']}")
		print(f"- Insider Own: {finviz_data['insider_own']}")
		inside_df = finviz_data.get("insider_trades")
		if inside_df is not None and hasattr(inside_df, "empty") and not inside_df.empty:
			print("busts [3. 내부자 거래] (오늘 데이터 없으면 최근 내역 표시)")
			print("-" * 80)
			# 최근 내역 5건만 표 형태로 출력
			try:
				print(inside_df.head(5).to_string())
			except Exception:
				# DataFrame 포맷에 문제가 있을 경우 간단히 딕셔너리 리스트로 출력
				for _, row in inside_df.head(5).iterrows():
					print(dict(row))
	except Exception as e:
		print(f"finviz 데이터 조회 실패: {e}")

	# 2) DuckDuckGo 뉴스
	print("\n[2] DuckDuckGo 실시간 뉴스")
	try:
		news_query = f"{company_en} stock news"
		news_items = get_latest_news_ddg(news_query, max_results=5)
		for idx, item in enumerate(news_items, start=1):
			print(f"[{idx}] {item.title}")
			if item.source:
				print(f"    Source: {item.source}")
			if item.date:
				print(f"    Date:   {item.date}")
			if item.link:
				print(f"    Link:   {item.link}")
	except Exception as e:
		print(f"뉴스 검색 실패: {e}")

	# 3) 연합인포맥스 Global Monitor 투자의견
	print("\n[3] 연합인포맥스 Global Monitor 투자의견")
	try:
		opinion_data = get_broker_opinion_from_einfomax(company_ko)
		print(f"- 종목명: {opinion_data['company_name']}")
		print(f"- 투자의견: {opinion_data.get('opinion')}")
		print(f"- 기준일: {opinion_data.get('as_of')}")
	except Exception as e:
		print(f"투자의견 크롤링 실패: {e}")


if __name__ == "__main__":
	try:
		main()
	except KeyboardInterrupt:
		print("\n사용자에 의해 중단되었습니다.")
		sys.exit(1)

