import pandas as pd
from duckduckgo_search import DDGS
from finvizfinance.quote import finvizfinance
from datetime import datetime

def get_stock_data_test(ticker):
    print(f"\n{'='*50}")
    print(f"  [{ticker}] 데이터 수집 테스트 (기준: {datetime.now().strftime('%Y-%m-%d %H:%M')})")
    print(f"{'='*50}\n")

    # ---------------------------------------------------------
    # 1. 뉴스 데이터 (지난 24시간 이내)
    # ---------------------------------------------------------
    print(f"📰 [1. 뉴스] 지난 24시간 이내 주요 기사")
    print("-" * 50)
    
    try:
        with DDGS() as ddgs:
            # time='d' 옵션이 핵심: 지난 1일(24시간) 데이터만 가져옴
            # time을 timelimit으로 변경
            news_results = list(ddgs.news(keywords=f"{ticker} stock", timelimit="d", max_results=5))
            
            if news_results:
                for i, news in enumerate(news_results, 1):
                    # 보기 좋게 날짜와 제목, 출처 출력
                    print(f"{i}. [{news['source']}] {news['title']}")
                    print(f"   (링크: {news['url']})")
                    print(f"   (시간: {news['date']})\n")
            else:
                print("   ※ 지난 24시간 동안 검색된 뉴스가 없습니다.")
                
    except Exception as e:
        print(f"   ※ 뉴스 수집 중 에러 발생: {e}")

    # ---------------------------------------------------------
    # 공통: Finviz 객체 생성
    # ---------------------------------------------------------
    try:
        stock = finvizfinance(ticker)
    except Exception as e:
        print(f"\n❌ Finviz 데이터 접속 실패: {e}")
        return

    # ---------------------------------------------------------
    # 2. 증권사 의견 (Analyst Ratings)
    # ---------------------------------------------------------
    print(f"\n📊 [2. 증권사 의견] (오늘 데이터 없으면 최근 내역 표시)")
    print("-" * 50)
    
    try:
        # 최신순으로 정렬된 데이터프레임을 가져옵니다.
        ratings_df = stock.ticker_outer_ratings()
        
        if not ratings_df.empty:
            # 상위 5개만 출력 (가장 최근 날짜가 맨 위)
            print(ratings_df.head(5).to_string(index=False))
        else:
            print("   ※ 증권사 의견 데이터가 없습니다.")
            
    except Exception as e:
        print(f"   ※ 증권사 의견 수집 중 에러 발생: {e}")

    # ---------------------------------------------------------
    # 3. 내부자 거래 (Insider Trading)
    # ---------------------------------------------------------
    print(f"\nbusts [3. 내부자 거래] (오늘 데이터 없으면 최근 내역 표시)")
    print("-" * 50)
    
    try:
        # 최신순으로 정렬된 데이터프레임을 가져옵니다.
        insider_df = stock.ticker_inside_trader()
        
        if not insider_df.empty:
            # 필요한 컬럼만 골라서 상위 5개 출력
            columns_to_show = ['Date', 'Relationship', 'Transaction', 'Cost', '#Shares', 'Value ($)', '#Shares Total']
            
            # 데이터프레임에 해당 컬럼들이 있는지 확인 후 출력
            available_cols = [c for c in columns_to_show if c in insider_df.columns]
            print(insider_df[available_cols].head(5).to_string(index=False))
        else:
            print("   ※ 내부자 거래 데이터가 없습니다.")
            
    except Exception as e:
        print(f"   ※ 내부자 거래 수집 중 에러 발생: {e}")

    print(f"\n{'-'*50}")
    print("✅ 테스트 완료")

# =========================================================
# 실행 부분
# =========================================================
if __name__ == "__main__":
    # 원하는 종목 티커 입력 (예: 테슬라 TSLA, 애플 AAPL, 엔비디아 NVDA)
    target_ticker = "TSLA"
    get_stock_data_test(target_ticker)