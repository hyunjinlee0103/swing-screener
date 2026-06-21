"""
fetch_kospi.py
코스피 전 종목의 스윙 스크리닝 지표를 계산해 kospi_data.json으로 저장한다.
필요 패키지: pip install finance-datareader pandas numpy requests
"""

import json
import time
import logging
import warnings
from datetime import datetime, timedelta
from typing import Optional, Dict

import numpy as np
import pandas as pd
import FinanceDataReader as fdr

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

LOOKBACK_DAYS = 100         # 시세 조회 기간 (영업일 기준 충분히)
DELAY_SEC = 0.5             # 종목 간 요청 딜레이
MIN_TRADE_AMOUNT = 5_000_000_000   # 50억원
MIN_INTRADAY_RANGE = 3.0           # 3%
MIN_MARKET_CAP = 100_000_000_000   # 1,000억원
VOLUME_RATIO_THRESHOLD = 1.2       # 상승일 거래량 / 평균 거래량


def fetch_kospi_listing() -> pd.DataFrame:
    df = fdr.StockListing("KOSPI")
    # 실제 컬럼: Code, ISU_CD, Name, Market, Dept, Close, Marcap, Amount, ...
    df = df[df["Code"].str.len() == 6].copy()
    return df.reset_index(drop=True)


def calc_indicators(code: str, end: datetime, kospi_ret_20d: float) -> Optional[Dict]:
    start = (end - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    end_s = end.strftime("%Y-%m-%d")

    try:
        df = fdr.DataReader(code, start, end_s)
    except Exception as e:
        log.warning(f"{code} 조회 실패: {e}")
        return None

    if df is None or len(df) < 20:
        return None

    df = df.copy()
    # 컬럼 정규화 (FDR 버전별로 대소문자 다를 수 있음)
    df.columns = [c.capitalize() for c in df.columns]
    required = {"Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(set(df.columns)):
        return None

    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    if len(df) < 20:
        return None

    close = df["Close"]
    volume = df["Volume"]
    high = df["High"]
    low = df["Low"]

    current_price = float(close.iloc[-1])

    # 거래대금 (종가 × 거래량) 20일 평균
    trade_amount_20 = float((close * volume).rolling(20).mean().iloc[-1])
    enough_liquidity = trade_amount_20 >= MIN_TRADE_AMOUNT

    # 일중 변동폭 20일 평균
    daily_range_pct = float(((high - low) / low * 100).rolling(20).mean().iloc[-1])
    enough_volatility = daily_range_pct >= MIN_INTRADAY_RANGE

    # 이동평균선
    ma20 = float(close.rolling(20).mean().iloc[-1])
    ma60 = float(close.rolling(60).mean().iloc[-1]) if len(df) >= 60 else None
    above_ma = bool(
        current_price > ma20
        and (ma60 is None or current_price > ma60)
    )

    # 상승일 거래량 동반
    price_up = close.diff() > 0
    if price_up.sum() >= 5:
        avg_up_vol = float(volume[price_up].rolling(10, min_periods=1).mean().iloc[-1])
        avg_vol = float(volume.rolling(20).mean().iloc[-1])
        volume_on_up = avg_up_vol >= avg_vol * VOLUME_RATIO_THRESHOLD
    else:
        volume_on_up = False

    # 업종 모멘텀: 종목 20일 수익률 vs KOSPI 20일 수익률 (개별 상대강도)
    if len(df) >= 20:
        stock_ret_20d = float(close.iloc[-1] / close.iloc[-20] - 1)
        sector_strong = stock_ret_20d > kospi_ret_20d
    else:
        sector_strong = False

    return {
        "current_price": current_price,
        "trade_amount_20d": round(trade_amount_20),
        "daily_range_pct": round(daily_range_pct, 2),
        "ma20": round(ma20, 2),
        "ma60": round(ma60, 2) if ma60 is not None else None,
        "enough_liquidity": bool(enough_liquidity),
        "enough_volatility": bool(enough_volatility),
        "above_ma": above_ma,
        "volume_on_up": bool(volume_on_up),
        "sector_strong": bool(sector_strong),
    }


def get_kospi_20d_return(end: datetime) -> float:
    start = (end - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    end_s = end.strftime("%Y-%m-%d")
    try:
        ki = fdr.DataReader("KS11", start, end_s)
        if ki is not None and len(ki) >= 20:
            return float(ki["Close"].iloc[-1] / ki["Close"].iloc[-20] - 1)
    except Exception as e:
        log.warning(f"KOSPI 지수 조회 실패: {e}")
    return 0.0


def main():
    today = datetime.today()
    log.info("코스피 종목 리스트 로드 중...")
    listing = fetch_kospi_listing()
    total = len(listing)
    log.info(f"총 {total}개 종목 처리 시작")

    log.info("KOSPI 지수 20일 수익률 계산 중...")
    kospi_ret = get_kospi_20d_return(today)
    log.info(f"KOSPI 20일 수익률: {kospi_ret*100:.2f}%")

    results = []
    for i, row in listing.iterrows():
        code = str(row["Code"])
        name = str(row["Name"])

        if i % 50 == 0:
            log.info(f"  진행 중: {i}/{total} ({len(results)}개 완료)")

        indic = calc_indicators(code, today, kospi_ret)
        if indic is None:
            time.sleep(DELAY_SEC)
            continue

        # 시가총액: 리스팅에서 직접 읽기
        marcap = None
        raw_marcap = row.get("Marcap")
        try:
            if raw_marcap is not None and not pd.isna(raw_marcap):
                marcap = int(raw_marcap)
        except (ValueError, TypeError):
            marcap = None
        large_cap = marcap is not None and marcap >= MIN_MARKET_CAP

        # 업종 정보 (Dept 컬럼, 없으면 빈 문자열)
        sector = ""
        for col in ("Dept", "Sector", "업종", "Industry"):
            if col in listing.columns:
                val = row.get(col)
                if val and not pd.isna(val):
                    sector = str(val)
                    break

        entry = {
            "code": code,
            "name": name,
            "current_price": indic["current_price"],
            "trade_amount_20d": indic["trade_amount_20d"],
            "daily_range_pct": indic["daily_range_pct"],
            "ma20": indic["ma20"],
            "ma60": indic["ma60"],
            "market_cap": marcap,
            "sector": sector,
            "sector_rs": round(kospi_ret, 4),
            # 6개 정량 기준
            "enough_liquidity": indic["enough_liquidity"],
            "enough_volatility": indic["enough_volatility"],
            "above_ma": indic["above_ma"],
            "volume_on_up": indic["volume_on_up"],
            "sector_strong": indic["sector_strong"],
            "large_cap": large_cap,
        }
        results.append(entry)
        time.sleep(DELAY_SEC)

    output = {
        "updated_at": today.strftime("%Y-%m-%d %H:%M:%S"),
        "kospi_ret_20d": round(kospi_ret * 100, 2),
        "stocks": results,
    }

    with open("kospi_data.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    pass_all = sum(
        1 for s in results
        if all(s[k] for k in ["enough_liquidity","enough_volatility","above_ma","volume_on_up","sector_strong","large_cap"])
    )
    log.info(f"완료: {len(results)}개 저장 → 6기준 충족: {pass_all}개 → kospi_data.json")


if __name__ == "__main__":
    main()
