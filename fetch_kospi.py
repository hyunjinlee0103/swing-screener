"""
fetch_kospi.py  —  상승추세 중 눌림목(조정) 후보 스크리너
필요 패키지: pip install finance-datareader pandas numpy requests
"""

import json
import os
import re
import time
import logging
import warnings
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from urllib.parse import quote

import numpy as np
import pandas as pd
import FinanceDataReader as fdr
import requests

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

# ── 기준값 ─────────────────────────────────────────────────────────────────
LOOKBACK_DAYS      = 120          # 시세 조회 기간 (영업일 60일 확보)
DELAY_SEC          = 0.45         # 종목 간 요청 딜레이
MIN_TRADE_AMOUNT   = 5_000_000_000   # 거래대금 50억원
MIN_MARKET_CAP     = 100_000_000_000 # 시가총액 1,000억원
PULLBACK_THRESHOLD = 0.95            # 20일 고점 대비 95% 이하 (≥-5%)

_HTML_TAG = re.compile(r"<[^>]+>")


# ── 1. 업종 매핑 ────────────────────────────────────────────────────────────
def fetch_sector_map() -> Dict[str, str]:
    """네이버 업종 목록(79개)을 순회해 {종목코드: 업종명} 반환"""
    h = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(
            "https://finance.naver.com/sise/sise_group.naver?type=upjong",
            headers=h, timeout=12)
        sectors = re.findall(r'no=(\d+)[^"]*">([^<]+)</a>', r.text)
    except Exception as e:
        log.warning(f"업종 목록 조회 실패: {e}")
        return {}

    log.info(f"업종 수: {len(sectors)}개 — 종목 매핑 중...")
    sector_map: Dict[str, str] = {}
    for no, name in sectors:
        try:
            r2 = requests.get(
                f"https://finance.naver.com/sise/sise_group_detail.naver?type=upjong&no={no}",
                headers=h, timeout=10)
            codes = list(set(re.findall(r"code=(\d{6})", r2.text)))
            for code in codes:
                sector_map[code] = name.strip()
            time.sleep(0.2)
        except Exception:
            pass

    log.info(f"업종 매핑 완료: {len(sector_map)}개 종목")
    return sector_map


# ── 2. 지표 계산 ────────────────────────────────────────────────────────────
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
    df.columns = [c.capitalize() for c in df.columns]
    if not {"Open", "High", "Low", "Close", "Volume"}.issubset(set(df.columns)):
        return None

    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    df = df[df["Low"] > 0]   # 거래정지 행 제거
    if len(df) < 20:
        return None

    close  = df["Close"]
    volume = df["Volume"]
    high   = df["High"]

    current_price   = float(close.iloc[-1])

    # ① 거래대금 (20일 평균)
    trade_amount_20 = float((close * volume).rolling(20).mean().iloc[-1])
    enough_liquidity = bool(np.isfinite(trade_amount_20) and trade_amount_20 >= MIN_TRADE_AMOUNT)

    # ② 60일선 위
    if len(df) >= 60:
        ma60_raw = float(close.rolling(60).mean().iloc[-1])
        ma60 = round(ma60_raw, 2) if np.isfinite(ma60_raw) else None
    else:
        ma60 = None
    above_ma60 = bool(ma60 is not None and current_price > ma60)

    # ③ 눌림목 조정: 최근 20일 고점 대비 -5% 이상 하락
    high_20d = float(high.iloc[-20:].max())
    if high_20d > 0 and np.isfinite(high_20d):
        pullback_pct = round((current_price / high_20d - 1) * 100, 2)
        pullback = bool(pullback_pct <= -5.0)
    else:
        pullback_pct = None
        pullback = False

    # ④ 조정 시 거래량 감소: 고점 이후 구간 평균 < 60일 평균
    high_20d_pos = len(df) - 20 + int(high.iloc[-20:].values.argmax())
    decline_vol_slice = volume.iloc[high_20d_pos:]
    avg_vol_60d_raw   = float(volume.rolling(60, min_periods=20).mean().iloc[-1])
    avg_vol_60d       = avg_vol_60d_raw if np.isfinite(avg_vol_60d_raw) else 0.0
    if len(decline_vol_slice) >= 3 and avg_vol_60d > 0:
        avg_decline_vol = float(decline_vol_slice.mean())
        volume_dry = bool(np.isfinite(avg_decline_vol) and avg_decline_vol < avg_vol_60d)
    else:
        volume_dry = False

    # ⑤ 상대강도 (RS): 종목 20일 수익률 > KOSPI 20일 수익률
    base_price = float(close.iloc[-20])
    if base_price > 0:
        stock_ret = float(close.iloc[-1] / base_price - 1)
        sector_strong = bool(np.isfinite(stock_ret) and stock_ret > kospi_ret_20d)
    else:
        sector_strong = False

    return {
        "current_price":   current_price,
        "trade_amount_20d": round(trade_amount_20) if np.isfinite(trade_amount_20) else None,
        "ma60":            ma60,
        "high_20d":        round(high_20d, 2) if np.isfinite(high_20d) else None,
        "pullback_pct":    pullback_pct,
        "enough_liquidity": enough_liquidity,
        "above_ma60":      above_ma60,
        "pullback":        pullback,
        "volume_dry":      volume_dry,
        "sector_strong":   sector_strong,
    }


# ── 3. KOSPI 기준 수익률 ─────────────────────────────────────────────────────
def get_kospi_20d_return(end: datetime) -> float:
    start = (end - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    try:
        ki = fdr.DataReader("KS11", start, end.strftime("%Y-%m-%d"))
        if ki is not None and len(ki) >= 20:
            return float(ki["Close"].iloc[-1] / ki["Close"].iloc[-20] - 1)
    except Exception as e:
        log.warning(f"KOSPI 지수 조회 실패: {e}")
    return 0.0


# ── 4. 네이버 뉴스 ──────────────────────────────────────────────────────────
def fetch_news(name: str, n: int = 3) -> List[Dict]:
    client_id     = os.environ.get("NAVER_CLIENT_ID", "")
    client_secret = os.environ.get("NAVER_CLIENT_SECRET", "")
    if not client_id or not client_secret:
        return []
    try:
        url = (f"https://openapi.naver.com/v1/search/news.json"
               f"?query={quote(name + ' 주가')}&display={n}&sort=date")
        r = requests.get(url, headers={
            "X-Naver-Client-Id": client_id,
            "X-Naver-Client-Secret": client_secret,
        }, timeout=10)
        if not r.ok:
            return []
        return [
            {
                "title":   _HTML_TAG.sub("", item.get("title", "")),
                "link":    item.get("originallink") or item.get("link", ""),
                "pubDate": item.get("pubDate", ""),
            }
            for item in r.json().get("items", [])[:n]
        ]
    except Exception as e:
        log.warning(f"뉴스 조회 실패 ({name}): {e}")
        return []


# ── 5. 메인 ─────────────────────────────────────────────────────────────────
QUANT_KEYS = ["enough_liquidity", "above_ma60", "pullback", "volume_dry", "sector_strong", "large_cap"]


def main():
    today = datetime.today()

    log.info("코스피 종목 리스트 로드 중...")
    listing = fdr.StockListing("KOSPI")
    listing = listing[listing["Code"].str.len() == 6].reset_index(drop=True)
    total = len(listing)
    log.info(f"총 {total}개 종목")

    log.info("업종 매핑 중...")
    sector_map = fetch_sector_map()

    log.info("KOSPI 20일 수익률 계산 중...")
    kospi_ret = get_kospi_20d_return(today)
    log.info(f"KOSPI 20일 수익률: {kospi_ret*100:.2f}%")

    results = []
    for i, row in listing.iterrows():
        code = str(row["Code"])
        name = str(row["Name"])

        if i % 50 == 0:
            log.info(f"  {i}/{total} — 완료 {len(results)}개")

        indic = calc_indicators(code, today, kospi_ret)
        if indic is None:
            time.sleep(DELAY_SEC)
            continue

        # 시가총액 (리스팅에서 직접)
        marcap = None
        raw = row.get("Marcap")
        try:
            if raw is not None and not pd.isna(raw):
                marcap = int(raw)
        except (ValueError, TypeError):
            pass
        large_cap = bool(marcap is not None and marcap >= MIN_MARKET_CAP)

        entry = {
            "code":           code,
            "name":           name,
            "current_price":  indic["current_price"],
            "trade_amount_20d": indic["trade_amount_20d"],
            "ma60":           indic["ma60"],
            "high_20d":       indic["high_20d"],
            "pullback_pct":   indic["pullback_pct"],
            "market_cap":     marcap,
            "sector":         sector_map.get(code, ""),
            "sector_rs":      round(kospi_ret, 4),
            # 6개 정량 기준
            "enough_liquidity": indic["enough_liquidity"],
            "above_ma60":     indic["above_ma60"],
            "pullback":       indic["pullback"],
            "volume_dry":     indic["volume_dry"],
            "sector_strong":  indic["sector_strong"],
            "large_cap":      large_cap,
        }
        results.append(entry)
        time.sleep(DELAY_SEC)

    # Top10 선정: 화면 표시 기준과 동일 (충족 기준 수 내림차순, 거래대금 내림차순)
    qualified = [s for s in results if all(s[k] for k in QUANT_KEYS)]
    results_sorted = sorted(
        results,
        key=lambda s: (-sum(1 for k in QUANT_KEYS if s.get(k)), -(s["trade_amount_20d"] or 0))
    )
    top10_codes = {s["code"] for s in results_sorted[:10]}

    log.info(f"6기준 충족: {len(qualified)}개 — Top10 뉴스 조회 중...")
    for s in results:
        if s["code"] in top10_codes:
            s["news"] = fetch_news(s["name"])
            time.sleep(0.3)

    output = {
        "updated_at":    today.strftime("%Y-%m-%d %H:%M:%S"),
        "kospi_ret_20d": round(kospi_ret * 100, 2),
        "stocks":        results,
    }

    with open("kospi_data.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, allow_nan=False)

    log.info(f"완료: {len(results)}개 저장 → 6기준 충족 {len(qualified)}개 → kospi_data.json")


if __name__ == "__main__":
    main()
