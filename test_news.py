"""
네이버 뉴스 API 단독 테스트
사용법:
  python test_news.py --id <CLIENT_ID> --secret <CLIENT_SECRET>
  또는 환경변수로:
  NAVER_CLIENT_ID=xxx NAVER_CLIENT_SECRET=yyy python test_news.py
"""

import argparse
import os
import re
import sys
import requests
from urllib.parse import quote

_HTML_TAG = re.compile(r"<[^>]+>")

def test_news(client_id: str, client_secret: str, query: str = "삼성전자 주가", n: int = 3):
    print(f"\n[테스트] 검색어: '{query}'")
    print(f"  CLIENT_ID    : {client_id[:4]}{'*' * (len(client_id) - 4) if len(client_id) > 4 else ''}")
    print(f"  CLIENT_SECRET: {client_secret[:4]}{'*' * (len(client_secret) - 4) if len(client_secret) > 4 else ''}")

    url = (f"https://openapi.naver.com/v1/search/news.json"
           f"?query={quote(query)}&display={n}&sort=date")
    try:
        r = requests.get(url, headers={
            "X-Naver-Client-Id": client_id,
            "X-Naver-Client-Secret": client_secret,
        }, timeout=10)
    except Exception as e:
        print(f"\n[실패] 네트워크 오류: {e}")
        sys.exit(1)

    print(f"\n  HTTP 상태: {r.status_code}")
    if not r.ok:
        print(f"[실패] API 오류 응답:\n{r.text}")
        sys.exit(1)

    items = r.json().get("items", [])
    if not items:
        print("[경고] 뉴스 결과가 0건입니다. 검색어나 키 권한을 확인하세요.")
        sys.exit(1)

    print(f"\n[성공] 뉴스 {len(items)}건:")
    for i, item in enumerate(items, 1):
        title = _HTML_TAG.sub("", item.get("title", ""))
        link  = item.get("originallink") or item.get("link", "")
        pub   = item.get("pubDate", "")
        print(f"  {i}. {title}")
        print(f"     {link}")
        print(f"     발행: {pub}")


def main():
    parser = argparse.ArgumentParser(description="네이버 뉴스 API 테스트")
    parser.add_argument("--id",     default=os.environ.get("NAVER_CLIENT_ID", ""),
                        help="NAVER_CLIENT_ID")
    parser.add_argument("--secret", default=os.environ.get("NAVER_CLIENT_SECRET", ""),
                        help="NAVER_CLIENT_SECRET")
    parser.add_argument("--query",  default="삼성전자 주가",
                        help="검색어 (기본: '삼성전자 주가')")
    args = parser.parse_args()

    if not args.id.strip() or not args.secret.strip():
        print("[오류] NAVER_CLIENT_ID와 NAVER_CLIENT_SECRET이 필요합니다.")
        print("  python test_news.py --id <ID> --secret <SECRET>")
        print("  또는: NAVER_CLIENT_ID=xxx NAVER_CLIENT_SECRET=yyy python test_news.py")
        sys.exit(1)

    test_news(args.id.strip(), args.secret.strip(), query=args.query)


if __name__ == "__main__":
    main()
