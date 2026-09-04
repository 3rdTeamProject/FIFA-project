import os
import json
import requests
import time
from dotenv import load_dotenv


# ==================================================
# 환경 설정
# ==================================================

load_dotenv()

API_KEY = os.getenv("NEXON_API_KEY")

headers = {
    "x-nxopen-api-key": API_KEY
}


# ==================================================
# 수집할 공식경기 1vs1 TOP 10 유저
# ==================================================

nicknames = [
    "극소수",
    "satthesails",
    "ZSia시아",
    "에이스더브라위너",
    "FIFA구독",
    "쿠마이누",
    "DCB호날두",
    "AbsoluteLegend",
    "뛰어봐",
    "으감"
]


# 한 유저당 최대 100경기
MATCH_LIMIT = 100

# 공식경기
MATCH_TYPE = 50

# JSON 저장 폴더
RAW_DIR = "match_data/raw_data"

os.makedirs(RAW_DIR, exist_ok=True)


# ==================================================
# 1. 닉네임 → OUID
# ==================================================

def get_ouid(nickname):

    url = "https://open.api.nexon.com/fconline/v1/id"

    params = {
        "nickname": nickname
    }

    response = requests.get(
        url,
        headers=headers,
        params=params
    )

    if response.status_code != 200:
        print(
            f"[OUID 조회 실패] {nickname}",
            "상태 코드:",
            response.status_code
        )
        return None

    return response.json().get("ouid")


# ==================================================
# 2. OUID → 최근 경기 ID
# ==================================================

def get_match_ids(ouid):

    url = "https://open.api.nexon.com/fconline/v1/user/match"

    params = {
        "ouid": ouid,
        "matchtype": MATCH_TYPE,
        "offset": 0,
        "limit": MATCH_LIMIT
    }

    response = requests.get(
        url,
        headers=headers,
        params=params
    )

    if response.status_code != 200:
        print(
            "[경기 목록 조회 실패]",
            "상태 코드:",
            response.status_code
        )
        return []

    return response.json()


# ==================================================
# 3. matchId → 경기 상세 데이터
# ==================================================

def get_match_detail(match_id):

    url = "https://open.api.nexon.com/fconline/v1/match-detail"

    params = {
        "matchid": match_id
    }

    max_retries = 3

    for attempt in range(max_retries):

        try:

            response = requests.get(
                url,
                headers=headers,
                params=params,
                timeout=15
            )

        except requests.exceptions.RequestException as e:

            print(
                "[네트워크 오류]",
                match_id,
                e
            )

            return None


        # 정상 응답
        if response.status_code == 200:
            return response.json()


        # 요청 제한
        if response.status_code == 429:

            print(
                f"[429 요청 제한] {match_id}"
                f" - {attempt + 1}/{max_retries}번째 시도"
            )

            time.sleep(10)

            continue


        # 그 외 오류
        print(
            "[경기 상세 조회 실패]",
            match_id,
            "상태 코드:",
            response.status_code
        )

        return None


    print(
        "[최종 실패]",
        match_id,
        "429 요청 제한이 계속 발생했습니다."
    )

    return None

# ==================================================
# 4. TOP 10 유저 순서대로 수집
# ==================================================

for user_number, nickname in enumerate(nicknames, start=1):

    print()
    print("=" * 70)
    print(f"{user_number}위 유저 수집 시작: {nickname}")
    print("=" * 70)


    # ----------------------------------------------
    # 닉네임 → OUID
    # ----------------------------------------------

    ouid = get_ouid(nickname)

    if ouid is None:
        print(f"{nickname} 수집을 건너뜁니다.")
        continue


    # ----------------------------------------------
    # 해당 유저의 최근 경기 목록
    # ----------------------------------------------

    match_ids = get_match_ids(ouid)

    print(
        f"{nickname}의 API 반환 경기 수:",
        len(match_ids)
    )


    new_count = 0
    duplicate_count = 0
    fail_count = 0


    # ----------------------------------------------
    # 각 경기 처리
    # ----------------------------------------------

    for match_number, match_id in enumerate(
        match_ids,
        start=1
    ):

        file_path = os.path.join(
            RAW_DIR,
            f"{match_id}.json"
        )


        # ==========================================
        # 이미 저장된 경기 → 상세 API 호출 안 함
        # ==========================================

        if os.path.exists(file_path):

            duplicate_count += 1

            print(
                f"{match_number}번째 경기 이미 존재:",
                match_id
            )

            continue


        # ==========================================
        # 새로운 경기 → 상세 API 호출
        # ==========================================

        match_data = get_match_detail(match_id)

        if match_data is None:

            fail_count += 1

            print(
                f"{match_number}번째 경기 저장 실패:",
                match_id
            )

            continue


        # ==========================================
        # JSON 저장
        # ==========================================

        with open(
            file_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                match_data,
                f,
                ensure_ascii=False,
                indent=2
            )


        new_count += 1

        print(
            f"{match_number}번째 경기 저장 완료:",
            match_id
        )


    # ----------------------------------------------
    # 현재 유저 결과
    # ----------------------------------------------

    print()
    print(f"[{nickname} 수집 결과]")
    print("API 반환 경기 수:", len(match_ids))
    print("새로 저장:", new_count)
    print("중복 경기:", duplicate_count)
    print("실패:", fail_count)


# ==================================================
# 5. 전체 저장 경기 수 확인
# ==================================================

json_files = [
    file_name
    for file_name in os.listdir(RAW_DIR)
    if file_name.endswith(".json")
]


print()
print("=" * 70)
print("TOP 10 경기 데이터 수집 완료")
print("현재 raw_data의 고유 경기 수:", len(json_files))
print("=" * 70)