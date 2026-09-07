import os
import json
import requests
import time

from dotenv import load_dotenv


# ==================================================
# 1. 환경 설정
# ==================================================

load_dotenv()

API_KEY = os.getenv("NEXON_API_KEY")

if not API_KEY:
    raise ValueError(
        "NEXON_API_KEY가 없습니다. "
        ".env 파일을 확인해주세요."
    )

headers = {
    "x-nxopen-api-key": API_KEY
}


# ==================================================
# 2. 수집할 유저
# ==================================================
#
# 기존 유저든 신규 유저든 여기에 닉네임만 넣으면 됨.
#
# 기존 데이터가 있으면:
# → 기존 JSON을 최대한 재사용
# → 부족한 정상 경기만 추가 수집
#
# 신규 유저면:
# → 0개부터 시작
# → 정상 종료 70경기까지 수집
# ==================================================

nicknames = [
    "보구리너지",
    "당신은승리자",
    "KRXChan",
    "디발라의낭만",
    "BenzJCW",
    "물안경남친",
    "복숭아맛사과",
    "극소수",
    "KRXTak",
    "snag1wonNoparent",
    "리바이브말맨",
    "걸리면빽태클",
    "리바이브이원희",
    "DRXSavior",
    "GCTCrong",
    "BFXKaiser",
    "태연",
    "2014MSN",
    
]


# ==================================================
# 3. 수집 설정
# ==================================================

# 최종 목표:
# 유저당 정상 종료 경기 100개
TARGET_NORMAL_MATCHES = 70


# 경기 목록을 한 번에 몇 개씩 가져올지
PAGE_SIZE = 100


# 한 유저당 최대 몇 경기까지 과거로 탐색할지
#
# 예:
# 0 ~ 99
# 100 ~ 199
# 200 ~ 299
# 300 ~ 399
# 400 ~ 499
#
MAX_SEARCH_MATCHES = 500


# 공식경기 1vs1
MATCH_TYPE = 50


# 정상 요청 사이 대기
REQUEST_DELAY = 0.15


# 일반 네트워크 / 서버 오류
MAX_RETRIES = 3

RETRY_DELAY = 5


# ==================================================
# 4. 저장 경로
# ==================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

RAW_DIR = os.path.join(
    BASE_DIR,
    "raw_data"
)

os.makedirs(
    RAW_DIR,
    exist_ok=True
)


RESULT_FILE = os.path.join(
    BASE_DIR,
    "normal_70_users.json"
)


# ==================================================
# 5. 전역 상태
# ==================================================

api_request_count = 0

API_LIMIT_REACHED = False


# 이번 실행에서 새로 저장한 경기
processed_match_ids = set()


# ==================================================
# 6. 기존 저장 경기 목록
# ==================================================

saved_match_ids = set()

for file_name in os.listdir(RAW_DIR):

    if file_name.endswith(".json"):

        match_id = os.path.splitext(
            file_name
        )[0]

        saved_match_ids.add(
            match_id
        )


print("=" * 70)
print("FC Online 정상 종료 경기 수집 시작")
print("=" * 70)

print(
    "기존 저장 경기 수:",
    len(saved_match_ids)
)

print(
    "유저당 목표 정상 경기:",
    TARGET_NORMAL_MATCHES
)

print(
    "최대 탐색 경기:",
    MAX_SEARCH_MATCHES
)


# ==================================================
# 7. 결과
# ==================================================

eligible_users = []

incomplete_users = []


# ==================================================
# 8. API 제한 처리
# ==================================================

def set_api_limit_reached(message):

    global API_LIMIT_REACHED

    API_LIMIT_REACHED = True

    print()
    print("=" * 70)
    print("API 요청 제한(429) 발생")
    print(message)
    print("추가 API 요청을 즉시 중단합니다.")
    print("=" * 70)


# ==================================================
# 9. 공통 GET 요청
# ==================================================

def api_get(url, params, description):

    global api_request_count
    global API_LIMIT_REACHED

    if API_LIMIT_REACHED:
        return None

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        try:

            response = requests.get(
                url,
                headers=headers,
                params=params,
                timeout=15
            )

            api_request_count += 1

        except requests.exceptions.RequestException as e:

            print(
                f"[네트워크 오류] {description} "
                f"({attempt}/{MAX_RETRIES}):",
                e
            )

            if attempt < MAX_RETRIES:

                time.sleep(
                    RETRY_DELAY
                )

                continue

            return None


        # ------------------------------------------
        # 성공
        # ------------------------------------------

        if response.status_code == 200:

            return response.json()


        # ------------------------------------------
        # 429
        # ------------------------------------------

        if response.status_code == 429:

            set_api_limit_reached(
                description
            )

            return None


        # ------------------------------------------
        # 서버 오류
        # ------------------------------------------

        if 500 <= response.status_code < 600:

            print(
                f"[서버 오류] {description} "
                f"상태 코드: {response.status_code} "
                f"({attempt}/{MAX_RETRIES})"
            )

            if attempt < MAX_RETRIES:

                time.sleep(
                    RETRY_DELAY
                )

                continue

            return None


        # ------------------------------------------
        # 기타 오류
        # ------------------------------------------

        print(
            f"[API 오류] {description} "
            f"상태 코드: {response.status_code}"
        )

        return None

    return None


# ==================================================
# 10. 닉네임 → OUID
# ==================================================

def get_ouid(nickname):

    url = (
        "https://open.api.nexon.com/"
        "fconline/v1/id"
    )

    params = {
        "nickname": nickname
    }

    return_data = api_get(
        url,
        params,
        f"OUID 조회: {nickname}"
    )

    if return_data is None:
        return None

    return return_data.get(
        "ouid"
    )


# ==================================================
# 11. 경기 목록 조회
# ==================================================
#
# ★ 기존 코드와 가장 중요한 차이
#
# offset을 외부에서 전달받음.
#
# offset = 0
# offset = 100
# offset = 200
# ...
#
# 이렇게 과거 경기로 계속 이동 가능
# ==================================================

def get_match_ids(
    ouid,
    offset,
    limit
):

    url = (
        "https://open.api.nexon.com/"
        "fconline/v1/user/match"
    )

    params = {
        "ouid": ouid,
        "matchtype": MATCH_TYPE,
        "offset": offset,
        "limit": limit
    }

    data = api_get(
        url,
        params,
        (
            f"경기 목록 조회 "
            f"ouid={ouid}, offset={offset}"
        )
    )

    if data is None:
        return []

    return data


# ==================================================
# 12. 경기 상세 조회
# ==================================================

def get_match_detail(match_id):

    url = (
        "https://open.api.nexon.com/"
        "fconline/v1/match-detail"
    )

    params = {
        "matchid": match_id
    }

    return api_get(
        url,
        params,
        f"경기 상세 조회: {match_id}"
    )


# ==================================================
# 13. 기존 JSON 읽기
# ==================================================

def load_saved_match(match_id):

    file_path = os.path.join(
        RAW_DIR,
        f"{match_id}.json"
    )

    if not os.path.exists(
        file_path
    ):
        return None

    try:

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    except (
        OSError,
        json.JSONDecodeError
    ) as e:

        print(
            "[기존 JSON 읽기 실패]",
            match_id,
            e
        )

        return None


# ==================================================
# 14. JSON 저장
# ==================================================

def save_match(
    match_id,
    match_data
):

    file_path = os.path.join(
        RAW_DIR,
        f"{match_id}.json"
    )

    try:

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

        saved_match_ids.add(
            match_id
        )

        processed_match_ids.add(
            match_id
        )

        return True

    except OSError as e:

        print(
            "[JSON 저장 실패]",
            match_id,
            e
        )

        return False


# ==================================================
# 15. 현재 경기에서 해당 유저가 정상 종료했는지 확인
# ==================================================

def is_normal_match_for_user(
    match_data,
    ouid
):

    if not isinstance(
        match_data,
        dict
    ):
        return False


    match_info_list = match_data.get(
        "matchInfo",
        []
    )


    if not isinstance(
        match_info_list,
        list
    ):
        return False


    for player_info in match_info_list:

        if not isinstance(
            player_info,
            dict
        ):
            continue


        # 현재 수집 대상 유저인지 확인
        if player_info.get(
            "ouid"
        ) != ouid:

            continue


        match_detail = player_info.get(
            "matchDetail",
            {}
        )


        # matchEndType == 0이면 정상 종료
        return (
            match_detail.get(
                "matchEndType"
            ) == 0
        )


    return False


# ==================================================
# 16. 유저별 수집
# ==================================================

for user_number, nickname in enumerate(
    nicknames,
    start=1
):

    if API_LIMIT_REACHED:
        break


    print()
    print("=" * 70)

    print(
        f"[{user_number}/{len(nicknames)}] "
        f"{nickname}"
    )

    print("=" * 70)


    # ==================================================
    # 16-1. OUID
    # ==================================================

    ouid = get_ouid(
        nickname
    )


    if API_LIMIT_REACHED:
        break


    if ouid is None:

        print(
            "OUID 조회 실패"
        )

        incomplete_users.append({
            "nickname": nickname,
            "reason": "OUID 조회 실패"
        })

        continue


    print(
        "OUID:",
        ouid
    )


    time.sleep(
        REQUEST_DELAY
    )


    # ==================================================
    # 16-2. 유저별 상태
    # ==================================================

    normal_match_ids = []

    seen_match_ids = set()

    existing_count = 0

    new_count = 0

    abnormal_count = 0

    fail_count = 0

    searched_count = 0


    # ==================================================
    # 16-3. 70경기씩 과거 방향으로 탐색
    # ==================================================

    for offset in range(
        0,
        MAX_SEARCH_MATCHES,
        PAGE_SIZE
    ):

        if API_LIMIT_REACHED:
            break


        # 이미 정상경기 100개면 끝
        if len(normal_match_ids) >= TARGET_NORMAL_MATCHES:
            break


        print()
        print(
            f"[경기 목록 탐색] "
            f"offset={offset}"
        )


        match_ids = get_match_ids(
            ouid,
            offset,
            PAGE_SIZE
        )


        if API_LIMIT_REACHED:
            break


        # 더 이상 경기 없음
        if not match_ids:

            print(
                "더 이상 조회 가능한 경기가 없습니다."
            )

            break


        # 실제 반환 개수
        page_match_count = len(
            match_ids
        )


        print(
            "이번 목록 경기 수:",
            page_match_count
        )


        time.sleep(
            REQUEST_DELAY
        )


        # ==================================================
        # 각 경기 확인
        # ==================================================

        for match_id in match_ids:

            if API_LIMIT_REACHED:
                break


            # 정상 70경기 확보 즉시 종료
            if len(normal_match_ids) >= TARGET_NORMAL_MATCHES:
                break


            # 같은 유저 탐색 중 중복 방지
            if match_id in seen_match_ids:
                continue


            seen_match_ids.add(
                match_id
            )

            searched_count += 1


            # ------------------------------------------
            # 기존 JSON 존재
            # ------------------------------------------

            if match_id in saved_match_ids:

                match_data = load_saved_match(
                    match_id
                )

                if match_data is not None:

                    existing_count += 1

                    if is_normal_match_for_user(
                        match_data,
                        ouid
                    ):

                        normal_match_ids.append(
                            match_id
                        )

                        print(
                            f"[기존] 정상 "
                            f"{len(normal_match_ids)}/"
                            f"{TARGET_NORMAL_MATCHES}"
                        )

                    else:

                        abnormal_count += 1

                    continue


            # ------------------------------------------
            # 새로운 경기
            # → 상세 API 호출
            # ------------------------------------------

            match_data = get_match_detail(
                match_id
            )


            if API_LIMIT_REACHED:
                break


            if match_data is None:

                fail_count += 1

                continue


            # ------------------------------------------
            # JSON 저장
            # ------------------------------------------

            if save_match(
                match_id,
                match_data
            ):

                new_count += 1

            else:

                fail_count += 1

                continue


            # ------------------------------------------
            # 정상 종료 여부 확인
            # ------------------------------------------

            if is_normal_match_for_user(
                match_data,
                ouid
            ):

                normal_match_ids.append(
                    match_id
                )

                print(
                    f"[신규] 정상 "
                    f"{len(normal_match_ids)}/"
                    f"{TARGET_NORMAL_MATCHES}"
                )

            else:

                abnormal_count += 1

                print(
                    "[신규] 비정상 종료"
                )


            time.sleep(
                REQUEST_DELAY
            )


        # ==================================================
        # 페이지 끝
        # ==================================================

        print()
        print(
            f"현재 정상경기: "
            f"{len(normal_match_ids)}/"
            f"{TARGET_NORMAL_MATCHES}"
        )


        # 100개 달성
        if len(normal_match_ids) >= TARGET_NORMAL_MATCHES:
            break


        # 100개보다 적게 반환되었다면
        # 더 과거 경기가 없는 것으로 판단
        if page_match_count < PAGE_SIZE:

            print(
                "마지막 경기 목록에 도달했습니다."
            )

            break


    # ==================================================
    # 16-4. 유저 결과
    # ==================================================

    normal_count = len(
        normal_match_ids
    )


    print()
    print("-" * 70)
    print(
        f"[{nickname} 결과]"
    )

    print(
        "탐색한 경기:",
        searched_count
    )

    print(
        "기존 JSON 사용:",
        existing_count
    )

    print(
        "새로 저장:",
        new_count
    )

    print(
        "비정상 종료:",
        abnormal_count
    )

    print(
        "조회 실패:",
        fail_count
    )

    print(
        "정상 종료 확보:",
        normal_count
    )

    print(
        "현재까지 API 요청:",
        api_request_count
    )

    print("-" * 70)


    # ==================================================
    # 16-5. 70경기 성공
    # ==================================================

    if normal_count >= TARGET_NORMAL_MATCHES:

        # 혹시 모르니 정확히 100개만 기록
        normal_match_ids = normal_match_ids[
            :TARGET_NORMAL_MATCHES
        ]


        print(
            f"✅ {nickname}: "
            "정상 종료 70경기 확보 완료"
        )


        eligible_users.append({

            "nickname":
                nickname,

            "ouid":
                ouid,

            "normal_match_count":
                TARGET_NORMAL_MATCHES,

            "normal_match_ids":
                normal_match_ids
        })


    # ==================================================
    # 16-6. 미완료
    # ==================================================

    else:

        if API_LIMIT_REACHED:

            reason = (
                "API 429 제한으로 수집 중단"
            )

        else:

            reason = (
                "정상 종료 70경기 미확보"
            )


        print(
            f"❌ {nickname}: "
            f"정상 종료 {normal_count}경기"
        )


        incomplete_users.append({

            "nickname":
                nickname,

            "ouid":
                ouid,

            "normal_match_count":
                normal_count,

            "reason":
                reason
        })


    # ==================================================
    # 16-7. 429이면 전체 종료
    # ==================================================

    if API_LIMIT_REACHED:

        print()
        print("=" * 70)
        print(
            "API 요청 제한으로 전체 수집을 종료합니다."
        )
        print(
            "지금까지 저장된 JSON은 그대로 유지됩니다."
        )
        print(
            "다음 실행에서 기존 JSON을 다시 사용합니다."
        )
        print("=" * 70)

        break


# ==================================================
# 17. 결과 JSON 저장
# ==================================================

result_data = {

    "target_normal_matches":
        TARGET_NORMAL_MATCHES,

    "match_type":
        MATCH_TYPE,

    "max_search_matches":
        MAX_SEARCH_MATCHES,

    "api_limit_reached":
        API_LIMIT_REACHED,

    "api_request_count":
        api_request_count,

    "eligible_user_count":
        len(eligible_users),

    "eligible_users":
        eligible_users,

    "incomplete_user_count":
        len(incomplete_users),

    "incomplete_users":
        incomplete_users
}


with open(
    RESULT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        result_data,
        f,
        ensure_ascii=False,
        indent=2
    )


# ==================================================
# 18. 최종 결과
# ==================================================

json_files = [

    file_name

    for file_name in os.listdir(
        RAW_DIR
    )

    if file_name.endswith(
        ".json"
    )
]


print()
print("=" * 70)
print("FC Online 정상 종료 경기 수집 종료")
print("=" * 70)

print(
    "현재 raw_data 고유 경기 수:",
    len(json_files)
)

print(
    "이번 실행 API 요청 수:",
    api_request_count
)

print(
    "이번 실행 새로 저장한 경기:",
    len(processed_match_ids)
)

print(
    "정상 종료 70경기 달성 유저:",
    len(eligible_users)
)

print(
    "미완료 유저:",
    len(incomplete_users)
)

print(
    "API 제한 발생:",
    API_LIMIT_REACHED
)

print(
    "결과 저장:",
    RESULT_FILE
)


print()
print("[정상 종료 100경기 달성]")

if eligible_users:

    for user in eligible_users:

        print(
            "-",
            user["nickname"],
            "/",
            user["normal_match_count"],
            "경기"
        )

else:

    print("없음")


print()
print("[미완료]")

if incomplete_users:

    for user in incomplete_users:

        print(
            "-",
            user["nickname"],
            "/",
            user.get(
                "normal_match_count",
                0
            ),
            "/",
            user["reason"]
        )

else:

    print("없음")


if API_LIMIT_REACHED:

    print()
    print(
        "⚠️ API 429 제한 때문에 중단되었습니다."
    )

    print(
        "이미 저장된 JSON은 삭제되지 않았습니다."
    )

    print(
        "제한이 풀린 뒤 같은 코드를 다시 실행하면 됩니다."
    )


print("=" * 70)