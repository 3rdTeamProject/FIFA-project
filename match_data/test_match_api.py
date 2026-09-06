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


# API KEY 확인
if not API_KEY:
    raise ValueError(
        "NEXON_API_KEY가 없습니다. "
        ".env 파일을 확인해주세요."
    )


headers = {
    "x-nxopen-api-key": API_KEY
}


# ==================================================
# 수집할 유저
# ==================================================
#
# 목표:
# 약 40~50명
# 유저당 최근 20경기
#
# 기존에 수집했던 유저보다
# 새로운 유저를 넣는 것을 권장
# ==================================================

nicknames = [
    "리센느원E",
    "빠다하트영철",
    "윤비뇨기과",
    "닭집닭똥집",
    "부캐키우기",
    "MadridistaCF",
    "로또일등시켜주라",
    "펠레친구팰래",
    "보정받는중입니다",
    "영악",
    "순갓",

    "ll아재ll",
    "킹오넬갓시",
    "강X수",
    "ASEN1",
    "창업주",
    "오빵달료",
    "축명",
    "차관",
    "기뭐녁",
    "영농의희망찬내일",

    "포람페911",
    "dop쭌",
    "신림동하늘다람쥐",
    "사딸란타",
    "영천퀵",
    "타겟터메시",
    "까칠한명수씨",
    "해외출장그만",
    "FIFA구독",
    "금나우지뉴",

    "하이도",
    "밍크왕",
    "명수는십잡스",
    "스주니",
    "순갓",
    "ckhy",
    "서휘재",
    "조지호마짤",
    "ZZFC",
    "앙리마미",

    "오픈ai",
    "가나가나가나나가",
    "방구석레알",
    "ACMilan밀라노",
    "쿠마이누",
    "유혹하는웃음폭탄",
    "바른구단주명9487",
    "ATM한국지부장",
    "DCB호날두",
    "IIIllIIlllIl",

    "AstonVillans",
    "WALTER11",
    "최강슈",
    "다크서클레인저스",
    "행복한베컴",
    "ll아재ll",
    "AbsoluteLegend",
    "하쿠지Hakuji",
    "채팅좀예쁘게해라",
    "감성티스푼",

    "엘에프씨",
    "고수르",
    "위송빠레",
    "EXCELLENTVISION",
    "사비에르난데스s",
    "원블루",
    "15일뒤사라짐",
    "판타스틱준",
    "뛰어봐",
    "TopPrice0"

    # 여기에 새로운 닉네임 추가
]


# ==================================================
# 수집 설정
# ==================================================

# 한 유저당 최근 경기 수
MATCH_LIMIT = 20

# 공식경기 1vs1
MATCH_TYPE = 50

# 정상 요청 사이 대기 시간
REQUEST_DELAY = 0.15

# 재시도 횟수
MAX_RETRIES = 3

# 재시도 대기 시간
RETRY_DELAY = 5


# ==================================================
# 저장 경로
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


# ==================================================
# API 요청 횟수 카운터
# ==================================================

api_request_count = 0


# ==================================================
# 이미 저장된 matchId 불러오기
# ==================================================
#
# raw_data 폴더에 이미 저장되어 있는 JSON 파일은
# 다시 API로 상세 조회하지 않는다.
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
print("FC Online 경기 데이터 수집 시작")
print("=" * 70)

print(
    "기존 저장 경기 수:",
    len(saved_match_ids)
)


# ==================================================
# 이번 실행에서 성공적으로 처리한 경기
# ==================================================
#
# 여러 유저의 경기 목록에 같은 matchId가
# 반복해서 등장할 수 있음.
#
# 성공적으로 상세 조회한 경기만 여기에 추가한다.
#
# 실패한 경기는 넣지 않기 때문에
# 다른 유저의 경기 목록에서 다시 발견되면
# 다시 시도할 수 있다.
# ==================================================

processed_match_ids = set()


# ==================================================
# OUID 조회
# ==================================================

def get_ouid(nickname):

    global api_request_count

    url = (
        "https://open.api.nexon.com/"
        "fconline/v1/id"
    )

    params = {
        "nickname": nickname
    }


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
                f"[OUID 네트워크 오류] "
                f"{nickname} "
                f"({attempt}/{MAX_RETRIES}):",
                e
            )

            if attempt < MAX_RETRIES:

                time.sleep(
                    RETRY_DELAY
                )

                continue

            return None


        # 정상 응답
        if response.status_code == 200:

            data = response.json()

            return data.get(
                "ouid"
            )


        # 요청 제한
        if response.status_code == 429:

            print(
                f"[OUID 429 요청 제한] "
                f"{nickname} "
                f"({attempt}/{MAX_RETRIES})"
            )

            if attempt < MAX_RETRIES:

                time.sleep(
                    10
                )

                continue

            return None


        # 서버 오류
        if 500 <= response.status_code < 600:

            print(
                f"[OUID 서버 오류] "
                f"{nickname} "
                f"상태 코드: "
                f"{response.status_code} "
                f"({attempt}/{MAX_RETRIES})"
            )

            if attempt < MAX_RETRIES:

                time.sleep(
                    RETRY_DELAY
                )

                continue

            return None


        # 그 외 오류
        print(
            f"[OUID 조회 실패] "
            f"{nickname} "
            f"상태 코드: "
            f"{response.status_code}"
        )

        return None


    return None


# ==================================================
# 경기 목록 조회
# ==================================================

def get_match_ids(ouid):

    global api_request_count

    url = (
        "https://open.api.nexon.com/"
        "fconline/v1/user/match"
    )

    params = {
        "ouid": ouid,
        "matchtype": MATCH_TYPE,
        "offset": 0,
        "limit": MATCH_LIMIT
    }


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
                f"[경기 목록 네트워크 오류] "
                f"({attempt}/{MAX_RETRIES}):",
                e
            )

            if attempt < MAX_RETRIES:

                time.sleep(
                    RETRY_DELAY
                )

                continue

            return []


        # 정상 응답
        if response.status_code == 200:

            return response.json()


        # 요청 제한
        if response.status_code == 429:

            print(
                f"[경기 목록 429 요청 제한] "
                f"({attempt}/{MAX_RETRIES})"
            )

            if attempt < MAX_RETRIES:

                time.sleep(
                    10
                )

                continue

            return []


        # 서버 오류
        if 500 <= response.status_code < 600:

            print(
                f"[경기 목록 서버 오류] "
                f"상태 코드: "
                f"{response.status_code} "
                f"({attempt}/{MAX_RETRIES})"
            )

            if attempt < MAX_RETRIES:

                time.sleep(
                    RETRY_DELAY
                )

                continue

            return []


        # 그 외 오류
        print(
            "[경기 목록 조회 실패]",
            "상태 코드:",
            response.status_code
        )

        return []


    return []


# ==================================================
# 경기 상세 조회
# ==================================================

def get_match_detail(match_id):

    global api_request_count

    url = (
        "https://open.api.nexon.com/"
        "fconline/v1/match-detail"
    )

    params = {
        "matchid": match_id
    }


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
                f"[경기 상세 네트워크 오류] "
                f"{match_id} "
                f"({attempt}/{MAX_RETRIES}):",
                e
            )

            # 네트워크 오류도 재시도
            if attempt < MAX_RETRIES:

                time.sleep(
                    RETRY_DELAY
                )

                continue

            return None


        # ------------------------------------------
        # 정상 응답
        # ------------------------------------------

        if response.status_code == 200:

            return response.json()


        # ------------------------------------------
        # 요청 제한
        # ------------------------------------------

        if response.status_code == 429:

            print(
                f"[429 요청 제한] "
                f"{match_id} "
                f"({attempt}/{MAX_RETRIES})"
            )

            if attempt < MAX_RETRIES:

                time.sleep(
                    10
                )

                continue

            return None


        # ------------------------------------------
        # 서버 오류
        # ------------------------------------------

        if 500 <= response.status_code < 600:

            print(
                f"[서버 오류] "
                f"{match_id} "
                f"상태 코드: "
                f"{response.status_code} "
                f"({attempt}/{MAX_RETRIES})"
            )

            if attempt < MAX_RETRIES:

                time.sleep(
                    RETRY_DELAY
                )

                continue

            return None


        # ------------------------------------------
        # 그 외 오류
        # ------------------------------------------

        print(
            "[경기 상세 조회 실패]",
            match_id,
            "상태 코드:",
            response.status_code
        )

        return None


    return None


# ==================================================
# 유저별 데이터 수집
# ==================================================

for user_number, nickname in enumerate(
    nicknames,
    start=1
):

    print()
    print("=" * 70)

    print(
        f"{user_number}/{len(nicknames)} "
        f"유저 수집 시작: {nickname}"
    )

    print("=" * 70)


    # ----------------------------------------------
    # 1. OUID 조회
    # ----------------------------------------------

    ouid = get_ouid(
        nickname
    )


    if ouid is None:

        print(
            f"{nickname} "
            f"OUID 조회 실패 → "
            f"이 유저는 건너뜁니다."
        )

        continue


    print(
        "OUID:",
        ouid
    )


    time.sleep(
        REQUEST_DELAY
    )


    # ----------------------------------------------
    # 2. 최근 경기 목록 조회
    # ----------------------------------------------

    match_ids = get_match_ids(
        ouid
    )


    print(
        f"{nickname}의 API 반환 경기 수:",
        len(match_ids)
    )


    if len(match_ids) == 0:

        print(
            f"{nickname}: "
            f"수집 가능한 경기가 없습니다."
        )

        continue


    time.sleep(
        REQUEST_DELAY
    )


    # ----------------------------------------------
    # 유저별 통계
    # ----------------------------------------------

    new_count = 0

    existing_count = 0

    duplicate_in_run_count = 0

    fail_count = 0


    # ----------------------------------------------
    # 3. 경기 상세 조회
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
        # 이미 과거에 저장된 경기
        # ==========================================

        if match_id in saved_match_ids:

            existing_count += 1

            print(
                f"{match_number}번째 경기 "
                f"기존 저장 데이터 사용:",
                match_id
            )

            continue


        # ==========================================
        # 이번 실행 중 이미 성공적으로 처리한 경기
        # ==========================================

        if match_id in processed_match_ids:

            duplicate_in_run_count += 1

            print(
                f"{match_number}번째 경기 "
                f"이번 실행 중 중복:",
                match_id
            )

            continue


        # ==========================================
        # 실제 API 상세 조회
        # ==========================================

        match_data = get_match_detail(
            match_id
        )


        # ==========================================
        # API 상세 조회 실패
        # ==========================================
        #
        # processed_match_ids에는 넣지 않는다.
        #
        # 따라서 다른 유저의 경기 목록에서
        # 같은 matchId가 다시 발견되면 재시도 가능.
        # ==========================================

        if match_data is None:

            fail_count += 1

            print(
                f"{match_number}번째 경기 "
                f"조회/저장 실패:",
                match_id
            )

            continue


        # ==========================================
        # JSON 저장
        # ==========================================

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


        except OSError as e:

            fail_count += 1

            print(
                f"{match_number}번째 경기 "
                f"파일 저장 실패:",
                match_id,
                e
            )

            continue


        # ==========================================
        # 저장까지 성공한 경기만 처리 완료
        # ==========================================

        processed_match_ids.add(
            match_id
        )

        saved_match_ids.add(
            match_id
        )


        new_count += 1


        print(
            f"{match_number}번째 경기 "
            f"저장 완료:",
            match_id
        )


        time.sleep(
            REQUEST_DELAY
        )


    # ----------------------------------------------
    # 유저별 결과 출력
    # ----------------------------------------------

    print()

    print(
        f"[{nickname} 수집 결과]"
    )

    print(
        "API 반환 경기 수:",
        len(match_ids)
    )

    print(
        "새로 저장:",
        new_count
    )

    print(
        "기존 저장 경기:",
        existing_count
    )

    print(
        "이번 실행 중 중복:",
        duplicate_in_run_count
    )

    print(
        "실패:",
        fail_count
    )

    print(
        "현재까지 API 요청 수:",
        api_request_count
    )


# ==================================================
# 전체 결과
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

print(
    "경기 데이터 수집 완료"
)

print("=" * 70)

print(
    "현재 raw_data 고유 경기 수:",
    len(json_files)
)

print(
    "이번 실행에서 API 요청 수:",
    api_request_count
)

print(
    "이번 실행에서 새로 저장한 고유 경기 수:",
    len(processed_match_ids)
)

print("=" * 70)