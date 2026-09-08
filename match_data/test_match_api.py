import os
import json
import time
import requests

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


HEADERS = {
    "x-nxopen-api-key": API_KEY
}


# ==================================================
# 2. 수집 대상 닉네임
# ==================================================
#
# 기존에 수집했던 유저 + 추가로 확인할 유저를
# 여기에 넣으면 됨.
#
# 기존 raw_data에서 정상 경기를 먼저 확인하고
# 부족한 유저만 추가 수집함.
# ==================================================

nicknames = [
    "15일뒤사라짐",
    "2014MSN",
    "ACMilan밀라노",
    "ASEN1",
    "ATM한국지부장",
    "AbsoluteLegend",
    "AstonVillans",
    "BFXKaiser",
    "BFXNoiZ",
    "BenzJCW",
    "DCB호날두",
    "DRXSavior",
    "EXCELLENTVISION",
    "Exit0",
    "FIFA구독",
    "GCTCrong",
    "GenGTen",
    "IIIllIIlllIl",
    "KRXChan",
    "KRXTak",
    "MadridistaCF",
    "T1정성민",
    "TopPrice0",
    "WALTER11",
    "ZSia시아",
    "ZZFC",
    "ckhy",
    "dop쭌",
    "ll아재ll",
    "satthesails",
    "snag1wonNoparent",
    "가나가나가나나가",
    "감성티스푼",
    "강X수",
    "걸리면빽태클",
    "고수르",
    "극소수",
    "금나우지뉴",
    "기뭐녁",
    "까칠한명수씨",
    "다크서클레인저스",
    "닭집닭똥집",
    "당신은승리자",
    "디발라의낭만",
    "뛰어봐",
    "로또일등시켜주라",
    "리바이브말맨",
    "리바이브이원희",
    "리센느원E",
    "명수는십잡스",
    "물안경남친",
    "밍크왕",
    "바른구단주명9487",
    "방구석레알",
    "보구리너지",
    "보정받는중입니다",
    "복숭아맛사과",
    "부캐키우기",
    "빠다하트영철",
    "사딸란타",
    "사비에르난데스s",
    "서휘재",
    "순갓",
    "스주니",
    "신림동하늘다람쥐",
    "앙리마미",
    "에이스더브라위너",
    "엘에프씨",
    "영농의희망찬내일",
    "영악",
    "영천퀵",
    "오빵달료",
    "오픈ai",
    "원블루",
    "위송빠레",
    "유혹하는웃음폭탄",
    "윤비뇨기과",
    "으감",
    "제이드",
    "조지호마짤",
    "차관",
    "창업주",
    "채팅좀예쁘게해라",
    "최강슈",
    "축명",
    "쿠마이누",
    "킹오넬갓시",
    "타겟터메시",
    "태연",
    "판타스틱준",
    "펠레친구팰래",
    "포람페911",
    "하이도",
    "하쿠지Hakuji",
    "해외출장그만",
    "행복한베컴"
]


# ==================================================
# 3. 수집 기준
# ==================================================

# 정상 종료 경기 목표
TARGET_NORMAL_MATCHES = 70


# 최종 사용 가능 최소 기준
#
# 주의:
# 65경기가 되었다고 수집을 멈추는 것이 아님.
# 항상 가능한 경우 70경기까지 수집함.
MIN_NORMAL_MATCHES = 65


# 공식경기 1vs1
MATCH_TYPE = 50


# 경기 목록 한 번 조회 개수
PAGE_SIZE = 100


# 한 유저당 최대 몇 경기까지 과거로 탐색할지
MAX_SEARCH_MATCHES = 500


# API 요청 사이 대기 시간
REQUEST_DELAY = 0.15


# 오류 발생 시 재시도
MAX_RETRIES = 3

RETRY_DELAY = 5


# ==================================================
# 4. 경로 설정
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


# 이번 수집 결과
RESULT_FILE = os.path.join(
    BASE_DIR,
    "match_collection_result.json"
)


# 신규 상세 API를 호출했지만
# 몰수/비정상 종료라서 저장하지 않은 matchId 기록
#
# 다음 실행 때 같은 경기를 또 상세 조회하는 것을 방지
ABNORMAL_ID_FILE = os.path.join(
    BASE_DIR,
    "abnormal_match_ids.json"
)


# ==================================================
# 5. 전역 상태
# ==================================================

api_request_count = 0

API_LIMIT_REACHED = False


# 이번 실행에서 실제로 새로 저장된 정상 경기
new_saved_match_ids = set()


# ==================================================
# 6. 비정상 경기 ID 목록 불러오기
# ==================================================

abnormal_match_ids = set()


if os.path.exists(ABNORMAL_ID_FILE):

    try:

        with open(
            ABNORMAL_ID_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            loaded_ids = json.load(f)


        if isinstance(loaded_ids, list):

            abnormal_match_ids = set(
                loaded_ids
            )


    except (
        OSError,
        json.JSONDecodeError
    ) as e:

        print(
            "[경고] abnormal_match_ids.json "
            "읽기 실패:",
            e
        )


# ==================================================
# 7. 비정상 경기 ID 저장
# ==================================================

def save_abnormal_ids():

    try:

        with open(
            ABNORMAL_ID_FILE,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                sorted(abnormal_match_ids),
                f,
                ensure_ascii=False,
                indent=2
            )


    except OSError as e:

        print(
            "[경고] 비정상 경기 ID 저장 실패:",
            e
        )


# ==================================================
# 8. 기존 raw_data 전체 검사
# ==================================================
#
# API 호출 없음.
#
# 기존 JSON을 직접 읽어서:
#
# - 어떤 OUID의 경기인지
# - 정상 종료인지
# - 기존 정상 경기가 몇 개인지
#
# 모두 파악함.
# ==================================================

saved_match_ids = set()


# OUID별 정상 경기 ID
existing_normal_by_ouid = {}


# 닉네임 → OUID 후보
nickname_to_ouids = {}


print("=" * 70)
print("기존 raw_data 검사 시작")
print("=" * 70)


json_files = sorted([
    file_name
    for file_name in os.listdir(RAW_DIR)
    if file_name.endswith(".json")
])


for index, file_name in enumerate(
    json_files,
    start=1
):

    match_id = os.path.splitext(
        file_name
    )[0]


    saved_match_ids.add(
        match_id
    )


    file_path = os.path.join(
        RAW_DIR,
        file_name
    )


    try:

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as f:

            match_data = json.load(f)


    except (
        OSError,
        json.JSONDecodeError
    ):

        continue


    match_info_list = match_data.get(
        "matchInfo",
        []
    )


    if not isinstance(
        match_info_list,
        list
    ):
        continue


    for user_info in match_info_list:

        if not isinstance(
            user_info,
            dict
        ):
            continue


        ouid = user_info.get(
            "ouid"
        )

        nickname = user_info.get(
            "nickname"
        )


        if not ouid:
            continue


        # ------------------------------------------
        # 닉네임 → OUID 관계 저장
        # ------------------------------------------

        if nickname:

            nickname_to_ouids.setdefault(
                nickname,
                set()
            ).add(
                ouid
            )


        # ------------------------------------------
        # 정상 종료 여부
        # ------------------------------------------

        match_detail = user_info.get(
            "matchDetail",
            {}
        )


        match_end_type = match_detail.get(
            "matchEndType"
        )


        if match_end_type == 0:

            existing_normal_by_ouid.setdefault(
                ouid,
                set()
            ).add(
                match_id
            )


    if (
        index % 500 == 0
        or index == len(json_files)
    ):

        print(
            f"{index}/{len(json_files)} "
            "기존 JSON 검사 완료"
        )


print()
print(
    "기존 raw_data JSON:",
    len(saved_match_ids)
)

print(
    "기록된 비정상 경기 ID:",
    len(abnormal_match_ids)
)

print("=" * 70)


# ==================================================
# 9. API 제한 처리
# ==================================================

def set_api_limit_reached(
    message
):

    global API_LIMIT_REACHED

    API_LIMIT_REACHED = True


    print()
    print("=" * 70)
    print("⚠️ API 429 요청 제한 발생")
    print(message)
    print("추가 API 요청을 중단합니다.")
    print("=" * 70)


# ==================================================
# 10. 공통 API GET
# ==================================================

def api_get(
    url,
    params,
    description
):

    global api_request_count


    if API_LIMIT_REACHED:
        return None


    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        try:

            response = requests.get(
                url,
                headers=HEADERS,
                params=params,
                timeout=15
            )


            api_request_count += 1


        except requests.exceptions.RequestException as e:

            print(
                f"[네트워크 오류] "
                f"{description} "
                f"({attempt}/{MAX_RETRIES}) "
                f"{e}"
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
        # API 제한
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
                f"[서버 오류] "
                f"{description} "
                f"status={response.status_code}"
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
            f"[API 오류] "
            f"{description} "
            f"status={response.status_code}"
        )

        return None


    return None


# ==================================================
# 11. 닉네임 → OUID API
# ==================================================

def get_ouid(
    nickname
):

    url = (
        "https://open.api.nexon.com/"
        "fconline/v1/id"
    )


    params = {
        "nickname": nickname
    }


    data = api_get(
        url,
        params,
        f"OUID 조회: {nickname}"
    )


    if data is None:
        return None


    return data.get(
        "ouid"
    )


# ==================================================
# 12. 경기 목록 API
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
            f"offset={offset}"
        )
    )


    if data is None:
        return []


    return data


# ==================================================
# 13. 경기 상세 API
# ==================================================

def get_match_detail(
    match_id
):

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
# 14. 기존 JSON 읽기
# ==================================================

def load_saved_match(
    match_id
):

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
    ):

        return None


# ==================================================
# 15. 정상 종료 여부
# ==================================================
#
# 현재 수집 대상 유저의
# matchEndType == 0일 때만 True
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


    for user_info in match_info_list:

        if not isinstance(
            user_info,
            dict
        ):
            continue


        if user_info.get(
            "ouid"
        ) != ouid:

            continue


        match_detail = user_info.get(
            "matchDetail",
            {}
        )


        return (
            match_detail.get(
                "matchEndType"
            ) == 0
        )


    return False


# ==================================================
# 16. 정상 경기 JSON 저장
# ==================================================

def save_normal_match(
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


        new_saved_match_ids.add(
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
# 17. 기존 데이터에서 OUID 찾기
# ==================================================

def get_existing_ouid(
    nickname
):

    ouids = nickname_to_ouids.get(
        nickname,
        set()
    )


    # 하나만 있으면 그대로 사용
    if len(ouids) == 1:

        return next(
            iter(ouids)
        )


    return None


# ==================================================
# 18. 결과 리스트
# ==================================================

target_users = []

usable_users = []

incomplete_users = []


# ==================================================
# 19. 유저별 처리
# ==================================================

print()
print("=" * 70)
print("FC Online 정상 경기 보충 수집 시작")
print("=" * 70)


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
    # 19-1. 기존 JSON에서 OUID 먼저 확인
    # ==================================================

    ouid = get_existing_ouid(
        nickname
    )


    # 기존 JSON에서 OUID를 찾을 수 없는 경우에만
    # 닉네임 → OUID API 호출
    if ouid is None:

        print(
            "기존 JSON에서 OUID 확인 불가 "
            "→ OUID API 조회"
        )


        ouid = get_ouid(
            nickname
        )


        if API_LIMIT_REACHED:
            break


        if ouid is None:

            print(
                "❌ OUID 조회 실패"
            )


            incomplete_users.append({
                "nickname": nickname,
                "normal_match_count": 0,
                "reason": "OUID 조회 실패"
            })

            continue


        time.sleep(
            REQUEST_DELAY
        )


    else:

        print(
            "기존 JSON에서 OUID 확인:",
            ouid
        )


    # ==================================================
    # 19-2. 기존 정상경기 확인
    # ==================================================

    normal_match_ids = set(
        existing_normal_by_ouid.get(
            ouid,
            set()
        )
    )


    existing_normal_count = len(
        normal_match_ids
    )


    print(
        "기존 정상 종료 경기:",
        existing_normal_count
    )


    # ==================================================
    # 이미 정상 70경기 이상
    # ==================================================

    if existing_normal_count >= TARGET_NORMAL_MATCHES:

        # 결과 기록은 최대 70개
        final_ids = sorted(
            normal_match_ids
        )[:TARGET_NORMAL_MATCHES]


        print(
            f"✅ 이미 정상 종료 "
            f"{TARGET_NORMAL_MATCHES}경기 이상 확보"
        )

        print(
            "→ 추가 경기 상세 API 호출 없음"
        )


        target_users.append({
            "nickname": nickname,
            "ouid": ouid,
            "normal_match_count":
                TARGET_NORMAL_MATCHES,
            "status": "목표 달성",
            "normal_match_ids": final_ids
        })

        continue


    # ==================================================
    # 19-3. 부족한 경우 70까지 추가 수집
    # ==================================================

    print(
        f"→ 정상 경기 "
        f"{TARGET_NORMAL_MATCHES - existing_normal_count}"
        f"개 부족"
    )

    print(
        "→ 정상 70경기까지 추가 수집 시도"
    )


    seen_match_ids = set()

    new_normal_count = 0

    new_abnormal_count = 0

    skipped_existing_count = 0

    skipped_abnormal_count = 0

    fail_count = 0


    # ==================================================
    # 경기 목록을 최근 → 과거 방향으로 탐색
    # ==================================================

    for offset in range(
        0,
        MAX_SEARCH_MATCHES,
        PAGE_SIZE
    ):

        if API_LIMIT_REACHED:
            break


        if len(normal_match_ids) >= TARGET_NORMAL_MATCHES:
            break


        print()
        print(
            f"[경기 목록 조회] offset={offset}"
        )


        match_ids = get_match_ids(
            ouid,
            offset,
            PAGE_SIZE
        )


        if API_LIMIT_REACHED:
            break


        if not match_ids:

            print(
                "더 이상 조회 가능한 경기가 없습니다."
            )

            break


        page_match_count = len(
            match_ids
        )


        print(
            "경기 목록 수:",
            page_match_count
        )


        time.sleep(
            REQUEST_DELAY
        )


        # ==================================================
        # 개별 경기 확인
        # ==================================================

        for match_id in match_ids:

            if API_LIMIT_REACHED:
                break


            if len(normal_match_ids) >= TARGET_NORMAL_MATCHES:
                break


            # ------------------------------------------
            # 같은 실행 내 중복
            # ------------------------------------------

            if match_id in seen_match_ids:
                continue


            seen_match_ids.add(
                match_id
            )


            # ------------------------------------------
            # 이미 정상경기로 확보한 경기
            # ------------------------------------------

            if match_id in normal_match_ids:

                skipped_existing_count += 1

                continue


            # ------------------------------------------
            # 기존 raw_data에 이미 있는 JSON
            # ------------------------------------------
            #
            # 기존 raw_data에는 과거에 수집한
            # 비정상 경기가 있을 수도 있음.
            #
            # 상세 API 재호출하지 않고
            # 기존 JSON을 직접 읽음.
            # ------------------------------------------

            if match_id in saved_match_ids:

                match_data = load_saved_match(
                    match_id
                )


                if match_data is None:

                    continue


                if is_normal_match_for_user(
                    match_data,
                    ouid
                ):

                    normal_match_ids.add(
                        match_id
                    )


                    print(
                        f"[기존 JSON 정상] "
                        f"{len(normal_match_ids)}/"
                        f"{TARGET_NORMAL_MATCHES}"
                    )


                else:

                    skipped_existing_count += 1


                continue


            # ------------------------------------------
            # 이전 실행에서 비정상으로 판정한 경기
            # ------------------------------------------

            if match_id in abnormal_match_ids:

                skipped_abnormal_count += 1

                continue


            # ==========================================
            # 여기부터 정말 새로운 경기
            # → 상세 API 호출
            # ==========================================

            match_data = get_match_detail(
                match_id
            )


            if API_LIMIT_REACHED:
                break


            if match_data is None:

                fail_count += 1

                continue


            # ==========================================
            # 가장 중요한 부분
            #
            # JSON 저장 전에 정상 종료 여부부터 확인
            # ==========================================

            if not is_normal_match_for_user(
                match_data,
                ouid
            ):

                new_abnormal_count += 1


                # raw_data에는 저장하지 않음
                abnormal_match_ids.add(
                    match_id
                )


                print(
                    "[제외] 몰수/비정상 종료 "
                    "→ JSON 저장 안 함"
                )


                time.sleep(
                    REQUEST_DELAY
                )

                continue


            # ==========================================
            # 정상 종료 경기만 raw_data 저장
            # ==========================================

            if save_normal_match(
                match_id,
                match_data
            ):

                normal_match_ids.add(
                    match_id
                )


                new_normal_count += 1


                print(
                    f"[신규 정상 저장] "
                    f"{len(normal_match_ids)}/"
                    f"{TARGET_NORMAL_MATCHES}"
                )


            else:

                fail_count += 1


            time.sleep(
                REQUEST_DELAY
            )


        # ==================================================
        # 현재 페이지 종료
        # ==================================================

        print(
            f"현재 정상 경기: "
            f"{len(normal_match_ids)}/"
            f"{TARGET_NORMAL_MATCHES}"
        )


        # 정상 70경기 달성
        if len(normal_match_ids) >= TARGET_NORMAL_MATCHES:

            break


        # 마지막 페이지
        if page_match_count < PAGE_SIZE:

            print(
                "마지막 경기 목록에 "
                "도달했습니다."
            )

            break


    # ==================================================
    # 비정상 ID 즉시 저장
    # ==================================================

    save_abnormal_ids()


    # ==================================================
    # 19-4. 유저 최종 결과
    # ==================================================

    normal_count = min(
        len(normal_match_ids),
        TARGET_NORMAL_MATCHES
    )


    final_ids = sorted(
        normal_match_ids
    )[:TARGET_NORMAL_MATCHES]


    print()
    print("-" * 70)

    print(
        f"[{nickname} 수집 결과]"
    )

    print(
        "기존 정상 경기:",
        existing_normal_count
    )

    print(
        "신규 정상 저장:",
        new_normal_count
    )

    print(
        "신규 몰수/비정상 제외:",
        new_abnormal_count
    )

    print(
        "기존 JSON 재사용/건너뜀:",
        skipped_existing_count
    )

    print(
        "기존 비정상 ID 건너뜀:",
        skipped_abnormal_count
    )

    print(
        "상세 조회 실패:",
        fail_count
    )

    print(
        "최종 정상 경기:",
        normal_count
    )

    print(
        "현재까지 API 요청:",
        api_request_count
    )

    print("-" * 70)


    # ==================================================
    # 70경기
    # ==================================================

    if normal_count >= TARGET_NORMAL_MATCHES:

        print(
            f"✅ 정상 종료 "
            f"{TARGET_NORMAL_MATCHES}경기 확보 완료"
        )


        target_users.append({
            "nickname": nickname,
            "ouid": ouid,
            "normal_match_count":
                TARGET_NORMAL_MATCHES,
            "status": "목표 달성",
            "normal_match_ids": final_ids
        })


    # ==================================================
    # 65 ~ 69경기
    # ==================================================

    elif normal_count >= MIN_NORMAL_MATCHES:

        print(
            f"⚠️ 정상 종료 "
            f"{normal_count}경기 확보"
        )

        print(
            f"→ 목표 {TARGET_NORMAL_MATCHES}경기는 "
            f"미달이지만 최소 "
            f"{MIN_NORMAL_MATCHES}경기 기준 충족"
        )


        usable_users.append({
            "nickname": nickname,
            "ouid": ouid,
            "normal_match_count":
                normal_count,
            "status": "최소 기준 충족",
            "normal_match_ids": final_ids
        })


    # ==================================================
    # 64경기 이하
    # ==================================================

    else:

        if API_LIMIT_REACHED:

            reason = (
                "API 429 제한으로 수집 중단"
            )

        else:

            reason = (
                f"정상 종료 "
                f"{MIN_NORMAL_MATCHES}경기 미만"
            )


        print(
            f"❌ 정상 종료 "
            f"{normal_count}경기"
        )


        incomplete_users.append({
            "nickname": nickname,
            "ouid": ouid,
            "normal_match_count":
                normal_count,
            "status": "최소 기준 미달",
            "reason": reason,
            "normal_match_ids": final_ids
        })


    if API_LIMIT_REACHED:
        break


# ==================================================
# 20. 비정상 ID 최종 저장
# ==================================================

save_abnormal_ids()


# ==================================================
# 21. 결과 JSON 저장
# ==================================================

result_data = {

    "target_normal_matches":
        TARGET_NORMAL_MATCHES,

    "minimum_normal_matches":
        MIN_NORMAL_MATCHES,

    "match_type":
        MATCH_TYPE,

    "max_search_matches":
        MAX_SEARCH_MATCHES,

    "api_limit_reached":
        API_LIMIT_REACHED,

    "api_request_count":
        api_request_count,

    "new_normal_json_count":
        len(new_saved_match_ids),

    "abnormal_id_count":
        len(abnormal_match_ids),

    "target_user_count":
        len(target_users),

    "target_users":
        target_users,

    "usable_user_count":
        len(usable_users),

    "usable_users":
        usable_users,

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
# 22. 최종 출력
# ==================================================

print()
print("=" * 70)
print("FC Online 정상 경기 보충 수집 종료")
print("=" * 70)


print(
    "이번 실행 API 요청:",
    api_request_count
)

print(
    "신규 정상 JSON 저장:",
    len(new_saved_match_ids)
)

print(
    "기록된 몰수/비정상 경기 ID:",
    len(abnormal_match_ids)
)

print(
    "정상 70경기 달성:",
    len(target_users)
)

print(
    "정상 65~69경기:",
    len(usable_users)
)

print(
    "정상 65경기 미만:",
    len(incomplete_users)
)


print()
print("[정상 70경기 달성]")


if target_users:

    for user in target_users:

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
print("[정상 65~69경기 - 사용 가능]")


if usable_users:

    for user in usable_users:

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
print("[정상 65경기 미만]")


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
            "경기 /",
            user.get(
                "reason",
                ""
            )
        )

else:

    print("없음")


print()
print(
    "수집 결과:",
    RESULT_FILE
)

print(
    "비정상 경기 ID:",
    ABNORMAL_ID_FILE
)


if API_LIMIT_REACHED:

    print()
    print(
        "⚠️ API 429 제한으로 중단되었습니다."
    )

    print(
        "저장된 정상 JSON과 "
        "비정상 경기 ID 기록은 유지됩니다."
    )

    print(
        "제한이 풀린 뒤 같은 코드를 "
        "다시 실행하면 이어서 수집할 수 있습니다."
    )


print("=" * 70)