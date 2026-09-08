import os
import json
import pandas as pd


# ==================================================
# 1. 경로 설정
# ==================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

RAW_DIR = os.path.join(
    BASE_DIR,
    "raw_data"
)

MATCH_TEAM_CSV = os.path.join(
    BASE_DIR,
    "match_team_data.csv"
)


# ==================================================
# 2. 저장할 데이터
# ==================================================

match_rows = []


# 몰수 / 비정상 종료로 제외된 경기
excluded_match_ids = set()


# 정상 종료로 처리된 경기
normal_match_ids = set()


# ==================================================
# 3. raw_data 확인
# ==================================================

if not os.path.exists(RAW_DIR):

    print(
        f"[오류] raw_data 폴더가 없습니다: {RAW_DIR}"
    )

    raise SystemExit


json_files = sorted([
    file_name
    for file_name in os.listdir(RAW_DIR)
    if file_name.endswith(".json")
])


print("=" * 70)
print("JSON → MATCH_TEAM CSV 변환 시작")
print("=" * 70)

print(
    "전체 JSON 파일 수:",
    len(json_files)
)


# ==================================================
# 4. JSON 파일 하나씩 처리
# ==================================================

for file_number, file_name in enumerate(
    json_files,
    start=1
):

    file_path = os.path.join(
        RAW_DIR,
        file_name
    )


    # --------------------------------------------------
    # JSON 읽기
    # --------------------------------------------------

    try:

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

    except Exception as e:

        print(
            f"[읽기 실패] {file_name}: {e}"
        )

        continue


    # --------------------------------------------------
    # 경기 기본 정보
    # --------------------------------------------------

    match_id = data.get(
        "matchId"
    )

    if not match_id:

        match_id = os.path.splitext(
            file_name
        )[0]


    match_date = data.get(
        "matchDate"
    )

    match_type = data.get(
        "matchType"
    )


    # --------------------------------------------------
    # 양쪽 유저 정보
    # --------------------------------------------------

    match_infos = data.get(
        "matchInfo",
        []
    )


    if not match_infos:

        print(
            f"[경고] matchInfo 없음: {file_name}"
        )

        continue


    # ==================================================
    # 5. 몰수승 / 몰수패 / 비정상 경기 제거
    # ==================================================
    #
    # 한 경기의 양쪽 유저 matchEndType을 확인
    #
    # 둘 중 하나라도 matchEndType != 0 이면
    # 해당 경기 전체를 CSV에서 제외
    #
    # matchEndType == 0
    # → 정상 종료 경기
    #
    # 기존 raw_data JSON 파일 자체는 삭제하지 않음.
    # ==================================================

    is_normal_match = True


    for user_data in match_infos:

        match_detail = user_data.get(
            "matchDetail",
            {}
        )

        match_end_type = match_detail.get(
            "matchEndType"
        )


        if match_end_type != 0:

            is_normal_match = False

            break


    # --------------------------------------------------
    # 비정상 종료 경기
    # → CSV 변환하지 않고 건너뜀
    # --------------------------------------------------

    if not is_normal_match:

        excluded_match_ids.add(
            match_id
        )

        continue


    # 정상 경기 기록
    normal_match_ids.add(
        match_id
    )


    # ==================================================
    # 6. 정상 종료 경기의 양쪽 유저 처리
    # ==================================================

    for user_data in match_infos:


        # ==================================================
        # 유저 기본 정보
        # ==================================================

        ouid = user_data.get(
            "ouid"
        )

        nickname = user_data.get(
            "nickname"
        )


        # ==================================================
        # matchDetail
        # ==================================================

        match_detail = user_data.get(
            "matchDetail",
            {}
        )


        # 시즌
        season_id = match_detail.get(
            "seasonId"
        )


        # 경기 결과
        match_result = match_detail.get(
            "matchResult"
        )


        # 정상 종료 여부
        #
        # 여기까지 온 경기는 정상 종료이므로
        # 기본적으로 0
        match_end_type = match_detail.get(
            "matchEndType"
        )


        # 등급 정보
        division = user_data.get(
            "division"
        )


        if division is None:

            division = match_detail.get(
                "division"
            )


        # ==================================================
        # 경기 기본 행동
        # ==================================================

        possession = match_detail.get(
            "possession"
        )

        dribble = match_detail.get(
            "dribble"
        )


        # ==================================================
        # 패스 데이터
        # ==================================================

        pass_data = user_data.get(
            "pass",
            {}
        )


        pass_try = pass_data.get(
            "passTry",
            0
        )

        pass_success = pass_data.get(
            "passSuccess",
            0
        )


        short_pass_try = pass_data.get(
            "shortPassTry",
            0
        )

        short_pass_success = pass_data.get(
            "shortPassSuccess",
            0
        )


        long_pass_try = pass_data.get(
            "longPassTry",
            0
        )

        long_pass_success = pass_data.get(
            "longPassSuccess",
            0
        )


        bouncing_lob_pass_try = pass_data.get(
            "bouncingLobPassTry",
            0
        )

        bouncing_lob_pass_success = pass_data.get(
            "bouncingLobPassSuccess",
            0
        )


        driven_ground_pass_try = pass_data.get(
            "drivenGroundPassTry",
            0
        )

        driven_ground_pass_success = pass_data.get(
            "drivenGroundPassSuccess",
            0
        )


        through_pass_try = pass_data.get(
            "throughPassTry",
            0
        )

        through_pass_success = pass_data.get(
            "throughPassSuccess",
            0
        )


        lobbed_through_pass_try = pass_data.get(
            "lobbedThroughPassTry",
            0
        )

        lobbed_through_pass_success = pass_data.get(
            "lobbedThroughPassSuccess",
            0
        )


        # ==================================================
        # 슈팅 데이터
        # ==================================================

        shoot_data = user_data.get(
            "shoot",
            {}
        )


        shoot_total = shoot_data.get(
            "shootTotal",
            0
        )

        effective_shoot_total = shoot_data.get(
            "effectiveShootTotal",
            0
        )

        shoot_heading = shoot_data.get(
            "shootHeading",
            0
        )

        shoot_freekick = shoot_data.get(
            "shootFreekick",
            0
        )

        shoot_in_penalty = shoot_data.get(
            "shootInPenalty",
            0
        )

        shoot_out_penalty = shoot_data.get(
            "shootOutPenalty",
            0
        )

        shoot_penalty_kick = shoot_data.get(
            "shootPenaltyKick",
            0
        )


        # ==================================================
        # 수비 데이터
        # ==================================================

        defence_data = user_data.get(
            "defence",
            {}
        )


        block_try = defence_data.get(
            "blockTry",
            0
        )

        block_success = defence_data.get(
            "blockSuccess",
            0
        )

        tackle_try = defence_data.get(
            "tackleTry",
            0
        )

        tackle_success = defence_data.get(
            "tackleSuccess",
            0
        )


        # ==================================================
        # MATCH_TEAM 행 생성
        # ==================================================
        #
        # 기존 컬럼은 그대로 유지
        # ==================================================

        match_rows.append({

            "matchId":
                match_id,

            "matchDate":
                match_date,

            "matchType":
                match_type,

            "ouid":
                ouid,

            "nickname":
                nickname,

            "division":
                division,

            "seasonId":
                season_id,

            "matchResult":
                match_result,

            "matchEndType":
                match_end_type,

            "possession":
                possession,

            "dribble":
                dribble,

            "passTry":
                pass_try,

            "passSuccess":
                pass_success,

            "shortPassTry":
                short_pass_try,

            "shortPassSuccess":
                short_pass_success,

            "longPassTry":
                long_pass_try,

            "longPassSuccess":
                long_pass_success,

            "bouncingLobPassTry":
                bouncing_lob_pass_try,

            "bouncingLobPassSuccess":
                bouncing_lob_pass_success,

            "drivenGroundPassTry":
                driven_ground_pass_try,

            "drivenGroundPassSuccess":
                driven_ground_pass_success,

            "throughPassTry":
                through_pass_try,

            "throughPassSuccess":
                through_pass_success,

            "lobbedThroughPassTry":
                lobbed_through_pass_try,

            "lobbedThroughPassSuccess":
                lobbed_through_pass_success,

            "shootTotal":
                shoot_total,

            "effectiveShootTotal":
                effective_shoot_total,

            "shootHeading":
                shoot_heading,

            "shootFreekick":
                shoot_freekick,

            "shootInPenalty":
                shoot_in_penalty,

            "shootOutPenalty":
                shoot_out_penalty,

            "shootPenaltyKick":
                shoot_penalty_kick,

            "blockTry":
                block_try,

            "blockSuccess":
                block_success,

            "tackleTry":
                tackle_try,

            "tackleSuccess":
                tackle_success
        })


    # ==================================================
    # 진행 상황 출력
    # ==================================================

    if (
        file_number % 50 == 0
        or file_number == len(json_files)
    ):

        print(
            f"{file_number}/"
            f"{len(json_files)} 처리 완료"
        )


# ==================================================
# 7. DataFrame 생성
# ==================================================

match_df = pd.DataFrame(
    match_rows
)


# ==================================================
# 8. 중복 제거
# ==================================================
#
# 같은 경기 + 같은 유저가
# 여러 번 들어가는 경우 방지
# ==================================================

if not match_df.empty:

    match_df = match_df.drop_duplicates(
        subset=[
            "matchId",
            "ouid"
        ]
    )


# ==================================================
# 9. 정렬
# ==================================================

if not match_df.empty:

    match_df = match_df.sort_values(
        by=[
            "matchDate",
            "matchId",
            "ouid"
        ],
        ascending=[
            False,
            True,
            True
        ]
    )


# ==================================================
# 10. CSV 저장
# ==================================================

match_df.to_csv(
    MATCH_TEAM_CSV,
    index=False,
    encoding="utf-8-sig"
)


# ==================================================
# 11. 결과 확인
# ==================================================

print()
print("=" * 70)
print("JSON → MATCH_TEAM CSV 변환 완료")
print("=" * 70)


print(
    "전체 원본 JSON:",
    len(json_files)
)

print(
    "정상 종료 경기:",
    len(normal_match_ids)
)

print(
    "몰수/비정상 제외 경기:",
    len(excluded_match_ids)
)

print(
    "최종 MATCH_TEAM 행 수:",
    len(match_df)
)


# ==================================================
# 12. matchEndType 최종 확인
# ==================================================

if not match_df.empty:

    print()
    print("[최종 matchEndType 분포]")

    print(
        match_df[
            "matchEndType"
        ].value_counts(
            dropna=False
        ).sort_index()
    )


# ==================================================
# 13. 주요 결측치 확인
# ==================================================

print()
print("[MATCH_TEAM 주요 결측치]")


check_columns = [
    "matchId",
    "ouid",
    "nickname",
    "division",
    "seasonId",
    "matchResult",
    "matchEndType",
    "possession",
    "dribble"
]


for column in check_columns:

    if column in match_df.columns:

        print(
            f"{column}:",
            match_df[
                column
            ].isna().sum()
        )


# ==================================================
# 14. 저장 위치
# ==================================================

print()
print(
    "MATCH_TEAM 저장:",
    MATCH_TEAM_CSV
)

print("=" * 70)