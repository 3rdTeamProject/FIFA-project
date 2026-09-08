import ast
import subprocess
import pandas as pd
import os


# ==================================================
# 설정
# ==================================================

TARGET_FILE = "match_data/test_match_api.py"

CSV_FILE = os.path.join(
    "match_data",
    "match_team_data.csv"
)

TARGET_NORMAL_MATCHES = 70
MIN_NORMAL_MATCHES = 65


# ==================================================
# 1. test_match_api.py 변경 이력이 있는 커밋 가져오기
# ==================================================

result = subprocess.run(
    [
        "git",
        "log",
        "--format=%H",
        "--all",
        "--",
        TARGET_FILE
    ],
    capture_output=True,
    text=True,
    encoding="utf-8"
)

commits = [
    line.strip()
    for line in result.stdout.splitlines()
    if line.strip()
]


print("=" * 70)
print("과거 test_match_api.py 수집 대상 복구")
print("=" * 70)

print("확인할 Git 버전:", len(commits))
print()


# ==================================================
# 2. 각 Git 버전의 nicknames 리스트 읽기
# ==================================================

all_nicknames = set()

nickname_history = {}


for commit in commits:

    show_result = subprocess.run(
        [
            "git",
            "show",
            f"{commit}:{TARGET_FILE}"
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )


    if show_result.returncode != 0:

        print(
            f"[건너뜀] {commit[:7]} "
            f"- 파일을 읽을 수 없음"
        )

        continue


    source_code = show_result.stdout


    try:

        tree = ast.parse(
            source_code
        )

    except SyntaxError as e:

        print(
            f"[파싱 실패] {commit[:7]}: {e}"
        )

        continue


    commit_nicknames = set()


    for node in ast.walk(tree):

        # nicknames = [...]
        if isinstance(node, ast.Assign):

            variable_names = []

            for target in node.targets:

                if isinstance(target, ast.Name):

                    variable_names.append(
                        target.id
                    )


            # 우리가 사용했던 닉네임 리스트 변수명 후보
            if not any(
                name in {
                    "nicknames",
                    "NICKNAMES",
                    "target_nicknames",
                    "TARGET_NICKNAMES"
                }
                for name in variable_names
            ):

                continue


            try:

                value = ast.literal_eval(
                    node.value
                )

            except Exception:

                continue


            if isinstance(
                value,
                (list, tuple, set)
            ):

                for nickname in value:

                    if isinstance(
                        nickname,
                        str
                    ):

                        nickname = (
                            nickname.strip()
                        )

                        if nickname:

                            commit_nicknames.add(
                                nickname
                            )


    nickname_history[
        commit[:7]
    ] = sorted(
        commit_nicknames
    )


    all_nicknames.update(
        commit_nicknames
    )


    print(
        f"{commit[:7]} : "
        f"{len(commit_nicknames)}명"
    )


# ==================================================
# 3. Git 기록에서 찾은 전체 닉네임
# ==================================================

all_nicknames = sorted(
    all_nicknames
)


print()
print("=" * 70)
print("Git 기록에서 복구한 전체 직접 수집 대상")
print("=" * 70)

print(
    "중복 제거 후:",
    len(all_nicknames),
    "명"
)


for i, nickname in enumerate(
    all_nicknames,
    start=1
):

    print(
        f"{i:>3}. {nickname}"
    )


# ==================================================
# 4. 현재 match_team_data.csv 읽기
# ==================================================

if not os.path.exists(
    CSV_FILE
):

    print()
    print(
        f"[오류] CSV가 없습니다: "
        f"{CSV_FILE}"
    )

    raise SystemExit


df = pd.read_csv(
    CSV_FILE
)


# 현재 CSV는 정상 경기만 들어 있지만
# 안전하게 matchEndType == 0 다시 확인
if "matchEndType" in df.columns:

    df = df[
        df["matchEndType"] == 0
    ].copy()


# ==================================================
# 5. 닉네임별 정상 경기 수
# ==================================================

counts = (
    df[
        df["nickname"].isin(
            all_nicknames
        )
    ]
    .groupby(
        "nickname"
    )["matchId"]
    .nunique()
)


results = []


for nickname in all_nicknames:

    normal_count = int(
        counts.get(
            nickname,
            0
        )
    )


    if normal_count >= TARGET_NORMAL_MATCHES:

        status = "70 이상"

    elif normal_count >= MIN_NORMAL_MATCHES:

        status = "65~69"

    else:

        status = "65 미만"


    results.append({
        "nickname":
            nickname,

        "normal_matches":
            normal_count,

        "status":
            status
    })


result_df = pd.DataFrame(
    results
)


# ==================================================
# 6. 결과 출력
# ==================================================

print()
print("=" * 70)
print("현재 정상 경기 보유 현황")
print("=" * 70)


for _, row in result_df.iterrows():

    print(
        f"{row['nickname']:<25} "
        f"{row['normal_matches']:>3} 경기   "
        f"{row['status']}"
    )


# ==================================================
# 7. 상태별 분류
# ==================================================

over_70 = result_df[
    result_df["normal_matches"] >= 70
]

between_65_69 = result_df[
    (
        result_df["normal_matches"] >= 65
    )
    &
    (
        result_df["normal_matches"] < 70
    )
]

under_65 = result_df[
    result_df["normal_matches"] < 65
]


print()
print("=" * 70)
print("요약")
print("=" * 70)

print(
    "전체 직접 수집 대상:",
    len(result_df)
)

print(
    "정상 70경기 이상:",
    len(over_70)
)

print(
    "정상 65~69경기:",
    len(between_65_69)
)

print(
    "정상 65경기 미만:",
    len(under_65)
)


# ==================================================
# 8. 추가 수집이 필요한 유저
# ==================================================

need_more = result_df[
    result_df["normal_matches"] < 70
].copy()


print()
print("=" * 70)
print("추가 수집 필요 유저 (< 70)")
print("=" * 70)


if need_more.empty:

    print(
        "없음"
    )

else:

    for _, row in need_more.iterrows():

        needed = (
            TARGET_NORMAL_MATCHES
            - row["normal_matches"]
        )

        print(
            f"- {row['nickname']} "
            f"/ 현재 {row['normal_matches']}경기 "
            f"/ 최대 {needed}경기 추가 필요"
        )


# ==================================================
# 9. 결과 CSV 저장
# ==================================================

OUTPUT_FILE = os.path.join(
    "match_data",
    "collected_user_match_counts.csv"
)


result_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


print()
print(
    "결과 저장:",
    OUTPUT_FILE
)

print("=" * 70)