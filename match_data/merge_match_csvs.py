import pandas as pd


# ==================================================
# 합칠 파일
# ==================================================

files = [
    "match_data/match_team_data.csv",
    "match_data/match_team_data(09.07).csv",
]


# ==================================================
# CSV 읽기
# ==================================================

dfs = []

for file in files:

    df = pd.read_csv(file)

    print(
        file,
        "행 수:",
        len(df)
    )

    dfs.append(df)


# ==================================================
# 전체 합치기
# ==================================================

merged_df = pd.concat(
    dfs,
    ignore_index=True
)


print()
print(
    "합치기 전 전체 행 수:",
    len(merged_df)
)


# ==================================================
# 중복 제거
# 같은 경기 + 같은 유저는 1행만 유지
# ==================================================

merged_df = merged_df.drop_duplicates(
    subset=[
        "matchId",
        "ouid"
    ],
    keep="last"
)


print(
    "중복 제거 후 행 수:",
    len(merged_df)
)


# ==================================================
# 최종 원본으로 저장
# ==================================================

OUTPUT_FILE = (
    "match_data/match_team_data_merged_temp.csv"
)


merged_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


print()
print(
    "통합 완료:",
    OUTPUT_FILE
)