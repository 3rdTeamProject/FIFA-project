import pandas as pd


# ==================================================
# 1. 데이터 불러오기
# ==================================================

# K=4 군집화 결과
cluster_path = "machine_learning/k4_cluster_result.csv"

# 원본 경기 데이터
original_path = "match_data/match_team_data.csv"


cluster_df = pd.read_csv(cluster_path)
original_df = pd.read_csv(original_path)


print("=" * 70)
print("K=4 저활동 Cluster 분석")
print("=" * 70)

print("K=4 데이터 크기:", cluster_df.shape)
print("원본 데이터 크기:", original_df.shape)


# ==================================================
# 2. K=4의 Cluster 3만 선택
# ==================================================

low_cluster = cluster_df[
    cluster_df["cluster"] == 3
].copy()


print("\n[Cluster 3 데이터 수]")
print(len(low_cluster))


# ==================================================
# 3. 원본 경기 데이터와 연결
# ==================================================

# matchId + ouid를 이용해서
# 같은 경기의 같은 유저 데이터를 연결한다.

merged = low_cluster.merge(
    original_df,
    on=["matchId", "ouid"],
    how="left",
    suffixes=("_ml", "_original")
)


print("\n[원본 데이터 연결 후]")
print("행 수:", len(merged))


# ==================================================
# 4. 분석에 필요한 원본 정보 선택
# ==================================================

check_columns = [
    "matchId",
    "ouid",
    "nickname_original",

    "matchResult",
    "matchEndType",
    "systemPause",

    "possession_original",
    "dribble_original",

    "passTry",
    "passSuccess",

    "shootTotal",
    "effectiveShootTotal",

    "tackleTry",
    "tackleSuccess",

    "blockTry",
    "blockSuccess",

    "cluster"
]


# 실제 존재하는 컬럼만 선택
check_columns = [
    col for col in check_columns
    if col in merged.columns
]

result = merged[check_columns].copy()


# ==================================================
# 5. 경기 종료 형태 확인
# ==================================================

print("\n" + "=" * 70)
print("matchEndType 분포")
print("=" * 70)

if "matchEndType" in result.columns:
    print(
        result["matchEndType"]
        .value_counts(dropna=False)
        .to_string()
    )


# ==================================================
# 6. 경기 결과 확인
# ==================================================

print("\n" + "=" * 70)
print("matchResult 분포")
print("=" * 70)

if "matchResult" in result.columns:
    print(
        result["matchResult"]
        .value_counts(dropna=False)
        .to_string()
    )


# ==================================================
# 7. 주요 행동량 평균 확인
# ==================================================

activity_columns = [
    "possession_original",
    "dribble_original",
    "passTry",
    "passSuccess",
    "shootTotal",
    "effectiveShootTotal",
    "tackleTry",
    "tackleSuccess",
    "blockTry",
    "blockSuccess"
]

activity_columns = [
    col for col in activity_columns
    if col in result.columns
]


print("\n" + "=" * 70)
print("Cluster 3 주요 행동량 통계")
print("=" * 70)

print(
    result[activity_columns]
    .describe()
    .round(2)
    .to_string()
)


# ==================================================
# 8. 슈팅 0회 경기 확인
# ==================================================

if "shootTotal" in result.columns:

    no_shoot_count = (
        result["shootTotal"] == 0
    ).sum()

    print("\n[슈팅 0회]")
    print(
        f"{no_shoot_count} / {len(result)} "
        f"({no_shoot_count / len(result) * 100:.2f}%)"
    )


# ==================================================
# 9. 결과 CSV 저장
# ==================================================

output_path = (
    "machine_learning/"
    "k4_cluster3_low_activity_analysis.csv"
)

result.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

print("\n분석 결과 저장 완료")
print("저장 위치:", output_path)