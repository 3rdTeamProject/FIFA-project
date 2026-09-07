import os
import joblib
import pandas as pd

from feature_pipeline import build_behavior_features


# ============================================================
# 1. 설정
# ============================================================

# 신규 유저 30경기 원본 CSV
# 기존 match_team_data.csv와 같은 54개 컬럼 구조
INPUT_FILE = "match_data/new_user_30matches.csv"

# 신규 유저는 최근 30경기 사용
MATCH_LIMIT = 30


# ============================================================
# 저장된 모델 경로
# ============================================================

MODEL_DIR = "machine_learning/models"

SCALER_FILE = os.path.join(
    MODEL_DIR,
    "style_scaler.pkl"
)

KMEANS_FILE = os.path.join(
    MODEL_DIR,
    "style_kmeans_k4.pkl"
)

FEATURE_FILE = os.path.join(
    MODEL_DIR,
    "style_features.pkl"
)


# ============================================================
# 결과 저장 경로
# ============================================================

OUTPUT_FILE = (
    "machine_learning/outputs/"
    "new_user_style_prediction.csv"
)


# ============================================================
# Cluster 이름
# ============================================================

CLUSTER_NAMES = {
    0: "빠른 전개·특수 패스형",
    1: "직선·롱패스 침투형",
    2: "공격 전개·수비 개입형",
    3: "점유·짧은 패스 연계형",
}


# ============================================================
# 시작
# ============================================================

print("=" * 70)
print("신규 유저 플레이스타일 예측")
print("=" * 70)


# ============================================================
# 2. 신규 30경기 원본 데이터 불러오기
# ============================================================

raw_df = pd.read_csv(
    INPUT_FILE
)

print()
print(
    "신규 원본 데이터:",
    raw_df.shape
)


# ============================================================
# 3. 신규 유저 14개 행동 Feature 생성
# ============================================================
#
# 기존 학습 유저:
#   최근 70경기
#
# 신규 예측 유저:
#   최근 30경기
#
# 계산 공식은 동일
#
# ============================================================

user_df = build_behavior_features(
    raw_df,
    match_limit=MATCH_LIMIT
)

print()
print(
    "생성된 유저 Feature:",
    user_df.shape
)


# ============================================================
# 4. 신규 CSV에 유저가 몇 명 있는지 확인
# ============================================================
#
# 신규 예측용 CSV는 기본적으로
# 한 명의 유저 데이터만 들어있는 것을 기준으로 한다.
#
# ============================================================

if len(user_df) == 0:

    raise ValueError(
        "예측할 수 있는 유저가 없습니다."
    )


if len(user_df) > 1:

    print()
    print("[오류] 신규 CSV에서 여러 유저가 발견되었습니다.")

    print()
    print(
        user_df[
            [
                "ouid",
                "nickname",
                "match_count"
            ]
        ].to_string(
            index=False
        )
    )

    raise ValueError(
        "신규 예측 CSV에는 한 명의 유저만 넣어주세요."
    )


# ============================================================
# 5. 저장된 모델 불러오기
# ============================================================

scaler = joblib.load(
    SCALER_FILE
)

kmeans = joblib.load(
    KMEANS_FILE
)

features = joblib.load(
    FEATURE_FILE
)


print()
print(
    "StandardScaler 불러오기 완료"
)

print(
    "K-Means 모델 불러오기 완료"
)

print(
    "저장된 Feature 수:",
    len(features)
)


# ============================================================
# 6. Feature 구조 확인
# ============================================================
#
# feature_pipeline.py가 만든 행동 Feature와
# 모델 학습 당시 사용했던 Feature가
# 같은지 확인한다.
#
# ============================================================

missing_features = [
    feature
    for feature in features
    if feature not in user_df.columns
]


if missing_features:

    print()
    print(
        "[오류] 모델에 필요한 Feature가 없습니다."
    )

    for feature in missing_features:

        print(
            "-",
            feature
        )

    raise ValueError(
        "Feature 구조를 확인해주세요."
    )


# ============================================================
# 7. K-Means 입력 14개 Feature 선택
# ============================================================

X_new = user_df[
    features
].copy()


print()
print("=" * 70)
print("K-Means 입력 Feature")
print("=" * 70)

print(
    X_new
    .round(4)
    .to_string(
        index=False
    )
)


# ============================================================
# 8. 기존 StandardScaler 기준으로 변환
# ============================================================
#
# 매우 중요
#
# 신규 유저에서는:
#
# fit()
# fit_transform()
#
# 사용하면 안 된다.
#
# 기존 183명 데이터로 학습된 scaler에
# transform()만 사용한다.
#
# ============================================================

X_scaled = scaler.transform(
    X_new
)


print()
print(
    "기존 StandardScaler 기준 변환 완료"
)


# ============================================================
# 9. 기존 K-Means 모델로 Cluster 예측
# ============================================================
#
# 신규 데이터에서는
#
# fit()
# fit_predict()
#
# 사용하지 않는다.
#
# predict()만 사용
#
# ============================================================

predicted_cluster = int(
    kmeans.predict(
        X_scaled
    )[0]
)


style_name = CLUSTER_NAMES.get(
    predicted_cluster,
    "알 수 없는 스타일"
)


# ============================================================
# 10. 각 Cluster 중심까지 거리 계산
# ============================================================
#
# 이 값은 확률이 아니다.
#
# 값이 작을수록
# 해당 Cluster 중심에 가까운 것이다.
#
# ============================================================

distances = kmeans.transform(
    X_scaled
)[0]


# ============================================================
# 11. 결과 DataFrame에 추가
# ============================================================

user_df[
    "cluster"
] = predicted_cluster

user_df[
    "style_name"
] = style_name


# ============================================================
# 12. 예측 결과 출력
# ============================================================

row = user_df.iloc[0]


print()
print("=" * 70)
print("플레이스타일 예측 결과")
print("=" * 70)

print()
print(
    "닉네임:",
    row["nickname"]
)

print(
    "OUID:",
    row["ouid"]
)

print(
    "사용 경기 수:",
    row["match_count"]
)

print(
    "예측 Cluster:",
    predicted_cluster
)

print(
    "플레이스타일:",
    style_name
)


# ============================================================
# 13. Cluster별 거리 출력
# ============================================================

print()
print("=" * 70)
print("Cluster 중심까지 거리")
print("=" * 70)


for cluster_number, distance in enumerate(
    distances
):

    print(
        f"Cluster {cluster_number}: "
        f"{distance:.4f}"
    )


# ============================================================
# 14. 최종 14개 행동 Feature 출력
# ============================================================

print()
print("=" * 70)
print("신규 유저 행동 Feature")
print("=" * 70)


result_columns = [
    "nickname",
    "match_count",
] + features + [
    "cluster",
    "style_name",
]


print(
    user_df[
        result_columns
    ]
    .round(4)
    .to_string(
        index=False
    )
)


# ============================================================
# 15. 결과 저장
# ============================================================

os.makedirs(
    os.path.dirname(
        OUTPUT_FILE
    ),
    exist_ok=True
)


user_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


print()
print("=" * 70)
print("예측 완료")
print("=" * 70)

print(
    "결과 저장:",
    OUTPUT_FILE
)