"""
플레이스타일 진단(K-means 군집화, 비지도학습) baseline 실행.

데이터 파싱/feature 조립은 preprocess_style.py, 저장은 utils.py에 맡기고, 여기서는
표준화 -> k 탐색 -> 최종 군집화 -> 결과 저장만 담당한다.

CLAUDE.md 원칙 #4("규칙 기반으로 미리 유형 개수를 정하지 않는다")를 그대로 따른다:
k=2~6을 전부 시도해 실루엣 점수가 가장 높은 k를 사후에 고르고, 각 군집에 사람이 붙일
이름(예: "스루패스 위주形")은 이 스크립트가 정하지 않는다 — cluster_summary에 군집별
feature 평균만 남겨서, 그 숫자를 보고 팀이 나중에 라벨을 붙인다.

2026-09-07 결정: "패스+슛 6개 feature 통합 1개 모델" 대신 **패스 모델 + 슛 모델을
독립적으로 군집화**한다 (preprocess_style.py docstring에 실험 근거 정리됨). 그래서 유저당
군집 라벨이 `pass_style_type`, `shoot_style_type` 두 개로 나온다. K-means 외 GMM 등도
비교해봤지만 최종적으로 K-means만 쓰기로 팀이 결정했다.

PCA는 군집화에 쓰지 않고, 모델별 시각화용 2차원 좌표를 뽑는 용도로만 별도로 돌린다.
"""

import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import preprocess_style as preprocess
import utils

# ============ CONFIG ============
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPO_ROOT, "data", "style")
MODEL_DIR = os.path.join(REPO_ROOT, "models")
RANDOM_SEED = 42
K_RANGE = range(2, 7)  # CLAUDE.md 원칙: k=2~6을 전부 시도
# ====================================================


def select_best_k(x_scaled, k_range=K_RANGE):
    """k별로 KMeans를 돌려 실루엣 점수를 계산하고, 가장 높은 k를 고른다.

    반환값: (best_k, {k: (labels, silhouette_score)} 전체 결과)
    """
    results = {}
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = kmeans.fit_predict(x_scaled)
        score = silhouette_score(x_scaled, labels)
        results[k] = (labels, score, kmeans)

    best_k = max(results, key=lambda k: results[k][1])
    return best_k, results


def run_style_clustering(style_df, feature_columns):
    """표준화 -> k=2~6 탐색 -> 최적 k 선택 -> 시각화용 PCA 좌표까지 한 번에 계산한다.

    패스 모델(4개 feature)과 슛 모델(2개 feature)에 똑같이 재사용한다.
    """
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(style_df[feature_columns].astype(float))

    best_k, k_search_results = select_best_k(x_scaled)
    labels, best_score, best_kmeans = k_search_results[best_k]

    n_components = min(2, len(feature_columns))
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    coords = pca.fit_transform(x_scaled)

    return {
        "labels": labels,
        "best_k": best_k,
        "best_score": best_score,
        "k_search_results": k_search_results,
        "scaler": scaler,
        "kmeans": best_kmeans,
        "pca": pca,
        "coords": coords,
    }


def build_cluster_summary(style_df, feature_columns, label_column):
    """군집별 표본 수와 feature 평균을 낸다 — 사후 라벨링(이름 붙이기)에 쓰는 참고 표."""
    summary = style_df.groupby(label_column)[feature_columns].mean()
    summary["n_users"] = style_df.groupby(label_column).size()
    return summary.reset_index()


def build_summary_text(n_users, min_matches, pass_result, shoot_result,
                        pass_summary, shoot_summary):
    lines = [
        "=== 플레이스타일 진단(K-means) Baseline 요약 ===",
        f"유저 최소 표본 경기 수(MIN_MATCHES_PER_USER): {min_matches}",
        f"군집화 대상 유저 수: {n_users}",
        "",
        "[설계 결정] 패스 모델과 슛 모델을 독립적으로 군집화한다. 6개 feature를 한 공간에서",
        "통합 군집화하면 실루엣 0.23~0.25로 구조가 거의 안 보였는데, 패스/슛을 분리하니",
        "각각 0.33/0.34로 개선됐다. dribble_intensity/possession/block_ratio 등도 시도했지만",
        "전부 실루엣을 낮춰(실력·경기 흐름과 얽혀 있어 스타일 신호를 희석시킴) 최종 feature에서",
        "뺐다 (preprocess_style.py docstring 참고).",
        "",
        "--- 패스 모델 ---",
        f"feature: {preprocess.PASS_FEATURE_COLUMNS}",
    ]
    for k in sorted(pass_result["k_search_results"]):
        _, score, _ = pass_result["k_search_results"][k]
        marker = "  <- 선택" if k == pass_result["best_k"] else ""
        lines.append(f"  k={k}: silhouette={score:.4f}{marker}")
    lines.append(f"[선택된 k] {pass_result['best_k']}")
    lines.append("[군집별 feature 평균]")
    lines.append(pass_summary.to_string(index=False))

    lines.append("")
    lines.append("--- 슛 모델 ---")
    lines.append(f"feature: {preprocess.SHOOT_FEATURE_COLUMNS}")
    for k in sorted(shoot_result["k_search_results"]):
        _, score, _ = shoot_result["k_search_results"][k]
        marker = "  <- 선택" if k == shoot_result["best_k"] else ""
        lines.append(f"  k={k}: silhouette={score:.4f}{marker}")
    lines.append(f"[선택된 k] {shoot_result['best_k']}")
    lines.append("[군집별 feature 평균]")
    lines.append(shoot_summary.to_string(index=False))

    lines.append("")
    lines.append("[한계] 최소 표본 경기 수 하한선(MIN_MATCHES_PER_USER)이 팀 논의로 "
                  "확정되지 않아 임시값을 쓰고 있다 (preprocess_style.py 참고).")
    lines.append("[한계] 실루엣 점수가 0.3대로 약한 군집 구조에 해당한다 — 유저의 플레이 "
                  "성향이 몇 개 뚜렷한 그룹으로 딱 떨어지기보다 연속적인 스펙트럼에 가깝다는 "
                  "뜻으로 해석하며, 완전히 분리된 유형이 아니라 '가장 가까운 성향' 정도의 "
                  "관찰적 진단으로 제시한다.")
    lines.append("[한계] long_pass_ratio/through_pass_ratio가 division(티어)과 상관관계가 "
                  "있음을 확인했다 — 관찰적 진단이라는 설계 선택에 따른 의도적 트레이드오프이며 "
                  "정규화로 보정하지 않는다 (style/README.md 참고).")
    return "\n".join(lines)


def main():
    print("1) match_team_data.csv 로드 및 (match, ouid) 행 추출")
    matches = preprocess.load_matches(preprocess.MATCHES_FILE)
    rows_df = preprocess.extract_match_style_rows(matches)

    print("2) 유저 단위 집계 및 패스/슛 비율 feature 계산")
    style_df = preprocess.aggregate_user_style(rows_df)
    if len(style_df) < max(K_RANGE):
        raise ValueError(
            f"군집화 대상 유저가 {len(style_df)}명뿐이라 k 최대값({max(K_RANGE)})보다 "
            "적다. 유저 수를 늘리거나 K_RANGE를 줄여야 한다."
        )

    print("3) 패스 모델 군집화 (표준화 -> k=2~6 탐색 -> 최적 k 선택 -> PCA 좌표)")
    pass_result = run_style_clustering(style_df, preprocess.PASS_FEATURE_COLUMNS)
    print(f"  선택된 k: {pass_result['best_k']} (silhouette={pass_result['best_score']:.4f})")

    print("4) 슛 모델 군집화 (표준화 -> k=2~6 탐색 -> 최적 k 선택 -> PCA 좌표)")
    shoot_result = run_style_clustering(style_df, preprocess.SHOOT_FEATURE_COLUMNS)
    print(f"  선택된 k: {shoot_result['best_k']} (silhouette={shoot_result['best_score']:.4f})")

    style_df = style_df.copy()
    style_df["pass_style_type"] = pass_result["labels"]
    style_df["shoot_style_type"] = shoot_result["labels"]
    style_df["pass_pca_x"] = pass_result["coords"][:, 0]
    style_df["pass_pca_y"] = pass_result["coords"][:, 1]
    style_df["shoot_pca_x"] = shoot_result["coords"][:, 0]
    style_df["shoot_pca_y"] = shoot_result["coords"][:, 1]

    print("5) 결과 저장")
    pass_summary = build_cluster_summary(style_df, preprocess.PASS_FEATURE_COLUMNS, "pass_style_type")
    shoot_summary = build_cluster_summary(style_df, preprocess.SHOOT_FEATURE_COLUMNS, "shoot_style_type")
    summary_text = build_summary_text(
        len(style_df), preprocess.MIN_MATCHES_PER_USER,
        pass_result, shoot_result, pass_summary, shoot_summary,
    )

    utils.save_model(
        {
            "pass": {"scaler": pass_result["scaler"], "kmeans": pass_result["kmeans"], "pca": pass_result["pca"]},
            "shoot": {"scaler": shoot_result["scaler"], "kmeans": shoot_result["kmeans"], "pca": shoot_result["pca"]},
        },
        os.path.join(MODEL_DIR, "style_diagnosis_baseline.joblib"),
    )
    utils.ensure_dir(OUTPUT_DIR)
    style_df.to_csv(os.path.join(OUTPUT_DIR, "user_style_profile.csv"), index=False)
    utils.save_text(summary_text, os.path.join(OUTPUT_DIR, "style_diagnosis_baseline_summary.txt"))

    print(f"  [저장] 모델(패스+슛 각각 scaler+kmeans+pca) -> {MODEL_DIR}/style_diagnosis_baseline.joblib")
    print(f"  [저장] 유저별 스타일 프로필 -> {OUTPUT_DIR}/user_style_profile.csv")
    print(f"  [저장] 요약 -> {OUTPUT_DIR}/style_diagnosis_baseline_summary.txt")


if __name__ == "__main__":
    main()
