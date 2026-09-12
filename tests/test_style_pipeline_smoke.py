"""
preprocess_style.py / train_style.py / utils.py 스모크 테스트 (pytest 불필요, plain assert).

data/style/match_team_data.csv(팀원이 팀 단위로 평탄화해 수집한 실 데이터)의 컬럼 구성을
본뜬 합성 데이터(tests/fake_data_style.py)로 파이프라인 각 단계(파싱 -> 유저 집계 ->
표준화 -> k 탐색 -> KMeans -> 저장)가 에러 없이 동작하는지, 그리고 뚜렷이 다른 스타일
그룹을 실제로 구분해내는지만 검증한다. 실제 데이터로의 최종 검증은 별도.

2026-09-08: 슛 모델(in_penalty+heading 통합)이 헤딩축/슛위치축 2개로 분리되면서
패스/헤딩/슛위치 3개 모델을 검증하도록 갱신했다.

실행: ./venv/Scripts/python.exe tests/test_style_pipeline_smoke.py
"""

import os
import sys
import tempfile

import joblib

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "style"))

import preprocess_style as preprocess
import train_style as train
import utils
from fake_data_style import make_fake_style_matches


def test_extract_match_style_rows_filters_forfeit():
    df = make_fake_style_matches(n_users_per_profile=2, n_matches_per_user=5)
    rows_df = preprocess.extract_match_style_rows(df)

    assert "match_forfeit" not in set(rows_df["match_id"])
    for col in ["pass_try", "short_pass_try", "long_pass_try", "through_pass_try",
                "driven_ground_pass_try", "dribble_yard", "shoot_total", "shoot_heading",
                "shoot_in_penalty"]:
        assert col in rows_df.columns
    print("OK: test_extract_match_style_rows_filters_forfeit")


def test_aggregate_user_style_filters_sparse_users():
    df = make_fake_style_matches(n_users_per_profile=2, n_matches_per_user=15)
    rows_df = preprocess.extract_match_style_rows(df)
    style_df = preprocess.aggregate_user_style(rows_df, min_matches=10)

    # 표본 부족 유저(3경기)는 min_matches=10 미만이라 제외되어야 한다.
    assert "sparse_user" not in set(style_df["ouid"])

    feature_columns = (
        preprocess.PASS_FEATURE_COLUMNS + preprocess.HEADING_FEATURE_COLUMNS
        + preprocess.SHOT_LOCATION_FEATURE_COLUMNS
    )
    for col in feature_columns:
        assert col in style_df.columns
        assert style_df[col].notna().all()
        assert (style_df[col] >= 0).all()
    print("OK: test_aggregate_user_style_filters_sparse_users")


def _mismatch_rate(style_df, label_column):
    """true_group(ouid 접두사)별 지배적 군집에 안 속한 유저 비율."""
    mismatch = 0
    for true_group, sub in style_df.groupby("true_group"):
        dominant_cluster = sub[label_column].mode()[0]
        mismatch += (sub[label_column] != dominant_cluster).sum()
    return mismatch / len(style_df)


def test_full_pipeline_with_fake_data():
    df = make_fake_style_matches(n_users_per_profile=5, n_matches_per_user=15)

    with tempfile.TemporaryDirectory() as tmp_dir:
        matches_path = os.path.join(tmp_dir, "match_team_data.csv")
        df.to_csv(matches_path, index=False)

        output_dir = os.path.join(tmp_dir, "output")
        model_dir = os.path.join(tmp_dir, "models")

        loaded_df = preprocess.load_matches(matches_path)
        rows_df = preprocess.extract_match_style_rows(loaded_df)
        # 실제 운영값(MIN_MATCHES_PER_USER)과 무관하게, 이 테스트의 가짜 데이터 규모
        # (유저당 15경기)에 맞춘 자체 min_matches를 명시한다.
        style_df = preprocess.aggregate_user_style(rows_df, min_matches=10)

        # 스타일 그룹 3개 x 유저 5명 = 15명 (표본 부족/몰수경기 유저는 별도라 안 섞임)
        assert len(style_df) == 15, f"expected 15 users, got {len(style_df)}"

        results = []
        for key, feature_columns, label_column in train.MODEL_SPECS:
            result = train.run_style_clustering(style_df, feature_columns)
            assert 2 <= result["best_k"] <= 6
            assert -1.0 <= result["best_score"] <= 1.0
            style_df[label_column] = result["labels"]
            style_df[f"{key}_pca_x"] = result["coords"][:, 0]
            if result["coords"].shape[1] > 1:
                style_df[f"{key}_pca_y"] = result["coords"][:, 1]
            results.append(result)

        # 뚜렷이 다른 3그룹으로 데이터를 만들었으니, 각 모델이 나눈 군집이 실제 프로필
        # 그룹(ouid 접두사)과 강하게 일치해야 한다 (완벽히 3이 아니어도 되지만 최소한
        # 같은 그룹 유저끼리는 대부분 같은 군집으로 묶여야 한다).
        style_df["true_group"] = style_df["ouid"].str.rsplit("_", n=1).str[0]
        mismatch_rates = {}
        for key, _, label_column in train.MODEL_SPECS:
            rate = _mismatch_rate(style_df, label_column)
            mismatch_rates[key] = rate
            assert rate <= 0.2, (
                f"{key} 모델이 뚜렷이 다른 스타일 그룹을 잘 못 갈랐음 (mismatch_rate={rate:.2f})"
            )

        summaries = [
            train.build_cluster_summary(style_df, feature_columns, label_column)
            for _, feature_columns, label_column in train.MODEL_SPECS
        ]
        summary_text = train.build_summary_text(
            len(style_df), preprocess.MIN_MATCHES_PER_USER, results, summaries,
        )

        model_path = os.path.join(model_dir, "style_diagnosis_baseline.joblib")
        csv_path = os.path.join(output_dir, "user_style_profile.csv")
        summary_path = os.path.join(output_dir, "style_diagnosis_baseline_summary.txt")

        model_dict = {
            key: {"scaler": result["scaler"], "kmeans": result["kmeans"], "pca": result["pca"]}
            for (key, _, _), result in zip(train.MODEL_SPECS, results)
        }
        utils.save_model(model_dict, model_path)
        utils.ensure_dir(output_dir)
        style_df.to_csv(csv_path, index=False)
        utils.save_text(summary_text, summary_path)

        assert os.path.exists(model_path)
        assert os.path.exists(csv_path)
        assert os.path.exists(summary_path)

        reloaded = joblib.load(model_path)
        for key, _, _ in train.MODEL_SPECS:
            assert key in reloaded
            assert "kmeans" in reloaded[key] and "scaler" in reloaded[key]

    summary_str = ", ".join(
        f"{key}: k={r['best_k']}, silhouette={r['best_score']:.3f}, mismatch={mismatch_rates[key]:.2f}"
        for (key, _, _), r in zip(train.MODEL_SPECS, results)
    )
    print(f"OK: test_full_pipeline_with_fake_data ({summary_str})")


if __name__ == "__main__":
    test_extract_match_style_rows_filters_forfeit()
    test_aggregate_user_style_filters_sparse_users()
    test_full_pipeline_with_fake_data()
    print("\n모든 스타일 진단 스모크 테스트 통과")
