"""
승부예측(승/패 이진분류) baseline 모델 파이프라인.

이번 baseline은 style_fit_score가 아직 없어 avg_stat_score, tier 두 feature만 사용한다.
나중에 style_fit_score가 추가된 버전과 "같은 데이터, 같은 split"으로 비교할 예정이므로,
split에 쓰는 random seed를 고정하고 어떤 match_id/ouid가 train/validation/test 중
어디로 갔는지 output/split_assignment.json에 저장해둔다.

matches.jsonl 필드 매핑은 2026-09-04에 실제 API 응답(match-detail)을 1건 조회해 확인한
스키마를 기준으로 한다:
    matchInfo[i]["division"]                       -> tier (정수 코드, 예: 2300, 2400)
    matchInfo[i]["matchDetail"]["matchResult"]      -> "승" / "패" / "무"
    matchInfo[i]["matchDetail"]["matchEndType"]     -> 0이 정상 종료
    matchInfo[i]["player"][j]["spPosition"] == 28   -> 교체선수(SUB), 스쿼드 계산에서 제외

player_1000_final.csv는 이 스크립트 작성 시점에 실물 파일이 없어 컬럼명이 확정되지 않았다.
CSV_SPID_COLUMN / CSV_STAT_COLUMNS는 data-schema.md 기준 추정값이며, 실제 파일을 확보하면
CONFIG 섹션만 고쳐서 맞추면 되도록 분리해뒀다. 필요한 컬럼이 없으면 조용히 넘어가지 않고
실제 컬럼 목록을 보여주며 에러를 낸다.

또한 이 파일 수치가 강화단계(spGrade)를 반영한 값인지 기본(0강) 값인지 불분명하다.
확인 전까지는 강화단계를 무시하고 spId만으로 매칭하는 근사치로 취급한다 (한계로 명시,
save_outputs()가 쓰는 summary 텍스트에도 남긴다).
"""

import json
import os
import pickle

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

# ============ CONFIG ============
MATCHES_FILE = "matches.jsonl"
PLAYER_CARD_FILE = "data/player_1000_final.csv"
OUTPUT_DIR = "output"
RANDOM_SEED = 42
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

# style_fit_score가 추가될 것을 감안해 feature 목록을 한 곳에서 관리한다.
FEATURE_COLUMNS = ["avg_stat_score", "tier"]

# 승패와 사실상 동일한 정보(결과)가 feature에 섞여 들어가는 것을 막기 위한 금지 키워드.
# CLAUDE.md의 leakage 원칙(슛수/골수/태클 성공률/경기평점 등 결과 관련 필드 금지)을 그대로 반영.
LEAKAGE_FORBIDDEN_KEYWORDS = [
    "goal", "shoot", "슛", "골", "rating", "tackle", "foul", "card",
    "offside", "controller", "penalty", "freekick",
]

# --- matches.jsonl 필드 매핑 (2026-09-04 실제 API 응답으로 확인됨) ---
SUB_POSITION_CODE = 28
WIN_LABEL = "승"
LOSE_LABEL = "패"
DRAW_LABEL = "무"
NORMAL_MATCH_END_TYPE = 0

# --- player_1000_final.csv 컬럼 매핑 (파일 미확보 상태의 추정값 — 실제 파일 확보 후 검증 필수) ---
CSV_SPID_COLUMN = "spid"
CSV_STAT_COLUMNS = ["stat_short_pass", "stat_long_pass", "stat_dribble", "stat_ball_control"]
# ====================================================


def load_matches(path):
    """matches.jsonl을 한 줄씩 읽어 dict 리스트로 반환한다."""
    matches = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                matches.append(json.loads(line))
    return matches


def extract_rows(matches):
    """matches.jsonl의 raw match-detail 리스트에서 (match_id, ouid) 단위 학습용 행을 뽑는다.

    매치 하나(matchInfo 2건, 양 팀)에서 최대 2행을 추출한다. matchEndType != 0(몰수경기),
    matchResult == "무"(무승부)는 제외하고, 같은 ouid가 여러 match_id에 걸쳐 중복 등장하면
    첫 번째만 남긴다. 각 필터링 사유별 제외 건수를 로그로 남긴다.
    """
    rows = []
    excluded_forfeit = 0
    excluded_draw = 0
    seen_ouid = set()
    excluded_dup_ouid = 0

    for match in matches:
        match_id = match.get("matchId")
        for info in match.get("matchInfo", []):
            match_detail = info.get("matchDetail", {})

            if match_detail.get("matchEndType") != NORMAL_MATCH_END_TYPE:
                excluded_forfeit += 1
                continue

            result_label = match_detail.get("matchResult")
            if result_label == DRAW_LABEL:
                excluded_draw += 1
                continue
            if result_label not in (WIN_LABEL, LOSE_LABEL):
                continue

            ouid = info.get("ouid")
            if ouid in seen_ouid:
                excluded_dup_ouid += 1
                continue
            seen_ouid.add(ouid)

            sp_ids = [
                p.get("spId")
                for p in info.get("player", [])
                if p.get("spPosition") != SUB_POSITION_CODE
            ]

            rows.append({
                "match_id": match_id,
                "ouid": ouid,
                "tier": info.get("division"),
                "sp_ids": sp_ids,
                "result": 1 if result_label == WIN_LABEL else 0,
            })

    print(f"  [필터링] 몰수경기 제외: {excluded_forfeit}건")
    print(f"  [필터링] 무승부 제외: {excluded_draw}건")
    print(f"  [필터링] 중복 ouid 제외: {excluded_dup_ouid}건")
    print(f"  [추출] 최종 행 수: {len(rows)}")
    return pd.DataFrame(rows)


def load_player_cards(csv_path):
    """PLAYER_CARD csv를 로드하고, 필요한 컬럼이 실제로 있는지 검증한다."""
    df = pd.read_csv(csv_path)
    required = [CSV_SPID_COLUMN] + CSV_STAT_COLUMNS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path}에 필요한 컬럼이 없습니다: {missing}\n"
            f"실제 컬럼 목록: {list(df.columns)}\n"
            "win_prediction_baseline.py 상단 CONFIG의 CSV_SPID_COLUMN/CSV_STAT_COLUMNS를 "
            "실제 파일에 맞게 수정하세요."
        )
    return df.set_index(CSV_SPID_COLUMN)


def compute_avg_stat_score(sp_ids, player_card_df):
    """spId 리스트를 받아 매칭된 선수들의 (짧은패스/긴패스/드리블/볼컨트롤) 평균을 낸다.

    매칭 안 되는 spId는 결측 처리 후 평균에서 제외한다. 매칭된 선수가 하나도 없으면 NaN.
    반환값: (avg_stat_score 또는 NaN, 조회 시도한 spId 수, 매칭된 spId 수)
    """
    matched_scores = []
    for sp_id in sp_ids:
        if sp_id in player_card_df.index:
            row = player_card_df.loc[sp_id, CSV_STAT_COLUMNS]
            matched_scores.append(float(np.mean(row.values.astype(float))))

    attempted = len(sp_ids)
    matched = len(matched_scores)
    avg_score = float(np.mean(matched_scores)) if matched_scores else np.nan
    return avg_score, attempted, matched


def assemble_feature_table(rows_df, player_card_df):
    """extract_rows 결과에 avg_stat_score를 붙여 최종 feature 테이블을 만든다."""
    avg_scores = []
    total_attempted = 0
    total_matched = 0

    for sp_ids in rows_df["sp_ids"]:
        avg_score, attempted, matched = compute_avg_stat_score(sp_ids, player_card_df)
        avg_scores.append(avg_score)
        total_attempted += attempted
        total_matched += matched

    rows_df = rows_df.copy()
    rows_df["avg_stat_score"] = avg_scores

    match_rate = (total_matched / total_attempted * 100) if total_attempted else 0.0
    print(f"  [avg_stat_score] spId 매칭률: {match_rate:.1f}% ({total_matched}/{total_attempted})")

    before = len(rows_df)
    feature_df = rows_df.dropna(subset=["avg_stat_score", "tier"]).copy()
    dropped = before - len(feature_df)
    print(f"  [avg_stat_score] 매칭된 선수가 하나도 없거나 tier 결측인 행 제외: {dropped}건")

    return feature_df[["match_id", "ouid", "avg_stat_score", "tier", "result"]], match_rate


def assert_no_leakage(feature_columns):
    """feature 목록에 결과와 사실상 동일한 정보(골/슛/평점 등)가 없는지 확인한다."""
    for col in feature_columns:
        col_lower = col.lower()
        for keyword in LEAKAGE_FORBIDDEN_KEYWORDS:
            assert keyword.lower() not in col_lower, (
                f"leakage 의심: feature '{col}'에 금지 키워드 '{keyword}'가 포함되어 있습니다."
            )


def split_dataset(feature_df, output_dir):
    """train/validation/test로 랜덤 분할(기본 60/20/20)하고, 나중 비교를 위해 어떤
    match_id/ouid가 어디로 갔는지 output/split_assignment.json에 저장한다.

    validation은 이번 baseline 자체 평가에는 쓰지 않지만, 이후 style_fit_score를 추가한
    버전과 "test셋은 절대 건드리지 않고" 모델을 비교/선택할 때 쓰기 위해 미리 분리해둔다.
    """
    train_val_df, test_df = train_test_split(
        feature_df, test_size=TEST_RATIO, random_state=RANDOM_SEED
    )
    val_ratio_within_train_val = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_ratio_within_train_val, random_state=RANDOM_SEED
    )

    assignment = (
        [{"match_id": r.match_id, "ouid": r.ouid, "split": "train"} for r in train_df.itertuples()]
        + [{"match_id": r.match_id, "ouid": r.ouid, "split": "val"} for r in val_df.itertuples()]
        + [{"match_id": r.match_id, "ouid": r.ouid, "split": "test"} for r in test_df.itertuples()]
    )
    os.makedirs(output_dir, exist_ok=True)
    split_path = os.path.join(output_dir, "split_assignment.json")
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump({
            "random_seed": RANDOM_SEED,
            "train_ratio": TRAIN_RATIO,
            "val_ratio": VAL_RATIO,
            "test_ratio": TEST_RATIO,
            "rows": assignment,
        }, f, ensure_ascii=False, indent=2)

    print(
        f"  [split] train {len(train_df)}건 / val {len(val_df)}건 / test {len(test_df)}건 "
        f"-> {split_path}"
    )
    return train_df, val_df, test_df


def train_model(train_df, feature_columns):
    """statsmodels Logit으로 학습해 계수/p-value를 확인할 수 있는 모델을 반환한다."""
    X_train = sm.add_constant(train_df[feature_columns].astype(float), has_constant="add")
    y_train = train_df["result"].astype(int)
    return sm.Logit(y_train, X_train).fit(disp=0)


def evaluate_model(model, eval_df, feature_columns):
    """주어진 split(val 또는 test)에 대해 accuracy와 50% 대비 유의성(binomial test)을 계산한다."""
    X_eval = sm.add_constant(eval_df[feature_columns].astype(float), has_constant="add")
    y_eval = eval_df["result"].astype(int)
    predicted = (model.predict(X_eval) >= 0.5).astype(int)

    correct = int((predicted == y_eval).sum())
    n = len(y_eval)
    accuracy = correct / n if n else float("nan")

    sig_test = binomtest(correct, n, p=0.5) if n else None
    p_value = sig_test.pvalue if sig_test else float("nan")

    return accuracy, p_value, correct, n


def build_summary_text(model, feature_columns, train_size, match_rate,
                        val_metrics, test_metrics):
    val_accuracy, val_p_value, val_correct, n_val = val_metrics
    test_accuracy, test_p_value, test_correct, n_test = test_metrics

    lines = [
        "=== 승부예측 Baseline 모델 요약 ===",
        f"feature: {feature_columns}",
        "",
        str(model.summary()),
        "",
        f"train 표본 수: {train_size}",
        f"validation 표본 수: {n_val} (정답 {val_correct}건)",
        f"validation accuracy: {val_accuracy:.4f}",
        f"validation 50% 우연 대비 이항검정 p-value: {val_p_value:.4f}",
        f"test 표본 수: {n_test} (정답 {test_correct}건)",
        f"test accuracy: {test_accuracy:.4f}",
        f"test 50% 우연 대비 이항검정 p-value: {test_p_value:.4f}",
        (
            "  -> p < 0.05이면 우연(50%)보다 유의미하게 낫다고 판단 가능."
            if not np.isnan(test_p_value) else ""
        ),
        f"player_1000_final.csv spId 매칭률: {match_rate:.1f}%",
        "",
        "[참고] validation은 이번 baseline 평가에는 쓰지 않았다. 이후 style_fit_score를 "
        "추가한 버전과 같은 split으로 비교/모델 선택할 때, test셋을 건드리지 않고 쓰기 "
        "위해 미리 분리해둔 것이다.",
        "[한계] player_1000_final.csv의 스탯 수치가 강화단계(spGrade)를 반영한 값인지 "
        "기본(0강) 값인지 확인되지 않았다. 확인 전까지는 강화단계를 무시하고 spId만으로 "
        "매칭한 근사치로 취급한다.",
        "[한계] 유저 실력(손가락) 차이는 feature로 통제하지 못하며, tier로 부분적으로만 "
        "간접 반영된다.",
    ]
    return "\n".join(line for line in lines if line != "")


def save_outputs(model, summary_text, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "win_prediction_baseline_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    summary_path = os.path.join(output_dir, "win_prediction_baseline_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(f"  [저장] 모델 -> {model_path}")
    print(f"  [저장] 요약 -> {summary_path}")


def main():
    assert_no_leakage(FEATURE_COLUMNS)

    print("1) matches.jsonl 로드 및 행 추출")
    matches = load_matches(MATCHES_FILE)
    rows_df = extract_rows(matches)

    print("2) player_1000_final.csv 로드 및 avg_stat_score 계산")
    player_card_df = load_player_cards(PLAYER_CARD_FILE)
    feature_df, match_rate = assemble_feature_table(rows_df, player_card_df)

    print("3) train/validation/test 분할")
    train_df, val_df, test_df = split_dataset(feature_df, OUTPUT_DIR)

    print("4) 로지스틱회귀 학습 및 평가")
    model = train_model(train_df, FEATURE_COLUMNS)
    val_metrics = evaluate_model(model, val_df, FEATURE_COLUMNS)
    test_metrics = evaluate_model(model, test_df, FEATURE_COLUMNS)
    print(f"  validation accuracy: {val_metrics[0]:.4f} (p-value vs 50%: {val_metrics[1]:.4f})")
    print(f"  test accuracy: {test_metrics[0]:.4f} (p-value vs 50%: {test_metrics[1]:.4f})")

    print("5) 결과 저장")
    summary_text = build_summary_text(
        model, FEATURE_COLUMNS, len(train_df), match_rate, val_metrics, test_metrics,
    )
    save_outputs(model, summary_text, OUTPUT_DIR)


if __name__ == "__main__":
    main()
