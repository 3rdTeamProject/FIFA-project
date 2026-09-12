"""
squad_fit_score(스쿼드 궁합 점수)가 실제 승률과 관련 있는지 검증한다.

CLAUDE.md 핵심 기능 6번("궁합 점수가 실제로 승률과 관련 있다는 근거를 뒷받침하는 용도")과
data-schema.md WIN_PREDICTION_MODEL_INPUT의 style_fit_score 계획을 실제로 실행하는
스크립트다. 이전에 한 번 같은 취지의 실험을 돌린 적이 있었으나 그때는 이 레포에 커밋되지
않은 일회성 스크립트였다 — 이번에 정식으로 다시 짜서 커밋한다(2026-09-12).

핵심 질문: "공식 승부예측 baseline과 같은 5개 feature(attack/mid/defense/gk_avg_score +
tier)를 아는 모델"과 "그 5개 + squad_fit_score를 아는 모델" 중 어느 쪽이 승패를 더 잘
맞히는가? 후자가 유의미하게 더 잘 맞히면 궁합 점수가 카드 품질과는 별개로 승률에 추가
정보를 준다는 근거가 된다.

2026-09-12 수정: 처음엔 "division만" vs "division+squad_fit_score"로 비교했는데(공식
5-feature baseline은 아예 빼고), 이러면 "squad_fit_score가 승률과 관련 있다"는 정도만
보여줄 뿐 "우리가 실제 쓰는 5-feature 모델에 얹었을 때도 도움이 되는가"는 확인이 안 된다는
지적을 받았다. 그래서 attack/mid/defense/gk_avg_score도 이 스크립트 안에서 직접 계산해
공식 baseline과 같은 feature 구성으로 맞췄다. winrate/preprocess_winrate.py의
compute_group_avg_scores()와 로직은 동일하지만 import는 하지 않고 그대로 복제했다 —
utils.py를 트랙마다 복제해둔 것과 같은 이유(트랙 독립성 유지, style/README.md 참고).

모집단(중요한 제약): squad_fit_score는 유저 본인의 스타일 비율(패스/슛)이 먼저 있어야
계산 가능한데, 이건 유저당 매치가 MIN_MATCHES_PER_USER(50)건 이상 쌓여야 나온다
(preprocess_style.aggregate_user_style). 승률 트랙(data/winrate/matches.jsonl)은 "1유저
1경기" 스노우볼이라 대부분 유저의 자기 과거 기록이 없어 이 feature를 쓸 수 없다. 그래서
이 검증은 승률 공식 baseline(수천 명)보다 훨씬 좁은 모집단 — "스타일 진단된 유저 자신이
뛴 경기"(data/style/matches_style.jsonl에서 그 유저가 등장하는 경기)로 한정한다.

데이터 흐름:
1) preprocess_style.py와 완전히 같은 방식으로 "스타일 진단 대상 유저 + 유저별 스타일 비율"
   을 구한다(train_style.py의 population과 동일 — 재현성을 위해 그대로 재사용).
2) data/style/matches_style.jsonl(원본, player[] 포함)을 스트리밍하며, matchInfo 두 쪽 중
   위 대상 유저가 낀 쪽만 골라 그 경기의 실제 11명 스쿼드를 뽑는다(몰수/SUB 포함 11명 미만
   제외 — live_diagnosis.build_current_squad와 같은 기준).
3) 그 유저의 스타일(1에서 구한 값)과 그 경기의 실제 스쿼드로 live_diagnosis.score_squad()를
   불러 squad_fit_score를, compute_group_avg_scores()로 attack/mid/defense/gk_avg_score를
   계산한다. 무승부는 승패 이진 분류 목적상 제외한다.
4) "공식 5-feature(attack/mid/defense/gk_avg_score+division)" vs "그 5개 + squad_fit_score"
   두 로지스틱회귀를 같은 train/val/test split(60/20/20, stratified, random_seed=42 — 위
   계획 문서의 "같은 split" 원칙)으로 학습해 정확도·계수 p-value·우도비검정(LR test)으로
   비교한다. winrate/train_winrate.py와 같은 원칙으로 test는 EVALUATE_TEST_SET=True일 때만
   평가한다("이제 진짜 최종"이라고 확신할 때 딱 한 번만 — 여러 버전을 시도하며 매번 test까지
   보면 leakage).

⚠️ 유저 본인의 과거 여러 경기 평균으로 스타일을 낸 뒤 그 평균을 각 경기에 그대로 쓰므로,
어떤 한 경기의 결과가 자기 자신의 feature에 새어 들어가는 강한 leakage는 아니다(원 경기가
많을수록 한 경기가 평균에 기여하는 비중은 작다) — 다만 완전히 독립은 아니라는 점은
한계로 명시한다(build_summary_text 참고).

실행: ./venv/Scripts/python.exe style/check_style_fit_winrate_correlation.py
"""

import json
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as scipy_stats
from scipy.stats import binomtest
from sklearn.model_selection import train_test_split

import live_diagnosis as ld
import position_fit as pf
import preprocess_style as preprocess
import utils

# ============ CONFIG ============
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STYLE_MATCHES_FILE = os.path.join(REPO_ROOT, "data", "style", "matches_style.jsonl")
OUTPUT_DIR = os.path.join(REPO_ROOT, "data", "style")
RANDOM_SEED = 42
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.6, 0.2, 0.2
SUB_POSITION = 28
NORMAL_MATCH_END_TYPE = 0
WIN_LABEL, LOSE_LABEL, DRAW_LABEL = "승", "패", "무"

# winrate/train_winrate.py와 같은 원칙: feature/모집단을 실험하는 동안은 False로 두고
# validation만 보고 판단한다. "이제 진짜 최종"이라고 확신할 때만 True로 바꿔 딱 한 번
# test를 확인한다. 2026-09-12: 카드 데이터 100% 매칭 + 스타일 모델(k) 수정 이후 팀 결정으로
# 최종 확인 시점이라 판단해 True로 전환, 한 번 확인했다 — 다시 실험하게 되면 False로 되돌릴 것.
EVALUATE_TEST_SET = True

# --- 포지션 그룹별 avg_score 계산 (winrate/preprocess_winrate.py에서 복제, 트랙 독립성
# 유지를 위해 import 대신 그대로 옮겨옴 — 로직/컬럼 구성은 완전히 동일해야 한다) ---
GK_POSITION_CODE = 0
POSITION_GROUP_DEFENSE = {1, 2, 3, 4, 5, 6, 7, 8}
POSITION_GROUP_MIDFIELD = {9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}
POSITION_GROUP_ATTACK = {20, 21, 22, 23, 24, 25, 26, 27}

CSV_STAT_COLUMNS = [
    "속력", "가속력", "골 결정력", "슛 파워", "중거리 슛", "위치 선정", "발리슛", "페널티 킥",
    "짧은 패스", "시야", "크로스", "긴 패스", "프리킥", "커브", "드리블", "볼 컨트롤",
    "민첩성", "밸런스", "반응 속도", "대인 수비", "태클", "가로채기", "헤더", "슬라이딩 태클",
    "몸싸움", "스태미너", "적극성", "점프", "침착성",
]
CSV_GK_STAT_COLUMNS = ["GK 다이빙", "GK 핸들링", "GK 킥", "GK 반응속도", "GK 위치 선정"]
CAT_PAC = ["속력", "가속력"]
CAT_SHO = ["골 결정력", "슛 파워", "중거리 슛", "위치 선정", "발리슛", "페널티 킥", "헤더"]
CAT_DRI = ["드리블", "볼 컨트롤", "민첩성", "밸런스", "반응 속도", "침착성"]
CAT_DEF = ["대인 수비", "태클", "가로채기", "헤더", "슬라이딩 태클"]
CAT_PHY = ["몸싸움", "스태미너", "적극성", "점프"]
ATTACK_STAT_COLUMNS = CAT_PAC + CAT_DRI + CAT_PHY + CAT_SHO
DEFENSE_STAT_COLUMNS = CAT_PAC + CAT_DRI + CAT_PHY + CAT_DEF
MID_STAT_COLUMNS = CSV_STAT_COLUMNS
_GROUP_TO_FEATURE_COLUMN = {
    "attack": "attack_avg_score", "midfield": "mid_avg_score",
    "defense": "defense_avg_score", "gk": "gk_avg_score",
}
_GROUP_STAT_COLUMNS = {
    "attack": ATTACK_STAT_COLUMNS, "midfield": MID_STAT_COLUMNS,
    "defense": DEFENSE_STAT_COLUMNS, "gk": CSV_GK_STAT_COLUMNS,
}
FIVE_FEATURE_COLUMNS = list(_GROUP_TO_FEATURE_COLUMN.values()) + ["division"]
SIX_FEATURE_COLUMNS = FIVE_FEATURE_COLUMNS + ["squad_fit_score"]
# ====================================================


def classify_position_group(sp_position):
    if sp_position == GK_POSITION_CODE:
        return "gk"
    if sp_position in POSITION_GROUP_DEFENSE:
        return "defense"
    if sp_position in POSITION_GROUP_MIDFIELD:
        return "midfield"
    if sp_position in POSITION_GROUP_ATTACK:
        return "attack"
    return None


def compute_group_avg_scores(squad, player_stats):
    """squad([{"sp_id","sp_position"}, ...])의 포지션 그룹별 평균 스탯 4개를 계산한다.

    winrate/preprocess_winrate.compute_group_avg_scores와 동일 로직(모듈 docstring 참고).
    매칭 안 되는 spId는 평균에서 제외, 그룹에 매칭된 선수가 없으면 그 그룹은 NaN.
    """
    grouped_sp_ids = {"attack": [], "midfield": [], "defense": [], "gk": []}
    for p in squad:
        group = classify_position_group(p["sp_position"])
        if group:
            grouped_sp_ids[group].append(p["sp_id"])

    scores = {}
    for group, sp_ids in grouped_sp_ids.items():
        stat_columns = _GROUP_STAT_COLUMNS[group]
        matched_scores = []
        for sp_id in sp_ids:
            if sp_id in player_stats.index:
                row = player_stats.loc[sp_id, stat_columns]
                matched_scores.append(float(np.mean(row.values.astype(float))))
        scores[_GROUP_TO_FEATURE_COLUMN[group]] = (
            float(np.mean(matched_scores)) if matched_scores else np.nan
        )
    return scores


def build_style_population():
    """train_style.py와 완전히 같은 방식으로 (대상 유저 집합, 유저별 스타일 dict)를 구한다."""
    matches = preprocess.load_matches(preprocess.MATCHES_FILE)
    rows_df = preprocess.extract_match_style_rows(matches)
    style_df = preprocess.aggregate_user_style(rows_df)
    style_by_ouid = style_df.set_index("ouid").to_dict("index")
    return style_by_ouid


def extract_squad_fit_rows(style_by_ouid, player_stats):
    """matches_style.jsonl을 스트리밍하며 대상 유저가 뛴 경기의 squad_fit_score와
    attack/mid/defense/gk_avg_score(공식 baseline과 같은 feature)를 함께 뽑는다.

    반환: [{"match_id", "ouid", "division", "result", "squad_fit_score",
            "attack_avg_score", "mid_avg_score", "defense_avg_score", "gk_avg_score"}, ...]
    """
    target_ouids = set(style_by_ouid)
    rows = []
    seen = set()  # (match_id, ouid) 중복 방지
    n_lines = 0

    with open(STYLE_MATCHES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            detail = json.loads(line)
            match_id = detail.get("matchId")

            for mi in detail.get("matchInfo", []):
                ouid = mi.get("ouid")
                if ouid not in target_ouids:
                    continue
                if (match_id, ouid) in seen:
                    continue

                match_detail = mi.get("matchDetail", {})
                if match_detail.get("matchEndType") != NORMAL_MATCH_END_TYPE:
                    continue
                result_label = match_detail.get("matchResult")
                if result_label == DRAW_LABEL or result_label not in (WIN_LABEL, LOSE_LABEL):
                    continue

                starters = [p for p in mi.get("player", []) if p.get("spPosition") != SUB_POSITION]
                if len(starters) != 11:
                    continue
                squad = [{"sp_id": p["spId"], "sp_position": p["spPosition"]} for p in starters]

                squad_scores = ld.score_squad(style_by_ouid[ouid], squad, player_stats)
                squad_fit_score = squad_scores["squad_fit_score"]
                if squad_fit_score != squad_fit_score:  # NaN
                    continue
                avg_scores = compute_group_avg_scores(squad, player_stats)

                seen.add((match_id, ouid))
                rows.append({
                    "match_id": match_id,
                    "ouid": ouid,
                    "division": mi.get("division"),
                    "result": 1 if result_label == WIN_LABEL else 0,
                    "squad_fit_score": squad_fit_score,
                    **avg_scores,
                })

    print(f"  [처리] matches_style.jsonl {n_lines}줄 스캔, 대상 유저 관여 경기 {len(rows)}건 추출")
    return pd.DataFrame(rows)


def split_dataset(df, output_dir):
    """train/validation/test로 랜덤 분할(60/20/20)하고, winrate/train_winrate.py와 같은
    방식으로 어떤 match_id/ouid가 어디로 갔는지 저장해둔다."""
    train_val_df, test_df = train_test_split(
        df, test_size=TEST_RATIO, random_state=RANDOM_SEED, stratify=df["result"]
    )
    val_ratio_within_train_val = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_ratio_within_train_val, random_state=RANDOM_SEED,
        stratify=train_val_df["result"],
    )

    assignment = (
        [{"match_id": r.match_id, "ouid": r.ouid, "split": "train"} for r in train_df.itertuples()]
        + [{"match_id": r.match_id, "ouid": r.ouid, "split": "val"} for r in val_df.itertuples()]
        + [{"match_id": r.match_id, "ouid": r.ouid, "split": "test"} for r in test_df.itertuples()]
    )
    utils.save_json({
        "random_seed": RANDOM_SEED,
        "train_ratio": TRAIN_RATIO, "val_ratio": VAL_RATIO, "test_ratio": TEST_RATIO,
        "rows": assignment,
    }, os.path.join(output_dir, "style_fit_split_assignment.json"))

    print(f"  [split] train {len(train_df)}건 / val {len(val_df)}건 / test {len(test_df)}건 "
          f"-> {output_dir}/style_fit_split_assignment.json")
    return train_df, val_df, test_df


def fit_model(train_df, feature_cols):
    """statsmodels Logit을 학습해 fitted_model을 반환한다."""
    x_train = sm.add_constant(train_df[feature_cols].astype(float))
    y_train = train_df["result"]
    return sm.Logit(y_train, x_train).fit(disp=0)


def evaluate_model(model, eval_df, feature_cols):
    """주어진 split(val 또는 test)에 대해 (accuracy, binom_pvalue)를 반환한다.

    binom_pvalue: winrate/train_winrate.py와 같은 방식(scipy.stats.binomtest)으로 정확도가
    50%(우연) 대비 유의미한지 검정한 p-value.
    """
    x_eval = sm.add_constant(eval_df[feature_cols].astype(float), has_constant="add")
    pred = (model.predict(x_eval) >= 0.5).astype(int)
    correct = int((pred == eval_df["result"]).sum())
    n = len(eval_df)
    accuracy = correct / n
    binom_pvalue = binomtest(correct, n, p=0.5).pvalue
    return accuracy, binom_pvalue


def build_summary_text(n_rows, n_users, model_a, acc_a, p_a, model_b, acc_b, p_b, val_n,
                        test_metrics_a=None, test_metrics_b=None, test_n=None):
    """test_metrics_a/b는 EVALUATE_TEST_SET=True로 최종 확인할 때만 넘긴다. None이면 test
    관련 줄을 아예 안 넣는다 — winrate/train_winrate.py와 같은 안전장치."""
    lr_stat = 2 * (model_b.llf - model_a.llf)
    lr_pvalue = float(scipy_stats.chi2.sf(lr_stat, df=1))

    lines = [
        "=== squad_fit_score 승률 상관관계 검증 ===",
        f"모집단: 스타일 진단 대상 유저(MIN_MATCHES_PER_USER={preprocess.MIN_MATCHES_PER_USER})가 "
        f"data/style/matches_style.jsonl에서 실제로 뛴 정상종료·승패 경기 {n_rows}건 "
        f"({n_users}명 관여)",
        f"train/val/test 분할: {TRAIN_RATIO:.0%}/{VAL_RATIO:.0%}/{TEST_RATIO:.0%}, stratified, "
        f"random_seed={RANDOM_SEED}",
        "",
        f"--- A. 공식 5-feature baseline과 동일 구성 ({FIVE_FEATURE_COLUMNS}) ---",
        model_a.summary2().as_text(),
        f"validation accuracy: {acc_a:.4f} ({val_n}건 중 {round(acc_a * val_n)}건 적중), "
        f"50% 우연 대비 이항검정 p-value: {p_a:.4f}",
        (
            f"test accuracy: {test_metrics_a[0]:.4f} ({test_n}건 중 "
            f"{round(test_metrics_a[0] * test_n)}건 적중), 50% 우연 대비 이항검정 p-value: "
            f"{test_metrics_a[1]:.4f}"
            if test_metrics_a is not None else
            "test: 아직 평가 안 함 (EVALUATE_TEST_SET=False)"
        ),
        "",
        f"--- B. A + squad_fit_score ({SIX_FEATURE_COLUMNS}) ---",
        model_b.summary2().as_text(),
        f"validation accuracy: {acc_b:.4f} ({val_n}건 중 {round(acc_b * val_n)}건 적중), "
        f"50% 우연 대비 이항검정 p-value: {p_b:.4f}",
        (
            f"test accuracy: {test_metrics_b[0]:.4f} ({test_n}건 중 "
            f"{round(test_metrics_b[0] * test_n)}건 적중), 50% 우연 대비 이항검정 p-value: "
            f"{test_metrics_b[1]:.4f}\n  -> p < 0.05이면 우연(50%)보다 유의미하게 낫다고 판단 가능."
            if test_metrics_b is not None else
            "test: 아직 평가 안 함 (EVALUATE_TEST_SET=False) — feature/모집단 확정 전에는 "
            "validation만 보고 판단할 것."
        ),
        "",
        f"[우도비검정(LR test, validation 기준)] LR={lr_stat:.4f}, df=1, p-value={lr_pvalue:.4f}",
        "  squad_fit_score를 추가했을 때 모델이 통계적으로 유의미하게 더 잘 설명하는지의 "
        "가장 정확한 기준. 개별 계수 p-value와 결론이 다를 수 있는데(표본이 작을 때 "
        "정확도 기반 검정과 계수 기반 검정이 어긋날 수 있음), 그 경우 LR test를 우선한다.",
        "",
        "[한계] squad_fit_score는 유저 본인의 여러 경기 평균 비율로 계산되고 그 평균을 "
        "각 경기에 동일하게 적용한다 — 한 경기의 결과가 자기 자신의 feature 평균에 "
        "약하게 기여하므로 완전한 독립은 아니다(경기 수가 많을수록 기여도는 작아짐).",
        "[한계] 모집단이 '스타일 진단된 유저가 실제로 뛴 경기'로 한정돼 공식 승률 baseline "
        "(수천 명, 1유저 1경기)보다 훨씬 좁고, 활동량이 많은(50경기 이상) 유저 쪽으로 "
        "편향돼 있을 수 있어 이 결과를 넓은 모집단에 그대로 일반화하기는 이르다.",
    ]
    return "\n".join(lines)


def main():
    print("1) 스타일 진단 대상 유저 + 유저별 스타일 비율 구성 (train_style.py와 동일 population)")
    style_by_ouid = build_style_population()
    print(f"  대상 유저 수: {len(style_by_ouid)}")

    print("2) data/style/matches_style.jsonl에서 대상 유저가 뛴 경기의 squad_fit_score 계산")
    player_stats = pf.load_player_stats()
    df = extract_squad_fit_rows(style_by_ouid, player_stats)
    df = df.dropna(subset=SIX_FEATURE_COLUMNS)
    n_users = df["ouid"].nunique()
    print(f"  최종 데이터셋: {len(df)}건 ({n_users}명)")

    print("3) train/validation/test 분할 및 로지스틱회귀 2종 학습")
    train_df, val_df, test_df = split_dataset(df, OUTPUT_DIR)

    model_a = fit_model(train_df, FIVE_FEATURE_COLUMNS)
    acc_a, p_a = evaluate_model(model_a, val_df, FIVE_FEATURE_COLUMNS)
    model_b = fit_model(train_df, SIX_FEATURE_COLUMNS)
    acc_b, p_b = evaluate_model(model_b, val_df, SIX_FEATURE_COLUMNS)
    print(f"  A. 공식 5-feature: validation accuracy={acc_a:.4f} (p={p_a:.4f})")
    print(f"  B. 5-feature+squad_fit_score: validation accuracy={acc_b:.4f} (p={p_b:.4f})")

    test_metrics_a = test_metrics_b = None
    if EVALUATE_TEST_SET:
        test_metrics_a = evaluate_model(model_a, test_df, FIVE_FEATURE_COLUMNS)
        test_metrics_b = evaluate_model(model_b, test_df, SIX_FEATURE_COLUMNS)
        print(f"  A. 공식 5-feature: test accuracy={test_metrics_a[0]:.4f} (p={test_metrics_a[1]:.4f})")
        print(f"  B. 5-feature+squad_fit_score: test accuracy={test_metrics_b[0]:.4f} "
              f"(p={test_metrics_b[1]:.4f})")
    else:
        print("  test: 평가 생략 (EVALUATE_TEST_SET=False)")

    print("4) 결과 저장")
    summary_text = build_summary_text(
        len(df), n_users, model_a, acc_a, p_a, model_b, acc_b, p_b, len(val_df),
        test_metrics_a, test_metrics_b, len(test_df),
    )
    utils.ensure_dir(OUTPUT_DIR)
    utils.save_text(summary_text, os.path.join(OUTPUT_DIR, "style_fit_winrate_summary.txt"))
    print(f"  [저장] {OUTPUT_DIR}/style_fit_winrate_summary.txt")


if __name__ == "__main__":
    main()
