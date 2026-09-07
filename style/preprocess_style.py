"""
match_team_data.csv 파싱 및 패스/슛 스타일 feature 조립.

2026-09-07 팀 결정: 스타일 트랙은 collect_style_matches.py로 matches_style.jsonl을
직접 수집하지 않고, 팀원이 이미 팀 단위로 평탄화해 수집한 data/style/match_team_data.csv
를 그대로 원본으로 쓴다.

2026-09-07 추가 결정: 애초 계획이던 "6개 feature 통합 1개 모델"을 버리고 **패스 모델 +
슛 모델 2개로 분리**했다. 실제 데이터로 여러 조합을 실험한 결과:
  - 6개 feature를 한 공간에서 같이 군집화 -> 실루엣 0.23~0.25 (구조 거의 없음)
  - 패스 4개만 단독 -> 실루엣 0.33 / 슛 2개만 단독 -> 실루엣 0.34 (둘 다 뚜렷이 개선)
  - dribble_intensity, possession, block_ratio(블락/(블락+태클)), shoot_intensity(경기당
    슛 시도 수)를 어느 모델에 섞어도 전부 실루엣이 떨어졌다 — 이 값들은 "선택"이 아니라
    "실력/경기 흐름"에 더 가깝게 얽혀 있어(예: shoot_intensity는 division과 상관계수 0.32)
    패스/슛 선택 신호를 오히려 희석시킨다. 그래서 이 값들은 군집화에는 넣지 않고, dribble
    관련 수치만 진단 문장에 참고용으로 곁들일 수 있게 dribble_intensity는 계속 계산해서
    같이 반환한다(군집 feature로는 쓰지 않음).
  - K-means 외 GMM/계층적/Spectral/DBSCAN도 비교해봤지만(팀 결정으로 최종은 K-means만
    사용) 이 표본 크기(수백 명)·차원(2~4개)에서 K-means보다 확실히 나은 대안은 없었다.

컬럼 매핑은 2026-09-07에 data/style/match_team_data.csv를 직접 확인해 정리했다:
    matchId                                -> 경기 ID
    ouid                                   -> 유저 ID
    matchEndType                           -> 0(정상종료)만 정상 경기로 쓴다.
        그 외 값(1/2=몰수 승/패로 확인됨, 4="오류"로 확인됨 — 이 경우 나머지 스탯
        필드가 전부 NULL)은 실력/스타일과 무관한 비정상 종료이므로 전부 제외한다.
    passTry / shortPassTry / longPassTry / drivenGroundPassTry -> 패스 시도
    throughPassTry + lobbedThroughPassTry  -> 스루 패스 시도 (합산, CLAUDE.md 방침)
    dribble                                -> 팀 드리블 야드 (참고용, 군집 feature 아님)
    shootTotal / shootHeading / shootInPenalty -> 슛 시도
⚠️ shootOutPenalty는 shootInPenalty와 거의 완전한 여집합(실측 89%가 정확히
   shootInPenalty+shootOutPenalty=shootTotal)이라 **군집화(K-means) feature로는**
   새 정보가 거의 없어 쓰지 않는다. 다만 position_fit.py의 궁합 점수 계산에서는
   "중거리슛" 카드 스탯을 매핑할 축이 필요해서 out_penalty_shoot_ratio로 계산은
   해서 반환한다(REFERENCE_COLUMNS 참고, PASS/SHOOT_FEATURE_COLUMNS에는 포함 안 함
   — 군집 결과 자체는 안 바뀜).
⚠️ bouncingLobPassTry도 시도해봤지만 대부분 유저가 0에 가까운 값만 가져(변별력 없음)
   빼기로 했다.
⚠️ shootPenaltyKick/shootFreekick(세트피스)는 상대 반칙에 좌우되는 값이라 CLAUDE.md의
   "유저의 선택이 아니라 상황이 만드는 값" 원칙에 따라 절대 쓰지 않는다.
⚠️ passSuccess/shootSuccess 등 "성공률" 계열은 절대 쓰지 않는다 — 이건 선택이 아니라
   실력(컨트롤 정확도)이라, 넣으면 스타일 군집이 아니라 실력(티어) 군집이 되어버린다.
⚠️ goalTotal/goalHeading 등 "골 여부"가 들어간 필드, foul/redCards/yellowCards/
   offsideCount/averageRating/controller/effectiveShootTotal 등도 스타일 진단과
   무관하거나(상황 의존적) 실력에 가까워 쓰지 않는다.

2026-09-07 결정: 팀원이 data/style/match_team_data.csv 자체를 70경기 이상 유저로만
정제해서 넘겨주기로 해, 파일 안 유저는 이미 다 70경기 이상이다. 아래
MIN_MATCHES_PER_USER=50은 그 정제를 신뢰하되, 혹시 다른(미정제) 파일이 들어오거나
정제 기준이 바뀌는 경우를 대비한 안전장치(safety net)다 — 정상적으로는 이 필터에
걸리는 유저가 없어야 한다.
"""

import os

import numpy as np
import pandas as pd

# ============ CONFIG ============
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATCHES_FILE = os.path.join(REPO_ROOT, "data", "style", "match_team_data.csv")

NORMAL_MATCH_END_TYPE = 0

# 안전장치용 하한선 (위 docstring 2026-09-07 결정 참고). 실제 하한은 팀원이
# match_team_data.csv를 정제할 때 적용하는 70경기 기준이며, 이 값은 그보다 낮게
# 잡은 방어선일 뿐이다.
MIN_MATCHES_PER_USER = 50

# --- 컬럼 매핑 (위 docstring 참고, 2026-09-07 확인) ---
MATCH_ID_COL = "matchId"
OUID_COL = "ouid"
MATCH_END_TYPE_COL = "matchEndType"
PASS_TRY_COL = "passTry"
SHORT_PASS_TRY_COL = "shortPassTry"
LONG_PASS_TRY_COL = "longPassTry"
THROUGH_PASS_TRY_COLS = ["throughPassTry", "lobbedThroughPassTry"]
DRIVEN_GROUND_PASS_TRY_COL = "drivenGroundPassTry"
DRIBBLE_YARD_COL = "dribble"
SHOOT_TOTAL_COL = "shootTotal"
SHOOT_HEADING_COL = "shootHeading"
SHOOT_IN_PENALTY_COL = "shootInPenalty"
SHOOT_OUT_PENALTY_COL = "shootOutPenalty"

# 패스 모델과 슛 모델을 분리해서 각각 K-means로 군집화한다(위 docstring 참고).
PASS_FEATURE_COLUMNS = [
    "short_pass_ratio",
    "long_pass_ratio",
    "through_pass_ratio",
    "driven_ground_pass_ratio",
]
SHOOT_FEATURE_COLUMNS = [
    "in_penalty_shoot_ratio",
    "heading_shoot_ratio",
]
# 군집(K-means) feature로는 안 쓰지만, 진단 문장 참고 수치 또는 position_fit.py의
# 궁합 점수 계산에 필요해서 계속 계산해서 반환한다.
REFERENCE_COLUMNS = ["dribble_intensity", "out_penalty_shoot_ratio"]
# ====================================================


def load_matches(path):
    """match_team_data.csv를 로드한다 (팀원이 팀 단위로 평탄화해 수집한 원본)."""
    return pd.read_csv(path)


def extract_match_style_rows(df):
    """match_team_data.csv의 팀 단위 원시 행에서 (match_id, ouid) 단위 원시 카운트를
    뽑는다.

    승/무/패는 스타일과 무관하므로 결과로는 필터링하지 않는다.
    몰수/오류 경기(matchEndType != 0)만 데이터 품질 문제로 제외한다.
    같은 유저가 여러 매치에 나오는 것은 여기서는 정상이다(오히려 그게 목적 —
    한 유저의 최근 최대 100경기를 모아 집계해야 하므로 dedup하지 않는다).
    """
    before = len(df)
    normal = df[df[MATCH_END_TYPE_COL] == NORMAL_MATCH_END_TYPE]
    excluded_forfeit = before - len(normal)

    through_pass_try = normal[THROUGH_PASS_TRY_COLS].fillna(0).sum(axis=1)

    rows_df = pd.DataFrame({
        "match_id": normal[MATCH_ID_COL].values,
        "ouid": normal[OUID_COL].values,
        "pass_try": normal[PASS_TRY_COL].fillna(0).values,
        "short_pass_try": normal[SHORT_PASS_TRY_COL].fillna(0).values,
        "long_pass_try": normal[LONG_PASS_TRY_COL].fillna(0).values,
        "through_pass_try": through_pass_try.values,
        "driven_ground_pass_try": normal[DRIVEN_GROUND_PASS_TRY_COL].fillna(0).values,
        "dribble_yard": normal[DRIBBLE_YARD_COL].fillna(0).values,
        "shoot_total": normal[SHOOT_TOTAL_COL].fillna(0).values,
        "shoot_heading": normal[SHOOT_HEADING_COL].fillna(0).values,
        "shoot_in_penalty": normal[SHOOT_IN_PENALTY_COL].fillna(0).values,
        "shoot_out_penalty": normal[SHOOT_OUT_PENALTY_COL].fillna(0).values,
    })

    print(f"  [필터링] 몰수/오류 경기 제외: {excluded_forfeit}건")
    print(f"  [추출] 최종 (match, ouid) 행 수: {len(rows_df)}")
    return rows_df


def _safe_ratio(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else np.nan


def aggregate_user_style(rows_df, min_matches=MIN_MATCHES_PER_USER):
    """(match, ouid) 원시 카운트를 유저 단위로 합산해 패스/슛 비율 feature를 계산한다.

    min_matches 미만인 유저는 비율이 노이즈에 크게 좌우되므로 제외한다.
    분모(pass_try/shoot_total)가 0인 유저도 비율 계산이 불가능해 제외한다.

    ⚠️ 티어(division) 정규화는 의도적으로 하지 않는다. 진단 화면에서 "관찰적 진단"을
    제시하기 위해, 유저 본인의 원본 비율 그대로 쓴다 (style/README.md 참고).
    """
    all_columns = ["ouid", "n_matches"] + PASS_FEATURE_COLUMNS + SHOOT_FEATURE_COLUMNS + REFERENCE_COLUMNS
    if rows_df.empty:
        return pd.DataFrame(columns=all_columns)

    grouped = rows_df.groupby("ouid").agg(
        n_matches=("match_id", "count"),
        pass_try=("pass_try", "sum"),
        short_pass_try=("short_pass_try", "sum"),
        long_pass_try=("long_pass_try", "sum"),
        through_pass_try=("through_pass_try", "sum"),
        driven_ground_pass_try=("driven_ground_pass_try", "sum"),
        dribble_yard=("dribble_yard", "sum"),
        shoot_total=("shoot_total", "sum"),
        shoot_heading=("shoot_heading", "sum"),
        shoot_in_penalty=("shoot_in_penalty", "sum"),
        shoot_out_penalty=("shoot_out_penalty", "sum"),
    ).reset_index()

    before = len(grouped)
    grouped = grouped[grouped["n_matches"] >= min_matches].copy()
    print(
        f"  [표본 필터] 최근 경기 {min_matches}건 미만 유저 제외: "
        f"{before - len(grouped)}명 (남은 유저: {len(grouped)}명)"
    )

    grouped["short_pass_ratio"] = grouped.apply(
        lambda r: _safe_ratio(r["short_pass_try"], r["pass_try"]), axis=1
    )
    grouped["long_pass_ratio"] = grouped.apply(
        lambda r: _safe_ratio(r["long_pass_try"], r["pass_try"]), axis=1
    )
    grouped["through_pass_ratio"] = grouped.apply(
        lambda r: _safe_ratio(r["through_pass_try"], r["pass_try"]), axis=1
    )
    grouped["driven_ground_pass_ratio"] = grouped.apply(
        lambda r: _safe_ratio(r["driven_ground_pass_try"], r["pass_try"]), axis=1
    )
    grouped["in_penalty_shoot_ratio"] = grouped.apply(
        lambda r: _safe_ratio(r["shoot_in_penalty"], r["shoot_total"]), axis=1
    )
    grouped["heading_shoot_ratio"] = grouped.apply(
        lambda r: _safe_ratio(r["shoot_heading"], r["shoot_total"]), axis=1
    )
    # 군집 feature로는 안 쓰는 참고 수치 (docstring 참고 — 실루엣을 오히려 낮춰 제외함).
    grouped["dribble_intensity"] = grouped["dribble_yard"] / grouped["n_matches"]
    grouped["out_penalty_shoot_ratio"] = grouped.apply(
        lambda r: _safe_ratio(r["shoot_out_penalty"], r["shoot_total"]), axis=1
    )

    feature_columns = PASS_FEATURE_COLUMNS + SHOOT_FEATURE_COLUMNS
    before = len(grouped)
    style_df = grouped.dropna(subset=feature_columns).copy()
    dropped = before - len(style_df)
    print(f"  [비율 계산] 분모가 0이라 비율을 못 낸 유저 제외: {dropped}명")

    return style_df[all_columns]
