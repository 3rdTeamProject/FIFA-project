"""
matches.jsonl / player_1000_final.csv 파싱 및 feature 조립.

matches.jsonl 필드 매핑은 2026-09-04에 실제 API 응답(match-detail)을 1건 조회해 확인한
스키마를 기준으로 한다:
    matchInfo[i]["division"]                       -> tier (정수 코드, 예: 2300, 2400)
    matchInfo[i]["matchDetail"]["matchResult"]      -> "승" / "패" / "무"
    matchInfo[i]["matchDetail"]["matchEndType"]     -> 0이 정상 종료
    matchInfo[i]["player"][j]["spPosition"] == 28   -> 교체선수(SUB), 스쿼드 계산에서 제외

player_1000_final.csv는 이 스크립트 작성 시점에 실물 파일이 없어 컬럼명이 확정되지 않았다.
CSV_SPID_COLUMN / CSV_STAT_COLUMNS는 data-schema.md 기준 추정값이며, 실제 파일을 확보하면
이 파일 상단의 CONFIG만 고쳐서 맞추면 되도록 분리해뒀다. 필요한 컬럼이 없으면 조용히 넘어가지
않고 실제 컬럼 목록을 보여주며 에러를 낸다.

또한 이 파일 수치가 강화단계(spGrade)를 반영한 값인지 기본(0강) 값인지 불분명하다.
확인 전까지는 강화단계를 무시하고 spId만으로 매칭하는 근사치로 취급한다 (한계로 명시,
train.build_summary_text()가 쓰는 summary 텍스트에도 남긴다).
"""

import json

import numpy as np
import pandas as pd

# ============ CONFIG ============
MATCHES_FILE = "matches.jsonl"
PLAYER_CARD_FILE = "data/player_1000_final.csv"

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
            "preprocess.py 상단 CONFIG의 CSV_SPID_COLUMN/CSV_STAT_COLUMNS를 "
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
