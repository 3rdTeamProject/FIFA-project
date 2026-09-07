"""
실시간 스타일 진단 + 스쿼드 궁합 점수 ("진단하기" 버튼의 백엔드 로직, CLAUDE.md 핵심 기능 1~5번).

닉네임 -> ouid -> 최근 30경기 조회(2026-09-07 확정, CLAUDE.md 핵심 기능 2번 참고) ->
스타일 비율 계산 -> 학습된 K-means로 군집 배정 -> 가장 최근 "정상종료" 경기로 현재 스쿼드
자동구성(몰수 경기는 player[]가 비어있는 경우가 많아 제외, CLAUDE.md 핵심 기능 3번 참고) ->
position_fit.py로 포지션별/스쿼드 궁합 점수 -> 규칙 기반 진단 문장까지 한 번에 처리한다.

API 호출은 fetch_recent_matches()에서만 발생한다. 그 이후 단계(비율 계산, 스쿼드 구성,
궁합 점수 계산)는 전부 순수 함수라 네트워크 없이 테스트 가능하고, 대시보드에서 유저가
스쿼드 슬롯을 다른 카드로 바꿨을 때도 score_squad()만 다시 부르면 된다 — user_style은
그대로 재사용하고 API를 다시 호출할 필요가 없다("대표 스쿼드를 게임 내 지정값에서 가져오자"는
안은 기각됨 — Nexon 공식 Open API/공식 사이트 어디에도 그런 엔드포인트가 없고, 있는 건 로그인
필요한 웹 프로필 페이지뿐이라 크롤링+자동로그인은 계정 정지 위험이 있어 채택하지 않았다).
"""

import os
import time

import joblib
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

import position_fit as pf
import preprocess_style as preprocess

# ============ CONFIG ============
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(REPO_ROOT, "models", "style_diagnosis_baseline.joblib")

API_KEY = os.environ.get("NEXON_API_KEY", "여기에_발급받은_API_키")
BASE_URL = "https://open.api.nexon.com/fconline/v1"
HEADERS = {"x-nxopen-api-key": API_KEY}
REQUEST_INTERVAL = 0.35  # winrate/snowball_collect.py와 동일 (초당 5건 제한 대응)

N_MATCHES = 30  # 2026-09-07 확정 (CLAUDE.md 핵심 기능 2번 참고)
MATCHTYPE = 50  # 공식경기

SUB_POSITION = 28
POS_NAME = {
    0: "GK", 1: "SW", 2: "RWB", 3: "RB", 4: "RCB", 5: "CB", 6: "LCB", 7: "LB", 8: "LWB",
    9: "RDM", 10: "CDM", 11: "LDM", 12: "RM", 13: "RCM", 14: "CM", 15: "LCM", 16: "LM",
    17: "RAM", 18: "CAM", 19: "LAM", 20: "RF", 21: "CF", 22: "LF", 23: "RW", 24: "RS",
    25: "ST", 26: "LS", 27: "LW", 28: "SUB",
}

# 군집 번호 -> 사후 라벨 (data/style/style_diagnosis_baseline_summary.txt의 [군집별 feature
# 평균]을 보고, 각 군집이 다른 군집 대비 뚜렷하게 높은 feature를 기준으로 붙였다 —
# CLAUDE.md 원칙 #4: 이름은 군집 결과가 나온 뒤 사후에 붙인다).
#   pass  0: through_pass_ratio가 세 군집 중 가장 높음(0.198 vs 0.128/0.132)
#   pass  1: short_pass_ratio가 압도적으로 높고 나머지가 다 낮음(0.764)
#   pass  2: driven_ground_pass_ratio가 세 군집 중 가장 높음(0.164 vs 0.058/0.059)
#   shoot 0: in_penalty는 2와 비슷하게 높지만 heading이 셋 중 가장 낮음(0.090)
#   shoot 1: in_penalty_shoot_ratio가 셋 중 가장 낮음(0.689) -> 상대적으로 중거리슛 비중 ↑
#   shoot 2: heading_shoot_ratio가 다른 군집의 두 배 이상(0.218)
# ⚠️ models/style_diagnosis_baseline.joblib을 train_style.py로 다시 학습하면 군집 번호
# 순서가 바뀔 수 있다 — 재학습 시 반드시 새 요약 파일을 다시 보고 이 매핑을 갱신할 것.
PASS_CLUSTER_LABELS = {
    0: "스루패스 위주",
    1: "숏패스 위주",
    2: "드리븐그라운드패스 위주",
}
SHOOT_CLUSTER_LABELS = {
    0: "박스 안(인패널티) 슈팅 위주",
    1: "중거리슛 위주",
    2: "헤딩슛 위주",
}
# ====================================================


class RateLimitedError(Exception):
    """429(rate limit) 응답. winrate/snowball_collect.py와 동일 의미."""


def _get(path, params):
    """공통 GET 요청 래퍼 (winrate/snowball_collect.py의 _get()과 동일 컨벤션)."""
    resp = requests.get(f"{BASE_URL}{path}", headers=HEADERS, params=params, timeout=10)
    time.sleep(REQUEST_INTERVAL)
    if resp.status_code == 429:
        raise RateLimitedError(f"{path} 429: {resp.text[:200]}")
    resp.raise_for_status()
    return resp.json()


def fetch_recent_matches(nickname, n_matches=N_MATCHES, matchtype=MATCHTYPE):
    """닉네임 -> ouid -> 최근 n_matches경기 상세 조회.

    API 호출이 발생하는 유일한 함수 (1 + 1 + n_matches회). 반환값은
    (ouid, [이 유저 기준 matchInfo dict, ...]) — 최신순 그대로.
    """
    ouid = _get("/id", {"nickname": nickname})["ouid"]

    match_ids = _get(
        "/user/match",
        {"ouid": ouid, "matchtype": matchtype, "offset": 0, "limit": n_matches},
    )

    match_infos = []
    for match_id in match_ids:
        detail = _get("/match-detail", {"matchid": match_id})
        mi = next(m for m in detail["matchInfo"] if m["ouid"] == ouid)
        match_infos.append(mi)

    return ouid, match_infos


def matches_to_style_rows(ouid, match_infos):
    """raw matchInfo 리스트 -> preprocess_style.extract_match_style_rows()가 기대하는
    컬럼의 DataFrame으로 변환한다 (비율 계산 로직은 중복 구현하지 않고 그대로 재사용하기
    위한 어댑터일 뿐).
    """
    rows = []
    for mi in match_infos:
        p, s = mi["pass"], mi["shoot"]
        rows.append({
            preprocess.MATCH_ID_COL: mi.get("matchId", ""),
            preprocess.OUID_COL: ouid,
            preprocess.MATCH_END_TYPE_COL: mi["matchDetail"]["matchEndType"],
            preprocess.PASS_TRY_COL: p.get("passTry", 0),
            preprocess.SHORT_PASS_TRY_COL: p.get("shortPassTry", 0),
            preprocess.LONG_PASS_TRY_COL: p.get("longPassTry", 0),
            "throughPassTry": p.get("throughPassTry", 0),
            "lobbedThroughPassTry": p.get("lobbedThroughPassTry", 0),
            preprocess.DRIVEN_GROUND_PASS_TRY_COL: p.get("drivenGroundPassTry", 0),
            preprocess.DRIBBLE_YARD_COL: mi["matchDetail"].get("dribble", 0),
            preprocess.SHOOT_TOTAL_COL: s.get("shootTotal", 0),
            preprocess.SHOOT_HEADING_COL: s.get("shootHeading", 0),
            preprocess.SHOOT_IN_PENALTY_COL: s.get("shootInPenalty", 0),
            preprocess.SHOOT_OUT_PENALTY_COL: s.get("shootOutPenalty", 0),
        })
    return pd.DataFrame(rows)


def compute_user_style(raw_rows_df):
    """matches_to_style_rows()가 만든 원시(camelCase) DataFrame -> position_fit.py가
    요구하는 7개 키를 가진 유저 스타일 비율 dict.

    preprocess_style.extract_match_style_rows()로 몰수/오류 경기를 거르고 snake_case
    컬럼으로 바꾼 뒤, aggregate_user_style()로 비율을 낸다 — 두 단계 다 재사용(중복 구현
    금지). 라이브 진단은 학습용 70/50경기 하한선(MIN_MATCHES_PER_USER)과 무관하므로
    min_matches=1로 호출한다 — 유저 표본이 몇 경기든 있는 그대로 비율을 낸다.
    """
    rows_df = preprocess.extract_match_style_rows(raw_rows_df)
    style_df = preprocess.aggregate_user_style(rows_df, min_matches=1)
    if style_df.empty:
        return None
    row = style_df.iloc[0]
    columns = preprocess.PASS_FEATURE_COLUMNS + preprocess.SHOOT_FEATURE_COLUMNS + preprocess.REFERENCE_COLUMNS
    return {col: row[col] for col in columns}


def predict_style_clusters(user_style):
    """학습된 style_diagnosis_baseline.joblib으로 pass/shoot 군집 라벨을 예측한다."""
    model = joblib.load(MODEL_PATH)

    x_pass = [[user_style[c] for c in preprocess.PASS_FEATURE_COLUMNS]]
    x_shoot = [[user_style[c] for c in preprocess.SHOOT_FEATURE_COLUMNS]]

    pass_scaled = model["pass"]["scaler"].transform(x_pass)
    shoot_scaled = model["shoot"]["scaler"].transform(x_shoot)

    pass_cluster = int(model["pass"]["kmeans"].predict(pass_scaled)[0])
    shoot_cluster = int(model["shoot"]["kmeans"].predict(shoot_scaled)[0])
    return pass_cluster, shoot_cluster


def build_current_squad(match_infos):
    """match_infos(최신순)에서 몰수가 아니고 11명이 온전히 채워진 가장 최근 경기를 찾아
    [{"sp_id": int, "sp_position": int}, ...] 11개를 반환한다.

    matchEndType==0만으로는 안전하지 않다 — 정상종료여도 드물게 player 수가 이상한
    레코드가 있을 수 있어, "SUB 제외 11명"까지 같이 확인한다. 못 찾으면 (None, 사유) 반환.
    """
    for mi in match_infos:
        if mi["matchDetail"]["matchEndType"] != 0:
            continue
        starters = [p for p in mi.get("player", []) if p["spPosition"] != SUB_POSITION]
        if len(starters) != 11:
            continue
        squad = [{"sp_id": p["spId"], "sp_position": p["spPosition"]} for p in starters]
        return squad, mi["matchDetail"].get("matchResult")

    return None, "정상종료 + 11명이 온전한 경기를 찾지 못했습니다"


def score_squad(user_style, squad, player_stats=None):
    """squad([{"sp_id", "sp_position"}, ...])의 포지션별/스쿼드 전체 궁합 점수.

    순수 함수 — API 호출 없음. 대시보드에서 유저가 슬롯 하나를 다른 카드로 바꾸면,
    바뀐 squad로 이 함수만 다시 부르면 된다(user_style은 재사용, 재계산 불필요).
    """
    if player_stats is None:
        player_stats = pf.load_player_stats()

    results = []
    for slot in squad:
        sp_id, sp_position = slot["sp_id"], slot["sp_position"]
        pos_name = POS_NAME.get(sp_position, str(sp_position))
        if sp_id not in player_stats.index:
            results.append({
                "sp_id": sp_id, "sp_position": sp_position, "pos_name": pos_name,
                "pass_fit": np.nan, "shoot_fit": np.nan, "position_fit": np.nan,
                "note": "카드 스탯 없음",
            })
            continue
        card = player_stats.loc[sp_id]
        pass_fit = pf.compute_pass_fit_score(user_style, card, sp_position)
        shoot_fit = pf.compute_shoot_fit_score(user_style, card, sp_position)
        position_fit = pf.compute_position_fit_score(user_style, card, sp_position)
        results.append({
            "sp_id": sp_id, "sp_position": sp_position, "pos_name": pos_name,
            "pass_fit": pass_fit, "shoot_fit": shoot_fit, "position_fit": position_fit,
            "note": None,
        })

    valid = [r for r in results if r["position_fit"] == r["position_fit"]]  # NaN 제외
    squad_fit_score = float(np.mean([r["position_fit"] for r in valid])) if valid else np.nan

    return {"slots": results, "squad_fit_score": squad_fit_score}


def generate_diagnosis_sentence(user_style, squad_scores, pass_cluster, shoot_cluster):
    """규칙 기반 진단 문장 (CLAUDE.md 핵심 기능 5번 예시 형식).

    특정 카드를 콕 집어 추천하지 않고, 가장 궁합이 낮은 포지션과 방향성만 제시한다.

    패스/슛 스타일은 유저 본인의 raw 비율 중 max()로 뽑지 않는다 — 축구 게임 특성상
    거의 모든 유저가 short_pass_ratio가 제일 커서, max()로는 사실상 항상 "숏패스 위주"만
    나오고 predict_style_clusters()가 계산한 K-means 군집 배정 결과가 버려지는 문제가
    있었다. 대신 그 유저가 실제로 배정된 pass_cluster/shoot_cluster 번호를
    PASS_CLUSTER_LABELS/SHOOT_CLUSTER_LABELS로 사후 라벨링해 사용한다.
    """
    pass_label = PASS_CLUSTER_LABELS.get(pass_cluster, f"패스 군집 {pass_cluster}")
    shoot_label = SHOOT_CLUSTER_LABELS.get(shoot_cluster, f"슛 군집 {shoot_cluster}")

    valid = [r for r in squad_scores["slots"] if r["position_fit"] == r["position_fit"]]
    lines = [f"당신은 {pass_label} · {shoot_label} 스타일입니다."]
    if valid:
        worst = min(valid, key=lambda r: r["position_fit"])
        lines.append(
            f"현재 스쿼드 궁합 점수는 {squad_scores['squad_fit_score']:.2f}(1.0 만점)이며, "
            f"{worst['pos_name']} 포지션의 궁합이 가장 낮습니다({worst['position_fit']:.2f}) — "
            f"이 자리에 맞는 스탯을 가진 카드를 고려해보세요."
        )
    return " ".join(lines)


def diagnose(nickname, n_matches=N_MATCHES):
    """최상위 오케스트레이터 — 대시보드는 이 함수 하나만 호출하면 된다."""
    ouid, match_infos = fetch_recent_matches(nickname, n_matches=n_matches)

    rows_df = matches_to_style_rows(ouid, match_infos)
    user_style = compute_user_style(rows_df)
    if user_style is None:
        return {"ouid": ouid, "error": "정상종료 경기가 없어 스타일을 계산할 수 없습니다"}

    pass_cluster, shoot_cluster = predict_style_clusters(user_style)

    squad, match_result = build_current_squad(match_infos)
    if squad is None:
        return {
            "ouid": ouid, "user_style": user_style,
            "pass_cluster": pass_cluster, "shoot_cluster": shoot_cluster,
            "squad": None, "squad_scores": None, "sentence": None,
            "error": match_result,
        }

    squad_scores = score_squad(user_style, squad)
    sentence = generate_diagnosis_sentence(user_style, squad_scores, pass_cluster, shoot_cluster)

    return {
        "ouid": ouid,
        "user_style": user_style,
        "pass_cluster": pass_cluster,
        "shoot_cluster": shoot_cluster,
        "squad": squad,
        "squad_match_result": match_result,
        "squad_scores": squad_scores,
        "sentence": sentence,
    }
