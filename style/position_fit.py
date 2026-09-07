"""
포지션별 스타일 궁합 점수(position_fit_score) 계산.

USER_STYLE_PROFILE(패스 4개 + 슛 2개 비율)과 PLAYER_CARD(player_stats_final.csv,
필드플레이어 스탯) 사이의 코사인 유사도로 "이 카드가 유저 스타일에 얼마나 맞는가"를
잰다. 설계 근거와 실험 과정은 Notion "포지션별 스타일 궁합 점수 설계 (2026-09-07)"
페이지에 정리되어 있다.

핵심 원칙:
- 점수는 style_type(K-means 군집 라벨)이 아니라 원래의 연속값 비율
  (preprocess_style.py의 PASS_FEATURE_COLUMNS/SHOOT_FEATURE_COLUMNS)로 계산한다.
  라벨은 진단 문장("당신은 스루패스 위주 스타일입니다") 전용이고, 점수는 경계선에
  걸친 유저도 부드럽게 나오도록 연속값을 그대로 쓴다.
- pass_fit_score와 shoot_fit_score는 처음부터 끝까지 완전히 독립적으로 계산한 뒤,
  포지션 성격에 따라 맨 마지막에만 합친다. 6개 feature를 한 공간에서 통합
  군집화했을 때 실루엣 점수가 떨어졌던 것과 같은 이유(서로 다른 축을 섞으면
  신호가 희석됨)로, 궁합 점수도 처음부터 섞지 않는다.
- 같은 스탯이 여러 축/포지션에 중복으로 들어가도 된다 — 각 축에 들어간 이유가
  그 축 고유의 이유이기만 하면 된다(다른 축의 이유를 빌려오면 안 됨).
- 좌/우 미러 포지션(RDM/LDM, RCM/LCM, RAM/LAM, RM/LM, RF/LF, RW/LW, RS/LS)은
  축구 역할이 완전히 같아 스탯 매핑도 동일하다.
- 스트라이커(RS/ST/LS)는 패스를 "보내는" 게 아니라 "받는" 입장이라, pass_fit의
  스탯 매핑이 다른 포지션과 근본적으로 다르다(볼 컨트롤/몸싸움/속력처럼 수신 관련
  스탯). 이름은 같은 pass_fit_score지만 의미가 다르다는 점에 주의.
- 수비(spPosition 1~8)와 GK(0)는 대응되는 스타일 축 자체가 없다고 보고 궁합 점수
  대상에서 완전히 제외한다.
- 축당 스탯을 2~3개로 제한한다. 스탯을 계속 추가하면 (1) 평균이 뭉개지고
  (2) 서로 상관관계 높은 FIFA 스탯 특성상 "스타일 궁합"이 아니라 그냥
  "종합 능력치"(avg_stat_score와 역할이 겹침)가 되어버린다.

2026-09-07 추가: shoot_fit은 preprocess_style.py가 군집화(K-means)에 쓰는
SHOOT_FEATURE_COLUMNS(in_penalty/heading 2개)이 아니라, 여기서만 쓰는 3축
(SHOOT_FIT_AXES = in_penalty/out_penalty/heading)을 쓴다. `out_penalty_shoot_ratio`는
군집화 실험 때는 in_penalty의 거의 완전한 여집합이라 정보량이 없어 뺐지만, 궁합
점수에서는 "중거리슛" 카드 스탯을 매핑할 자리가 필요해서 되살렸다 — 군집 라벨과
궁합 점수는 어차피 독립적으로 계산되므로(모듈 docstring 원칙 참고) 서로 다른
feature 집합을 써도 문제없다.

2026-09-07 추가: BOTH_GROUPS(CAM/DEEP_FORWARD/WIDE_FORWARD/STRIKER)의 position_fit_score
결합은 원래 pass_fit_score/shoot_fit_score를 무조건 50:50 평균했는데, CAM이 순수
스트라이커와 같은 비중으로 슛 궁합을 받는 게 축구 상식과 안 맞는다는 지적으로 역할별
가중치(BOTH_GROUP_WEIGHTS)를 도입했다. 자세한 근거는 그 상수 정의부 주석 참고.
"""

import os

import numpy as np
import pandas as pd

import preprocess_style as preprocess

# ============ CONFIG ============
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLAYER_STATS_FILE = os.path.join(REPO_ROOT, "data", "player_stats_final.csv")
PLAYER_ID_COL = "spid"

# --- 포지션 그룹 (api-constraints.md의 spPosition 28개 코드 기준) ---
GK_POSITION = 0
DEFENSE_POSITIONS = {1, 2, 3, 4, 5, 6, 7, 8}       # SW/RWB/RB/RCB/CB/LCB/LB/LWB
SUB_POSITION = 28

DM_POSITIONS = {9, 10, 11}             # RDM, CDM, LDM
CM_POSITIONS = {13, 14, 15}            # RCM, CM, LCM
WIDE_MID_POSITIONS = {12, 16}          # RM, LM
CAM_POSITIONS = {17, 18, 19}           # RAM, CAM, LAM
DEEP_FORWARD_POSITIONS = {20, 21, 22}  # RF, CF, LF
WIDE_FORWARD_POSITIONS = {23, 27}      # RW, LW
STRIKER_POSITIONS = {24, 25, 26}       # RS, ST, LS

# 포지션 그룹별 position_fit_score 결합 방식.
PASS_ONLY_GROUPS = {"DM", "CM", "WIDE_MID"}
BOTH_GROUPS = {"CAM", "DEEP_FORWARD", "WIDE_FORWARD", "STRIKER"}

# BOTH_GROUPS의 pass_fit_score/shoot_fit_score 결합 가중치 (pass_weight, shoot_weight),
# 2026-09-07 팀 결정. 원래는 전부 50:50 평균이었는데, CAM이 순수 스트라이커와 같은
# 비중으로 슛 궁합을 받는 게 축구 상식과 안 맞는다는 지적으로 역할별 차등을 도입했다:
# CAM은 플레이메이킹이 본업이라 pass 비중을 높게, STRIKER는 마무리가 본업이라 shoot
# 비중을 높게 잡는다. DEEP_FORWARD(CF)는 연계와 마무리를 둘 다 맡는 롤이라 50:50을
# 유지하고, WIDE_FORWARD는 측면 크로스/서비스가 침투 마무리보다 살짝 우선이라고 봐서
# pass 쪽에 약간 무게를 둔다. 데이터로 학습된 값이 아니라 팀이 축구 지식으로 정한
# 값이며, avg_stat_score 같은 종합 능력치가 되지 않도록 각 그룹의 축(pass_fit_score,
# shoot_fit_score) 자체는 여전히 완전히 독립적으로 계산한 뒤 마지막에만 합친다.
BOTH_GROUP_WEIGHTS = {
    "CAM": (0.7, 0.3),
    "DEEP_FORWARD": (0.5, 0.5),
    "WIDE_FORWARD": (0.6, 0.4),
    "STRIKER": (0.2, 0.8),
}

# shoot_fit은 K-means 군집화용 preprocess.SHOOT_FEATURE_COLUMNS(2축)이 아니라
# 여기서만 쓰는 3축을 쓴다 (모듈 docstring 2026-09-07 추가 참고).
SHOOT_FIT_AXES = ["in_penalty_shoot_ratio", "out_penalty_shoot_ratio", "heading_shoot_ratio"]


def group_of(sp_position):
    """spPosition 코드를 7개 궁합 점수 그룹 중 하나로 분류한다.

    수비/GK/SUB이거나 알 수 없는 코드면 None을 반환한다(궁합 점수 대상 아님).
    """
    if sp_position in DM_POSITIONS:
        return "DM"
    if sp_position in CM_POSITIONS:
        return "CM"
    if sp_position in WIDE_MID_POSITIONS:
        return "WIDE_MID"
    if sp_position in CAM_POSITIONS:
        return "CAM"
    if sp_position in DEEP_FORWARD_POSITIONS:
        return "DEEP_FORWARD"
    if sp_position in WIDE_FORWARD_POSITIONS:
        return "WIDE_FORWARD"
    if sp_position in STRIKER_POSITIONS:
        return "STRIKER"
    return None


# --- 패스 축 스탯 매핑 (그룹 -> 축 -> 카드 스탯 컬럼명 리스트) ---
# 스트라이커만 "송신"이 아니라 "수신" 기준 스탯을 쓴다 (모듈 docstring 참고).
PASS_STAT_MAP = {
    "DM": {
        "short_pass_ratio": ["짧은 패스", "볼 컨트롤"],
        "long_pass_ratio": ["긴 패스", "시야"],
        "through_pass_ratio": ["시야"],
        "driven_ground_pass_ratio": ["볼 컨트롤"],
    },
    "CM": {
        "short_pass_ratio": ["짧은 패스", "볼 컨트롤"],
        "long_pass_ratio": ["긴 패스", "시야"],
        "through_pass_ratio": ["시야"],
        "driven_ground_pass_ratio": ["볼 컨트롤"],
    },
    "WIDE_MID": {
        "short_pass_ratio": ["짧은 패스", "볼 컨트롤"],
        "long_pass_ratio": ["긴 패스", "크로스"],
        "through_pass_ratio": ["시야"],
        "driven_ground_pass_ratio": ["크로스", "볼 컨트롤"],
    },
    "CAM": {
        "short_pass_ratio": ["짧은 패스", "볼 컨트롤"],
        "long_pass_ratio": ["긴 패스", "시야"],
        "through_pass_ratio": ["시야", "가속력"],
        "driven_ground_pass_ratio": ["볼 컨트롤"],
    },
    "DEEP_FORWARD": {
        # short_pass/driven_ground은 콤비네이션 플레이라 송신 기준 유지.
        # long_pass/through_pass는 CAM(창조자)과 달리 CF는 최전방에서 그 패스를
        # "받는" 쪽에 훨씬 가까워서 ST와 같은 수신 기준 스탯을 쓴다.
        "short_pass_ratio": ["짧은 패스", "볼 컨트롤"],
        "long_pass_ratio": ["볼 컨트롤", "몸싸움"],
        "through_pass_ratio": ["속력", "가속력"],
        "driven_ground_pass_ratio": ["볼 컨트롤"],
    },
    "WIDE_FORWARD": {
        "short_pass_ratio": ["짧은 패스", "볼 컨트롤"],
        "long_pass_ratio": ["긴 패스", "크로스"],
        "through_pass_ratio": ["시야"],
        "driven_ground_pass_ratio": ["크로스", "볼 컨트롤"],
    },
    "STRIKER": {
        # 패스를 보내는 게 아니라 받는 입장 — 수신 관련 스탯으로 매핑.
        "short_pass_ratio": ["볼 컨트롤"],
        "long_pass_ratio": ["볼 컨트롤", "몸싸움"],
        "through_pass_ratio": ["속력", "가속력"],
        "driven_ground_pass_ratio": ["볼 컨트롤", "반응 속도"],
    },
}

# --- 슛 축 스탯 매핑 (DM/CM/WIDE_MID는 shoot_fit 자체를 안 씀) ---
# out_penalty_shoot_ratio(중거리슛 비중)는 포지션 성격과 무관하게 "중거리 슛" 기술
# 자체가 핵심이라 4개 그룹 모두 동일하게 매핑한다 (다른 축처럼 역할별 차별화 안 함).
_OUT_PENALTY_STATS = ["중거리 슛", "슛 파워"]

SHOOT_STAT_MAP = {
    "CAM": {
        "in_penalty_shoot_ratio": ["골 결정력", "위치 선정"],
        "out_penalty_shoot_ratio": _OUT_PENALTY_STATS,
        "heading_shoot_ratio": ["헤더"],
    },
    "DEEP_FORWARD": {
        "in_penalty_shoot_ratio": ["골 결정력", "위치 선정", "발리슛"],
        "out_penalty_shoot_ratio": _OUT_PENALTY_STATS,
        "heading_shoot_ratio": ["헤더"],
    },
    "WIDE_FORWARD": {
        "in_penalty_shoot_ratio": ["골 결정력", "위치 선정", "속력"],
        "out_penalty_shoot_ratio": _OUT_PENALTY_STATS,
        # 본인이 헤딩하는 게 아니라 크로스로 헤딩 기회를 만들어주는 역할.
        "heading_shoot_ratio": ["크로스", "속력"],
    },
    "STRIKER": {
        "in_penalty_shoot_ratio": ["골 결정력", "위치 선정", "반응 속도"],
        "out_penalty_shoot_ratio": _OUT_PENALTY_STATS,
        "heading_shoot_ratio": ["헤더", "점프", "몸싸움"],
    },
}


def load_player_stats(path=PLAYER_STATS_FILE):
    """player_stats_final.csv를 로드해 spid를 인덱스로 한 DataFrame을 반환한다."""
    df = pd.read_csv(path)
    return df.set_index(PLAYER_ID_COL)


def _cosine_similarity(user_vector, card_vector):
    user = np.asarray(user_vector, dtype=float)
    card = np.asarray(card_vector, dtype=float)
    denom = np.linalg.norm(user) * np.linalg.norm(card)
    if denom == 0:
        return np.nan
    return float(np.dot(user, card) / denom)


def _invert_axis_stat_map(axis_stat_map):
    """{axis: [stat, ...]} -> {stat: [axis, ...]}로 뒤집는다.

    한 스탯이 여러 축에 걸쳐 있으면(예: 볼 컨트롤이 short_pass_ratio와
    driven_ground_pass_ratio 둘 다에 관련 있음), 그 스탯은 여러 축의 리스트를 갖는다.
    """
    stat_to_axes = {}
    for axis, stats in axis_stat_map.items():
        for stat in stats:
            stat_to_axes.setdefault(stat, []).append(axis)
    return stat_to_axes


def _fit_score(user_style, card_stats, axis_stat_map):
    """관련 스탯을 축 단위로 평균 내서 뭉개지 않고, 스탯 하나하나를 벡터의 개별
    차원으로 펼쳐서 코사인 유사도를 계산한다.

    스탯을 축별로 평균 내던 예전 방식(_card_axis_value)은, 두 축이 완전히 같은
    스탯 조합을 쓰면(예: short_pass_ratio와 driven_ground_pass_ratio가 둘 다
    [짧은 패스, 볼 컨트롤]) 두 축의 카드 값이 항상 똑같아져서 유저가 어느 쪽을
    선호하든 카드 점수를 구분 못 하는 문제가 있었다.

    이 방식은 스탯별로 "그 스탯이 관련 있는 유저 축들의 비율값을 더한 것"을
    목표값으로 삼는다 — 한 스탯이 여러 축에 관련 있으면 그만큼 목표값이 커지고,
    임의의 숫자 가중치를 새로 만드는 게 아니라 유저의 실제 비율값을 그대로 더하는
    것뿐이다. 다만 두 축의 관련 스탯 목록이 100% 동일하면 이 방식으로도 두 축을
    구분 못 하므로(목표값이 항상 같아짐), 그 경우엔 스탯 목록 자체를 다르게
    잡아야 한다(PASS_STAT_MAP/SHOOT_STAT_MAP 주석 참고).
    """
    stat_to_axes = _invert_axis_stat_map(axis_stat_map)
    if not stat_to_axes:
        return np.nan

    stats = sorted(stat_to_axes)  # 순서 고정(유저 벡터·카드 벡터가 같은 순서여야 함)
    target_vector = [
        sum(user_style[axis] for axis in stat_to_axes[stat]) for stat in stats
    ]
    card_vector = [card_stats[stat] for stat in stats]
    return _cosine_similarity(target_vector, card_vector)


def compute_pass_fit_score(user_style, card_stats, sp_position):
    """유저의 패스 스타일 비율과 카드의 패스 관련 스탯 사이의 코사인 유사도.

    user_style: short_pass_ratio/long_pass_ratio/through_pass_ratio/
        driven_ground_pass_ratio 키를 가진 dict-like (예: user_style_profile.csv 한 행).
    card_stats: player_stats_final.csv 컬럼명을 키로 하는 dict-like (카드 한 장).
    sp_position: api-constraints.md 기준 spPosition 코드.

    해당 포지션 그룹이 pass_fit을 쓰지 않으면(GK/수비/스트라이커 외 매핑 없음) NaN.
    """
    group = group_of(sp_position)
    if group is None or group not in PASS_STAT_MAP:
        return np.nan
    return _fit_score(user_style, card_stats, PASS_STAT_MAP[group])


def compute_shoot_fit_score(user_style, card_stats, sp_position):
    """유저의 슛 스타일 비율과 카드의 슛 관련 스탯 사이의 코사인 유사도.

    user_style: in_penalty_shoot_ratio/out_penalty_shoot_ratio/heading_shoot_ratio
        키를 가진 dict-like (SHOOT_FIT_AXES 참고 — K-means 군집화용 2축이 아니라
        여기서만 쓰는 3축).
    해당 포지션 그룹이 shoot_fit을 쓰지 않으면(DM/CM/WIDE_MID/GK/수비) NaN.
    """
    group = group_of(sp_position)
    if group is None or group not in SHOOT_STAT_MAP:
        return np.nan
    return _fit_score(user_style, card_stats, SHOOT_STAT_MAP[group])


def _weighted_combine(pass_fit, shoot_fit, pass_weight, shoot_weight):
    """pass_fit/shoot_fit을 역할별 가중치로 합친다.

    카드 스탯이 전부 0이라 코사인 유사도 분모가 0이 되는 등, 드물게 둘 중 하나가
    NaN으로 나오는 경우에는 가중치를 무시하고 남은 값을 그대로 쓴다(전부 NaN이면
    NaN 유지) — 예전 np.nanmean 방식의 안전장치를 그대로 가져온 것.
    """
    if np.isnan(pass_fit) and np.isnan(shoot_fit):
        return np.nan
    if np.isnan(pass_fit):
        return shoot_fit
    if np.isnan(shoot_fit):
        return pass_fit
    return pass_weight * pass_fit + shoot_weight * shoot_fit


def compute_position_fit_score(user_style, card_stats, sp_position):
    """포지션 성격에 따라 pass_fit_score/shoot_fit_score를 계산하고 결합한다.

    - DM/CM/WIDE_MID: pass_fit_score 그대로
    - CAM/DEEP_FORWARD/WIDE_FORWARD/STRIKER: BOTH_GROUP_WEIGHTS의 역할별 가중치로
      pass_fit_score와 shoot_fit_score를 합친다 (더 이상 전부 50:50 평균 아님)
    - GK/수비/SUB/알 수 없는 코드: NaN (궁합 점수 대상 아님)
    """
    group = group_of(sp_position)
    if group is None:
        return np.nan

    if group in PASS_ONLY_GROUPS:
        return compute_pass_fit_score(user_style, card_stats, sp_position)

    if group in BOTH_GROUPS:
        pass_fit = compute_pass_fit_score(user_style, card_stats, sp_position)
        shoot_fit = compute_shoot_fit_score(user_style, card_stats, sp_position)
        pass_weight, shoot_weight = BOTH_GROUP_WEIGHTS[group]
        return _weighted_combine(pass_fit, shoot_fit, pass_weight, shoot_weight)

    return np.nan
