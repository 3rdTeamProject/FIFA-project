"""
preprocess_style.py / train_style.py 단위 테스트용 합성(fake) 데이터 생성기.

2026-09-07 팀 결정으로 스타일 진단 원본이 matches_style.jsonl(원본 API 응답)에서
팀원이 이미 팀 단위로 평탄화해 수집한 data/style/match_team_data.csv로 바뀌었다.
이 CSV를 실제로 확인해 정리한 컬럼 구성(matchId, ouid, matchEndType, passTry,
shortPassTry, longPassTry, throughPassTry, lobbedThroughPassTry, drivenGroundPassTry,
dribble, shootTotal, shootHeading, shootInPenalty)을 그대로 본떠 DataFrame을 만든다.

2026-09-07 추가 결정으로 패스 모델(short/long/through/driven_ground)과 슛 모델
(in_penalty/heading)을 독립적으로 군집화하므로, 프로필도 두 모델을 각각 구분해낼 수
있도록 짰다: 패스 축(short/long/through/driven)과 슛 축(in_pen/heading)이 profile마다
서로 다르게 뚜렷이 갈린다.

서로 다른 "진짜 스타일" 3그룹의 유저를 만들어, 유저별로 같은 그룹 안에서는 비율이
비슷하지만 그룹 사이에는 뚜렷이 다르게 잔차(jitter)를 준다 — K-means가 그룹을 구분해낼
수 있는지 스모크 테스트로 확인하기 위함이다. 몰수/오류 경기 1건과 표본이 모자란
(min_matches 미만) 유저 1명도 섞어 필터링 로직을 검증한다.
"""

import random

import pandas as pd

STYLE_PROFILES = {
    "short_pass_heavy": {"short": 70, "long": 10, "through": 10, "driven": 10, "dribble": 40, "in_pen": 60, "heading": 5},
    "long_pass_heavy": {"short": 20, "long": 55, "through": 10, "driven": 15, "dribble": 30, "in_pen": 50, "heading": 15},
    "through_pass_heavy": {"short": 30, "long": 10, "through": 45, "driven": 15, "dribble": 60, "in_pen": 70, "heading": 5},
}


def _make_team_row(match_id, ouid, profile, rng, match_end_type=0):
    """profile 비율(%) 근처에서 잔차를 줘서 pass/shoot try 카운트를 만든다."""
    pass_try = 100 + rng.randint(-10, 10)
    short_pass_try = int(pass_try * (profile["short"] + rng.uniform(-3, 3)) / 100)
    long_pass_try = int(pass_try * (profile["long"] + rng.uniform(-3, 3)) / 100)
    through_pass_try = int(pass_try * (profile["through"] + rng.uniform(-3, 3)) / 100)
    lobbed_through_pass_try = max(0, int(through_pass_try * 0.1))
    driven_ground_pass_try = int(pass_try * (profile["driven"] + rng.uniform(-3, 3)) / 100)

    shoot_total = max(1, 5 + rng.randint(-2, 2))
    shoot_in_penalty = min(shoot_total, max(0, int(shoot_total * (profile["in_pen"] + rng.uniform(-5, 5)) / 100)))
    shoot_out_penalty = max(0, shoot_total - shoot_in_penalty)
    shoot_heading = min(shoot_total, max(0, int(shoot_total * (profile["heading"] + rng.uniform(-3, 3)) / 100)))

    dribble_yard = max(0, int(profile["dribble"] + rng.uniform(-5, 5)))

    return {
        "matchId": match_id,
        "ouid": ouid,
        "matchEndType": match_end_type,
        "passTry": pass_try,
        "shortPassTry": short_pass_try,
        "longPassTry": long_pass_try,
        "throughPassTry": through_pass_try,
        "lobbedThroughPassTry": lobbed_through_pass_try,
        "drivenGroundPassTry": driven_ground_pass_try,
        "dribble": dribble_yard,
        "shootTotal": shoot_total,
        "shootHeading": shoot_heading,
        "shootInPenalty": shoot_in_penalty,
        "shootOutPenalty": shoot_out_penalty,
    }


def make_fake_style_matches(n_users_per_profile=4, n_matches_per_user=15, seed=13):
    """스타일 그룹별 n_users_per_profile명 x n_matches_per_user경기 + 몰수/오류 경기 1건
    (2행) + 표본 부족(min_matches 미만) 유저 1명(3경기만)을 담은 DataFrame을 만든다."""
    rng = random.Random(seed)
    rows = []
    match_counter = 0

    for profile_name, profile in STYLE_PROFILES.items():
        for u in range(n_users_per_profile):
            ouid = f"{profile_name}_{u}"
            for _ in range(n_matches_per_user):
                match_id = f"match_{match_counter}"
                opp_ouid = f"opponent_{match_counter}"
                rows.append(_make_team_row(match_id, ouid, profile, rng))
                rows.append(_make_team_row(match_id, opp_ouid, profile, rng))
                match_counter += 1

    # 몰수/오류 경기 - extract_match_style_rows에서 제외되어야 한다.
    # matchEndType 1/2(몰수 승/패)를 각각 한쪽씩 섞어 실제 데이터 패턴을 본뜬다.
    forfeit_profile = STYLE_PROFILES["short_pass_heavy"]
    rows.append(_make_team_row("match_forfeit", "forfeit_a", forfeit_profile, rng, match_end_type=2))
    rows.append(_make_team_row("match_forfeit", "forfeit_b", forfeit_profile, rng, match_end_type=1))

    # 표본 부족 유저 - aggregate_user_style의 min_matches 필터에서 제외되어야 한다.
    sparse_profile = STYLE_PROFILES["long_pass_heavy"]
    for i in range(3):
        match_id = f"match_sparse_{i}"
        rows.append(_make_team_row(match_id, "sparse_user", sparse_profile, rng))
        rows.append(_make_team_row(match_id, f"sparse_opp_{i}", sparse_profile, rng))

    return pd.DataFrame(rows)
