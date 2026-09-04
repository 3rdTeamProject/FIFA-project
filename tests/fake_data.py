"""
win_prediction_baseline.py 단위 테스트용 합성(fake) 데이터 생성기.

2026-09-04에 실제 Nexon API(match-detail)를 1회 호출해 확인한 스키마를 그대로 본떠
matchInfo/matchDetail/player 구조를 만든다. 정상 케이스뿐 아니라 몰수경기, 무승부,
SUB(spPosition==28) 포함, ouid 중복 등장 케이스도 함께 만들어 필터링 로직을 검증한다.
"""

import random

SUB_POSITION_CODE = 28


def _make_player(sp_id, sp_position, sp_grade=9):
    return {
        "spId": sp_id,
        "spPosition": sp_position,
        "spGrade": sp_grade,
        "status": {"shoot": 0, "goal": 0, "spRating": 0.0},
    }


def _make_squad(base_sp_id, starting_positions):
    """starting_positions에 있는 포지션 코드로 11명 + SUB 몇 명을 만든다."""
    players = [
        _make_player(base_sp_id + i, pos) for i, pos in enumerate(starting_positions)
    ]
    # 교체선수(SUB) 몇 명 추가 - avg_stat_score 계산에서 제외돼야 한다.
    for i in range(3):
        players.append(_make_player(base_sp_id + 100 + i, SUB_POSITION_CODE))
    return players

# 11자리 포지션 코드(대충 4-4-2 형태, GK 포함) - 실제 포메이션 조합표는 아직 없어 테스트용 값
STARTING_POSITIONS = [0, 3, 4, 6, 7, 9, 12, 14, 16, 25, 26]


def _make_match_info(ouid, nickname, division, result, base_sp_id, match_end_type=0):
    return {
        "ouid": ouid,
        "nickname": nickname,
        "division": division,
        "matchDetail": {
            "seasonId": 202604,
            "matchResult": result,
            "matchEndType": match_end_type,
            "systemPause": 0,
            "foul": 0,
            "injury": 0,
            "redCards": 0,
            "yellowCards": 0,
            "dribble": 50,
            "cornerKick": 1,
            "possession": 50,
            "offsideCount": 0,
            "averageRating": 6.5,
            "controller": "keyboard",
        },
        "shoot": {"shootTotal": 5, "goalTotal": 1},
        "shootDetail": [],
        "pass": {"passTry": 100, "passSuccess": 80},
        "defence": {"tackleTry": 10, "tackleSuccess": 5},
        "player": _make_squad(base_sp_id, STARTING_POSITIONS),
    }


def make_normal_match(match_id, ouid_a, ouid_b, division_a=2300, division_b=2400,
                       base_sp_id_a=100000, base_sp_id_b=200000):
    """정상 종료, 승/패가 명확히 갈리는 매치 1건 (matchInfo 2건)."""
    return {
        "matchId": match_id,
        "matchDate": "2026-09-03T12:00:00",
        "matchType": 50,
        "matchInfo": [
            _make_match_info(ouid_a, f"user_{ouid_a}", division_a, "승", base_sp_id_a),
            _make_match_info(ouid_b, f"user_{ouid_b}", division_b, "패", base_sp_id_b),
        ],
    }


def make_draw_match(match_id, ouid_a, ouid_b):
    """무승부 매치 - extract_rows에서 제외되어야 한다."""
    return {
        "matchId": match_id,
        "matchDate": "2026-09-03T12:00:00",
        "matchType": 50,
        "matchInfo": [
            _make_match_info(ouid_a, f"user_{ouid_a}", 2300, "무", 300000),
            _make_match_info(ouid_b, f"user_{ouid_b}", 2300, "무", 400000),
        ],
    }


def make_forfeit_match(match_id, ouid_a, ouid_b):
    """몰수경기(matchEndType != 0) - extract_rows에서 제외되어야 한다."""
    return {
        "matchId": match_id,
        "matchDate": "2026-09-03T12:00:00",
        "matchType": 50,
        "matchInfo": [
            _make_match_info(ouid_a, f"user_{ouid_a}", 2300, "승", 500000, match_end_type=3),
            _make_match_info(ouid_b, f"user_{ouid_b}", 2300, "패", 600000, match_end_type=3),
        ],
    }


def make_fake_matches(n_normal=40, seed=7):
    """정상 매치 n_normal건 + 무승부 1건 + 몰수경기 1건 + ouid 중복 매치 1건을 만든다.

    승/패는 avg_stat_score와 상관관계를 갖도록 설계한다: base_sp_id가 클수록(스탯이 좋은
    선수 카드와 매칭되도록 player_1000_final.csv를 구성) 이기는 쪽으로 편향을 준다.
    """
    rng = random.Random(seed)
    matches = []

    for i in range(n_normal):
        ouid_a = f"ouid_a_{i}"
        ouid_b = f"ouid_b_{i}"
        # 절반은 A가 이기고(스탯 좋은 쪽), 절반은 B가 이기도록 구성해 학습이 가능하게 한다.
        a_wins = i % 2 == 0
        base_a = 700000 if a_wins else 100000
        base_b = 100000 if a_wins else 700000
        result_a = "승" if a_wins else "패"
        result_b = "패" if a_wins else "승"
        matches.append({
            "matchId": f"match_{i}",
            "matchDate": "2026-09-03T12:00:00",
            "matchType": 50,
            "matchInfo": [
                _make_match_info(ouid_a, f"user_{ouid_a}", rng.choice([2200, 2300, 2400]),
                                  result_a, base_a),
                _make_match_info(ouid_b, f"user_{ouid_b}", rng.choice([2200, 2300, 2400]),
                                  result_b, base_b),
            ],
        })

    matches.append(make_draw_match("match_draw", "ouid_draw_a", "ouid_draw_b"))
    matches.append(make_forfeit_match("match_forfeit", "ouid_forfeit_a", "ouid_forfeit_b"))

    # 같은 ouid(ouid_a_0)가 다른 match_id에도 등장하는 케이스 - 중복 제거 로직 검증용
    dup_match = make_normal_match("match_dup", "ouid_a_0", "ouid_c_dup")
    matches.append(dup_match)

    return matches


def make_fake_player_card_rows():
    """base_sp_id(100000/700000)에서 파생된 spId들을 커버하는 소규모 player 카드 데이터.

    spId가 클수록(700000대) 스탯이 좋게, 작을수록(100000대) 스탯이 낮게 설정해
    avg_stat_score와 승패 사이에 학습 가능한 상관관계를 만든다.
    """
    rows = []
    for base, quality in [(100000, "low"), (700000, "high"), (200000, "low"),
                           (300000, "low"), (400000, "low"), (500000, "low"),
                           (600000, "low")]:
        stat = 60 if quality == "low" else 90
        for offset in list(range(11)) + [100, 101, 102]:
            sp_id = base + offset
            rows.append({
                "spid": sp_id,
                "stat_short_pass": stat,
                "stat_long_pass": stat,
                "stat_dribble": stat,
                "stat_ball_control": stat,
            })
    return rows
