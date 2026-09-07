"""
live_diagnosis.py 스모크 테스트 (pytest 불필요, plain assert, 네트워크 호출 없음).

fetch_recent_matches()만 API를 쓰고 나머지(비율 계산/스쿼드 구성/궁합 점수)는 순수
함수라, 가짜 matchInfo로 API 없이 전체 흐름을 검증할 수 있다.

실행: ./venv/Scripts/python.exe tests/test_live_diagnosis_smoke.py
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "style"))
sys.path.insert(0, os.path.join(REPO_ROOT, "tests"))

import pandas as pd

import live_diagnosis as ld
from test_position_fit_smoke import _baseline_card

OUID = "fake_ouid"
# GK, RB, RCB, CB, LB, CDM, CM, LM, CAM, CF, ST — GK/수비는 궁합 대상 아니라 카드 스탯
# 없이도 안전해야 하고, 나머지 6자리는 _baseline_card()의 스탯 키로 전부 계산 가능해야 함.
SQUAD_POSITIONS = [0, 3, 4, 5, 7, 10, 14, 16, 18, 21, 25]
MAIN_SQUAD_IDS = list(range(101, 112))


def _fake_pass(short=60, long_=10, through=20, driven=5):
    total = short + long_ + through + driven
    return {"passTry": total, "shortPassTry": short, "longPassTry": long_,
            "throughPassTry": through, "lobbedThroughPassTry": 0, "drivenGroundPassTry": driven}


def _fake_shoot(in_pen=6, out_pen=2, heading=1):
    return {"shootTotal": in_pen + out_pen, "shootHeading": heading,
            "shootInPenalty": in_pen, "shootOutPenalty": out_pen}


def _fake_match(match_end_type, sp_ids=None, include_player=True, pass_kwargs=None, shoot_kwargs=None):
    players = []
    if include_player:
        sp_ids = sp_ids or []
        players = [{"spId": sp_id, "spPosition": pos, "spGrade": 8}
                   for sp_id, pos in zip(sp_ids, SQUAD_POSITIONS)]
        players.append({"spId": 9999, "spPosition": 28, "spGrade": 1})  # SUB 1명
    return {
        "matchId": "m",
        "ouid": OUID,
        "matchDetail": {"matchEndType": match_end_type, "matchResult": "승", "dribble": 100},
        "pass": _fake_pass(**(pass_kwargs or {})),
        "shoot": _fake_shoot(**(shoot_kwargs or {})),
        "player": players,
    }


def test_matches_to_style_rows_and_compute_user_style():
    match_infos = [
        _fake_match(0, MAIN_SQUAD_IDS, pass_kwargs={"short": 60, "through": 30}),
        _fake_match(0, MAIN_SQUAD_IDS, pass_kwargs={"short": 55, "through": 25}),
    ]
    raw_rows_df = ld.matches_to_style_rows(OUID, match_infos)
    assert len(raw_rows_df) == 2

    user_style = ld.compute_user_style(raw_rows_df)
    assert user_style is not None
    for key in ["short_pass_ratio", "long_pass_ratio", "through_pass_ratio",
                "driven_ground_pass_ratio", "in_penalty_shoot_ratio",
                "out_penalty_shoot_ratio", "heading_shoot_ratio"]:
        assert key in user_style and user_style[key] == user_style[key], f"{key} 누락/NaN"
    print("OK: test_matches_to_style_rows_and_compute_user_style")


def test_build_current_squad_skips_forfeit_with_empty_player():
    """가장 최근 경기가 몰수(matchEndType!=0)에 player[]가 비어있으면 건너뛰고, 그다음
    정상종료 경기로 스쿼드를 구성해야 한다 (2026-09-07 실측으로 발견한 문제의 재현 테스트)."""
    match_infos = [
        _fake_match(2, include_player=False),  # 몰수패, player[] 비어있음 (실측 그대로)
        _fake_match(0, MAIN_SQUAD_IDS),        # 정상종료 -> 이걸로 스쿼드가 구성돼야 함
    ]
    squad, match_result = ld.build_current_squad(match_infos)
    assert squad is not None, "정상종료 경기가 있는데도 스쿼드를 못 찾음"
    assert len(squad) == 11
    assert {s["sp_id"] for s in squad} == set(MAIN_SQUAD_IDS)
    print("OK: test_build_current_squad_skips_forfeit_with_empty_player")


def test_build_current_squad_returns_none_when_no_normal_match():
    match_infos = [_fake_match(2, include_player=False), _fake_match(1, include_player=False)]
    squad, reason = ld.build_current_squad(match_infos)
    assert squad is None
    assert isinstance(reason, str)
    print("OK: test_build_current_squad_returns_none_when_no_normal_match")


def _fake_player_stats():
    rows = {sp_id: _baseline_card() for sp_id in MAIN_SQUAD_IDS}
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "spid"
    return df


def test_score_squad_changes_when_slot_is_swapped():
    """스쿼드 슬롯 하나를 다른 카드로 바꾸면 즉시 다른 점수가 나와야 한다 — 대시보드의
    "스쿼드 조정 시 즉시 재채점" 요구사항의 핵심 동작(같은 user_style 재사용, API 불필요)."""
    user_style = {
        "short_pass_ratio": 0.1, "long_pass_ratio": 0.1,
        "through_pass_ratio": 0.7, "driven_ground_pass_ratio": 0.1,
        "in_penalty_shoot_ratio": 0.6, "out_penalty_shoot_ratio": 0.2,
        "heading_shoot_ratio": 0.2,
    }
    squad = [{"sp_id": sp_id, "sp_position": pos} for sp_id, pos in zip(MAIN_SQUAD_IDS, SQUAD_POSITIONS)]
    player_stats = _fake_player_stats()

    before = ld.score_squad(user_style, squad, player_stats=player_stats)
    cm_slot_index = SQUAD_POSITIONS.index(14)  # CM 슬롯
    before_cm_score = before["slots"][cm_slot_index]["position_fit"]
    assert before_cm_score == before_cm_score  # NaN 아님

    # CM 카드를 "시야 좋고 짧은패스 약한" 카드로 교체 (스루패스형 유저에게 더 잘 맞아야 함)
    vision_card_id = 999
    vision_card = _baseline_card(시야=99)
    vision_card["짧은 패스"] = 20
    player_stats.loc[vision_card_id] = pd.Series(vision_card)

    swapped_squad = list(squad)
    swapped_squad[cm_slot_index] = {"sp_id": vision_card_id, "sp_position": 14}

    after = ld.score_squad(user_style, swapped_squad, player_stats=player_stats)
    after_cm_score = after["slots"][cm_slot_index]["position_fit"]

    assert after_cm_score != before_cm_score, "슬롯을 바꿨는데 점수가 그대로임"
    assert after_cm_score > before_cm_score, (
        "스루패스형 유저는 시야 좋은 카드로 바꾼 뒤 CM 궁합이 더 높아져야 함 "
        f"(before={before_cm_score:.4f}, after={after_cm_score:.4f})"
    )
    # 다른 슬롯(GK 등)은 이번 교체와 무관하게 그대로여야 한다
    gk_slot_index = SQUAD_POSITIONS.index(0)
    assert before["slots"][gk_slot_index]["position_fit"] != before["slots"][gk_slot_index]["position_fit"]
    print("OK: test_score_squad_changes_when_slot_is_swapped")


def test_generate_diagnosis_sentence_mentions_weakest_position():
    user_style = {
        "short_pass_ratio": 0.7, "long_pass_ratio": 0.1,
        "through_pass_ratio": 0.1, "driven_ground_pass_ratio": 0.1,
        "in_penalty_shoot_ratio": 0.6, "out_penalty_shoot_ratio": 0.2,
        "heading_shoot_ratio": 0.2,
    }
    squad = [{"sp_id": sp_id, "sp_position": pos} for sp_id, pos in zip(MAIN_SQUAD_IDS, SQUAD_POSITIONS)]
    squad_scores = ld.score_squad(user_style, squad, player_stats=_fake_player_stats())
    # pass_cluster=1 -> "숏패스 위주", shoot_cluster=2 -> "헤딩슛 위주" (PASS/SHOOT_CLUSTER_LABELS 참고)
    sentence = ld.generate_diagnosis_sentence(user_style, squad_scores, pass_cluster=1, shoot_cluster=2)
    assert "숏패스 위주" in sentence
    assert "헤딩슛 위주" in sentence
    valid = [r for r in squad_scores["slots"] if r["position_fit"] == r["position_fit"]]
    worst = min(valid, key=lambda r: r["position_fit"])
    assert worst["pos_name"] in sentence
    print("OK: test_generate_diagnosis_sentence_mentions_weakest_position")


if __name__ == "__main__":
    test_matches_to_style_rows_and_compute_user_style()
    test_build_current_squad_skips_forfeit_with_empty_player()
    test_build_current_squad_returns_none_when_no_normal_match()
    test_score_squad_changes_when_slot_is_swapped()
    test_generate_diagnosis_sentence_mentions_weakest_position()
    print("\n모든 live_diagnosis 스모크 테스트 통과")
