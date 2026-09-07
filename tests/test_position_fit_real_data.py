"""
position_fit.py 실제 카드 데이터 face-validity 검증 (pytest 불필요, plain assert).

test_position_fit_smoke.py는 합성 데이터로 "계산 로직 자체가 맞는가"를 검증하고,
여기서는 data/player_stats_final.csv의 실제 극단(스탯이 한쪽으로 치우친) 카드를
찾아 "실제 데이터에서도 의도한 방향대로 점수가 갈리는가"를 검증한다.

카드는 spId를 하드코딩하지 않고 매번 스탯 차이(idxmax/idxmin)로 동적으로 찾는다 —
player_stats_final.csv가 나중에 갱신돼도 테스트가 계속 유효하게 동작하도록.

실행: ./venv/Scripts/python.exe tests/test_position_fit_real_data.py
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "style"))

import position_fit as pf

_STATS = pf.load_player_stats()

# player_stats_final.csv는 GK 카드와 수비수 카드도 섞여 있다. idxmax/idxmin을 전체
# 풀에 그냥 돌리면 "스타일이 치우친 공격형 카드"가 아니라 GK(시야 외 나머지가 전부
# 낮아 상대적으로 튀어 보임)나 수비수(공격 스탯이 원래 낮음)가 뽑혀버려서, 비교
# 자체가 무의미해진다(실제로 처음 이 테스트를 만들 때 그렇게 걸렸다 — 아래 두 필터로
# GK와 수비형 카드를 먼저 제외한 풀에서만 극단값을 찾는다).
_GK_COLS = ["GK 다이빙", "GK 핸들링", "GK 킥", "GK 반응속도", "GK 위치 선정"]
_is_gk = _STATS[_GK_COLS].mean(axis=1) > 30
_is_defender_like = _STATS["대인 수비"] > 60
_OUTFIELD_ATTACKERS = _STATS[~_is_gk & ~_is_defender_like]

THROUGH_STYLE = {"short_pass_ratio": 0.1, "long_pass_ratio": 0.1,
                  "through_pass_ratio": 0.7, "driven_ground_pass_ratio": 0.1}
SHORT_STYLE = {"short_pass_ratio": 0.7, "long_pass_ratio": 0.1,
               "through_pass_ratio": 0.1, "driven_ground_pass_ratio": 0.1}
HEADING_STYLE = {"in_penalty_shoot_ratio": 0.2, "out_penalty_shoot_ratio": 0.1,
                  "heading_shoot_ratio": 0.7}
IN_PENALTY_STYLE = {"in_penalty_shoot_ratio": 0.7, "out_penalty_shoot_ratio": 0.1,
                     "heading_shoot_ratio": 0.2}


def test_vision_vs_technician_card_for_pass_styles():
    """시야가 짧은패스보다 압도적으로 높은 카드(비전형) vs 그 반대(테크니션형)를
    CM 포지션에 놓고, 스루패스형/숏패스형 유저에게 각각 예상대로 갈리는지 확인한다."""
    diff = _OUTFIELD_ATTACKERS["시야"] - _OUTFIELD_ATTACKERS["짧은 패스"]
    vision_card = _STATS.loc[diff.idxmax()]
    technician_card = _STATS.loc[diff.idxmin()]

    v_through = pf.compute_pass_fit_score(THROUGH_STYLE, vision_card, 14)  # CM
    t_through = pf.compute_pass_fit_score(THROUGH_STYLE, technician_card, 14)
    assert v_through > t_through, (
        "스루패스형 유저는 비전형 카드를 더 높게 쳐야 함 "
        f"(vision={v_through:.4f}, technician={t_through:.4f})"
    )

    v_short = pf.compute_pass_fit_score(SHORT_STYLE, vision_card, 14)
    t_short = pf.compute_pass_fit_score(SHORT_STYLE, technician_card, 14)
    assert t_short > v_short, (
        "숏패스형 유저는 테크니션형 카드를 더 높게 쳐야 함 "
        f"(vision={v_short:.4f}, technician={t_short:.4f})"
    )
    print("OK: test_vision_vs_technician_card_for_pass_styles")


def test_header_vs_poacher_card_for_shoot_styles():
    """헤더가 골결정력보다 압도적으로 높은 카드(헤더형) vs 그 반대(포처형)를
    ST 포지션에 놓고, 헤딩형/박스안형 유저에게 각각 예상대로 갈리는지 확인한다."""
    diff = _OUTFIELD_ATTACKERS["헤더"] - _OUTFIELD_ATTACKERS["골 결정력"]
    header_card = _STATS.loc[diff.idxmax()]
    poacher_card = _STATS.loc[diff.idxmin()]

    h_heading = pf.compute_shoot_fit_score(HEADING_STYLE, header_card, 25)  # ST
    p_heading = pf.compute_shoot_fit_score(HEADING_STYLE, poacher_card, 25)
    assert h_heading > p_heading, (
        "헤딩형 유저는 헤더형 카드를 더 높게 쳐야 함 "
        f"(header={h_heading:.4f}, poacher={p_heading:.4f})"
    )

    h_inpen = pf.compute_shoot_fit_score(IN_PENALTY_STYLE, header_card, 25)
    p_inpen = pf.compute_shoot_fit_score(IN_PENALTY_STYLE, poacher_card, 25)
    assert p_inpen > h_inpen, (
        "박스안형 유저는 포처형 카드를 더 높게 쳐야 함 "
        f"(header={h_inpen:.4f}, poacher={p_inpen:.4f})"
    )
    print("OK: test_header_vs_poacher_card_for_shoot_styles")


if __name__ == "__main__":
    test_vision_vs_technician_card_for_pass_styles()
    test_header_vs_poacher_card_for_shoot_styles()
    print("\n모든 position_fit 실데이터 검증 테스트 통과")
