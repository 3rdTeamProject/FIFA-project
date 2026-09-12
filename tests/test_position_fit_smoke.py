"""
position_fit.py 스모크 테스트 (pytest 불필요, plain assert).

실제 카드/유저 데이터 없이도 계산 로직 자체가 맞는지(포지션 분류, pass_fit/shoot_fit
독립 계산, 결합 방식, 스타일이 뚜렷이 다른 카드끼리 점수가 실제로 갈리는지)를
합성 데이터로 검증한다.

실행: ./venv/Scripts/python.exe tests/test_position_fit_smoke.py
"""

import math
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "style"))

import position_fit as pf


def _baseline_card(**overrides):
    """position_fit.py의 모든 매핑에 쓰이는 카드 스탯을 중간값(50)으로 채우고,
    overrides로 특정 스탯만 바꾼다."""
    stats = {
        "짧은 패스": 50, "침착성": 50, "긴 패스": 50, "시야": 50, "커브": 50,
        "볼 컨트롤": 50, "크로스": 50, "드리블": 50, "속력": 50, "가속력": 50,
        "몸싸움": 50, "골 결정력": 50, "위치 선정": 50, "발리슛": 50,
        "반응 속도": 50, "헤더": 50, "점프": 50, "중거리 슛": 50, "슛 파워": 50,
    }
    stats.update(overrides)
    return stats


def _uniform_style():
    """6개 축이 전부 동일한(0.5) 유저 스타일 — 방향성이 없는 기준점."""
    return {
        "short_pass_ratio": 0.5, "long_pass_ratio": 0.5,
        "through_pass_ratio": 0.5, "driven_ground_pass_ratio": 0.5,
        "in_penalty_shoot_ratio": 0.5, "out_penalty_shoot_ratio": 0.5,
        "heading_shoot_ratio": 0.5,
    }


def test_group_of_classifies_mirror_positions_identically():
    assert pf.group_of(9) == pf.group_of(10) == pf.group_of(11) == "DM"
    assert pf.group_of(13) == pf.group_of(14) == pf.group_of(15) == "CM"
    assert pf.group_of(12) == pf.group_of(16) == "WIDE_MID"
    assert pf.group_of(17) == pf.group_of(18) == pf.group_of(19) == "CAM"
    assert pf.group_of(20) == pf.group_of(21) == pf.group_of(22) == "DEEP_FORWARD"
    assert pf.group_of(23) == pf.group_of(27) == "WIDE_FORWARD"
    assert pf.group_of(24) == pf.group_of(25) == pf.group_of(26) == "STRIKER"
    print("OK: test_group_of_classifies_mirror_positions_identically")


def test_gk_defense_sub_excluded():
    excluded = [0, 1, 2, 3, 4, 5, 6, 7, 8, 28]
    for sp in excluded:
        assert pf.group_of(sp) is None, f"spPosition {sp}는 궁합 점수 대상이 아니어야 함"
        assert math.isnan(pf.compute_pass_fit_score(_uniform_style(), _baseline_card(), sp))
        assert math.isnan(pf.compute_shoot_fit_score(_uniform_style(), _baseline_card(), sp))
        assert math.isnan(pf.compute_position_fit_score(_uniform_style(), _baseline_card(), sp))
    print("OK: test_gk_defense_sub_excluded")


def test_dm_cm_wide_mid_only_reflect_out_penalty_shoot():
    """DM/CM/WIDE_MID는 2026-09-08부터 shoot_fit을 쓰지만, out_penalty_shoot_ratio
    (중거리슛) 축만 있다 — 박스 안 마무리(in_penalty)/헤더는 매핑 자체가 없어서
    극단적으로 바꿔도 shoot_fit_score에 영향이 없어야 하고, 중거리슛/슛파워를
    바꾸면 영향이 있어야 한다."""
    style = _uniform_style()
    baseline_card = _baseline_card()
    # 헤더/점프(heading 관련)와 골결정력/위치선정(in_penalty 관련)을 극단적으로
    # 바꿔도 이 세 그룹은 매핑이 없어서 shoot_fit_score가 그대로여야 함.
    irrelevant_card = _baseline_card(**{"헤더": 99, "점프": 99, "골 결정력": 99, "위치 선정": 99})
    for sp in [10, 14, 12]:  # CDM, CM, RM
        base_shoot = pf.compute_shoot_fit_score(style, baseline_card, sp)
        irrelevant_shoot = pf.compute_shoot_fit_score(style, irrelevant_card, sp)
        assert not math.isnan(base_shoot), f"spPosition={sp}: shoot_fit이 이제 NaN이면 안 됨"
        assert abs(base_shoot - irrelevant_shoot) < 1e-9, (
            f"spPosition={sp}: 헤더/골결정력 등은 이 그룹의 shoot_fit에 영향 없어야 함"
        )

        long_range_card = _baseline_card(**{"중거리 슛": 99, "슛 파워": 99})
        long_range_shoot = pf.compute_shoot_fit_score(style, long_range_card, sp)
        assert long_range_shoot > base_shoot, (
            f"spPosition={sp}: 중거리슛/슛파워를 올리면 shoot_fit_score도 올라가야 함"
        )

        pass_fit = pf.compute_pass_fit_score(style, baseline_card, sp)
        position_fit = pf.compute_position_fit_score(style, baseline_card, sp)
        pass_weight, shoot_weight = pf.BOTH_GROUP_WEIGHTS[pf.group_of(sp)]
        expected = pass_weight * pass_fit + shoot_weight * base_shoot
        assert abs(position_fit - expected) < 1e-9, (
            f"spPosition={sp}: position_fit_score가 BOTH_GROUP_WEIGHTS 가중결합과 안 맞음"
        )
    print("OK: test_dm_cm_wide_mid_only_reflect_out_penalty_shoot")


def test_both_groups_combine_pass_and_shoot_with_role_weights():
    """BOTH_GROUPS는 더 이상 전부 50:50 평균이 아니라 BOTH_GROUP_WEIGHTS의 역할별
    가중치로 합쳐진다(2026-09-07 변경 — CAM이 순수 스트라이커와 같은 비중으로 슛
    궁합을 받는 게 이상하다는 지적으로 도입됨)."""
    style = _uniform_style()
    card = _baseline_card()
    sp_by_group = {
        "DM": 10, "CM": 14, "WIDE_MID": 12,
        "CAM": 18, "DEEP_FORWARD": 21, "WIDE_FORWARD": 23, "STRIKER": 25,
    }
    for group, sp in sp_by_group.items():
        pass_fit = pf.compute_pass_fit_score(style, card, sp)
        shoot_fit = pf.compute_shoot_fit_score(style, card, sp)
        position_fit = pf.compute_position_fit_score(style, card, sp)
        assert not math.isnan(pass_fit) and not math.isnan(shoot_fit)
        pass_weight, shoot_weight = pf.BOTH_GROUP_WEIGHTS[group]
        expected = pass_weight * pass_fit + shoot_weight * shoot_fit
        assert abs(position_fit - expected) < 1e-9, (
            f"{group}: 가중치({pass_weight}/{shoot_weight})로 계산한 기대값과 다름 "
            f"(position_fit={position_fit:.4f}, expected={expected:.4f})"
        )
    print("OK: test_both_groups_combine_pass_and_shoot_with_role_weights")


def test_through_pass_style_prefers_vision_card_for_cm():
    """스루패스 위주 유저는 시야 좋은 카드를 짧은패스 위주 카드보다 높게 쳐야 한다.

    through_pass_ratio는 2026-09-07 재검토로 시야 단일 스탯만 쓴다(커브/침착성은
    특정 축 고유의 이유가 없는 범용 스탯이라 제외됨 — position_fit.py 참고)."""
    style = dict(_uniform_style())
    style.update({
        "short_pass_ratio": 0.1, "long_pass_ratio": 0.1,
        "through_pass_ratio": 0.7, "driven_ground_pass_ratio": 0.1,
    })

    vision_card = _baseline_card(시야=95)
    vision_card["짧은 패스"] = 30
    short_pass_card = _baseline_card(시야=20)
    short_pass_card["짧은 패스"] = 95

    score_vision = pf.compute_pass_fit_score(style, vision_card, 14)   # CM
    score_short = pf.compute_pass_fit_score(style, short_pass_card, 14)
    assert score_vision > score_short, (
        f"스루패스형 유저는 시야 좋은 카드를 더 높게 쳐야 함 "
        f"(vision={score_vision:.4f}, short={score_short:.4f})"
    )
    print("OK: test_through_pass_style_prefers_vision_card_for_cm")


def test_striker_pass_fit_uses_receiving_stats():
    """스트라이커는 '보내는' 스탯이 아니라 '받는' 스탯(속력/가속력 등)으로 pass_fit을 잰다."""
    style = dict(_uniform_style())
    style.update({
        "short_pass_ratio": 0.1, "long_pass_ratio": 0.1,
        "through_pass_ratio": 0.7, "driven_ground_pass_ratio": 0.1,
    })

    fast_card = _baseline_card(속력=95, 가속력=95)
    slow_card = _baseline_card(속력=20, 가속력=20)

    score_fast = pf.compute_pass_fit_score(style, fast_card, 25)  # ST
    score_slow = pf.compute_pass_fit_score(style, slow_card, 25)
    assert score_fast > score_slow, (
        "스루패스형 스타일에서는 빠른 스트라이커가 더 높은 pass_fit을 받아야 함 "
        f"(fast={score_fast:.4f}, slow={score_slow:.4f})"
    )
    print("OK: test_striker_pass_fit_uses_receiving_stats")


def test_heading_style_prefers_target_man_for_striker_but_crosser_for_winger():
    style = dict(_uniform_style())
    style.update({"in_penalty_shoot_ratio": 0.3, "heading_shoot_ratio": 0.9})

    target_man = _baseline_card(헤더=95, 점프=95, 몸싸움=95)
    weak_air = _baseline_card(헤더=20, 점프=20, 몸싸움=20)
    assert pf.compute_shoot_fit_score(style, target_man, 25) > pf.compute_shoot_fit_score(style, weak_air, 25)

    # RW/LW는 헤더 스탯 자체가 아니라 크로스+속력으로 heading_shoot_ratio를 잰다.
    good_crosser = _baseline_card(크로스=95, 속력=95, 헤더=20)  # 헤더는 낮아도 됨
    poor_crosser = _baseline_card(크로스=20, 속력=20, 헤더=95)  # 헤더가 높아도 무관해야 함
    score_good = pf.compute_shoot_fit_score(style, good_crosser, 23)  # RW
    score_poor = pf.compute_shoot_fit_score(style, poor_crosser, 23)
    assert score_good > score_poor, (
        "RW/LW는 헤더 스탯이 아니라 크로스+속력으로 헤딩 스타일 궁합을 재야 함"
    )
    print("OK: test_heading_style_prefers_target_man_for_striker_but_crosser_for_winger")


def test_no_two_axes_share_identical_stat_set():
    """같은 포지션 안에서 두 축의 관련 스탯 목록이 100% 동일하면, 그 두 축의 카드
    점수가 항상 똑같아져서 유저가 어느 쪽을 선호하든 구분을 못 하게 된다(2026-09-07
    발견한 버그 — CM 등에서 short_pass_ratio와 driven_ground_pass_ratio가 둘 다
    [짧은 패스, 볼 컨트롤]이었음). 회귀 방지용 테스트."""
    for group, axis_map in pf.PASS_STAT_MAP.items():
        stat_sets = {axis: frozenset(stats) for axis, stats in axis_map.items()}
        axes = list(stat_sets)
        for i in range(len(axes)):
            for j in range(i + 1, len(axes)):
                assert stat_sets[axes[i]] != stat_sets[axes[j]], (
                    f"{group}: {axes[i]}와 {axes[j]}의 관련 스탯이 완전히 동일함 "
                    f"({stat_sets[axes[i]]}) — 두 축을 구분 못 함"
                )
    for group, axis_map in pf.SHOOT_STAT_MAP.items():
        stat_sets = {axis: frozenset(stats) for axis, stats in axis_map.items()}
        axes = list(stat_sets)
        for i in range(len(axes)):
            for j in range(i + 1, len(axes)):
                assert stat_sets[axes[i]] != stat_sets[axes[j]], (
                    f"{group}: {axes[i]}와 {axes[j]}의 관련 스탯이 완전히 동일함 "
                    f"({stat_sets[axes[i]]}) — 두 축을 구분 못 함"
                )
    print("OK: test_no_two_axes_share_identical_stat_set")


def test_identical_stat_pair_now_differentiates_cards():
    """2026-09-07 버그 재현 + 수정 확인: driven_ground_pass_ratio 위주 유저에게
    짧은패스만 좋은 카드보다 볼컨트롤만 좋은 카드가 더 높은 점수를 받아야 한다."""
    style = dict(_uniform_style())
    style.update({
        "short_pass_ratio": 0.05, "long_pass_ratio": 0.1,
        "through_pass_ratio": 0.15, "driven_ground_pass_ratio": 0.7,
    })

    technique_card = _baseline_card(**{"짧은 패스": 95, "볼 컨트롤": 40})
    touch_card = _baseline_card(**{"짧은 패스": 40, "볼 컨트롤": 95})

    for sp in [14, 18, 21]:  # CM, CAM, CF
        score_technique = pf.compute_pass_fit_score(style, technique_card, sp)
        score_touch = pf.compute_pass_fit_score(style, touch_card, sp)
        assert score_touch > score_technique, (
            f"spPosition={sp}: 드리븐형 유저는 볼컨트롤 좋은 카드를 더 높게 쳐야 함 "
            f"(technique={score_technique:.4f}, touch={score_touch:.4f})"
        )
    print("OK: test_identical_stat_pair_now_differentiates_cards")


def test_out_penalty_style_prefers_long_range_shooter():
    """중거리슛 위주 유저(out_penalty_shoot_ratio 높음)는 중거리슛/슛파워 좋은 카드를
    선호해야 한다 — 박스 안 마무리력(골결정력/위치선정)만 좋은 카드보다."""
    style = dict(_uniform_style())
    style.update({
        "in_penalty_shoot_ratio": 0.2, "out_penalty_shoot_ratio": 0.7,
        "heading_shoot_ratio": 0.1,
    })

    long_range_shooter = _baseline_card(**{"중거리 슛": 95, "슛 파워": 95})
    box_poacher = _baseline_card(**{"골 결정력": 95, "위치 선정": 95, "중거리 슛": 20, "슛 파워": 20})

    score_long_range = pf.compute_shoot_fit_score(style, long_range_shooter, 25)  # ST
    score_poacher = pf.compute_shoot_fit_score(style, box_poacher, 25)
    assert score_long_range > score_poacher, (
        "중거리슛형 유저는 중거리슛/슛파워 좋은 카드를 더 높게 쳐야 함 "
        f"(long_range={score_long_range:.4f}, poacher={score_poacher:.4f})"
    )
    print("OK: test_out_penalty_style_prefers_long_range_shooter")


if __name__ == "__main__":
    test_group_of_classifies_mirror_positions_identically()
    test_gk_defense_sub_excluded()
    test_dm_cm_wide_mid_only_reflect_out_penalty_shoot()
    test_both_groups_combine_pass_and_shoot_with_role_weights()
    test_through_pass_style_prefers_vision_card_for_cm()
    test_striker_pass_fit_uses_receiving_stats()
    test_heading_style_prefers_target_man_for_striker_but_crosser_for_winger()
    test_no_two_axes_share_identical_stat_set()
    test_identical_stat_pair_now_differentiates_cards()
    test_out_penalty_style_prefers_long_range_shooter()
    print("\n모든 position_fit 스모크 테스트 통과")
