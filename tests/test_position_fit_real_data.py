"""
position_fit.py 실제 카드 데이터 face-validity 검증 (pytest 불필요, plain assert).

test_position_fit_smoke.py는 합성 데이터로 "계산 로직 자체가 맞는가"를 검증하고,
여기서는 data/player_stats_final.csv의 실제 카드로 "실제 데이터에서도 의도한
방향대로 점수가 갈리는가"를 검증한다.

2026-09-08 전면 재작성: 원래는 idxmax/idxmin(스탯 차이)만으로 "특화 카드"를 찾았는데,
_fit_score가 코사인 유사도(방향만 봄)에서 "품질(quality) x 스타일_배율" 방식으로
바뀌면서 이 방법이 더 이상 안 통한다 — 스탯 차이가 큰 카드는 종종 "특화 카드"가
아니라 그냥 전체적으로 약한 카드(예: 헤더가 골결정력보다 상대적으로 나은 카드를
찾았더니 그냥 평범한 카드가 뽑히고, 실제로는 전방위로 뛰어난 카드가 헤더 자체도
더 높은 경우가 흔했다 — 대화 로그의 "윤석주 vs 호드리구" 사례 참고).

새 방식은 (1)먼저 품질이 비슷한 카드끼리만 묶고, (2)그 안에서 실제로 두 스타일
방향에 대해 점수가 반대로 갈리는 쌍을 직접 찾는다 — "품질 차이가 크지 않을 때
스타일이 실제로 순위를 가른다"는, 이 설계가 원래 하려던 것 그 자체를 검증한다.
spId는 여전히 하드코딩하지 않고 매번 탐색한다.

실행: ./venv/Scripts/python.exe tests/test_position_fit_real_data.py
"""

import itertools
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "style"))

import position_fit as pf

_STATS = pf.load_player_stats()

# player_stats_final.csv는 GK 카드와 수비수 카드도 섞여 있다. GK/수비형 카드를
# 먼저 제외한 풀에서만 찾는다 (공격 스탯이 원래 낮아 비교 자체가 무의미해짐).
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


def _find_crossover_pair(pool, relevant_stats, score_fn, style_a, style_b, sp_position,
                          quality_quantile=0.85, quality_tolerance=1.0):
    """품질이 비슷한 카드 중, style_a에서는 card1이 이기고 style_b에서는 card2가
    이기는(즉 스타일에 따라 순위가 실제로 뒤집히는) 쌍을 찾는다.

    품질(=relevant_stats 단순평균) 상위 (1-quality_quantile) 비율 안에서만, 그리고
    둘의 품질 차이가 quality_tolerance 이내인 쌍만 후보로 본다 — 품질 차이가 크면
    스타일이 아니라 품질 차이로 순위가 갈릴 수 있어(2026-09-08 실측으로 확인된
    한계) 의미 있는 검증이 안 된다.

    반환: (card1, card2) — card1은 style_a 선호, card2는 style_b 선호. 못 찾으면
    AssertionError.
    """
    quality = pool[relevant_stats].mean(axis=1)
    elite = pool[quality >= quality.quantile(quality_quantile)].copy()
    elite["_quality"] = quality[elite.index]

    ids = elite.index.tolist()
    best = None
    for i, j in itertools.combinations(ids, 2):
        if abs(elite.loc[i, "_quality"] - elite.loc[j, "_quality"]) > quality_tolerance:
            continue
        ci, cj = _STATS.loc[i], _STATS.loc[j]
        score_i_a = score_fn(style_a, ci, sp_position)
        score_j_a = score_fn(style_a, cj, sp_position)
        score_i_b = score_fn(style_b, ci, sp_position)
        score_j_b = score_fn(style_b, cj, sp_position)

        if score_i_a > score_j_a and score_j_b > score_i_b:
            margin = (score_i_a - score_j_a) + (score_j_b - score_i_b)
            candidate = (margin, i, j)
        elif score_j_a > score_i_a and score_i_b > score_j_b:
            margin = (score_j_a - score_i_a) + (score_i_b - score_j_b)
            candidate = (margin, j, i)
        else:
            continue
        if best is None or candidate[0] > best[0]:
            best = candidate

    assert best is not None, (
        f"품질 허용치({quality_tolerance}) 안에서 스타일에 따라 순위가 뒤집히는 쌍을 "
        "못 찾음 — quality_tolerance를 넓히거나 elite_quantile을 낮춰야 할 수 있음"
    )
    _, a_id, b_id = best
    return _STATS.loc[a_id], _STATS.loc[b_id]


def test_vision_vs_technician_card_for_pass_styles():
    """품질이 비슷한 CM 카드 중, 시야 위주 카드는 스루패스형 유저에게,
    짧은패스 위주 카드는 숏패스형 유저에게 더 높은 점수를 받아야 한다."""
    cm_stats = ["짧은 패스", "볼 컨트롤", "긴 패스", "시야"]
    vision_card, technician_card = _find_crossover_pair(
        _OUTFIELD_ATTACKERS, cm_stats, pf.compute_pass_fit_score,
        THROUGH_STYLE, SHORT_STYLE, sp_position=14,  # CM
    )

    v_through = pf.compute_pass_fit_score(THROUGH_STYLE, vision_card, 14)
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
    """품질이 비슷한 ST 카드 중, 헤더 위주 카드는 헤딩형 유저에게,
    박스 안 마무리 위주 카드는 박스안형 유저에게 더 높은 점수를 받아야 한다."""
    st_stats = ["골 결정력", "위치 선정", "반응 속도", "헤더", "점프", "몸싸움"]
    header_card, poacher_card = _find_crossover_pair(
        _OUTFIELD_ATTACKERS, st_stats, pf.compute_shoot_fit_score,
        HEADING_STYLE, IN_PENALTY_STYLE, sp_position=25,  # ST
    )

    h_heading = pf.compute_shoot_fit_score(HEADING_STYLE, header_card, 25)
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
