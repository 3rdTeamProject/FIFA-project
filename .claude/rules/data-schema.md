# 데이터 스키마 (ERD 상세)

이 프로젝트에서 데이터베이스/전처리 코드를 작성할 때는 아래 스키마를 그대로 따른다.
새 필드를 추가하기 전에 반드시 "Feature 선정 원칙"(CLAUDE.md)에 부합하는지 확인할 것.

## USER
- ouid (PK), nickname, tier(division)

## MATCH (팀 단위 집계, 실제 API matchDetail+pass+shoot 기반)
- match_id (PK), ouid (FK), season_id, match_result(승/무/패)
- possession
- pass_try, short_pass_try, long_pass_try, through_pass_try (스루+로빙스루 합산)
- dribble_yard
- shoot_total, shoot_heading, shoot_in_penalty, shoot_out_penalty
- ⚠️ goal_total 등 "골 여부"가 들어간 필드는 절대 포함하지 않는다 (승패와 거의 동일한 정보, leakage).
- ⚠️ 파울/카드/오프사이드/컨트롤러타입/경기평점 등 스타일·승률과 무관한 필드는 포함하지 않는다.
- ⚠️ 프리킥/페널티킥 관련 슛 필드는 포함하지 않는다 (유저 선택이 아니라 상대 반칙에 좌우됨).

## MATCH_PLAYER (경기별 사용 선수, 실제 API player[] 반영)
- match_id (FK), card_id(spId, FK), sp_position(spPosition), sp_grade
- ⚠️ player[].status(개인별 패스/드리블/태클 등 세부 기록)는 저장하지 않는다 — 어느 계산에도 쓰이지 않음.

## PLAYER_CARD (선수 카드 마스터, 실제 선수 스탯 파일 기준)
- card_id(spid, PK), player_name, season
- stat_short_pass(짧은 패스), stat_long_pass(긴 패스)
- stat_dribble(드리블), stat_ball_control(볼 컨트롤)
- ⚠️ 오버롤 필드가 원본 파일에 없음 — 미해결 항목 참고. 확정 전까지 avg_overall 계산 로직을 임의로 만들지 말 것.

## USER_STYLE_PROFILE (K-means 결과, 유저 단위 집계)
- ouid (FK), style_type (K-means 군집 결과, 사후 라벨링)
- short_pass_ratio, long_pass_ratio, through_pass_ratio, dribble_intensity
- in_penalty_shoot_ratio, heading_shoot_ratio
- ⚠️ 비율 feature는 **티어별 정규화 없이** 유저 본인 데이터로만 계산한다 (style/README.md 참고).
  관찰적 진단이라는 설계 선택에 의한 의도적 제약이며, 사후 검증 시에도 정규화 로직을
  추가하면 안 된다.

## WIN_PREDICTION_MODEL_INPUT (경기 단위, 모델 학습용 최종 feature)
- match_id (FK), ouid (FK)
- attack_avg_score / mid_avg_score / defense_avg_score (spPosition으로 나눈 공격/미드필더/수비
  그룹의 세부스탯 단순평균 — 오버롤 필드가 선수 스탯 파일에 없어 확정된 대체 방식).
  그룹별로 어떤 스탯을 쓸지는 임의로 정하지 않고 EA/FC 온라인의 공식 6분류(게임 UI의
  스피드/슈팅/패스/드리블/수비/피지컬, 즉 PAC/SHO/PAS/DRI/DEF/PHY)를 그대로 근거로 삼는다:
    - attack_avg_score = 스피드 + 드리블 + 피지컬 + 슈팅(특화, 헤더 포함)
    - defense_avg_score = 스피드 + 드리블 + 피지컬 + 수비(특화, 헤더 포함)
    - mid_avg_score = 6개 카테고리 전체(29개 스탯 그대로)
  헤더는 공식 분류상 수비(DEF) 카테고리에만 속하지만, 헤딩골도 명백한 득점 루트라는
  이유로 슈팅(SHO) 카테고리에도 중복 포함시켰다 — 공식 분류에서 벗어나는 유일한 예외이며
  이유가 명확해 임의 가중치 문제로 보지 않는다. 그룹 내부에서는 여전히 전부 동일 가중치로
  평균한다(가중치를 준 게 아니라 "어떤 스탯이 그 포지션과 관련 있는가"만 고른 것).
- gk_avg_score (GK 전용 스탯 5개만 평균)
- tier
- result (타겟)
- ~~formation~~: 영구 제외(2026-09-12) — formation_fit_score 자체를 범위에서 뺐으므로
  (CLAUDE.md "명시적으로 범위에서 제외한 것" 참고) 이 feature의 원천 데이터가 없다.
- ~~team_color~~: 현재 baseline에서는 제외(API 응답 존재 여부 미확인). 있으면 재검토.
- **style_fit_score(=squad_fit_score) — 다음 작업 대상(2026-09-12)**: "궁합 점수가 실제
  승률과 관련 있다"는 예비 검증(「스타일 궁합 승부예측 실험」, 201명·구 데이터 기준
  val 정확도 50.9%→54.1%, squad_fit_score 계수 p=0.083 경계선)까지는 끝났고, 공식
  WIN_PREDICTION_MODEL_INPUT에 정식 feature로 편입해 같은 split(random_seed 고정)으로
  이전 baseline과 비교하는 작업이 남았다. USER_STYLE_PROFILE 표본(현재 319명, 50경기
  이상)이 승률 트랙(수천 명, 1유저 1경기)보다 훨씬 좁다는 제약을 그대로 안고 간다.

⚠️ avg_score 계산 전 확인 필요:
1. player_stats_final.csv가 4,187개 카드를 담고 있음 (2026-09-04 수집분 2,916개 +
   2026-09-07 병합한 v2 1,271개, spid 중복 없음 확인됨) — 수집된 매치의 spId 매칭률 100% 확인됨
2. 이 파일 수치가 강화단계(spGrade) 반영값인지 기본(0강)값인지 불분명 —
   불분명하면 강화단계 무시하고 근사치로 쓰는 것을 한계로 명시

## 궁합 점수 (진단 화면에 표시, 별도 저장 테이블 없음 — 진단 시점에 즉시 계산)
- position_fit_score: 포지션별 선수 스탯 벡터 vs 유저 스타일 벡터 유사도 (품질×스타일_배율, style/position_fit.py)
- squad_fit_score: 스쿼드 11명 전체 단위로 집계한 궁합 점수
- ~~formation_fit_score~~: 범위에서 제외(2026-09-12, CLAUDE.md 참고) — 포메이션 추천은 하지 않는다.

## 제거된 엔티티 (다시 만들지 말 것)
- MY_SQUAD, MY_SQUAD_SLOT: 유저가 직접 스쿼드를 등록/저장하는 기능은 없음.
  최근 경기 1건의 MATCH_PLAYER 11명 조합을 그대로 "현재 스쿼드"로 사용한다.
- SHOOT_DETAIL(슈팅별 좌표): 현재 어느 기능에도 쓰이지 않아 제거됨.
