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

## WIN_PREDICTION_MODEL_INPUT (경기 단위, 모델 학습용 최종 feature)
- match_id (FK), ouid (FK)
- avg_stat_score (포지션별 세부스탯 단순평균 — 오버롤 필드가 선수 스탯 파일에 없어 확정된 대체 방식.
  포지션별 가중평균은 가중치를 임의로 정해야 해 근거가 약해져 채택 안 함)
- formation (spPosition 조합으로 추정, 자리 조합표 필요)
- tier
- style_fit_score (USER_STYLE_PROFILE과 PLAYER_CARD 스탯 벡터 간 유사도)
- result (타겟)

⚠️ avg_stat_score 계산 전 확인 필요:
1. player_1000_final.csv가 1,000개 카드만 담고 있음 — 수집된 매치의 spId 중 매칭률 확인 필요
2. 이 파일 수치가 강화단계(spGrade) 반영값인지 기본(0강)값인지 불분명 —
   불분명하면 강화단계 무시하고 근사치로 쓰는 것을 한계로 명시

## 궁합 점수 (진단 화면에 표시, 별도 저장 테이블 없음 — 진단 시점에 즉시 계산)
- position_fit_score: 포지션별 선수 스탯 벡터 vs 유저 스타일 벡터 유사도
- squad_fit_score: 스쿼드 11명 전체 단위로 집계한 궁합 점수
- formation_fit_score: 포메이션(전술 구조) 자체의 궁합 점수 — 후보 포메이션들 중 최고점을 "추천 포메이션"으로 제시

## 제거된 엔티티 (다시 만들지 말 것)
- MY_SQUAD, MY_SQUAD_SLOT: 유저가 직접 스쿼드를 등록/저장하는 기능은 없음.
  최근 경기 1건의 MATCH_PLAYER 11명 조합을 그대로 "현재 스쿼드"로 사용한다.
- SHOOT_DETAIL(슈팅별 좌표): 현재 어느 기능에도 쓰이지 않아 제거됨.
