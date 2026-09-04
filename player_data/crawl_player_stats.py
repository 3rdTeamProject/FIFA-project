import os
import json
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup


# ============================================================
# 설정값
# ============================================================

# 팀원이 준 SPID 목록
SPID_FILE = "needed_spids.json"

# 저장 파일
CHECKPOINT_FILE = "player_data/player_stats_checkpoint.csv"
FINAL_FILE = "player_data/player_stats_final.csv"
FAILED_FILE = "player_data/player_stats_failed.csv"

# 신규 성공 몇 개마다 중간 저장할지
SAVE_INTERVAL = 50

# 정상 요청 사이 대기시간
SLEEP_SECONDS = 0.5

# 요청 실패 시 최대 재시도 횟수
MAX_RETRIES = 3

# 재시도 전 대기시간
RETRY_WAIT_SECONDS = 3


# ============================================================
# 1. 선수 세부 능력치를 가져오는 함수
# ============================================================

def get_player_stats(spid):

    url = "https://fconline.nexon.com/datacenter/PlayerAbility"

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": (
            "https://fconline.nexon.com/"
            f"DataCenter/PlayerInfo?spid={spid}"
        ),
    }

    payload = {
        "spid": spid,
        "n1Strong": 1,
        "n1Grow": 0,
        "n4TeamColorId": 0,
        "n4TeamColorLv": 0,
        "n1Change": 0,
    }

    # --------------------------------------------------------
    # 요청 실패 시 최대 MAX_RETRIES번 재시도
    # --------------------------------------------------------

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        try:

            response = requests.post(
                url,
                headers=headers,
                data=payload,
                timeout=10
            )

            response.raise_for_status()

            # ------------------------------------------------
            # HTML 파싱
            # ------------------------------------------------

            soup = BeautifulSoup(
                response.text,
                "html.parser"
            )

            ability_areas = soup.select(
                "ul.data_wrap_playerinfo"
            )

            # 요청은 성공했지만 능력치 영역이 없는 경우
            if not ability_areas:
                return None

            # ------------------------------------------------
            # 선수 능력치 저장
            # ------------------------------------------------

            player_stats = {
                "spid": spid
            }

            for ability_area in ability_areas:

                for item in ability_area.select("li.ab"):

                    name_tag = item.select_one(".txt")
                    value_tag = item.select_one(".value")

                    if name_tag and value_tag:

                        stat_name = name_tag.get_text(
                            strip=True
                        )

                        stat_value = value_tag.get_text(
                            " ",
                            strip=True
                        )

                        # 숫자로 변환 가능한 경우 정수로 변환
                        try:
                            stat_value = int(
                                stat_value
                            )

                        except ValueError:
                            pass

                        player_stats[
                            stat_name
                        ] = stat_value

            # 정상적으로 수집되었으면 반환
            return player_stats


        # ----------------------------------------------------
        # 네트워크 또는 HTTP 요청 오류
        # ----------------------------------------------------

        except requests.RequestException as e:

            print(
                f"  → 요청 오류 "
                f"({attempt}/{MAX_RETRIES}): "
                f"{e}"
            )

            # 마지막 시도가 아니면 기다렸다가 재시도
            if attempt < MAX_RETRIES:

                print(
                    f"  → {RETRY_WAIT_SECONDS}초 후 "
                    f"재시도..."
                )

                time.sleep(
                    RETRY_WAIT_SECONDS
                )

            # 마지막 시도까지 실패한 경우
            else:

                raise

    return None


# ============================================================
# 2. 프로그램 시작
# ============================================================

print("=" * 60)
print("FC 온라인 선수 세부 능력치 크롤링")
print("=" * 60)

print(
    "\nSPID 목록 불러오는 중..."
)


# ============================================================
# 3. needed_spids.json 읽기
# ============================================================

with open(
    SPID_FILE,
    "r",
    encoding="utf-8"
) as f:

    spids = json.load(f)


# ============================================================
# 4. SPID 정리
# ============================================================

# 혹시 중복된 SPID가 있어도 한 번만 수집
# dict.fromkeys()를 사용하여 기존 순서는 유지
spids = list(
    dict.fromkeys(
        int(spid)
        for spid in spids
    )
)


total = len(spids)


print(
    "수집 대상 SPID:",
    total
)


# ============================================================
# 5. 기존 체크포인트 확인
# ============================================================

all_stats = []

completed_spids = set()


if os.path.exists(
    CHECKPOINT_FILE
):

    print(
        "\n기존 체크포인트 발견"
    )

    df_checkpoint = pd.read_csv(
        CHECKPOINT_FILE
    )

    all_stats = df_checkpoint.to_dict(
        orient="records"
    )

    completed_spids = set(
        df_checkpoint["spid"]
        .astype(int)
    )

    print(
        "이미 수집된 선수 카드:",
        len(completed_spids)
    )


else:

    print(
        "\n기존 체크포인트 없음"
    )


# ============================================================
# 6. 기존 실패 목록 확인
# ============================================================

failed_players = []


if os.path.exists(
    FAILED_FILE
):

    df_failed_old = pd.read_csv(
        FAILED_FILE
    )

    failed_players = df_failed_old.to_dict(
        orient="records"
    )

    print(
        "기존 실패 기록:",
        len(failed_players)
    )


# ============================================================
# 7. 크롤링 시작
# ============================================================

new_success_count = 0


for count, spid in enumerate(
    spids,
    start=1
):

    # --------------------------------------------------------
    # 이미 성공적으로 수집한 SPID는 건너뛰기
    # --------------------------------------------------------

    if spid in completed_spids:

        print(
            f"[{count}/{total}] "
            f"{spid} "
            f"이미 수집됨 → 건너뜀"
        )

        continue


    print(
        f"[{count}/{total}] "
        f"{spid} "
        f"수집 중..."
    )


    try:

        stats = get_player_stats(
            spid
        )


        # ----------------------------------------------------
        # 정상적으로 능력치를 가져온 경우
        # ----------------------------------------------------

        if stats is not None:

            all_stats.append(
                stats
            )

            completed_spids.add(
                spid
            )

            new_success_count += 1

            print(
                f"  → 성공 "
                f"({len(stats) - 1}개 능력치)"
            )


        # ----------------------------------------------------
        # 요청은 성공했지만 능력치 영역이 없는 경우
        # ----------------------------------------------------

        else:

            print(
                "  → 능력치 영역 없음"
            )

            failed_players.append(
                {
                    "spid": spid,
                    "이유": "능력치 영역 없음"
                }
            )


    # --------------------------------------------------------
    # 최대 재시도 횟수까지 요청이 실패한 경우
    # --------------------------------------------------------

    except Exception as e:

        print(
            f"  → 최종 수집 실패: {e}"
        )

        failed_players.append(
            {
                "spid": spid,
                "이유": str(e)
            }
        )


    # ========================================================
    # 8. 일정 개수마다 체크포인트 저장
    # ========================================================

    if (
        new_success_count > 0
        and
        new_success_count % SAVE_INTERVAL == 0
    ):

        df_temp = pd.DataFrame(
            all_stats
        )

        df_temp.to_csv(
            CHECKPOINT_FILE,
            index=False,
            encoding="utf-8-sig"
        )

        print(
            f"\n  >>> 체크포인트 저장 완료 "
            f"(총 {len(all_stats)}장 수집)\n"
        )


    # ========================================================
    # 9. 다음 선수 요청 전 대기
    # ========================================================

    time.sleep(
        SLEEP_SECONDS
    )


# ============================================================
# 10. 최종 DataFrame 생성
# ============================================================

df_final = pd.DataFrame(
    all_stats
)


# ============================================================
# 11. 최종 결과 저장
# ============================================================

df_final.to_csv(
    FINAL_FILE,
    index=False,
    encoding="utf-8-sig"
)


# 체크포인트도 마지막 상태로 갱신
df_final.to_csv(
    CHECKPOINT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 12. 실패 목록 정리 및 저장
# ============================================================

if failed_players:

    df_failed = pd.DataFrame(
        failed_players
    )

    # 같은 SPID가 여러 번 실패 기록된 경우
    # 가장 마지막 기록만 남김
    df_failed = (
        df_failed
        .drop_duplicates(
            subset=["spid"],
            keep="last"
        )
    )

    df_failed.to_csv(
        FAILED_FILE,
        index=False,
        encoding="utf-8-sig"
    )


# ============================================================
# 13. 최종 결과 출력
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "선수 세부 능력치 크롤링 결과"
)

print(
    "=" * 60
)


print(
    "전체 수집 대상:",
    total
)

print(
    "최종 수집 성공:",
    len(df_final)
)

print(
    "이번 실행 신규 성공:",
    new_success_count
)

print(
    "최종 데이터 크기:",
    df_final.shape
)


if failed_players:

    failed_count = len(
        pd.DataFrame(
            failed_players
        )
        .drop_duplicates(
            subset=["spid"],
            keep="last"
        )
    )

    print(
        "실패 기록:",
        failed_count
    )


print(
    f"\n최종 파일: "
    f"{FINAL_FILE}"
)

print(
    f"체크포인트: "
    f"{CHECKPOINT_FILE}"
)


if failed_players:

    print(
        f"실패 목록: "
        f"{FAILED_FILE}"
    )


print(
    "\n크롤링 작업 종료"
)