"""parse_vision_pdf_notes()가 실패한 6페이지(다이어그램 폭주, LengthFinishReasonError)
를 Claude가 직접 읽어서 수동 전사한 내용을 삽입한다 -- 2026-10-14.

실패 페이지: 5, 21, 38, 43, 46, 51 (21/46/51은 2026-09-30 1차 시도 때도 실패했던
바로 그 페이지와 동일 -- 다이어그램 위주 페이지의 고질적 실패 패턴 재확인).
origin_ref에 ":claude_manual_ocr" 태그를 붙여 자동 OCR과 구분한다(기존 관례).
"""

from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

from src.api import ingest_note  # noqa: E402

_PATH = "data/personal/Scan_20260930_145133.pdf"
_COURSE_ID = "sesac_bootcamp"

_PAGES = [
    {
        "page": 5,
        "timestamp": datetime(2026, 6, 30),
        "content": (
            "프로젝트 유포니아(project Euphoria) 관련 메모. teachable Machine 시도, "
            "자율비행 관련 아이디어. GPT 5.6, SOL, 앤트로픽 fable:mythos 메모. "
            "LLM(Large Language Model) = 챗GPT 기반 대규모 언어 모델 -- 대규모 데이터로 "
            "사전학습된 모델이며, 모델별로 사전학습 데이터에 따라 성능이 다르다. "
            "오픈소스/폐쇄형 모델 구분. OpenAI, GPT 플랫폼 관련 메모. "
            "Credential(로그인 정보): API key, 지갑 비밀번호 등 메모 관리 관련."
        ),
    },
    {
        "page": 21,
        "timestamp": datetime(2026, 7, 9),
        "content": (
            "create-sub-workflow 관련 메모. 마크다운(.md) 문서 작성법: #(샵)은 heading, "
            "**(별표 두 개)는 bold, 백틱(BackTick/Grave Accent)과 코드블록, latex 수식 "
            "입력법. Visual Studio Code에서 탐색(explore) 기능. Markup과 Markdown의 "
            "구분. AI 사용이 점점 늘어나고 여러 사람이 같이 쓰게 된다는 메모."
        ),
    },
    {
        "page": 38,
        "timestamp": datetime(2026, 9, 12),
        "content": "Open API, 공공데이터 활용, 공공데이터 포털 API 관련 메모.",
    },
    {
        "page": 43,
        "timestamp": datetime(2026, 8, 18),
        "content": (
            "Git 저장소 초기화 실습. Initialize Repository = git init 명령어. "
            "rm -rf .git으로 초기화 git 폴더 삭제 가능(CLI). git init하면 버전관리가 "
            "시작됨(watching 상태). 최초의 commit은 스냅샷(snapshot)을 저장하는 "
            "기록이다. 흐름: staged change -> commit. 기본 흐름(init -> add -> commit), "
            "src 폴더 생성. .gitignore 설정 -- API key 같은 민감정보는 GitHub에 "
            "올라가면 안 되므로 여기에 등록. README.md는 해당 프로젝트의 설명서 역할을 "
            "하는 특수한 파일. 내 컴퓨터(Local)는 .git 폴더가 숨어있어서 버전관리 "
            "기능이 활성화됨. Repository(Repo)는 로컬/원격 둘 다 버전을 저장하는 곳 -- "
            "cloud의 GitHub은 원격 저장소(Remote Repository) 역할. git을 통해 "
            "Local과 Remote를 동기화(A=B). origin은 remote 연결을 가리키는 이름."
        ),
    },
    {
        "page": 46,
        "timestamp": None,  # 이 페이지엔 날짜가 안 적혀있음 -- 임의로 채우지 않음
        "content": (
            "Git 브랜치/머지 실습. mkdir src, git add, git commit -m '메시지'. "
            "커밋마다 고유한 해시값(ID)이 있다. 브랜치 생성과 tag(태그) 기능. "
            "git switch b1로 브랜치 전환 -- head가 가리키는 브랜치가 바뀐다. "
            "Branching(병합) 결과는 3가지: 1.FF(Fast-Forward) 2.Auto merge(새 커밋 "
            "생성) 3.Conflict(충돌). git branch -d b1로 브랜치 삭제, git merge 실행, "
            "git log로 커밋 기록 확인. head는 지금 작업 중인 브랜치를 가리키고, "
            "main은 기준이 되는 브랜치."
        ),
    },
    {
        "page": 51,
        "timestamp": datetime(2026, 8, 24),
        "content": (
            "반복문(while/for) 실습. alt는 반복기 -- 동일한 패턴을 반복적으로 실행. "
            "brute force(무식하게 대입)도 반복문의 일종이라는 메모. while 조건: 조건이 "
            "참인 동안 반복 실행. if 조건: 조건에 해당할 때만 실행. "
            "members = [1,2,3,4,5], Total을 구하기 위해 for num in members로 "
            "리스트 순회. while문의 구조는 종료조건/처리/판단 3단계. "
            "for 반복문의 in/out 흐름도(루프 진입과 탈출)와 조건 분기(if)에 대한 "
            "그림 메모."
        ),
    },
]


def main() -> None:
    for item in _PAGES:
        origin_ref = f"{_PATH}:page{item['page']}:claude_manual_ocr"
        state = ingest_note(
            content=item["content"],
            source="personal",
            medium="paper_ocr",
            origin_ref=origin_ref,
            course_id=_COURSE_ID,
            timestamp=item["timestamp"],
        )
        print(f"{origin_ref}: routing={state.get('routing_decision')}")
    print(f"완료: {len(_PAGES)}건 수동 전사 삽입.")


if __name__ == "__main__":
    main()
