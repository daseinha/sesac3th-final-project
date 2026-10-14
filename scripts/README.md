# DB 백업/복구 (2026-10-14 추가)

Supabase 무료 플랜 egress 경고(월 5GB 중 20.88GB 사용)를 계기로 만든
안전망. **egress 초과 자체는 데이터 삭제로 이어지지 않는다**(알림 →
유예기간 → 서비스 제한 순서, [Supabase Billing
FAQ](https://supabase.com/docs/guides/platform/billing-faq) 기준)지만,
만약을 대비해 로컬에 전체 데이터를 복사해두고, 필요하면 다른 Postgres(새
Supabase 프로젝트든 로컬 설치든)로 옮길 수 있게 해둔다.

## 평소: 백업만

```
uv run python -m scripts.backup_supabase
```

`note_chunks`, `deleted_sessions`, `failed_ingestions`, `period_summaries`,
`session_events`, `token_usage` 6개 테이블 전체를 `data/backup_<테이블명>_<오늘날짜>.json`
으로 저장한다. `data/`는 gitignore 대상이라 커밋에 안 섞인다. 이 시점의
스냅샷일 뿐 자동으로 최신 상태를 유지하지 않으므로, 안심하고 싶을 때마다
다시 실행하면 된다.

## 비상시: 다른 DB로 복구

Supabase가 막히거나(제한/정지), 아예 로컬 Postgres로 옮기고 싶을 때.

**전제**: 복구 대상 DB(새 Supabase 프로젝트 또는 로컬 Postgres)가 이미
준비돼 있어야 함 -- 로컬이면 PostgreSQL 설치 + `pgvector` 확장까지.
스크립트는 Postgres 자체를 설치해주지 않는다.

### 1단계: 스키마 만들기

대상 DB에 연결해서 `scripts/schema.sql`을 실행 -- 테이블/확장/인덱스를
전부 만든다. 둘 중 편한 방법으로:

- Supabase면: 대시보드 → SQL Editor → `scripts/schema.sql` 내용 붙여넣고 실행
- 로컬/다른 Postgres면: `psql "<연결 문자열>" -f scripts/schema.sql`
  (이 컴퓨터엔 `psql`이 안 깔려 있어서, 로컬로 옮길 거면 PostgreSQL
  클라이언트 도구까지 같이 설치해야 함)

### 2단계: `.env`의 `POSTGRES_URI`를 복구 대상으로 맞추기

복구 스크립트는 **`.env`에 지금 설정된 `POSTGRES_URI`가 가리키는 DB에
그대로 씀**. 원래 Supabase 프로젝트로 되돌리는 거면 안 바꿔도 되고, 새
프로젝트/로컬로 옮기는 거면 `POSTGRES_URI`를 그 대상으로 바꿔야 함.

### 3단계: 복구 실행

```
uv run python -m scripts.restore_supabase
```

- 실행하면 지금 `.env` 기준으로 어느 DB에 쓸지 먼저 보여주고, `yes`를
  입력해야 진행된다(확인 없이 바로 하려면 `--yes`).
- 대상 DB의 `note_chunks`에 이미 행이 있으면(= 빈 DB가 아닌 것 같으면)
  안전을 위해 중단한다 -- 정말 그 DB에 계속 복구하려면 `--force`.
- `data/`에서 각 테이블의 **가장 최근 날짜** 백업 파일을 자동으로 찾아서
  씀(`backup_<테이블명>_<YYYY-MM-DD>.json` 형태만 인식 -- 과거에 다른
  이름 규칙으로 수동 백업해둔 파일이 있다면 헷갈리지 않게 그 파일명을
  바꾸거나 지워둘 것).
- 전체 테이블 복구는 지금 데이터 규모(`note_chunks` 약 2,700행 기준)로
  약 1분 걸림(여러 행을 한 번에 묶어서 넣는 방식이라, 행 하나씩 넣으면
  Supabase 왕복 지연 때문에 수십 분까지도 걸릴 수 있어서 이렇게 만들었음).
- 같은 행을 또 복구해도(재실행) 중복으로 쌓이지 않는다 -- 각 테이블의
  자연키 기준으로 이미 있는 행은 조용히 건너뜀.

## 알아둘 것

- 백업은 **그 순간의 스냅샷**이다. Supabase가 계속 바뀌면 다시 백업해야
  최신 상태가 된다.
- 복구 스크립트는 **데이터만** 넣는다 -- 스키마(1단계)가 먼저 없으면 실패.
- `note_chunks.embedding`(RAG 벡터, 1536차원)도 백업/복구에 포함된다 --
  복구 후 바로 벡터 검색이 정상 동작한다(재임베딩 불필요).
