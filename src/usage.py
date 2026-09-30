"""토큰 사용량 추적 -- 모든 LLM/임베딩 호출을 token_usage 테이블에 기록한다.

프로젝트 자체가 테스트베드라, "확인은 외부 대시보드에서" 대신 우리 DB에
남겨서 나중에 프론트엔드나 다른 도구에서도 바로 조회할 수 있게 한다.
"""

from src.db import get_pool


def record_usage(
    call_type: str,
    model: str,
    input_tokens: int,
    output_tokens: int = 0,
    purpose: str | None = None,
) -> None:
    """호출 하나의 토큰 사용량을 기록. call_type은 'chat' 또는 'embedding'."""
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO token_usage (call_type, model, purpose, input_tokens, output_tokens)
                VALUES (%(call_type)s, %(model)s, %(purpose)s, %(input_tokens)s, %(output_tokens)s);
                """,
                {
                    "call_type": call_type,
                    "model": model,
                    "purpose": purpose,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                },
            )


def get_usage_summary() -> dict:
    """지금까지 누적된 사용량을 call_type/model/purpose별로 집계해서 반환."""
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT call_type, model, purpose,
                       count(*) AS calls,
                       sum(input_tokens) AS input_tokens,
                       sum(output_tokens) AS output_tokens
                FROM token_usage
                GROUP BY call_type, model, purpose
                ORDER BY call_type, model, purpose;
                """
            )
            columns = [c.name for c in cur.description]
            rows = [dict(zip(columns, row)) for row in cur.fetchall()]

            cur.execute(
                "SELECT count(*), sum(input_tokens), sum(output_tokens) FROM token_usage;"
            )
            total_calls, total_input, total_output = cur.fetchone()

    return {
        "total_calls": total_calls or 0,
        "total_input_tokens": total_input or 0,
        "total_output_tokens": total_output or 0,
        "by_group": rows,
    }
