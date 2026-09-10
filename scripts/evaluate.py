"""성능 평가 — Precision, Recall, F1 산출 (README.md 참고).

사용법
    python scripts/evaluate.py --pred results/ --gt data/labels/
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SafeWatch AI 예측 결과 평가")
    parser.add_argument("--pred", required=True, help="예측 결과(이벤트 JSON) 디렉터리")
    parser.add_argument("--gt", required=True, help="정답 라벨 디렉터리")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(
        "TODO(S4 이후): 라벨 포맷 확정 후 event_id 매칭 기반 Precision/Recall/F1 계산. "
        f"pred={args.pred} gt={args.gt}"
    )


if __name__ == "__main__":
    main()
