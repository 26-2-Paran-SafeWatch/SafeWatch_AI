"""성능 벤치마크.

단계별 처리 시간, fps, CPU 온도를 측정하여 results/benchmark/에 기록한다
(README.md 참고). 파이프라인 각 단계가 구현된 뒤 작성한다.

사용법
    python scripts/benchmark.py --config configs/pi5.yaml
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SafeWatch AI 성능 벤치마크")
    parser.add_argument("--config", required=True, help="configs/*.yaml 경로")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(
        "TODO(S1 이후): DETECTION~RISK 단계 구현 후 src.utils.profiler 결과를 집계해 "
        f"results/benchmark/에 기록. config={args.config}"
    )


if __name__ == "__main__":
    main()
