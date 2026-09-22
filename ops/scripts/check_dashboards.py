"""ダッシュボードの全パネルのクエリを Grafana の API で流し、データが返るかを確かめる。

python3 scripts/check_dashboards.py [--grafana http://127.0.0.1:3000] [--minutes 15]
データが返らないパネルがあれば終了コード 1 で終わる。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.request
from pathlib import Path

DASHBOARDS = Path(__file__).resolve().parent.parent / "grafana" / "dashboards"


def substitute(expr: str) -> str:
    for name in ("$service", "$agent", "$model"):
        expr = expr.replace(name, ".+")
    return expr.replace("$__rate_interval", "1m")


def query(grafana: str, datasource: dict, expr: str, minutes: int, instant: bool) -> list:
    now = int(time.time() * 1000)
    q = {"refId": "A", "datasource": datasource, "expr": substitute(expr), "maxLines": 20}
    if datasource["type"] == "prometheus":
        q.update({"instant": instant, "range": not instant, "intervalMs": 15000})
    else:
        q.update({"queryType": "range"})
    body = json.dumps({"queries": [q], "from": str(now - minutes * 60_000), "to": str(now)}).encode()
    req = urllib.request.Request(f"{grafana}/api/ds/query", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.load(resp)["results"]["A"]
    if "error" in result:
        raise RuntimeError(result["error"])
    return result.get("frames", [])


def has_data(frames: list) -> bool:
    for frame in frames:
        values = frame.get("data", {}).get("values", [])
        for column in values[1:] or values:
            for v in column:
                if v is None:
                    continue
                if isinstance(v, float) and math.isnan(v):
                    continue
                return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grafana", default="http://127.0.0.1:3000")
    parser.add_argument("--minutes", type=int, default=15)
    args = parser.parse_args()
    failed = 0
    for path in sorted(DASHBOARDS.glob("*.json")):
        board = json.loads(path.read_text())
        print(f"== {board['title']}")
        for panel in board["panels"]:
            if panel["type"] == "row":
                continue
            empty = []
            for target in panel.get("targets", []):
                try:
                    frames = query(args.grafana, target["datasource"], target["expr"], args.minutes, target.get("instant", False))
                except Exception as e:  # noqa: BLE001
                    empty.append(f"{target['refId']} error: {e}")
                    continue
                if not has_data(frames):
                    empty.append(target["refId"])
            status = "ok" if not empty else f"データなし {empty}"
            failed += bool(empty)
            print(f"  {status:<24} {panel['title']}")
    print(f"データの無いパネル {failed} 件")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
