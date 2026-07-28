import csv
import os
import shutil
import sys
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from math import atan2, degrees

import sumolib
import traci
import traci.constants as tc

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


PROJECT_ROOT = Path(r"C:\Users\Tsukasa\Desktop\研究\計測")
SUMOCFG_FILE = PROJECT_ROOT / "toyama_shouwa.sumocfg"
ACTIVE_ROUTE_FILE = PROJECT_ROOT / "random_trips_with_highway.rou.xml"
SAVED_ROUTE_DIR = PROJECT_ROOT / "車両生成データ" / "★車両データ" / "6月28日作成 (24hデータ)" / "生成された車_北南10%削減_2026-07-14"
OUTPUT_ROOT = PROJECT_ROOT / "output" / "交差点流入量_北南10%削減_2026-07-14"

INTERVAL_SECONDS = 3600
DAYS = 1

# 既存の待ち台数計測で使っている信号機だけを計測対象にする。
# None にすると、SUMOネットワーク内の全信号機を対象にする。
TARGET_TRAFFIC_LIGHT_IDS = ["E", "F", "I", "J", "K", "M", "N"]
DIRECTION_COLUMNS = ["東", "西", "南", "北"]


def get_sumo_binary():
    # 自動実行のためヘッドレス(sumo)を使用。sumo_cmd に --start が無いため
    # sumo-gui だと再生待ちでハングする。
    try:
        return sumolib.checkBinary("sumo")
    except Exception:
        return "sumo"


def prepare_route_file_for_day(day):
    saved_route = SAVED_ROUTE_DIR / f"{day}日目.rou.xml"
    if saved_route.exists():
        shutil.copy2(saved_route, ACTIVE_ROUTE_FILE)
        print(f"✅ Day {day}: 保存済みルートを使用します -> {saved_route}")
        return

    if ACTIVE_ROUTE_FILE.exists():
        print(f"✅ Day {day}: 現在のルートファイルを使用します -> {ACTIVE_ROUTE_FILE}")
        return

    raise FileNotFoundError(
        f"{ACTIVE_ROUTE_FILE} が見つかりません。"
        f"{saved_route} を用意するか、車両生成を先に実行してください。"
    )


def lane_to_edge(lane_id):
    return traci.lane.getEdgeID(lane_id)


def get_lane_shape(lane_id):
    try:
        return traci.lane.getShape(lane_id)
    except Exception:
        return []


def direction_from_lane(lane_id):
    shape = get_lane_shape(lane_id)
    if len(shape) < 2:
        return "不明"

    upstream = shape[0]
    downstream = shape[-1]
    dx = upstream[0] - downstream[0]
    dy = upstream[1] - downstream[1]
    if dx == 0 and dy == 0:
        return "不明"

    angle = (degrees(atan2(dy, dx)) + 360) % 360
    if 45 <= angle < 135:
        return "北"
    if 135 <= angle < 225:
        return "西"
    if 225 <= angle < 315:
        return "南"
    return "東"


def add_unique(items, value):
    if value and value not in items:
        items.append(value)


def get_target_traffic_light_ids():
    all_tls_ids = set(traci.trafficlight.getIDList())
    if TARGET_TRAFFIC_LIGHT_IDS is None:
        return sorted(all_tls_ids)

    missing_ids = [tls_id for tls_id in TARGET_TRAFFIC_LIGHT_IDS if tls_id not in all_tls_ids]
    for tls_id in missing_ids:
        print(f"⚠ 計測対象信号IDがSUMOに見つかりません: {tls_id}")

    return [tls_id for tls_id in TARGET_TRAFFIC_LIGHT_IDS if tls_id in all_tls_ids]


def build_signal_inflow_config():
    direction_map = {}
    edge_map = {}
    via_lane_to_targets = defaultdict(list)

    for tls_id in get_target_traffic_light_ids():
        controlled_links = traci.trafficlight.getControlledLinks(tls_id)

        for link_group in controlled_links:
            for link in link_group:
                if len(link) < 2:
                    continue

                from_lane = link[0]
                to_lane = link[1]
                via_lane = link[2] if len(link) >= 3 else ""

                if not from_lane:
                    continue

                incoming_edge = lane_to_edge(from_lane)
                if incoming_edge.startswith(":"):
                    continue

                count_lane = via_lane if via_lane and via_lane.startswith(":") else to_lane
                if not count_lane:
                    count_lane = from_lane

                source_direction = direction_from_lane(from_lane)
                direction_key = (tls_id, source_direction)
                edge_key = (tls_id, incoming_edge)

                direction_info = direction_map.setdefault(
                    direction_key,
                    {"tls_id": tls_id, "direction": source_direction, "incoming_edges": []}
                )
                add_unique(direction_info["incoming_edges"], incoming_edge)

                edge_map.setdefault(
                    edge_key,
                    {"tls_id": tls_id, "direction": source_direction, "incoming_edge": incoming_edge}
                )

                target = {
                    "tls_id": tls_id,
                    "direction": source_direction,
                    "incoming_edge": incoming_edge,
                }
                if target not in via_lane_to_targets[count_lane]:
                    via_lane_to_targets[count_lane].append(target)

    return direction_map, edge_map, via_lane_to_targets


def subscribe_count_lanes(lane_targets):
    subscribed_count = 0
    for lane_id in lane_targets:
        try:
            traci.lane.subscribe(lane_id, (tc.LAST_STEP_VEHICLE_ID_LIST,))
            subscribed_count += 1
        except Exception as e:
            print(f"⚠ レーン購読をスキップしました: {lane_id} - {e}")
    return subscribed_count


def safe_traci_close():
    try:
        traci.close(False)
    except Exception as e:
        print(f"⚠ TraCI closeをスキップしました: {e}")


def new_counts(keys):
    return {key: 0 for key in keys}


def new_seen(keys):
    return {key: set() for key in keys}


def safe_filename(value):
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in value)


def cleanup_old_combined_outputs(day_dir, day):
    old_paths = [
        day_dir / f"交差点流入量_5分_方向別_day{day}.csv",
        day_dir / f"交差点流入量_5分_エッジ別_day{day}.csv",
        day_dir / f"交差点流入方向一覧_day{day}.csv",
    ]
    for path in old_paths:
        if not path.exists():
            continue
        try:
            path.unlink()
        except PermissionError:
            print(f"⚠ 不要ファイルを削除できませんでした。閉じてから再実行してください: {path}")


def open_intersection_writers(stack, day_dir, day, direction_map):
    writers = {}
    tls_ids = sorted({info["tls_id"] for info in direction_map.values()})
    for tls_id in tls_ids:
        csv_path = day_dir / f"交差点_{safe_filename(tls_id)}_流入量_5分_day{day}.csv"
        f = stack.enter_context(open(csv_path, mode="w", newline="", encoding="utf-8-sig"))
        writer = csv.writer(f)
        writer.writerow(["開始時間(s)", "終了時間(s)", *DIRECTION_COLUMNS])
        writers[tls_id] = writer
    return writers


def write_intersection_direction_rows(writers, interval_start, interval_end, counts):
    for tls_id in sorted(writers):
        row = [interval_start, interval_end]
        for direction in DIRECTION_COLUMNS:
            row.append(counts.get((tls_id, direction), 0))
        writers[tls_id].writerow(row)


def measure_day(day):
    prepare_route_file_for_day(day)

    day_dir = OUTPUT_ROOT / f"day{day}"
    day_dir.mkdir(parents=True, exist_ok=True)
    cleanup_old_combined_outputs(day_dir, day)

    sumo_cmd = [
        get_sumo_binary(),
        "-c",
        str(SUMOCFG_FILE),
        "--quit-on-end",
        "--no-step-log",
        "true",
    ]

    traci.start(sumo_cmd)
    try:
        direction_map, _edge_map, lane_targets = build_signal_inflow_config()
        subscribed_count = subscribe_count_lanes(lane_targets)
        print(f"✅ Day {day}: 監視対象レーン {subscribed_count} 本")

        direction_counts = new_counts(direction_map.keys())
        direction_seen = new_seen(direction_map.keys())

        current_interval_start = 0

        with ExitStack() as stack:
            intersection_writers = open_intersection_writers(stack, day_dir, day, direction_map)

            while traci.simulation.getMinExpectedNumber() > 0:
                traci.simulationStep()
                sim_time = traci.simulation.getTime()
                interval_start = int(sim_time // INTERVAL_SECONDS) * INTERVAL_SECONDS

                while interval_start > current_interval_start:
                    interval_end = current_interval_start + INTERVAL_SECONDS
                    write_intersection_direction_rows(
                        intersection_writers,
                        current_interval_start,
                        interval_end,
                        direction_counts,
                    )
                    current_interval_start = interval_end
                    direction_counts = new_counts(direction_map.keys())
                    direction_seen = new_seen(direction_map.keys())

                for lane_id, targets in lane_targets.items():
                    results = traci.lane.getSubscriptionResults(lane_id) or {}
                    vehicle_ids = results.get(tc.LAST_STEP_VEHICLE_ID_LIST, ())

                    for veh_id in vehicle_ids:
                        for target in targets:
                            direction_key = (target["tls_id"], target["direction"])

                            if veh_id not in direction_seen[direction_key]:
                                direction_counts[direction_key] += 1
                                direction_seen[direction_key].add(veh_id)

            final_interval_end = current_interval_start + INTERVAL_SECONDS
            write_intersection_direction_rows(
                intersection_writers,
                current_interval_start,
                final_interval_end,
                direction_counts,
            )

    finally:
        safe_traci_close()

    print(f"✅ Day {day}: 5分間流入量を出力しました -> {day_dir}")


def main():
    for day in range(1, DAYS + 1):
        print(f"=== Day {day}: 交差点流入量5分計測開始 ===")
        measure_day(day)


if __name__ == "__main__":
    main()
