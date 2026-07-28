import traci
import sumolib
import csv
import os
import xml.etree.ElementTree as ET

# --- ユーザーが設定する部分 ---
sumocfg_file = r"C:\Users\tslab\Desktop\町モデル\toyama_shouwa.sumocfg"  # .sumocfgファイルの名前を指定
output_dir1 = r"C:\Users\tslab\Desktop\町モデル\output\交通環境調査\車両流入"        # 出力フォルダを自由に指定（例："./data/counts/" など）

# --- 固定設定 ---
step_length = 1           # 1ステップ = 1秒
interval = 300            # 記録間隔（秒）

# === sumocfgからnetファイル名を取得 ===
def get_net_file_from_sumocfg(sumocfg_file):
    tree = ET.parse(sumocfg_file)
    root = tree.getroot()
    for elem in root.findall("input/net-file"):
        return elem.attrib["value"]
    raise ValueError("net-file not found in sumocfg")

# === 出力フォルダ作成 ===
def prepare_output_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# === TraCIから信号機ごとの制御レーンを取得 ===
def get_signal_lanes_traci():
    signal_lanes = {}
    for tls_id in traci.trafficlight.getIDList():
        controlled_links = traci.trafficlight.getControlledLinks(tls_id)
        lanes = set()
        for link_list in controlled_links:
            for link in link_list:
                lanes.add(link[0])  # 進入レーンID
        signal_lanes[tls_id] = list(lanes)
    return signal_lanes

# === メイン処理 ===
def run():
    net_file = get_net_file_from_sumocfg(sumocfg_file)
    prepare_output_dir(output_dir1)

    # SUMOシミュレーション開始
    traci.start(["sumo-gui", "-c", sumocfg_file, "--start", "--quit-on-end"])

    # 信号機ごとの進入レーンを取得
    signal_lanes = get_signal_lanes_traci()

    # CSV初期化（ヘッダー書き込み）
    for tls_id in signal_lanes:
        with open(f"{output_dir1}/traffic_{tls_id}.csv", "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["time", "vehicle_count"])  # ← 時刻ベースに変更

    step = 0
    signal_passed_vehicles = {tls: set() for tls in signal_lanes}

    # シミュレーションループ
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()

        # 各信号ごとの進入レーンに進入した車両を記録
        for tls_id, lanes in signal_lanes.items():
            for lane_id in lanes:
                veh_ids = traci.lane.getLastStepVehicleIDs(lane_id)
                signal_passed_vehicles[tls_id].update(veh_ids)

        step += step_length

        # intervalごとにCSVへ出力
        if step % interval == 0:
            for tls_id, vehs in signal_passed_vehicles.items():
                with open(f"{output_dir1}/traffic_{tls_id}.csv", "a", newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([step, len(vehs)])  # ← step: 計測タイミング
                signal_passed_vehicles[tls_id] = set()  # リセット

    traci.close()

if __name__ == "__main__":
    run()
