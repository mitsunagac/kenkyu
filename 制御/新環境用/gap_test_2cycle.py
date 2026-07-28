import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import csv
import sys
import numpy as np
import matplotlib.pyplot as plt
import traci
import datetime
SUMO_EXE = rf"C:\Users\tslab\Downloads\sumo-win64-1.6.0\sumo-1.6.0\bin\sumo.exe"
SUMO_NET = rf"C:\Users\tslab\Desktop\単一(卒論)\intersection2_sotsu.net.xml"
SUMO_ROUTE = rf"C:\Users\tslab\Desktop\単一(卒論)\intersection2_sotsu.rou.xml"
SUMO_ADDITIONAL = rf"C:\Users\tslab\Desktop\単一(卒論)\gap.add.xml"
if 'SUMO_HOME' not in os.environ:
    os.environ['SUMO_HOME'] = 'C:\\Users\\tslab\\Downloads\\sumo-win64-1.6.0\\sumo-1.6.0'
    
sys.path.append('C:\\Users\\tslab\\Downloads\\sumo-win64-1.6.0\\sumo-1.6.0\\tools')
try:
    import sumolib
except ImportError:
    sys.exit("Please check SUMO_HOME environment variable or install sumolib")

WARMUP_TIME = 600
TEST_TIME = 4200
TOTAL_SIMULATION_TIME = WARMUP_TIME + TEST_TIME

MAX_GAP = 4
MIN_GREEN = 28
MAX_GREEN = 44
YELLOW_TIME = 3
ALL_RED_TIME = 3
LEFT_ARROW_TIME = 5

INTERSECTION_IDS = ["intersection1", "intersection2"]
print(f"使用する信号機ID: {INTERSECTION_IDS}")

manual_lane_data = {
    "intersection1": {
        "NS1_4": ["NS1_4_0", "NS1_4_1"],
        "NS1_3": ["NS1_3_0"],
        "NS1_2": ["NS1_2_0"],
        "SN1_4": ["SN1_4_0", "SN1_4_1"],
        "SN1_3": ["SN1_3_0"],
        "SN1_2": ["SN1_2_0"],
        "EW7": ["EW7_0", "EW7_1"],
        "EW6": ["EW6_0"],
        "EW5": ["EW5_0"],
        "WE4": ["WE4_0", "WE4_1"],
        "WE3": ["WE3_0"],
        "WE2": ["WE2_0"],
    },
    "intersection2": {
        "NS2_4": ["NS2_4_0", "NS2_4_1"],
        "NS2_3": ["NS2_3_0"],
        "NS2_2": ["NS2_2_0"],
        "SN2_4": ["SN2_4_0", "SN2_4_1"],
        "SN2_3": ["SN2_3_0"],
        "SN2_2": ["SN2_2_0"],
        "EW2": ["EW2_0", "EW2_1"],
        "EW1": ["EW1_0"],
        "EW0": ["EW0_0"],
        "WE9": ["WE9_0", "WE9_1"],
        "WE8": ["WE8_0"],
        "WE7": ["WE7_0"],
        "WE6": ["WE6_0"],
        "WE5": ["WE5_0"]
    },
}

DETECTORS = {
    "intersection1": {
        "EW": ["det_int1_EW7_0"],
        "WE": ["det_int1_WE4_0"],
        "NS": ["det_int1_NS1_4_0"],
        "SN": ["det_int1_SN1_4_0"]
    },
    "intersection2": {
        "EW": ["det_int2_EW2_0"],
        "WE": ["det_int2_WE9_0"],
        "NS": ["det_int2_NS2_4_0"],
        "SN": ["det_int2_SN2_4_0"]
    }
}

SIGNAL_PHASES = {
    "EW_NS_green": "rrrGGgrrrGGg",
    "EW_NS_yellow": "rrryyyrrryyy",
    "EW_NS_left": "rrrrrGrrrrrG",
    "EW_NS_left_yellow": "rrrrryrrrrry",
    "all_red": "rrrrrrrrrrrr",
    "NS_EW_green": "GGgrrrGGgrrr",
    "NS_EW_yellow": "yyyrrryyyrrr",
    "NS_EW_left": "rrGrrrrrGrrr",
    "NS_EW_left_yellow": "rryrrrrryrrr"
}

class GapActuatedController:
    def __init__(self, intersection_id):
        self.intersection_id = intersection_id
        self.current_phase = 0
        self.phase_start_time = 0
        self.last_vehicle_time = 0
        self.cycle_count = 0
        self.previous_duration = {0: None, 1: None}
        self.target_duration = {0: None, 1: None}
        
        self.signal_state = "green"
        self.state_start_time = 0
        self.green_duration = 0
        
    def check_gap(self, current_time):
        if self.signal_state != "green":
            return False
            
        direction_group = ["EW", "WE"] if self.current_phase == 0 else ["NS", "SN"]
        
        for direction in direction_group:
            for detector_id in DETECTORS[self.intersection_id][direction]:
                try:
                    if traci.inductionloop.getLastStepVehicleNumber(detector_id) > 0:
                        self.last_vehicle_time = current_time
                        return False
                except:
                    pass
        
        if current_time - self.last_vehicle_time >= MAX_GAP:
            return True
        return False
    
    def get_green_duration(self, current_time):
        if self.signal_state == "green":
            return current_time - self.phase_start_time
        else:
            return self.green_duration
    
    def calculate_target_duration(self, actual_duration, phase_index):
        baseline = 36  
        deviation = actual_duration - baseline
        
        target = baseline - deviation
        
        target = max(MIN_GREEN, min(MAX_GREEN, target))
        
        return target
    
    def should_end_green(self, current_time):
        if self.signal_state != "green":
            return False
            
        duration = self.get_green_duration(current_time)
    
        if self.cycle_count == 0:
            if duration >= MAX_GREEN:
                return True
            if duration >= MIN_GREEN and self.check_gap(current_time):
                return True
        else:
            target = self.target_duration[self.current_phase]
            if target is not None:
                if duration >= target:
                    return True
            else:
                if duration >= MAX_GREEN:
                    return True
                if duration >= MIN_GREEN and self.check_gap(current_time):
                    return True
        
        return False
    
    def start_phase_change(self, current_time):
        self.green_duration = current_time - self.phase_start_time
        
        if self.cycle_count == 0:
            self.previous_duration[self.current_phase] = self.green_duration
            next_target = self.calculate_target_duration(self.green_duration, self.current_phase)
            self.target_duration[self.current_phase] = next_target
            
            print(f"{self.intersection_id} [サイクル1] Phase {self.current_phase}: "
                  f"実際={self.green_duration:.1f}秒 → 次回目標={next_target:.1f}秒")
        else:
            print(f"{self.intersection_id} [サイクル2] Phase {self.current_phase}: "
                  f"目標={self.target_duration[self.current_phase]:.1f}秒, "
                  f"実際={self.green_duration:.1f}秒")
        self.signal_state = "yellow"
        self.state_start_time = current_time
    
    def update_signal_state(self, current_time):
        state_duration = current_time - self.state_start_time
        
        if self.signal_state == "green":
            if self.should_end_green(current_time):
                self.start_phase_change(current_time)
                
        elif self.signal_state == "yellow":
            if state_duration >= YELLOW_TIME:
                self.signal_state = "left_arrow"
                self.state_start_time = current_time
        
        elif self.signal_state == "left_arrow":
            if state_duration >= LEFT_ARROW_TIME:
                self.signal_state = "left_yellow"
                self.state_start_time = current_time
        
        elif self.signal_state == "left_yellow":
            if state_duration >= YELLOW_TIME:
                self.signal_state = "all_red"
                self.state_start_time = current_time
                
        elif self.signal_state == "all_red":
            if state_duration >= ALL_RED_TIME:
                self.current_phase = 1 - self.current_phase
                self.signal_state = "green"
                self.phase_start_time = current_time
                self.state_start_time = current_time
                self.last_vehicle_time = current_time
                self.green_duration = 0

                if self.current_phase == 0:
                    self.cycle_count = 1 - self.cycle_count
                    if self.cycle_count == 0:
                        print(f"{self.intersection_id}: 新しいサイクルペア開始")
    
    def get_current_signal_state(self):
        if self.signal_state == "green":
            if self.current_phase == 0:
                return SIGNAL_PHASES["EW_NS_green"]
            else:
                return SIGNAL_PHASES["NS_EW_green"]
                
        elif self.signal_state == "yellow":
            if self.current_phase == 0:
                return SIGNAL_PHASES["EW_NS_yellow"]
            else:
                return SIGNAL_PHASES["NS_EW_yellow"]
        
        elif self.signal_state == "left_arrow":
            if self.current_phase == 0:
                return SIGNAL_PHASES["EW_NS_left"]
            else:
                return SIGNAL_PHASES["NS_EW_left"]
        
        elif self.signal_state == "left_yellow":
            if self.current_phase == 0:
                return SIGNAL_PHASES["EW_NS_left_yellow"]
            else:
                return SIGNAL_PHASES["NS_EW_left_yellow"]
                
        elif self.signal_state == "all_red":
            return SIGNAL_PHASES["all_red"]

class TrafficEnvironment:
    def __init__(self, sumo_exe, net_file, route_file, additional_file):
        self.sumo_exe = sumo_exe
        self.net_file = net_file
        self.route_file = route_file
        self.additional_file = additional_file
        self.intersection_ids = INTERSECTION_IDS
        self.intersection_lanes = self._initialize_manual_lanes()
        self.prev_speed = {}
        self.stopped_vehicles = set()
        
        self.controllers = {
            id: GapActuatedController(id) for id in self.intersection_ids
        }

        self.current_step = 0
        self.total_stopped_count = {id: 0 for id in self.intersection_ids}
        self.total_delay = {id: 0 for id in self.intersection_ids}

        self.direction_stopped_count = {
            id: {"east": 0, "west": 0, "north": 0, "south": 0}
            for id in self.intersection_ids
        }
        self.direction_delay = {
            id: {"east": 0, "west": 0, "north": 0, "south": 0}
            for id in self.intersection_ids
        }
    
    def _initialize_manual_lanes(self):
        manual_lanes = {}
        
        for intersection_id in self.intersection_ids:
            if intersection_id == "intersection1":
                manual_lanes[intersection_id] = {
                    "east": [l for lanes in ["EW5", "EW6", "EW7"]
                        for l in manual_lane_data["intersection1"][lanes]],
                    "west": [l for lanes in ["WE2", "WE3", "WE4"]
                        for l in manual_lane_data["intersection1"][lanes]],
                    "north": [l for lanes in ["NS1_4", "NS1_3", "NS1_2"]
                        for l in manual_lane_data["intersection1"][lanes]],
                    "south": [l for lanes in ["SN1_4", "SN1_3", "SN1_2"]
                        for l in manual_lane_data["intersection1"][lanes]]
                }
            else:
                manual_lanes[intersection_id] = {
                    "east": [l for lanes in ["EW0", "EW1", "EW2"]
                        for l in manual_lane_data["intersection2"][lanes]],
                    "west": [l for lanes in ["WE5", "WE6", "WE7", "WE8", "WE9"]
                        for l in manual_lane_data["intersection2"][lanes]],
                    "north": [l for lanes in ["NS2_4", "NS2_3", "NS2_2"]
                        for l in manual_lane_data["intersection2"][lanes]],
                    "south": [l for lanes in ["SN2_4", "SN2_3", "SN2_2"]
                        for l in manual_lane_data["intersection2"][lanes]]
                }
        
        print("手動設定したレーン情報:")
        for tls_id, directions in manual_lanes.items():
            print(f"交差点 {tls_id}:")
            for direction, lanes in directions.items():
                print(f"  {direction}: {lanes}")
    
        return manual_lanes
    
    def start_simulation(self):
        sumo_cmd = [
            self.sumo_exe,
            "-n", self.net_file,
            "-r", self.route_file,
            "-a", self.additional_file,
            "--no-step-log", "true",
            "--step-length", "1",
            "--start",
            "--quit-on-end"
        ]
        traci.start(sumo_cmd)
        print(f"SUMOシミュレーション開始: ギャップ感応制御（右折フェーズ付き）")
        
        for intersection_id in self.intersection_ids:
            traci.trafficlight.setProgram(intersection_id, "0")
    
    def reset(self):
        if traci.isLoaded():
            traci.close()
        self.start_simulation()
        
        self.prev_speed = {}
        self.stopped_vehicles = set()
        self.current_step = 0
        
        self.controllers = {
            id: GapActuatedController(id) for id in self.intersection_ids
        }
        
        self.total_stopped_count = {id: 0 for id in self.intersection_ids}
        self.total_delay = {id: 0 for id in self.intersection_ids}
        self.direction_stopped_count = {
            id: {"east": 0, "west": 0, "north": 0, "south": 0} 
            for id in self.intersection_ids
        }
        self.direction_delay = {
            id: {"east": 0, "west": 0, "north": 0, "south": 0} 
            for id in self.intersection_ids
        }
    
    def get_lane_delay(self, intersection_id, direction):
        stop_threshold = 0.0
        step_length = 1

        if intersection_id not in self.intersection_lanes:
            return 0

        lanes = self.intersection_lanes[intersection_id][direction]
        total_delay = 0

        for lane_id in lanes:
            try:
                vehicle_ids = traci.lane.getLastStepVehicleIDs(lane_id)
                for vehicle_id in vehicle_ids:
                    current_speed = traci.vehicle.getSpeed(vehicle_id)
                    if current_speed == stop_threshold:
                        total_delay += step_length
            except traci.exceptions.TraCIException as e:
                print(f"TraCIエラー (lane_id={lane_id}): {e}")

        return total_delay
    
    def get_lane_stopped_vehicles(self, intersection_id, direction):
        stop_threshold = 0.0
        if intersection_id not in self.intersection_lanes:
            return 0

        lanes = self.intersection_lanes[intersection_id][direction]
        newly_stopped_count = 0

        for lane_id in lanes:
            try:
                vehicle_ids = traci.lane.getLastStepVehicleIDs(lane_id)
                for vehicle_id in vehicle_ids:
                    current_speed = traci.vehicle.getSpeed(vehicle_id)
                    if (current_speed == stop_threshold and 
                        vehicle_id not in self.stopped_vehicles):
                        self.stopped_vehicles.add(vehicle_id)
                        newly_stopped_count += 1
                    self.prev_speed[vehicle_id] = current_speed
            except traci.exceptions.TraCIException as e:
                print(f"TraCIエラー (lane_id={lane_id}): {e}")

        return newly_stopped_count
    
    def update_traffic_lights(self):
        for intersection_id in self.intersection_ids:
            controller = self.controllers[intersection_id]
            
            controller.update_signal_state(self.current_step)
            
            signal_state = controller.get_current_signal_state()
            traci.trafficlight.setRedYellowGreenState(intersection_id, signal_state)
    
    def simulation_step(self, count_metrics=False):
        if traci.simulation.getMinExpectedNumber() <= 0:
            return False
        
        self.update_traffic_lights()

        for intersection_id in self.intersection_ids:
            for direction in ["east", "west", "north", "south"]:
                newly_stopped = self.get_lane_stopped_vehicles(intersection_id, direction)
                current_delay = self.get_lane_delay(intersection_id, direction)
                
                if count_metrics:
                    self.total_stopped_count[intersection_id] += newly_stopped
                    self.total_delay[intersection_id] += current_delay
                    self.direction_stopped_count[intersection_id][direction] += newly_stopped
                    self.direction_delay[intersection_id][direction] += current_delay
        
        traci.simulationStep()
        self.current_step += 1
        return True

def run_gap_control_test():
    env = TrafficEnvironment(SUMO_EXE, SUMO_NET, SUMO_ROUTE, SUMO_ADDITIONAL)
    env.reset()
    
    print("ウォームアップ期間中...")
    while env.current_step < WARMUP_TIME:
        if not env.simulation_step(count_metrics=False):
            print("シミュレーション終了: 車両なし")
            break
        
        if env.current_step % 10 == 0:
            print(f"  経過時間: {env.current_step}秒 / {WARMUP_TIME}秒")
    
    print("\nテスト期間中...")
    while env.current_step < TOTAL_SIMULATION_TIME:
        if not env.simulation_step(count_metrics=True):
            print("シミュレーション終了: 車両なし")
            break
        
        if (env.current_step - WARMUP_TIME) % 100 == 0 and env.current_step > WARMUP_TIME:
            elapsed_test_time = env.current_step - WARMUP_TIME
            print(f"  テスト経過時間: {elapsed_test_time}秒 / {TEST_TIME}秒")
    
    results = {
        "total_stopped_count": env.total_stopped_count.copy(),
        "total_delay": env.total_delay.copy(),
        "direction_stopped_count": env.direction_stopped_count.copy(),
        "direction_delay": env.direction_delay.copy()
    }
    
    if traci.isLoaded():
        traci.close()
    
    print("\n=== テスト結果 ===")
    for intersection_id in INTERSECTION_IDS:
        print(f"\n{intersection_id}:")
        print(f"  総停車台数: {results['total_stopped_count'][intersection_id]}")
        print(f"  総遅れ時間: {results['total_delay'][intersection_id]:.2f} 秒")
        print(f"  方向別停車台数:")
        for direction in ["east", "west", "north", "south"]:
            print(f"    {direction}: {results['direction_stopped_count'][intersection_id][direction]}")
        print(f"  方向別遅れ時間:")
        for direction in ["east", "west", "north", "south"]:
            print(f"    {direction}: {results['direction_delay'][intersection_id][direction]:.2f} 秒")
    
    return results

def save_results(results):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"gap_control_left_turn_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    csv_path = os.path.join(results_dir, "total_results.csv")
    with open(csv_path, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Intersection", "Total Stopped Count", "Total Delay (s)"])
        
        for intersection_id in INTERSECTION_IDS:
            writer.writerow([
                intersection_id,
                results["total_stopped_count"][intersection_id],
                f"{results['total_delay'][intersection_id]:.2f}"
            ])
    
    print(f"\n総計結果をCSVに保存しました: {csv_path}")
    
    detail_csv_path = os.path.join(results_dir, "direction_results.csv")
    with open(detail_csv_path, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Intersection", "Direction", "Stopped Count", "Delay (s)"])
        
        for intersection_id in INTERSECTION_IDS:
            for direction in ["east", "west", "north", "south"]:
                writer.writerow([
                    intersection_id,
                    direction,
                    results["direction_stopped_count"][intersection_id][direction],
                    f"{results['direction_delay'][intersection_id][direction]:.2f}"
                ])
    
    print(f"方向別詳細結果をCSVに保存しました: {detail_csv_path}")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    x = np.arange(len(INTERSECTION_IDS))
    width = 0.6
    
    stopped_counts = [results["total_stopped_count"][id] for id in INTERSECTION_IDS]
    axes[0, 0].bar(x, stopped_counts, width, alpha=0.8, color='steelblue')
    axes[0, 0].set_xlabel('Intersection')
    axes[0, 0].set_ylabel('Total Stopped Count')
    axes[0, 0].set_title('Total Stopped Count by Intersection')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(INTERSECTION_IDS)
    axes[0, 0].grid(True, alpha=0.3)
    
    delays = [results["total_delay"][id] for id in INTERSECTION_IDS]
    axes[0, 1].bar(x, delays, width, alpha=0.8, color='coral')
    axes[0, 1].set_xlabel('Intersection')
    axes[0, 1].set_ylabel('Total Delay (s)')
    axes[0, 1].set_title('Total Delay by Intersection')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(INTERSECTION_IDS)
    axes[0, 1].grid(True, alpha=0.3)
    
    directions = ["east", "west", "north", "south"]
    int1_stopped = [results["direction_stopped_count"]["intersection1"][d] for d in directions]
    x_dir = np.arange(len(directions))
    axes[1, 0].bar(x_dir, int1_stopped, width, alpha=0.8, color='lightgreen')
    axes[1, 0].set_xlabel('Direction')
    axes[1, 0].set_ylabel('Stopped Count')
    axes[1, 0].set_title('Stopped Count by Direction (intersection1)')
    axes[1, 0].set_xticks(x_dir)
    axes[1, 0].set_xticklabels(directions)
    axes[1, 0].grid(True, alpha=0.3)
    
    int1_delay = [results["direction_delay"]["intersection1"][d] for d in directions]
    axes[1, 1].bar(x_dir, int1_delay, width, alpha=0.8, color='plum')
    axes[1, 1].set_xlabel('Direction')
    axes[1, 1].set_ylabel('Delay (s)')
    axes[1, 1].set_title('Delay by Direction (intersection1)')
    axes[1, 1].set_xticks(x_dir)
    axes[1, 1].set_xticklabels(directions)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    graph_path = os.path.join(results_dir, "results_graph.png")
    plt.savefig(graph_path, dpi=150)
    plt.close()
    
    print(f"結果グラフを保存しました: {graph_path}")
    
    return results_dir

def main():
    print("=== ギャップ感応制御評価プログラム（右折フェーズ付き） ===")
    print(f"準備時間: {WARMUP_TIME}秒")
    print(f"テスト時間: {TEST_TIME}秒")
    print(f"総シミュレーション時間: {TOTAL_SIMULATION_TIME}秒")
    print(f"制御方式: ギャップ感応制御（サイクル相殺機能）")
    print(f"信号現示順序:")
    print(f"  1. 青時間（可変: {MIN_GREEN}～{MAX_GREEN}秒、ギャップ: {MAX_GAP}秒）")
    print(f"  2. 黄色時間（固定: {YELLOW_TIME}秒）")
    print(f"  3. 全赤時間（固定: {ALL_RED_TIME}秒）")
    print(f"  4. 右折矢印（固定: {LEFT_ARROW_TIME}秒）")
    print(f"  5. 右折黄色（固定: {YELLOW_TIME}秒）")
    print(f"  6. 全赤時間（固定: {ALL_RED_TIME}秒）\n")
    
    results = run_gap_control_test()
    results_dir = save_results(results)
    
    print(f"\n全ての結果を保存しました: {results_dir}")
    print("\n評価完了")

if __name__ == "__main__":
    main()