import os

# SUMOの tools フォルダのパス（変更する）
sumo_tools = r"C:\Users\takut\Desktop\町モデル\tools"
random_trips = os.path.join(sumo_tools, "randomTrips.py")

# 町モデルのネットワークファイル
network_file = r"C:\Users\takut\Desktop\町モデル\toyama_shouwa.net.xml"

# ルートファイルの出力先
output_folder = r"C:\Users\takut\Desktop\町モデル\output"
os.makedirs(output_folder, exist_ok=True)  # フォルダがない場合は作成

# ルートファイルのパス
route_file = os.path.join(output_folder, "toyama_shouwa.rou.xml")

# 端から端のルートを作成（`--fringe-factor` を使用）
command = f'python "{random_trips}" -n "{network_file}" -r "{route_file}" -e 3600 --fringe-factor 10 -l'
os.system(command)

print(f"端から端のルートファイルが {route_file} に出力されました！")
