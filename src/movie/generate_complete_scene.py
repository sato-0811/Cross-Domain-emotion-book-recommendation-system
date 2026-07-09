import os
import json
import re
from datetime import datetime

# パス定義
COMMON_MOVIES_PATH = "src/movie/common_movies.txt"
DATA_BASE_DIR = "datasets/movienet/datas"
OUTPUT_DIR = "datasets/movienet/scene_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def srt_time_to_seconds(time_str):
    try:
        time_str = time_str.replace(',', '.')
        t = datetime.strptime(time_str, "%H:%M:%S.%f")
        return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1000000.0
    except ValueError:
        return 0.0

def load_script_content(script_path):
    """ .scriptファイルを読み込み、行ごとのリストを返す """
    if not os.path.exists(script_path):
        return []
    with open(script_path, 'r', encoding='utf-8', errors='ignore') as f:
        return [line.rstrip() for line in f]

def find_situation_before_dialog(script_lines, dialog_text):
    """ .script 内でセリフを検索し、その直前にあるト書き(状況説明)を大まかに抽出する """
    cleaned_dialog = dialog_text.lower().replace(" ", "").replace("'", "").replace(".", "")
    
    for idx, line in enumerate(script_lines):
        cleaned_line = line.lower().replace(" ", "").replace("'", "").replace(".", "")
        if cleaned_dialog in cleaned_line and len(cleaned_dialog) > 3:
            # セリフが見つかったら、そこから上に向かってト書き（空行ではないインデントの浅い行など）を探す
            descriptions = []
            for check_idx in range(max(0, idx - 5), idx):
                current_check = script_lines[check_idx].strip()
                # 完全にセリフではないト書きっぽい行（大文字始まりや、MS/LSなどのカメラワーク記述）
                if current_check and not current_check.isupper() and len(current_check) > 10:
                    descriptions.append(current_check)
            if descriptions:
                return " ".join(descriptions)
    return None

# メイン処理開始
with open(COMMON_MOVIES_PATH, "r") as f:
    common_movies = [line.strip() for line in f if line.strip()]

for movie_id in common_movies:
    print(f"Building final integrated JSON for: {movie_id}")
    
    # 各種ファイルパス
    srt_path = os.path.join(DATA_BASE_DIR, "subtitle", f"{movie_id}.srt")
    anno_path = os.path.join(DATA_BASE_DIR, "annotation", f"{movie_id}.json")
    script_path = os.path.join(DATA_BASE_DIR, "script", f"{movie_id}.script")  # .scriptに対応！
    
    # 1. 各種データの事前読み込み
    script_lines = load_script_content(script_path)
    
    with open(anno_path, "r", encoding="utf-8") as f:
        anno_data = json.load(f)
        
    # 字幕の読み込み（3分窓、30秒オーバーラップ）
    scenes_dict = {}
    window_size = 180.0
    overlap = 30.0
    step_size = window_size - overlap
    
    if os.path.exists(srt_path):
        with open(srt_path, 'r', encoding='utf-8', errors='ignore') as f:
            blocks = f.read().strip().split('\n\n')
            
        for block in blocks:
            lines = block.split('\n')
            if len(lines) >= 3:
                time_match = re.search(r'(\d+:\d+:\d+,\d+)\s+-->\s+(\d+:\d+:\d+,\d+)', lines[1])
                if time_match:
                    start_sec = srt_time_to_seconds(time_match.group(1))
                    end_sec = srt_time_to_seconds(time_match.group(2))
                    text = " ".join(line.strip() for line in lines[2:] if line.strip())
                    
                    # 窓判定
                    for i in range(150): # 映画の長さに応じてループ
                        scene_start = i * step_size
                        scene_end = scene_start + window_size
                        
                        if scene_start <= start_sec < scene_end:
                            scene_id = i + 1
                            
                            if scene_id not in scenes_dict:
                                scenes_dict[scene_id] = {
                                    "scene_id": scene_id,
                                    "time_range": f"{int(scene_start)}s - {int(scene_end)}s",
                                    "start_second": scene_start,
                                    "end_second": scene_end,
                                    "situation_descriptions": [],
                                    "dialogs": [],
                                    "visual_characters": []
                                }
                            
                            # セリフの追加
                            scenes_dict[scene_id]["dialogs"].append(text)
                            
                            # 【scriptから状況説明を引っ張って合流させる】
                            situation = find_situation_before_dialog(script_lines, text)
                            if situation and situation not in scenes_dict[scene_id]["situation_descriptions"]:
                                scenes_dict[scene_id]["situation_descriptions"].append(situation)
                        
                        if start_sec < scene_start:
                            break

    # 2. annotation ("cast") からキャラクター情報のマッピング
    # 30ショット = 1シーン (3分) のロジックで流し込む
    cast_list = anno_data.get("cast")
    if cast_list is not None:  # ← ここで None じゃない場合だけ処理するようにガードします
        for cast in cast_list:
            shot_idx = cast.get("shot_idx")
            if shot_idx is None:
                continue
                
            # 30ショット = 1シーンの計算ルール
            scene_id = (shot_idx // 30) + 1
            
            # もし字幕がなくてシーンが初期化されていなければスキップ、または作成
            if scene_id not in scenes_dict:
                scene_start = (scene_id - 1) * step_size
                scene_end = scene_start + window_size
                scenes_dict[scene_id] = {
                    "scene_id": scene_id,
                    "time_range": f"{int(scene_start)}s - {int(scene_end)}s",
                    "start_second": scene_start,
                    "end_second": scene_end,
                    "situation_descriptions": ["No dialogue / Situation missing"],
                    "dialogs": [],
                    "visual_characters": []
                }
                
            # visual_characters キーの中に指定通りのフォーマットで格納
            #
            # 重要:
            #   MovieNet の body.bbox は「顔」ではなく「全身」の座標です。
            #   また bbox は annotation の resolution 基準なので、240P keyframe に
            #   合わせて縮尺変換するため resolution も一緒に保存します。
            scenes_dict[scene_id]["visual_characters"].append({
                "shot_idx": shot_idx,
                "img_idx": cast.get("img_idx"),
                "pid": cast.get("pid"),
                "bbox": cast.get("body", {}).get("bbox") if cast.get("body") else None,
                "bbox_type": "body",
                "resolution": cast.get("resolution")
            })

    # 3. リスト形式に綺麗に並び替えてJSON保存
    output_data = []
    for s_id in sorted(scenes_dict.keys()):
        # もし状況説明が空っぽだったらデフォルトテキストを入れておく
        if not scenes_dict[s_id]["situation_descriptions"]:
            scenes_dict[s_id]["situation_descriptions"].append("No situation description matched from script.")
            
        output_data.append(scenes_dict[s_id])
        
    # 最終統合ファイルの書き出し（例: tt0032138_integrated.json）
    output_file_path = os.path.join(OUTPUT_DIR, f"{movie_id}_integrated.json")
    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

print("\n【完了】すべてのモーダルを統合したシーンJSONファイルが完成しました")
