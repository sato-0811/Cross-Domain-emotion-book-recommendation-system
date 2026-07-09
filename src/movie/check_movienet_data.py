import os

# MovieNetの各フォルダが格納されているルートパスを指定してください
# 例: base_path = "./MovieNet" (現在のディレクトリにある場合)
base_path = "datasets/movienet/datas" 

# 5つのフォルダ名（image_8f847a.pngに示されているもの）
folders = ['annotation', 'keyframes_data', 'meta', 'script', 'subtitle']

# 各フォルダに存在する映画ID（拡張子を除いた名前）を格納するセットのリスト
movie_id_sets = []

for folder in folders:
    folder_path = os.path.join(base_path, folder)
    
    if os.path.exists(folder_path):
        # フォルダ内のファイル名から拡張子を除いた「映画ID」を抽出
        # (例: "tt0111161.json" や "tt0111161/" から "tt0111161" を取得)
        ids = {os.path.splitext(f)[0] for f in os.listdir(folder_path) if not f.startswith('.')}
        movie_id_sets.append(ids)
        print(f"[{folder}] 内の映画数: {len(ids)}")
    else:
        print(f"エラー: フォルダ '{folder}' が見つかりません。パスを確認してください。")
        movie_id_sets.append(set())

# 5つのフォルダすべてに共通して存在する映画IDを抽出 (積集合)
if movie_id_sets:
    common_movies = set.intersection(*movie_id_sets)
    
    print("\n--- 結果 ---")
    print(f"5つのデータがすべて揃っている映画の総数: {len(common_movies)} 本")
    
    # 抽出された映画IDのリストを表示（数が多い場合は上位10件のみ表示）
    common_movies_list = sorted(list(common_movies))
    print("揃っている映画ID（一部）:", common_movies_list[:10])
    
    # テキストファイルに保存したい場合
    with open("common_movies.txt", "w") as f:
        for movie_id in common_movies_list:
            f.write(f"{movie_id}\n")
    print("すべての共通映画IDを 'common_movies.txt' に保存しました。")