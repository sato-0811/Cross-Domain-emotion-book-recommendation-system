# Cross-Domain Emotion Book Recommendation System

映画と本をつなげて、映画を入れたら原作本を推薦するための実験用リポジトリです。

このプロジェクトでは、映画と本をそれぞれ別々に埋め込み化してから、最終的に Siamese Network で「映画 → 原作本」の対応を学習します。

## 何をやっているか

ざっくり言うと、次の流れです。

1. 映画から3種類の特徴を作る
   - 顔の表情ベクトル
   - 状況説明テキストのベクトル
   - 台詞・字幕テキストのベクトル
2. それらを結合して、シーンごとの映画ベクトルを作る
3. 本をスライディングウィンドウで分割して、ブロックごとの本ベクトルを作る
4. 映画と本を対応づけて疑似ラベルを作る
5. Siamese Network で「映画に近い原作本」を学習する
6. テストデータで、原作本を推薦できるか評価する

## ディレクトリ構成

```text
src/
  movie/   映画側の特徴量作成
  book/    本側の特徴量作成と映画→本マッチング
  fusion/  映画本データセット作成と Siamese 学習

datasets/
  movienet/        MovieNet 関連データと中間生成物
  pg19/            本の元テキスト
  pg19_embeddings/ 本の埋め込み
  book_movie_dataset/  映画→本の教師データ
  siamese_runs/    Siamese 学習結果
```

## 前提

- Python 3.10+ を想定
- PyTorch
- Hugging Face Transformers
- MovieNet の関連データ
- PG19 または Hugging Face の `emozilla/pg19`

GPU があるなら使えます。`--device cuda` を指定してください。

## インストール

必要な依存は用途ごとに分けています。

```bash
pip install -r src/movie/requirements-face.txt
pip install -r src/movie/requirements-script.txt
pip install -r src/movie/requirements-subtitle.txt
pip install -r src/book/requirements-book.txt
pip install -r src/book/requirements-match.txt
```

学習系は追加で `torch` が必要です。

## 映画側の特徴量作成

### 1. 表情ベクトル

映画のキーフレームから顔を検出し、感情ベクトルを作ります。

```bash
python3 src/movie/make_face_vectors.py --movie-id tt0032138 --device cuda
```

出力先:

```text
datasets/movienet/face_vectors/
```

### 2. 状況説明ベクトル

シーンの script テキストを埋め込み化します。

```bash
python3 src/movie/make_script_embeddings.py --device cuda
```

出力先:

```text
datasets/movienet/script_embeddings/
```

### 3. 字幕ベクトル

字幕をシーン単位で埋め込み化します。

```bash
python3 src/movie/make_subtitle_embeddings.py --device cuda
```

出力先:

```text
datasets/movienet/subtitle_embeddings/
```

### 4. シーン融合ベクトル

表情 + script + subtitle を結合して、映画のシーンベクトルを作ります。

```bash
python3 src/movie/build_scene_fusion.py
```

出力先:

```text
datasets/movienet/scene_fusion/
```

ここには各映画の

- `*_scene_matrix.npy`
- `*_movie_vector.npy`

などが保存されます。

## 本側の特徴量作成

### PG19 の埋め込み

本を 1000 文字ずつ、前後 100 文字重複で区切って埋め込み化します。

```bash
python3 src/book/make_book_embeddings.py --device cuda
```

必要なら Hugging Face のデータセットも使えます。

```bash
python3 src/book/make_book_embeddings.py --dataset-id emozilla/pg19 --device cuda
```

出力先:

```text
datasets/pg19_embeddings/
```

ここには以下が出ます。

- `*_block_embeddings.npy`
- `*_block_valid_mask.npy`
- `*_block_spans.json`
- `*_block_texts.json`
- `*_book_vector.npy`
- `*_book_metadata.json`

## 映画→本のマッチング

映画1本に対して候補本をランキングし、教師ラベルを作ります。

```bash
python3 src/fusion/build_movie_book_dataset.py --device cuda
```

出力先:

```text
datasets/book_movie_dataset/
```

生成されるファイル:

- `movie_book_labels.jsonl`
- `train.jsonl`
- `val.jsonl`
- `test.jsonl`
- `manifest.json`

## Siamese Network 学習

映画ベクトルと本ベクトルを共通空間に写して、原作本を推薦するモデルを学習します。

```bash
python3 src/fusion/train_siamese_movie_book.py --epochs 20 --batch-size 32 --device cuda
```

ハードネガティブを強めるなら:

```bash
python3 src/fusion/train_siamese_movie_book.py \
  --epochs 20 \
  --batch-size 32 \
  --device cuda \
  --hard-negative-sample-size 10000
```

出力先:

```text
datasets/siamese_runs/
```

保存物:

- `siamese_state.pt`
- `siamese_best_state.pt`
- `siamese_metrics.json`

## 評価の見方

このタスクは「映画を入れたら原作本を上位に出せるか」を見るので、分類精度よりランキング指標が重要です。

主な指標:

- `Recall@1` : 原作本が1位に来た割合
- `Recall@5` : 原作本が上位5件に入った割合
- `MRR` : 原作本の順位の良さ

実験では、`best_epoch` のモデルを使うのが基本です。

## いまの実装の特徴

- 顔・script・字幕を分けて扱う
- 本はスライディングウィンドウで区切る
- movie / book のベクトルは事前計算して再利用する
- Siamese 学習は PyTorch で実装
- hard negative mining を入れている
- `Recall@1 / Recall@5 / MRR` を評価する

## 注意点

- `datasets/` は大きいので Git には入れない
- このリポジトリは「再現用コード」と「中間生成コード」が主
- GPU はこの環境から直接見えない場合があるので、手元の端末で `--device cuda` を使ってください
- `main` ブランチへの push は SSH 鍵登録後に行っています

## 参考

- MovieNet
- PG19
- NarrativeQA
- Siamese Network

