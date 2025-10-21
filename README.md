
# 胸部X線ノーマルジェネレーター（Diffusers + LoRA）

Stable Diffusion の UNet を LoRA で微調整し、**256×256 の胸部 X 線（正常）画像**を生成する研究用プロジェクトです。テキストエンコーダーと VAE は固定し、`"chest x-ray, normal"` という定常プロンプトで UNet の LoRA モジュールのみを学習します。

## クイックスタート

### 0) uv と Python 環境の準備
1. uv のインストール  
   ```bash
   curl -Ls https://astral.sh/uv/install.sh | sh
   # Windows の場合は PowerShell 用インストーラを参照してください
   ```
2. プロジェクトルートで依存関係を同期  
   ```bash
   uv sync
   ```
   `pyproject.toml` に定義された依存関係が `.venv/` にインストールされ、必要に応じて `uv.lock` が生成されます。

### 1) データセットの用意
約 20,000 枚の胸部 X 線 **正常**画像を以下のようなディレクトリに配置します。
```
/path/to/dataset/
  000001.png
  000002.png
  ...
```
- png/jpg/jpeg など一般的なフォーマットに対応しています。
- 画像はセンタークロップとリサイズにより **256×256** に変換されます。
- グレースケール画像は自動的に 3 チャンネルへ複製され、Stable Diffusion 互換の RGB として扱われます。

### 2) Accelerate の設定（任意）
```bash
uv run accelerate config default
```

### 3) 学習（単一 GPU 想定）
```bash
uv run accelerate launch src/train_lora.py \
  --dataset_dir /path/to/dataset \
  --model_id runwayml/stable-diffusion-v1-5 \
  --output_dir ./outputs/exp1 \
  --resolution 256 \
  --train_batch_size 16 \
  --grad_accum_steps 1 \
  --mixed_precision bf16 \
  --learning_rate 5e-5 \
  --max_train_steps 20000 \
  --lora_rank 16 \
  --lora_alpha 16 \
  --checkpointing_steps 1000
```

メモ:
- **H100** 環境では `--mixed_precision bf16` が推奨です。
- GPU メモリに応じて `--train_batch_size` を調整してください。LoRA を用いることでメモリ効率が高く保たれます。
- `--max_train_steps` を 20,000 程度から開始するのが目安ですが、データセットや目的に応じて増減できます。
- **Weights & Biases** でログを取りたい場合は、`config/defaults.yaml` で `wandb_enabled: true` にするか、起動時に `--wandb --wandb_project プロジェクト名` を付与してください。`--wandb_entity`、`--wandb_group`、`--wandb_tags` も使用できます。

### 4) 推論（サンプリング）
```bash
uv run python src/inference.py \
  --base_model runwayml/stable-diffusion-v1-5 \
  --lora_path ./outputs/exp1/lora_unet \
  --out_dir ./samples \
  --num_images 8 \
  --height 256 \
  --width 256 \
  --prompt "chest x-ray, normal"
```

生成結果は RGB 画像として保存されます。単一チャネルの PNG が必要な場合は `--grayscale_out` を指定してください。

## uv を使ったパッケージ管理
- 依存関係は `pyproject.toml` にまとめてあります。新しいライブラリが必要になったら `pyproject.toml` の `dependencies` に追記し、`uv sync` を再実行します。
- ロックファイルを更新する場合は `uv lock` を実行してください。
- 既存の `requirements.txt` は uv からエクスポートする際の参考として残しています。必要に応じて `uv pip export --format requirements-txt > requirements.txt` を利用してください。
- 環境変数や実行コマンドを付与したい場合は `uv run -- <command>` 形式で実行できます。

## 設計上のポイント
- **UNet の注意機構のみを LoRA で学習**することで、1 GPU でも効率的に収束させやすい設計です。
- **定常プロンプト**を用いることでドメイン内変動を抑え、テキストエンコーダーを凍結して安定した学習を実現しています。
- **グレースケール対応**は学習時に 3 チャネルへ複製し、推論時には必要に応じて 1 チャネルで出力可能としました。

## プロジェクト構成
```
chestxray_sd_lora/
  README.md
  pyproject.toml
  requirements.txt
  config/
    defaults.yaml
  src/
    dataset.py
    utils.py
    train_lora.py
    inference.py
```

---

**注意**: データセットを用いた生成モデルの学習には各種ライセンス・倫理審査が必要となる場合があります。研究機関や施設のガイドラインに従って運用してください。
