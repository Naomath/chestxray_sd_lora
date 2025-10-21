
# 胸部X線ノーマルジェネレーター（Diffusers + LoRA / Full Finetune）

Stable Diffusion の UNet を **LoRA** もしくは **通常のフル微調整**で再学習し、**256×256 の胸部 X 線（正常）画像**を生成する研究用プロジェクトです。テキストエンコーダーと VAE は固定し、`"chest x-ray, normal"` という定常プロンプトで UNet の挙動を最適化します。LoRA の軽量更新をデフォルトとしつつ、必要に応じて UNet 全体の再学習にも切り替えられるようにしました。

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

3. compileとimportが通るかの確認
   ```bash
   uv run python -m compileall src
   uv run python -c "import src.train_lora"
   ```

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
nohup uv run accelerate launch --module src.train_lora \
  --dataset_dir /home/goto/mask_diffusion/data/images/train_normal_256 \
  --model_id runwayml/stable-diffusion-v1-5 \
  --output_dir ./outputs/exp1 \
  --resolution 256 \
  --train_batch_size 16 \
  --grad_accum_steps 1 \
  --mixed_precision bf16 \
  --learning_rate 5e-5 \
  --max_train_steps 20000 \
  --checkpointing_steps 1000 \
  --training_mode lora \
  --lora_rank 16 \
  --lora_alpha 16 \
  --wandb \
  --wandb_project sd_sora_train \
  > train.log 2>&1 &
```

メモ:
- `--training_mode` は `lora`（デフォルト）と `full` を切り替え可能です。`full` を指定すると UNet 全体を微調整します。
- **LoRA** では `./outputs/exp1/lora_unet/` にアテンションプロセッサが保存されます。**フル微調整**では `./outputs/exp1/unet/` に再学習済みの UNet 重みが保存されます。
- **Weights & Biases** を有効化すると、損失やチェックポイント生成画像が自動でアップロードされます。`--wandb_*` フラグでプロジェクト情報を渡せます。
- **H100** など BF16 対応 GPU では `--mixed_precision bf16` が推奨です。GPU メモリに応じて `--train_batch_size` を調整してください。
- `--max_train_steps` の初期値は 20,000 を目安に、データセット規模や目的に応じて調整してください。

### 4) 推論（サンプリング）
#### LoRA で学習した場合
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

#### フル微調整した場合
`src/inference.py` は LoRA を対象にしています。フル微調整した UNet (`./outputs/exp1/unet/`) を利用する場合は、以下のような簡単なスクリプトで UNet の置き換えを行ってください。

```python
import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel

base_model = "runwayml/stable-diffusion-v1-5"
finetuned_unet_dir = "./outputs/exp1/unet"

pipe = StableDiffusionPipeline.from_pretrained(base_model, torch_dtype=torch.float16).to("cuda")
pipe.unet = UNet2DConditionModel.from_pretrained(finetuned_unet_dir, torch_dtype=pipe.unet.dtype).to(pipe.device)
pipe.safety_checker = None

image = pipe("chest x-ray, normal", height=256, width=256).images[0]
image.save("sample_full_ft.png")
```

生成結果は RGB 画像として保存されます。単一チャネルの PNG が必要な場合は `--grayscale_out` を指定するか、保存前に `img.convert('L')` を用いてください。

## uv を使ったパッケージ管理
- 依存関係は `pyproject.toml` にまとめてあります。新しいライブラリが必要になったら `pyproject.toml` の `dependencies` に追記し、`uv sync` を再実行します。
- ロックファイルを更新する場合は `uv lock` を実行してください。
- `requirements.txt` が必要な場面では、`uv pip export --format requirements-txt > requirements.txt` でエクスポートできます。
- 環境変数や実行コマンドを付与したい場合は `uv run -- <command>` 形式で実行できます。

## 設計上のポイント
- **UNet の注意機構のみを LoRA で学習**することで、1 GPU でも効率的に収束させやすい設計です。
- **LoRA / フル微調整を CLI で切り替え**できるため、軽量な探索から高精度な最終学習まで同一コードで運用できます。
- **定常プロンプト**を用いることでドメイン内変動を抑え、テキストエンコーダーを凍結して安定した学習を実現しています。
- **グレースケール対応**は学習時に 3 チャネルへ複製し、推論時には必要に応じて 1 チャネルで出力可能としました。
- **Weights & Biases** のチェックポイントログには各ステップで生成した画像も添付されるため、学習過程を視覚的に監視できます。

## プロジェクト構成
```
chestxray_sd_lora/
  README.md
  pyproject.toml
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
