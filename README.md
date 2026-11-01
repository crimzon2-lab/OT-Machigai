# OT-Machigai

ローカルAI（SAM 2 + LaMa）と PhotoCraft を組み合わせ、外部クラウドAPIを一切使わずに「間違い探し」の問題・解答・レイヤー分離済みPSDを生成するツールです。

設計経緯および初期構想の対話ログは [`docs/DESIGN_LOG.md`](docs/DESIGN_LOG.md) を参照してください。

## 特徴
- **完全ローカル動作**: 初回セットアップでモデル重みを取得した後はオフラインで稼働します。
- **非改変領域の完全維持**: 変更箇所以外は元画像のピクセルを100%維持します。
- **PhotoCraft での手直し前提の設計**: 自動生成で不自然な箇所が出た場合、出力された `*_project.psd` を PhotoCraft で開き、該当レイヤーを非表示にするだけで調整が完了します。

## ディレクトリ構成
```text
OT-Machigai/
├── .gitignore
├── README.md
├── requirements.txt
├── docs/
│   └── DESIGN_LOG.md            # 初期構想・アーキテクチャ選定の対話記録
├── scripts/
│   └── setup_models.py          # 初回のみモデル重みをローカル保存（以降はオフライン稼働）
└── ot_machigai/
    ├── __init__.py
    ├── modifiers.py             # 局所改変処理（LaMa消去・色相変換・背景補完付き反転）
    ├── psd_builder.py           # レイヤー分離PSD構築・比較シート画像生成
    ├── photocraft_bridge.py     # photocraft-cli 連携ラッパー（info / convert / run / commands）
    └── cli.py                   # CLIエントリポイント
```

## 別PCでのセットアップ手順

### 1. リポジトリの取得と仮想環境作成
```bash
git clone https://github.com/crimzon2-lab/OT-Machigai.git
cd OT-Machigai
python3 -m venv .venv
```

- **macOS / Linux**: `source .venv/bin/activate`
- **Windows (PowerShell)**: `.\.venv\Scripts\Activate.ps1`

### 2. 依存パッケージの導入
Windows（NVIDIA GPU搭載機）の場合は、先にCUDA対応版PyTorchを入れてから残りを導入します（MacやCPU環境はそのまま `pip install -r requirements.txt` のみで動作します）。

```bash
# Windows (CUDA 12.4 の例) のみ先に実行
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 共通パッケージ導入
pip install -r requirements.txt

# LaMa消去モデルの導入（PyTorchのダウングレードを防ぐため --no-deps を付与）
pip install simple-lama-inpainting --no-deps
```

### 3. PhotoCraft CLI の配置
[storytold/photocraft の Releases](https://github.com/storytold/photocraft/releases) から OS に合った `photocraft-cli` を取得して PATH を通すか、環境変数 `PHOTOCRAFT_CLI` に実行ファイルのパスを設定します（未導入のPCでもPSD生成および画像出力自体は動作します）。

### 4. 初回モデルキャッシュ（オンライン時に1回のみ実行）
```bash
python scripts/setup_models.py
```

## 使い方

```bash
# 基本実行（間違いを6個生成し、PhotoCraftで手直しして5個残す運用を推奨）
python -m ot_machigai.cli sample.jpg -n 6 -o output

# インストール済みPhotoCraftの内部コマンド一覧を確認する場合
python -m ot_machigai.cli --list-commands layer
```

## 出力ファイル（`output/<画像名>/`）
- `<画像名>_project.psd`: PhotoCraft でレイヤーごとにON/OFF・微調整できるマスターファイル
- `original.png`: 元画像
- `puzzle.png`: 問題画像（PhotoCraftコンポジタ出力）
- `answer.png`: 正解赤丸付き画像
- `printable_side_by_side.png`: 左右比較シート（A4横印刷・画面提示用）
- `metadata.json`: 間違い座標・改変種別の記録データ
