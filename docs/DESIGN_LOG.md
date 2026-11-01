# OT-Machigai 設計経緯・対話ログ

- **記録日**: 2026-10-31
- **対象リポジトリ**: `https://github.com/crimzon2-lab/OT-Machigai`
- **目的**: 外部クラウドAIに依存せず、ローカル環境（macOS / Windows）と PhotoCraft を組み合わせて実用的な「間違い探し」教材・画像を生成するパイプラインの構築。

---

## 1. アーキテクチャ決定事項サマリー（別PC・開発エージェント引継ぎ用）

1. **PhotoCraft と Python の連携方式**
   - `storytold/photocraft` は内部に画像生成AIを持つのではなく、内部エンジン操作を外部公開する設計（MCPサーバー / TCP JSON Control Channel / CLIバッチ実行）。
   - 定型パイプラインではオーバーヘッドの少ない `subprocess` 経由の `photocraft-cli`（`info`, `convert`, `run`, `commands`）呼び出しを採用。
   - アルファ版であり外部PNGを直接レイヤー挿入する内部コマンドIDが公開仕様上未確定なため、PSD構築自体は安定している Python の `psd-tools` で行い、検証・コンポジット書き出し・追加フィルタ処理を `photocraft-cli` に担わせるハイブリッド構成とする。

2. **ローカル間違い探し生成の技術選定**
   - 画像全体を拡散モデル（Stable Diffusion等）で再生成すると非改変領域のピクセルが変化し、正解座標も特定不能になるため不採用。
   - **要素抽出**: `ultralytics` 版 **SAM 2 (`sam2.1_b.pt`)** を採用（Meta公式リポジトリ版のWindows環境でのC++/CUDAビルド問題を回避し、CUDA / Apple Silicon MPS / CPU に単一コードで対応）。
   - **局所改変**:
     - 消去: **LaMa (`simple-lama-inpainting`)**（未導入時はOpenCV Telea Inpaintへ自動フォールバック）。
     - 色変更: **OpenCV HSV色相シフト**（彩度・明度が低い白壁や黒枠は自動でスキップし消去へフォールバック）。
     - 反転: **背景Inpaint補完 + 左右反転アルファブレンド**。

3. **実運用ワークフロー（レイヤー分離PSD出力）**
   - 完全自動抽出では影や境界線の誤抽出が一定確率で混ざるため、一枚絵に焼き込まず「元画像 + 個別間違い透過レイヤー + 非表示の正解赤丸レイヤー」を持つPSDを出力する。
   - 目標数より1〜2個多めに生成（例: `-n 7`）し、PhotoCraft GUI上で不自然なレイヤーを非表示にする運用で手戻りコストを最小化する。

---

## 2. 対話ログ記録

### Q1. PhotoCraft の Python 操作可否について

> **User**: PhotocraftってAIで動かせるって聞いてるけど、Pythonでも動かせる？

**要点整理**:
- **直接操作・自動化が可能**。Rust製のPhotoshopクローン「PhotoCraft」（`storytold/photocraft`）は「すべての内部操作を外部コマンドとして公開する」設計になっており、外部スクリプトからAPI/CLI経由で描画エンジンを操作できる。
- Pythonからの操作ルートは以下の3通り:
  1. **Python MCPクライアント経由 (`photocraft-cli mcp`)**: `mcp` Python SDK 等からヘッドレスMCPサーバーとして起動し、LLMエージェントにツール群を呼び出させる方式。
  2. **JSON Control Channel（TCPソケット / JSON-lines）**: `photocraft --control` や `photocraft-cli serve` が公開するループバックTCPソケットへコマンドID（例: `filter.blur.gaussianBlur`）とJSONパラメータを送る方式。
  3. **`subprocess` による CLIバッチ実行 (`photocraft-cli run` / `convert`)**: 定型バッチ処理で最もシンプルかつ堅牢な方式。

```python
import subprocess
import json

cmd_params = json.dumps({"radius": 8.0})

res = subprocess.run([
    "photocraft-cli", "run", "input.psd",
    "--cmd", "filter.blur.gaussianBlur",
    "--params", cmd_params,
    "--out", "output.png"
], capture_output=True, text=True)
```

---

### Q2. 完全ローカル運用での「間違い探し自動生成」の実現可能性

> **User**: そしたら、とある画像を提示して「間違い探し作って」的なプログラムを作ることはできる？ネット上のAIは使わないでローカル運用できるのか気になる。

**要点整理**:
- 外部APIやクラウドAIを一切使わずローカル環境のみで構築可能。
- ただし、画像全体を生成AIに渡してプロンプト一発で生成させる手法は以下の2点から破綻する:
  1. **非改変箇所の完全一致**: 再生成時に画像全体の線や質感が微妙に変わり、全面が間違いになってしまう。
  2. **正解座標の絶対的な特定**: ピクセル単位でどこを変えたかをプログラム側が保持する必要がある。
- **推奨ローカル構成（レイヤー分離型パイプライン）**:
  - **要素抽出**: SAM 2 (Segment Anything Model 2) — 被写体を自動で輪郭マスク化。
  - **要素消去**: LaMa (Large Mask Inpainting) — 軽量・高速なGANベースInpaintingで自然に消去。
  - **要素改変**: OpenCV / NumPy — マスク内の色相シフト・左右反転などを決定論的に処理。
  - **画像合成・出力**: 元画像のピクセルを100%維持したまま改変パーツのみを透過レイヤーとして重ね、バウンディングボックスから正解赤丸とJSONメタデータを出力。

---

### Q3. PhotoCraft 連携プログラムの実装とマルチPC（GitHub）展開

> **User**: このプログラムをphotocraftを使って作るPythonプログラムを作ってほしい。で、この計画を他のPCでできるように、https://github.com/crimzon2-lab/OT-Machigaiで動かせるようにしてほしい。

**実装仕様**:
- `.gitignore` で大容量モデル重み（`weights/`, `*.pt`）と出力（`output/`, `*.psd`）を除外し、初回のみ `python scripts/setup_models.py` でモデルをキャッシュ、以降は完全オフライン動作とする構成を設計。
- `ot_machigai/photocraft_bridge.py` にて PATH・環境変数 `PHOTOCRAFT_CLI`・`~/.cargo/bin` から `photocraft-cli` を自動検出し、未導入PCでもPythonコンポジットへ安全にフォールバックする仕組みを実装。
- `ot_machigai/psd_builder.py` にて `00_Original_Base`、各間違いレイヤー `01_Diff_*`、およびデフォルト非表示の `99_Answer_Markers` を格納したPSDを出力。あわせて印刷・画面提示用の左右比較画像 `printable_side_by_side.png` を自動生成する仕様とした。
