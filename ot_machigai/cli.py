import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
from PIL import Image

from ot_machigai.modifiers import LocalModifier
from ot_machigai.photocraft_bridge import PhotoCraftBridge
from ot_machigai.psd_builder import (
    build_answer_overlay,
    composite_images,
    save_layered_psd,
)


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def select_candidate_masks(
    masks_np: np.ndarray,
    boxes_np: np.ndarray,
    img_shape: Tuple[int, int],
    max_count: int,
    rng: random.Random,
) -> List[Dict]:
    """極端に小さいノイズや背景全体を除外し、位置が重ならない候補を選出する"""
    h, w = img_shape
    total_area = float(h * w)
    min_dist = min(h, w) * 0.12
    valid = []

    for idx in range(len(masks_np)):
        mask_bool = masks_np[idx].astype(bool)
        area_ratio = float(np.sum(mask_bool)) / total_area
        # 画像全体の 0.3%〜10% のサイズに収まるオブジェクトのみ対象にする
        if not (0.003 <= area_ratio <= 0.10):
            continue

        x1, y1, x2, y2 = [int(v) for v in boxes_np[idx]]
        bw, bh = x2 - x1, y2 - y1
        if bw < 16 or bh < 16:
            continue
        # 極端に細長い境界線ノイズを除外
        aspect = max(bw / bh, bh / bw)
        if aspect > 5.0:
            continue

        valid.append(
            {
                "mask": mask_bool,
                "bbox": (x1, y1, x2, y2),
                "center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                "area_ratio": round(area_ratio, 5),
            }
        )

    rng.shuffle(valid)
    selected: List[Dict] = []
    for cand in valid:
        cx, cy = cand["center"]
        too_close = any(
            ((cx - s["center"][0]) ** 2 + (cy - s["center"][1]) ** 2) ** 0.5 < min_dist
            for s in selected
        )
        if not too_close:
            selected.append(cand)
        if len(selected) >= max_count:
            break

    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ローカルAI + PhotoCraft レイヤー分離型 間違い探し生成ツール"
    )
    parser.add_argument("input", nargs="?", help="入力画像パス (PNG/JPG)")
    parser.add_argument("-o", "--out-dir", default="output", help="出力先ディレクトリ")
    parser.add_argument("-n", "--count", type=int, default=5, help="生成する間違いの数（推奨: 3〜7）")
    parser.add_argument("--seed", type=int, default=42, help="乱数シード")
    parser.add_argument("--sam-model", default="weights/sam2.1_b.pt", help="SAM 2 重みパス")
    parser.add_argument("--photocraft-cli", default=None, help="photocraft-cli 実行ファイルのパス")
    parser.add_argument(
        "--list-commands",
        metavar="KEYWORD",
        nargs="?",
        const="",
        help="PhotoCraftの内部コマンド一覧を検索して終了",
    )
    args = parser.parse_args()

    bridge = PhotoCraftBridge(args.photocraft_cli)

    if args.list_commands is not None:
        kw = args.list_commands if args.list_commands != "" else None
        print(bridge.search_registry(kw))
        return

    if not args.input:
        parser.error("入力画像パスを指定してください。")

    in_path = Path(args.input).resolve()
    if not in_path.is_file():
        raise FileNotFoundError(f"入力画像が見つかりません: {in_path}")

    out_dir = Path(args.out_dir).resolve() / in_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    device = detect_device()
    print(f"[INFO] 推論デバイス: {device}")

    base_pil = Image.open(in_path).convert("RGB")
    base_rgb = np.array(base_pil, dtype=np.uint8)
    h, w = base_rgb.shape[:2]

    # 1. SAM 2 による自動セグメンテーション
    from ultralytics import SAM

    sam_weight = Path(args.sam_model)
    sam_weight.parent.mkdir(parents=True, exist_ok=True)
    sam = SAM(str(sam_weight))
    results = sam(str(in_path), device=device, verbose=False)

    if not results or results[0].masks is None:
        raise RuntimeError("画像からセグメンテーション候補を抽出できませんでした。")

    masks_np = results[0].masks.data.cpu().numpy()
    boxes_np = results[0].boxes.xyxy.cpu().numpy()

    candidates = select_candidate_masks(
        masks_np, boxes_np, (h, w), max_count=args.count, rng=rng
    )
    if not candidates:
        raise RuntimeError("条件を満たすサイズの間違い候補が見つかりませんでした。")

    # 2. 各候補に対する局所改変レイヤー作成
    modifier = LocalModifier(use_lama=True)
    diff_layers: List[Tuple[str, Image.Image]] = []
    metadata_diffs: List[Dict] = []
    modes = ["erase", "color", "flip"]

    for idx, cand in enumerate(candidates, start=1):
        chosen_mode = rng.choice(modes)
        mask_bool = cand["mask"]
        bbox = cand["bbox"]
        patch_img = None

        if chosen_mode == "color":
            patch_img = modifier.shift_color(base_rgb, mask_bool, hue_shift=rng.choice([50, 75, 110]))
            if patch_img is None:
                chosen_mode = "erase"

        if chosen_mode == "erase":
            patch_img = modifier.erase_object(base_rgb, mask_bool)
        elif chosen_mode == "flip":
            patch_img = modifier.flip_object(base_rgb, mask_bool, bbox)

        layer_name = f"{idx:02d}_Diff_{chosen_mode}"
        diff_layers.append((layer_name, patch_img))
        metadata_diffs.append(
            {
                "id": idx,
                "layer_name": layer_name,
                "mode": chosen_mode,
                "bbox": list(bbox),
                "area_ratio": cand["area_ratio"],
            }
        )

    # 3. PSD出力と確認用画像の書き出し
    answer_overlay = build_answer_overlay((w, h), metadata_diffs)
    psd_path = out_dir / f"{in_path.stem}_project.psd"
    save_layered_psd(base_pil, diff_layers, answer_overlay, psd_path)

    puzzle_img, answer_img, sheet_img = composite_images(base_pil, diff_layers, answer_overlay)
    base_pil.save(out_dir / "original.png")
    answer_img.save(out_dir / "answer.png")
    sheet_img.save(out_dir / "printable_side_by_side.png")

    # 4. PhotoCraft 連携によるレンダリングと検証
    puzzle_out = out_dir / "puzzle.png"
    if bridge.is_available:
        print(f"[INFO] photocraft-cli を検出: {bridge.cli_bin}")
        doc_info = bridge.inspect_document(psd_path)
        rendered = bridge.render_or_run(psd_path, puzzle_out)
        if not rendered:
            puzzle_img.save(puzzle_out)
    else:
        print("[INFO] photocraft-cli 未検出のため Python コンポジットで puzzle.png を出力します。")
        doc_info = None
        puzzle_img.save(puzzle_out)

    meta = {
        "source": in_path.name,
        "size": [w, h],
        "seed": args.seed,
        "device": device,
        "photocraft_used": bridge.is_available,
        "photocraft_info": doc_info,
        "differences": metadata_diffs,
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[DONE] 生成数: {len(metadata_diffs)}箇所 -> 出力先: {out_dir}")
    print(f"       手直し用PSD: {psd_path}")


if __name__ == "__main__":
    main()
