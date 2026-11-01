import inspect
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image, ImageDraw
from psd_tools import PSDImage
from psd_tools.api.layers import PixelLayer


def _append_rgba_layer(
    psd: PSDImage, pil_img: Image.Image, name: str, visible: bool = True
) -> None:
    rgba = pil_img.convert("RGBA")
    if hasattr(psd, "create_pixel_layer"):
        sig = inspect.signature(psd.create_pixel_layer)
        if "image" in sig.parameters:
            layer = psd.create_pixel_layer(image=rgba, name=name)
            layer.visible = visible
            return
    layer = PixelLayer.frompil(rgba, psd, name)
    layer.visible = visible
    psd.append(layer)


def build_answer_overlay(
    size: Tuple[int, int], diffs: List[Dict]
) -> Image.Image:
    """正解位置を示す赤丸オーバーレイ（透過RGBA）を作成する"""
    overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    stroke = max(3, min(size) // 180)

    for item in diffs:
        x1, y1, x2, y2 = item["bbox"]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        radius = int(max(x2 - x1, y2 - y1) * 0.65) + 8
        draw.ellipse(
            (cx - radius, cy - radius, cx + radius, cy + radius),
            outline=(230, 30, 30, 255),
            width=stroke,
        )
    return overlay


def save_layered_psd(
    base_pil: Image.Image,
    diff_layers: List[Tuple[str, Image.Image]],
    answer_overlay: Image.Image,
    out_psd_path: Path,
) -> None:
    """元画像・各間違いレイヤー・正解赤丸レイヤーを統合したPSDを保存する"""
    w, h = base_pil.size
    psd = PSDImage.new(mode="RGBA", size=(w, h), depth=8)

    _append_rgba_layer(psd, base_pil, name="00_Original_Base", visible=True)
    for layer_name, patch_img in diff_layers:
        _append_rgba_layer(psd, patch_img, name=layer_name, visible=True)

    # 正解レイヤーは非表示状態で最上位に保持（PhotoCraft上で切り替え可能）
    _append_rgba_layer(psd, answer_overlay, name="99_Answer_Markers", visible=False)
    psd.save(str(out_psd_path))


def composite_images(
    base_pil: Image.Image,
    diff_layers: List[Tuple[str, Image.Image]],
    answer_overlay: Image.Image,
) -> Tuple[Image.Image, Image.Image, Image.Image]:
    """問題画像・正解赤丸付き画像・左右並列プリント用シートを生成する"""
    puzzle = base_pil.convert("RGBA")
    for _, patch in diff_layers:
        puzzle = Image.alpha_composite(puzzle, patch.convert("RGBA"))

    answer = Image.alpha_composite(puzzle, answer_overlay)

    # 印刷・タブレット提示用の左右比較シート（左：元画像 / 右：問題画像）
    w, h = base_pil.size
    gap = max(16, w // 40)
    sheet = Image.new("RGB", (w * 2 + gap * 3, h + gap * 2), (255, 255, 255))
    sheet.paste(base_pil.convert("RGB"), (gap, gap))
    sheet.paste(puzzle.convert("RGB"), (w + gap * 2, gap))

    return puzzle.convert("RGB"), answer.convert("RGB"), sheet
