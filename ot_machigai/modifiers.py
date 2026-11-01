from typing import Optional, Tuple
import cv2
import numpy as np
from PIL import Image


class LocalModifier:
    """マスク領域のみを改変し、元画像と同サイズの透過RGBAパッチを生成するクラス"""

    def __init__(self, use_lama: bool = True) -> None:
        self.lama = None
        if use_lama:
            try:
                from simple_lama_inpainting import SimpleLama

                self.lama = SimpleLama()
            except Exception as e:
                print(f"[WARN] SimpleLama 読み込みスキップ（OpenCVで代替）: {e}")

    def _inpaint_rgb(self, base_rgb: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
        h, w = base_rgb.shape[:2]
        if self.lama is not None:
            res_pil = self.lama(Image.fromarray(base_rgb), Image.fromarray(mask_u8))
            # SimpleLamaが8の倍数にパディングするケースへの対策として元サイズで切り出す
            res_pil = res_pil.crop((0, 0, w, h)).convert("RGB")
            return np.array(res_pil, dtype=np.uint8)
        # フォールバック: OpenCV Telea Inpaint
        bgr = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2BGR)
        inpainted_bgr = cv2.inpaint(bgr, mask_u8, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
        return cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _build_rgba_patch(patch_rgb: np.ndarray, alpha_u8: np.ndarray) -> Image.Image:
        rgba = np.zeros((patch_rgb.shape[0], patch_rgb.shape[1], 4), dtype=np.uint8)
        active = alpha_u8 > 0
        rgba[active, :3] = patch_rgb[active]
        rgba[:, :, 3] = alpha_u8
        return Image.fromarray(rgba, mode="RGBA")

    def erase_object(self, base_rgb: np.ndarray, mask_bool: np.ndarray) -> Image.Image:
        """対象オブジェクトを背景で塗りつぶして消去する透過レイヤーを返す"""
        mask_u8 = (mask_bool.astype(np.uint8)) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        dilated = cv2.dilate(mask_u8, kernel, iterations=1)

        inpainted_rgb = self._inpaint_rgb(base_rgb, dilated)
        feathered_alpha = cv2.GaussianBlur(dilated, (5, 5), 0)
        return self._build_rgba_patch(inpainted_rgb, feathered_alpha)

    def shift_color(
        self, base_rgb: np.ndarray, mask_bool: np.ndarray, hue_shift: int = 60
    ) -> Optional[Image.Image]:
        """対象オブジェクトの色相を回転させる。無彩色に近い場合はNoneを返す"""
        hsv = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2HSV)
        sat_vals = hsv[:, :, 1][mask_bool]
        val_vals = hsv[:, :, 2][mask_bool]

        # 彩度・明度が低すぎる対象（白壁・黒枠・グレーの影など）は色変え不適として弾く
        if sat_vals.size == 0 or float(np.mean(sat_vals)) < 40.0 or float(np.mean(val_vals)) < 40.0:
            return None

        shifted_hsv = hsv.copy()
        shifted_hsv[:, :, 0] = ((shifted_hsv[:, :, 0].astype(np.int16) + hue_shift) % 180).astype(
            np.uint8
        )
        shifted_rgb = cv2.cvtColor(shifted_hsv, cv2.COLOR_HSV2RGB)

        mask_u8 = (mask_bool.astype(np.uint8)) * 255
        feathered_alpha = cv2.GaussianBlur(mask_u8, (3, 3), 0)
        return self._build_rgba_patch(shifted_rgb, feathered_alpha)

    def flip_object(
        self,
        base_rgb: np.ndarray,
        mask_bool: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> Image.Image:
        """背景を一旦補完した上で、対象オブジェクトを左右反転して再配置する"""
        x1, y1, x2, y2 = bbox
        mask_u8 = (mask_bool.astype(np.uint8)) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        dilated = cv2.dilate(mask_u8, kernel, iterations=1)

        # 反転前のシルエットが残らないよう背景を補完
        canvas_rgb = self._inpaint_rgb(base_rgb, dilated)

        obj_rgb = base_rgb[y1:y2, x1:x2][:, ::-1]
        obj_mask = mask_u8[y1:y2, x1:x2][:, ::-1]
        obj_alpha = (cv2.GaussianBlur(obj_mask, (3, 3), 0).astype(np.float32)) / 255.0

        roi_bg = canvas_rgb[y1:y2, x1:x2].astype(np.float32)
        blended_roi = obj_rgb.astype(np.float32) * obj_alpha[..., None] + roi_bg * (
            1.0 - obj_alpha[..., None]
        )
        canvas_rgb[y1:y2, x1:x2] = np.clip(blended_roi, 0, 255).astype(np.uint8)

        union_mask = dilated.copy()
        union_mask[y1:y2, x1:x2] = np.maximum(union_mask[y1:y2, x1:x2], obj_mask)
        feathered_alpha = cv2.GaussianBlur(union_mask, (5, 5), 0)
        return self._build_rgba_patch(canvas_rgb, feathered_alpha)
