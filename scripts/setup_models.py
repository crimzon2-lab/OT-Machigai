from pathlib import Path
from PIL import Image
from ultralytics import SAM


def main() -> None:
    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)

    sam_path = weights_dir / "sam2.1_b.pt"
    print(f"[1/2] SAM 2 モデルを準備しています: {sam_path}")
    model = SAM(str(sam_path))
    model.info()

    print("[2/2] LaMa モデルを準備しています...")
    try:
        from simple_lama_inpainting import SimpleLama

        lama = SimpleLama()
        dummy_img = Image.new("RGB", (64, 64), (128, 128, 128))
        dummy_mask = Image.new("L", (64, 64), 0)
        _ = lama(dummy_img, dummy_mask)
        print("LaMa モデルのキャッシュが完了しました。")
    except ImportError:
        print(
            "simple-lama-inpainting が未導入です。"
            "消去処理は OpenCV Inpaint にフォールバックします。"
        )

    print("初期セットアップ完了。以降はオフライン環境で実行可能です。")


if __name__ == "__main__":
    main()
