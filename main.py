import mimetypes
from pathlib import Path
from PIL import Image
from base_tagger import BaseTagger

def is_valid_image(path: Path) -> bool:
    mime, _ = mimetypes.guess_type(path)
    if mime and mime.startswith("image/"):
        return True
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False

def append_caption(caption_file: Path, new_content: str) -> None:
    if not new_content.strip():
        return

    separator = ""
    if caption_file.exists() and caption_file.stat().st_size > 0:
        with caption_file.open("r", encoding="utf-8", errors="ignore") as f:
            existing = f.read()
        if existing.strip():
            # 3 lignes blanches de séparation = 4 sauts de ligne
            separator = "\n\n----\n\n"

    with caption_file.open("a", encoding="utf-8") as f:
        f.write(f"{separator}{new_content.strip()}\n")

def run_pipeline(image_dir: Path, taggers: list[BaseTagger]) -> None:
    if not image_dir.is_dir():
        raise NotADirectoryError(f"Directory doesn't exist : {image_dir}")

    # Récupération et tri de tous les fichiers du dossier
    candidates = sorted([p for p in image_dir.iterdir() if p.is_file()])
    images = [p for p in candidates if is_valid_image(p)]

    print(f"[{len(images)} image(s) detected in {image_dir}]")

    for tagger in taggers:
        print(f"\n--- Execution of tagger : {tagger.name} ---")
        for idx, img_path in enumerate(images, start=1):
            print(f"[{idx}/{len(images)}] {img_path.name}...")
            try:
                caption = tagger.predict(img_path)
                txt_path = img_path.with_suffix(".txt")
                append_caption(txt_path, caption)
            except Exception as e:
                print(f"  Error while processing {img_path.name} : {e}")
        tagger.cleanup()
