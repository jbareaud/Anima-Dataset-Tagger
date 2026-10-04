import base64
import mimetypes
from pathlib import Path

def _image_to_base64_uri(image_path: Path) -> str:
    path = Path(image_path)
    mime_type, _ = mimetypes.guess_type(path)
    if not mime_type:
        mime_type = "image/png"  # safe fallback if extension is unknown
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"