from pathlib import Path
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Gemma4ChatHandler
from base_tagger import BaseTagger
from config import Gemma4Config
import math
import base64
from io import BytesIO
from PIL import Image

MAX_IMAGE_PIXELS = 1024**2

PROMPT = "Write a descriptive caption for this image."

class Gemma4Tagger(BaseTagger):
    def __init__(
        self,
        config: Gemma4Config,
    ):
        self.config = config
        chat_handler = Gemma4ChatHandler(clip_model_path=config.mmproj_path)
        self.llm = Llama(
            model_path=config.model_path,
            chat_handler=chat_handler,
            n_ctx=config.n_ctx,
            n_gpu_layers=config.n_gpu_layers,
            verbose=False,
        )

    @property
    def name(self) -> str:
        return "Gemma 4 tagger"

    def rgba_to_rgb(self, image: Image.Image, background_color=(255, 255, 255)) -> Image.Image:
        """Convert an RGBA image to RGB with filled background color."""
        assert image.mode == "RGBA"
        converted = Image.new("RGB", image.size, background_color)
        converted.paste(image, mask=image.split()[3])  # 3 is the alpha channel
        return converted

    def convert_image_mode(self, image: Image.Image, to_mode: str):
        if image.mode == to_mode:
            return image
        elif image.mode == "RGBA" and to_mode == "RGB":
            return self.rgba_to_rgb(image)
        else:
            return image.convert(to_mode)

    def limit_size(self, img):
        w, h = img.size
        num_pixels = w * h
        if num_pixels > MAX_IMAGE_PIXELS:
            scale = math.sqrt(num_pixels / MAX_IMAGE_PIXELS)
            img.thumbnail((int(w / scale), int(h / scale)))
        return img

    def predict(self, image_path: Path) -> str:
        img = Image.open(image_path)
        image_pil = self.convert_image_mode(self.limit_size(img), 'RGB')

        buffered = BytesIO()
        image_pil.save(buffered, format="JPEG", quality=90)
        b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        image_uri = f"data:image/jpeg;base64,{b64_str}"

        response = self.llm.create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_uri}},
                        {"type": "text", "text": PROMPT}
                    ]
                }
            ],
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature
        )
        return response["choices"][0]["message"]["content"].strip()

    def cleanup(self) -> None:
        pass