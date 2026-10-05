from pathlib import Path
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler

from base_tagger import BaseTagger
from config import JoyCaptionConfig
from utils import _image_to_base64_uri

SYSTEM_PROMPT = "You are a helpful image captioner."

PROMPTS = {
    "Descriptive": "Write a detailed description for this image.",
    "Casual" : 		"Write a descriptive caption for this image in a casual tone.",
    "Straightforward" : 		"Write a straightforward caption for this image. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what's absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with “This image is…” or similar phrasing.",
    "SD" : 		"Output a stable diffusion prompt that is indistinguishable from a real stable diffusion prompt.",
    "MidJourney" : 		"Write a MidJourney prompt for this image.",
    "Danbooru" : 		"Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: `artist:`, `copyright:`, `character:`, `meta:`, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text.",
    "e621" :     		"Write a comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags (if any), prefixed by 'artist:', 'copyright:', 'character:', 'species:', 'meta:', and 'lore:'. Then all the general tags.",
    "Rule34" : 		"Write a comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags (if any), prefixed by 'artist:', 'copyright:', 'character:', and 'meta:'. Then all the general tags.",
}

class JoyCapTagger(BaseTagger):
    def __init__(
        self,
        config: JoyCaptionConfig,
    ):
        self.config = config
        chat_handler = Llava15ChatHandler(clip_model_path=config.mmproj_path)
        self.llm = Llama(
            model_path=config.model_path,
            chat_handler=chat_handler,
            n_ctx=config.n_ctx,
            n_gpu_layers=config.n_gpu_layers,
            verbose=False
        )

    @property
    def name(self) -> str:
        return "JoyCaption Beta One tagger"

    def predict(self, image_path: Path) -> str:
        image_uri = _image_to_base64_uri(image_path)
        response = self.llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_uri}},
                        {"type": "text", "text": PROMPTS.get(self.config.mode)},
                    ]
                }
            ],
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature
        )
        return response["choices"][0]["message"]["content"].strip()

    def cleanup(self) -> None:
        pass