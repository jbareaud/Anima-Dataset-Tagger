from pathlib import Path
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Qwen25VLChatHandler
from base_tagger import BaseTagger
from config import Qwen3VLConfig
from utils import _image_to_base64_uri

SYSTEM_PROMPT = "You are a helpful image captioner."

OLD_PROMPT = ("""
Write one factual training caption for this image as a single sentence, covering these in order: the subject and what they are doing; the camera viewpoint (e.g. 'viewed from behind', 'side profile', 'close-up') and whether the face is visible; their pose; their clothing; the setting; the lighting. Use the same order and the same plain phrasing every time. Name the subject with the specific term that is visually apparent — 'a woman', 'a man', 'a girl', 'a boy' — and not the vague 'a person'. Use 'a person' only when the image genuinely does not show enough to tell. Begin directly with the subject. Never open with 'This image shows', 'The image depicts', 'In this image', 'Here we see', 'The photo shows' or any similar preamble. Name anything prominent a viewer would notice, especially anything unusual about the angle, the framing, or what is hidden or cropped. State only what is visible — no speculation, no proper names, no style or quality commentary.
"""),

SUBJECT_RULE = (
    "Name the subject with the specific term that is visually apparent — 'a woman', 'a man', "
    "'a girl', 'a boy' — and not the vague 'a person'. Use 'a person' only when the image "
    "genuinely does not show enough to tell. "
)
NO_PREAMBLE_RULE = (
    "Begin directly with the subject. Never open with 'This image shows', 'The image depicts', "
    "'In this image', 'Here we see', 'The photo shows' or any similar preamble. "
)

CAPTION_INSTRUCTION = (
    "Write one factual training caption for this image as a single sentence, covering these in "
    "order: the subject and what they are doing; the camera viewpoint (e.g. 'viewed from behind', "
    "'side profile', 'close-up') and whether the face is visible; their pose; their clothing; the "
    "setting; the lighting. Use the same order and the same plain phrasing every time. "
    + SUBJECT_RULE + NO_PREAMBLE_RULE +
    "Name anything prominent a viewer would notice, especially anything unusual about the angle, "
    "the framing, or what is hidden or cropped. State only what is visible — no speculation, no "
    "proper names, no style or quality commentary."
)

class Qwen3VLTagger(BaseTagger):
    def __init__(
        self,
        config: Qwen3VLConfig,
    ):
        self.config = config
        chat_handler = Qwen25VLChatHandler(clip_model_path=config.mmproj_path)
        self.llm = Llama(
            model_path=config.model_path,
            chat_handler=chat_handler,
            n_ctx=config.n_ctx,
            n_gpu_layers=config.n_gpu_layers,
            verbose=False,
        )

    @property
    def name(self) -> str:
        return "Qwen 3 VL tagger"

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
                        {"type": "text", "text": CAPTION_INSTRUCTION}
                    ]
                }
            ],
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature
        )
        return response["choices"][0]["message"]["content"].strip()

    def cleanup(self) -> None:
        pass