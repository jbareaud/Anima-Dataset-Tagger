from pathlib import Path
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Qwen25VLChatHandler
from base_tagger import BaseTagger
from config import QwenConfig
from utils import _image_to_base64_uri

SYSTEM_PROMPT = "You are a helpful image captioner."

PROMPT = ("""
Analyze the image, and output the following information:
 - sentiment expressed by the subject, like 'angry', 'sad', 'happy', 'ecstasy', etc. 
 - state of the eyes : 'open eyes', 'half-closed eyes', 'closed eyes', 'narrowed eyes', etc.
 - state of the mouth : 'open mouth', 'closed mouth', 'parted lips', 'clenched teeth', 'grin', 'smile', 'light smile', etc.
 - direction of the gaze : pick one of 'averting eyes', 'facing to the side', 'looking ahead', 'looking at viewer', 'looking up', 'looking down', 'looking back', 'shaft look', 'sideways glance', 'turning head'.  
Don't output anything but the requested information.
"""),

class QwenTagger(BaseTagger):
    def __init__(
        self,
        config: QwenConfig
    ):
        self.config = config
        chat_handler = Qwen25VLChatHandler(clip_model_path=self.config.mmproj_path)
        self.llm = Llama(
            model_path=self.config.model_path,
            chat_handler=chat_handler,
            n_ctx=self.config.n_ctx,
            n_gpu_layers=self.config.n_gpu_layers,
            verbose=False
        )

    @property
    def name(self) -> str:
        return "Qwen tagger"

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