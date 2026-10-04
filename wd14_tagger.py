from pathlib import Path
from PIL import Image
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
import timm
from timm.data import create_transform, resolve_data_config
from huggingface_hub import hf_hub_download, login
from huggingface_hub.utils import HfHubHTTPError
from base_tagger import BaseTagger

MODEL_REPO_MAP = {
    "vit": "SmilingWolf/wd-vit-tagger-v3",
    "swinv2": "SmilingWolf/wd-swinv2-tagger-v3",
    "convnext": "SmilingWolf/wd-convnext-tagger-v3",
    "eva02": "SmilingWolf/wd-eva02-large-tagger-v3",
}

GENERAL_THRESHOLD = 0.2
CHARACTER_THRESHOLD = 0.8

@dataclass
class LabelData:
    names: list[str]
    rating: list[np.int64]
    general: list[np.int64]
    character: list[np.int64]

class WD14Tagger(BaseTagger):
    def __init__(
            self,
            model_name: str = "vit",
            general_threshold = GENERAL_THRESHOLD,
            character_threshold = CHARACTER_THRESHOLD,
            device: str | None = None,
            prefixes: list[str] | None = None,
            forbidden_tags: set[str] | None = None, # set of Danbooru tags, with underscores
    ):
        login()
        if prefixes is None:
            prefixes = [""]
        print(f"Using model {model_name}")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.general_threshold = general_threshold
        self.character_threshold = character_threshold
        self.prefixes = prefixes
        self.forbidden_tags = forbidden_tags

        repo_id = MODEL_REPO_MAP.get(model_name, None)
        if repo_id is None:
            raise AssertionError(f"model {repo_id} not found")

        print(f"Loading model '{model_name}' from '{repo_id}'...")
        self.model: nn.Module = timm.create_model("hf-hub:" + repo_id).eval()
        state_dict = timm.models.load_state_dict_from_hf(repo_id)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()

        # Prétraitements standards recommandés par timm pour ce modèle
        data_config = timm.data.resolve_model_data_config(self.model)
        self.transform = timm.data.create_transform(**data_config, is_training=False)

        # Chargement des labels
        #self.tags = self._load_tags(model_repo)
        self.labels = self._load_labels_hf(repo_id)

    @property
    def name(self) -> str:
        return "Smiling Wolf WD tagger"

    def _load_labels_hf(self, repo_id: str) -> LabelData:
        try:
            revision = None
            token = None
            csv_path = hf_hub_download(
                repo_id=repo_id, filename="selected_tags.csv", revision=revision, token=token
            )
            csv_path = Path(csv_path).resolve()
        except HfHubHTTPError as e:
            raise FileNotFoundError(f"selected_tags.csv failed to download from {repo_id}") from e
        df: pd.DataFrame = pd.read_csv(csv_path, usecols=["name", "category"])
        tag_data = LabelData(
            names=df["name"].tolist(),
            rating=list(np.where(df["category"] == 9)[0]),
            general=list(np.where(df["category"] == 0)[0]),
            character=list(np.where(df["category"] == 4)[0]),
        )
        return tag_data

    def pil_ensure_rgb(self, image: Image.Image) -> Image.Image:
        # convert to RGB/RGBA if not already (deals with palette images etc.)
        if image.mode not in ["RGB", "RGBA"]:
            image = image.convert("RGBA") if "transparency" in image.info else image.convert("RGB")
        # convert RGBA to RGB with white background
        if image.mode == "RGBA":
            canvas = Image.new("RGBA", image.size, (255, 255, 255))
            canvas.alpha_composite(image)
            image = canvas.convert("RGB")
        return image

    def pil_pad_square(self, image: Image.Image) -> Image.Image:
        w, h = image.size
        # get the largest dimension so we can pad to a square
        px = max(image.size)
        # pad to square with white background
        canvas = Image.new("RGB", (px, px), (255, 255, 255))
        canvas.paste(image, ((px - w) // 2, (px - h) // 2))
        return canvas

    def get_tags(
            self,
            probs: Tensor,
    ) -> str:
        # Convert indices+probs to labels
        probs = list(zip(self.labels.names, probs.numpy()))

        # First 4 labels are actually ratings
        rating_labels = dict([probs[i] for i in self.labels.rating])

        # General labels, pick any where prediction confidence > threshold
        gen_labels = [probs[i] for i in self.labels.general]
        gen_labels = dict([x for x in gen_labels if x[1] > self.general_threshold])
        gen_labels = dict(sorted(gen_labels.items(), key=lambda item: item[1], reverse=True))

        # Character labels, pick any where prediction confidence > threshold
        char_labels = [probs[i] for i in self.labels.character]
        char_labels = dict([x for x in char_labels if x[1] > self.character_threshold])
        char_labels = dict(sorted(char_labels.items(), key=lambda item: item[1], reverse=True))

        # Combine general and character labels, sort by confidence
        combined_names = [ x for x in list(gen_labels) + list(char_labels) ]

        forbidden = self.forbidden_tags or set()
        filtered = [x for x in combined_names if x not in forbidden]

        caption = [*self.prefixes, *filtered] if self.prefixes else filtered

        caption =  ", ".join(caption)
        caption = caption.replace("_", " ").replace("(", "\\(").replace(")", "\\)")

        return caption


    @torch.no_grad()
    def predict(self, image_path: Path) -> str:
        if not image_path.is_file():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        transform = create_transform(**resolve_data_config(self.model.pretrained_cfg, model=self.model))
        img_input: Image.Image = Image.open(image_path)
        img_input = self.pil_ensure_rgb(img_input)
        img_input = self.pil_pad_square(img_input)
        inputs: Tensor = transform(img_input).unsqueeze(0).to(self.device)
        inputs = inputs[:, [2, 1, 0]]

        with torch.inference_mode():
            logits = self.model.forward(inputs)
            outputs = torch.sigmoid(logits)

        caption = self.get_tags(
            probs=outputs.squeeze(0).cpu(),
        )

        del inputs
        del outputs

        return caption

    def cleanup(self) -> None:
        del self.model
        torch.cuda.empty_cache()