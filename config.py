import json
from pathlib import Path
from pydantic import BaseModel, Field

class WD14Config(BaseModel):
    device: str
    model_name: str
    general_threshold: float
    character_threshold: float
    forbidden_tags: set[str] = Field(default_factory=set)
    prefixes: list[str] = Field(default_factory=list)

class JoyCaptionConfig(BaseModel):
    model_path: str
    mmproj_path: str
    n_ctx: int
    n_gpu_layers: int
    temperature: float
    mode: str
    max_tokens: int

class Gemma4Config(BaseModel):
    model_path: str
    mmproj_path: str
    n_ctx: int
    n_gpu_layers: int
    max_tokens: int
    temperature: float

class QwenConfig(BaseModel):
    model_path: str
    mmproj_path: str
    n_ctx: int
    n_gpu_layers: int
    max_tokens: int
    temperature: float

class AppConfig(BaseModel):
    wd14: WD14Config #Field(default_factory=WD14Config)
    joycaption: JoyCaptionConfig
    gemma4: Gemma4Config
    qwen: QwenConfig

def load_config(config_path: Path | str = "config.json") -> AppConfig:
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found : {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return AppConfig.model_validate(data)