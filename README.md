# Anima-Dataset-Tagger

Caption generator for Anima Datasets.

See config.json for basic settings and model locations.

## Installation 

Install torch/torchaudio/torchvision and llama-cpp-python. Make sure they use the same backend. Then :

```commandline
uv pip install Pillow pydantic pandas numpy timm huggingface-hub
```

## Usage
 
```commandline
uv run cli.py --image-dir "path/to/image_directory"
```

Output is raw, and is intended to be manually checked and edited.

## Models used 

Models run on llama cpp python with gguf models, except SmilingWolf which runs on timm and requires a Hugging Face account. 

Models list :
- SmilingWolf for Danbooru tags
- Joy Caption Beta One for straightforward description
- Gemma 4 31b with vision enabled
- Danbooru API for images taht exist in the Danbooru database  

- (qwen 3.5 9b exists but disabled)

Recommended taggers : DanbooruTagger,WD14Tagger,Gemma4Tagger   
