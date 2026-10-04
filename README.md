# Anima-Dataset-Tagger

Caption generator for Anima Datasets.

This is a WIP, and modifying parameters and editing the models locs will require editing the python scripts directly.

Tool built in Python 3.13.2 with torch 2.13.0+rocm10.0.0.

## Installation 

Install the version of torch for your machine, then the requirements.txt file.

## Usage
 
```
uv run cli.py --image-dir "path/to/image_directory"
```

Output is raw, and is intended to be manually checked and edited.

## Models used 

Models run on llama cpp python with gguf models, except SmilingWolf which runs on timm and requires a Hugging Face account. 

Models list :
- SmilingWolf for Danbooru tags
- Joy Caption Beta One for straightforward description
- Gemma 4 e4b with vision enabled to focus on specific elements, like direction of gaze.

(qwen 3.5 9b exists but disabled)




