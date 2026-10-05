import argparse
from pathlib import Path

from gemma4_tagger import Gemma4Tagger
from main import run_pipeline
from config import load_config
from qwen_tagger import QwenTagger
from wd14_tagger import WD14Tagger
from joycap_tagger import JoyCapTagger

def parse_args():
    parser = argparse.ArgumentParser(
        description="Automatic caption generator."
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Path to images directory."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.json"),
        help="Path to JSON configuration file (default: config.json)"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    cfg = load_config(args.config)

    # Init taggers
    taggers = [
        WD14Tagger(cfg.wd14),
        JoyCapTagger(cfg.joycaption),
        #Gemma4Tagger(cfg.gemma4),
        #QwenTagger(cfg.qwen),
    ]

    run_pipeline(image_dir=args.image_dir, taggers=taggers)

if __name__ == "__main__":
    main()