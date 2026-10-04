
import argparse
from pathlib import Path

from gemma4_tagger import Gemma4Tagger
from main import run_pipeline
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
    return parser.parse_args()

def main():
    args = parse_args()

    # Init taggers
    taggers = [
        WD14Tagger(model_name="eva02"),
        JoyCapTagger(mode="Straightforward"),
        Gemma4Tagger(),
        # QwenTagger(),
    ]

    run_pipeline(image_dir=args.image_dir, taggers=taggers)

if __name__ == "__main__":
    main()