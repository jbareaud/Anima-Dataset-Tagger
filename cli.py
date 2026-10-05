import argparse
from pathlib import Path

from gemma4_tagger import Gemma4Tagger
from main import run_pipeline
from config import load_config
from qwen_tagger import QwenTagger
from wd14_tagger import WD14Tagger
from joycap_tagger import JoyCapTagger

def parse_comma_separated(value: str) -> list[str]:
    """Parse comma-separated strings into a clean list of names."""
    return [item.strip() for item in value.split(",") if item.strip()]

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
    parser.add_argument(
        "--taggers",
        type=parse_comma_separated,
        default=None,
        help="Comma-separated taggers to run (e.g. --taggers WD14Tagger,JoyCapTagger,Gemma4Tagger). Runs all by default."
    )
    return parser.parse_args()

def main():
    args = parse_args()

    cfg = load_config(args.config)

    tagger_factories = {
        "WD14Tagger": lambda: WD14Tagger(cfg.wd14),
        "JoyCapTagger": lambda: JoyCapTagger(cfg.joycaption),
        "Gemma4Tagger": lambda: Gemma4Tagger(cfg.gemma4),
        #"QwenTagger": lambda: QwenTagger(cfg.qwen),
    }

    selected_taggers = args.taggers or list(tagger_factories.keys())

    invalid = [name for name in selected_taggers if name not in tagger_factories]
    if invalid:
        raise ValueError(
            f"Unknown tagger(s): {', '.join(invalid)}. "
            f"Valid options: {', '.join(tagger_factories.keys())}"
        )

    taggers = [tagger_factories[name]() for name in selected_taggers]

    run_pipeline(image_dir=args.image_dir, taggers=taggers)

if __name__ == "__main__":
    main()