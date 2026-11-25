from base_tagger import BaseTagger
from config import DanbooruConfig
import hashlib
import os
import re
import time
from pathlib import Path
import requests
from dotenv import load_dotenv

load_dotenv()
DANBOORU_USER_ID = os.getenv("DANBOORU_USER_ID")
RATE_LIMIT = 1
HEADERS = {"User-Agent": f"DatasetTagFetcher/1.0 (user #{DANBOORU_USER_ID})"}
ORDER_CATEGORIES = [
    "tag_string_meta",
    "tag_string_character",
    "tag_string_copyright",
    "tag_string_artist",
    "tag_string_general",
]
URL = "https://danbooru.donmai.us/posts.json"

class DanbooruTagger(BaseTagger):
    def __init__(
        self,
        config: DanbooruConfig,
    ):
        self.config = config

    @property
    def name(self) -> str:
        return "Danbooru web tagger"

    def extract_md5(self, file_path: Path) -> str:
        match = re.search(r"([a-fA-F0-9]{32})$", file_path.stem)
        if match:
            return match.group(1).lower()

        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for bloc in iter(lambda: f.read(4096), b""):
                hash_md5.update(bloc)
        return hash_md5.hexdigest()

    def format_tag_list(self, raw_tags: str, prefix: str = "") -> list[str]:
        if not raw_tags:
            return []
        return [
            f"{prefix}{tag.replace('_', ' ')}" for tag in raw_tags.split()
        ]

    def fetch_danbooru_tags(self, md5: str) -> tuple[str, str] | None:
        params = {"tags": f"md5:{md5}"}
        try:
            response = requests.get(URL, params=params, headers=HEADERS, timeout=10)
            response.raise_for_status()
            data = response.json()

            if not data:
                return None

            post = data[0]

            # 1. all tags
            list_all = self.format_tag_list(post.get("tag_string", ""))
            tags_all = self.config.separator.join(list_all)

            # 2. ordered tags
            list_in_order = []
            for category in ORDER_CATEGORIES:
                content_cat = post.get(category, "")
                if content_cat:
                    prefix = "@" if category == "tag_string_artist" else ""
                    list_in_order.extend(self.format_tag_list(content_cat, prefix=prefix))

            tags_in_order = self.config.separator.join(list_in_order)

            return tags_all, tags_in_order

        except requests.RequestException as e:
            print(f"  [network error] {e}")
            return None

    def predict(self, image_path: Path) -> str:
        time.sleep(RATE_LIMIT)
        md5 = self.extract_md5(image_path)
        print(f"searching for {image_path.name} (MD5: {md5})")
        result = self.fetch_danbooru_tags(md5)
        if result:
            tags_all, tags_in_order = result
            return tags_in_order
        else:
            print("  -> No post found on Danbooru for that MD5.")
            return ""

    def cleanup(self) -> None:
        pass