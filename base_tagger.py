from abc import ABC, abstractmethod
from pathlib import Path

class BaseTagger(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Tagger name."""
        pass

    @abstractmethod
    def predict(self, image_path: Path) -> str:
        """Generate captions for a given image"""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup if needed."""
        pass
