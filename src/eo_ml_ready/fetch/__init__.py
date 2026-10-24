"""Provider adapters for downloading satellite scenes."""

from eo_ml_ready.fetch.base import Provider, SceneRef
from eo_ml_ready.fetch.goes_aws import GoesAws

__all__ = ["GoesAws", "Provider", "SceneRef"]
