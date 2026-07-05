from .ml import get_classifier

__all__ = ["get_classifier"]

try:
    from .dl import CNN1DClassifier

    _TORCH_AVAILABLE = True
    __all__.append("CNN1DClassifier")
except ImportError:
    _TORCH_AVAILABLE = False
