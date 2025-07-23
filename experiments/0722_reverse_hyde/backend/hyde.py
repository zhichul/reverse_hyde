import importlib.util
from pathlib import Path
from typing import Callable

def load_extractor(py_path: Path) -> Callable[[str], list]:
    """
    Dynamically import an extractor script expected to expose:
        def extract_reverse_hyde_keys(text: str) -> List[str]
    Returns that function.
    """
    spec = importlib.util.spec_from_file_location("extractor_mod", py_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)            # type: ignore
    try:
        return mod.extract_reverse_hyde_keys
    except AttributeError as e:
        raise AttributeError(
            f"{py_path} must define 'extract_reverse_hyde_keys(text) -> list[str]'"
        ) from e