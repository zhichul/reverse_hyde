from gritlm import GritLM
import numpy as np

class GritEmbedder:
    """Lazy‑loads the GritLM embedding model and exposes .__call__(text) -> np.ndarray."""
    _model = None         # class‑level cache

    def __init__(self, model_name: str = "GritLM/GritLM-7B"):
        if GritEmbedder._model is None:
            GritEmbedder._model = GritLM(model_name, torch_dtype="auto", device_map="auto", mode='embedding')
        self.model = GritEmbedder._model

    def __call__(self, text: str) -> np.ndarray:
        return self.model.encode([text], instruction=self._instruction())[0]

    @staticmethod
    def _instruction():
        instruction = "Given a research query, retrieve the title and abstract of the relevant research paper"
        return "<|user|>\n" + instruction + "\n<|embed|>\n" if instruction else "<|embed|>\n"

def get_embedder(name: str) -> GritEmbedder:
    """Signature kept from earlier design (embed_model arg ignored for now)."""
    if name == 'grit':
        return GritEmbedder()
    else:
        raise NotImplementedError(name)