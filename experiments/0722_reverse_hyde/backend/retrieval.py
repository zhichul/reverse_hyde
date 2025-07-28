from pathlib import Path
from typing import List, Tuple

import numpy as np
import faiss


class Retriever:
    """
    FAISS‑only retriever. The FAISS index **row order MUST align**
    with the corpus row order (0‑based). Results are (row_idx, distance).
    """

    def __init__(self, backend: str, index_path: Path, embedder):
        if backend != "faiss":
            raise ValueError("Only 'faiss' backend is implemented for now.")
        self.index = faiss.read_index(str(index_path))
        self.embed = embedder

    def search(self, text: str, k: int) -> List[Tuple[int, float]]:
        vec = self.embed(text, type='query').astype("float32")
        D, I = self.index.search(vec[None, :], k)
        return list(zip(I[0].tolist(), (1-D[0]).tolist())) # turn into cosine distance