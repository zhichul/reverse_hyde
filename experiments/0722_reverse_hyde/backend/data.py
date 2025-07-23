from pathlib import Path
from typing import Dict, Any, Union, List

from datasets import load_dataset, Dataset

# ------------------------------------------------------------------ #
# Helper to load local path *or* HF hub name ----------------------- #
# ------------------------------------------------------------------ #
def _load_dataset(source: str, type: str) -> Dataset:
    if type not in ['query', 'corpus']:
        raise ValueError(type)
    if source == "princeton-nlp/LitSearch":
        if type == 'query':
            return load_dataset("princeton-nlp/LitSearch", "query", split="full")
        else:
            return load_dataset("princeton-nlp/LitSearch", "corpus_clean", split="full")
    elif source.endswith('.parquet'):
        return load_dataset('parquet', data_files=source, split='train')
    else:
        raise NotImplementedError(source)


# ------------------------------------------------------------------ #
# Corpus with ID → index mapping ----------------------------------- #
# ------------------------------------------------------------------ #
class Corpus:
    """
    Holds the document corpus (title, abstract, …) and a mapping from an
    external `id_field` (e.g. 'doc_id') to row indices.
    """

    def __init__(self, source: str, id_field: str = "id"):
        self.ds: Dataset = _load_dataset(source, 'corpus')
        self.id_field = id_field
        # build {doc_id: row_idx}
        self.id2idx: Dict[int, int] = {
            row[id_field]: idx for idx, row in enumerate(self.ds)
        }

    # -- retrieval helpers --------------------------------------------------
    def get_by_index(self, row_idx: int) -> Dict[str, Any]:
        return self.ds[row_idx]

    def get_by_id(self, doc_id: int) -> Dict[str, Any]:
        try:
            return self.ds[self.id2idx[doc_id]]
        except KeyError as e:
            raise KeyError(f"doc_id {doc_id} not found in corpus") from e


# ------------------------------------------------------------------ #
# Query set (unchanged API) ---------------------------------------- #
# ------------------------------------------------------------------ #
class QuerySet:
    """
    Each row must have:
        - 'query': str
        - 'relevant_documents': List[int]    (IDs or row indices)

    This class doesn’t need to know the corpus; it just returns query objects.
    """

    def __init__(self, source: str):
        self.ds: Dataset = _load_dataset(source, 'query')

    def __len__(self) -> int:
        return len(self.ds)

    def get_query(self, idx: int) -> Dict[str, Any]:
        return self.ds[idx]
    
    def get_all(self) -> list[str]:
        return list(self.ds['query'])
    