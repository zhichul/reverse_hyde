from dataclasses import dataclass
from pathlib import Path

@dataclass
class Settings:
    embed_model: str
    index_path: Path
    backend: str
    query_dataset: str
    corpus_dataset: str
    prompt_dir: Path
    extractor_dir: Path
    annotation_dir: Path
    ui_config: Path
    host: str
    port: int
    id_field: str
    relevant_documents_field: str
