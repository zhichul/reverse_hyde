import argparse, pathlib, uvicorn
from .config import Settings
from .main import make_app

def main():
    ap = argparse.ArgumentParser(prog="prompt-hyde")
    add = ap.add_argument
    add("--embed-model", default='grit')
    add("--index-path", type=pathlib.Path, default='../0721_litsearch_example/faiss/litsearch.index')
    add("--backend", default="faiss")
    add("--query-dataset", default='princeton-nlp/LitSearch')
    add("--corpus-dataset", default='../0721_litsearch_example/corpus_clean_dedup.parquet')
    add("--id-field", default="corpusid",)
    add("--relevant-documents-field", default="corpusids",)
    add("--prompt-dir", type=pathlib.Path, default='./prompts')
    add("--extractor-dir", type=pathlib.Path, default='./extractors')
    add("--ui-config", type=pathlib.Path, default='configs/ui_config.json')
    add("--host", default="0.0.0.0")
    add("--port", type=int, default=8200)

    ns = ap.parse_args()
    cfg = Settings(**vars(ns))  # just a dataclass, no validation
    app = make_app(cfg)
    uvicorn.run(app, host=ns.host, port=ns.port)

if __name__ == "__main__":
    main()