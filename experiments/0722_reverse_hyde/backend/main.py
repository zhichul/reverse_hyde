from __future__ import annotations

from dataclasses import asdict
import glob
import html
import json
import math
from pathlib import Path
from typing import Dict, Any, List

from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
from fastapi import FastAPI, HTTPException, Body
import yaml
from datetime import datetime, timezone
import base64, json, os

from .config import Settings
from .data import QuerySet, Corpus
from .embeddings import get_embedder
from .retrieval import Retriever
from .llm import LLMClient
from .hyde import load_extractor


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _settings_to_json(cfg: Settings) -> Dict[str, Any]:
    """Convert dataclass → JSON‑serialisable dict (Paths → str)."""
    return {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()}

def _build_annotation_index(base_dir: Path) -> None:
    rows = []
    for p in base_dir.rglob("*"):
        if p.suffix.lower() not in {".html", ".txt", ".json"} or not p.is_file():
            continue
        rel = p.relative_to(base_dir).as_posix()
        rows.append(f"<li><a href=\"/annotations/{rel}\">{html.escape(rel)}</a></li>")
    rows.sort()

    (base_dir / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'>"
        "<h1>Saved annotations</h1>"
        "<button onclick=\"fetch('/annotation/reindex',"
        "{method:'POST'}).then(()=>location.reload())\">🔄 Refresh index</button>"
        "<ul>" + "\n".join(rows) + "</ul>",
        encoding="utf-8"
    )


def make_app(cfg: Settings) -> FastAPI:
    app     = FastAPI(title="Prompt‑HyDE")
    queries = QuerySet(cfg.query_dataset)
    corpus  = Corpus(cfg.corpus_dataset, id_field=cfg.id_field)
    embed   = get_embedder(cfg.embed_model)
    retr    = Retriever(cfg.backend, cfg.index_path, embed)
    prompt_glob     = str(cfg.prompt_dir / "*.txt")
    extractor_glob  = str(cfg.extractor_dir / "*.py")
    _build_annotation_index(cfg.annotation_dir)


    @app.get("/doc/{doc_id}", tags=["data"])
    def get_document(doc_id: int):
        """
        Return the full corpus row whose id_field == doc_id.
        Raises 404 if the ID is missing.
        """
        try:
            print(type(doc_id))
            return corpus.get_by_id(doc_id)
        except KeyError:
            raise HTTPException(404, f"doc id {doc_id} not found")  # type: ignore

    @app.get("/queries", tags=["data"])
    def list_queries():
        """
        Returns [{idx:int, query:str}] so the UI can populate its dropdown
        without 100 separate /query/{i} calls.
        """
        return [
            {"idx": i, **q} for i, q in enumerate(queries.get_all())
        ]

    def run_retrieval_only(q_obj: dict, k: int, *, rel_field: str, id_field: str):
        """
        Returns docs_before, ranks_before (dict[doc_id -> rank|inf]), recall_before
        """
                # ---------------- original retrieval --------------------------- #
        print('Running [basic retrieval]')
        rel_ids = q_obj[rel_field]
        rel_ids_set = set(rel_ids)

        hits = retr.search(q_obj["query"], k)          # [(row, dist), ...]
        ranks = {doc_id: 9999999999 for doc_id in rel_ids}
        docs  = []

        for r, (row, dist) in enumerate(hits):
            doc = corpus.get_by_index(row)
            doc_id = doc[id_field]
            docs.append(doc)
            if doc_id in rel_ids_set:
                ranks[doc_id] = r + 1

        tp = sum(1 for doc_id in rel_ids if ranks[doc_id] <= k)
        recall = tp / len(rel_ids)

        print(ranks)
        for r, (doc, (row, d)) in enumerate(zip(docs, hits)):
            print(f"rank {r+1} ({d}): {doc['title']}")

        return docs, ranks, recall, hits      # hits reused by run_prompt
    
    @app.post("/retrieve", tags=["core"])
    def retrieve(payload: dict):
        """
        payload: { "query_idx": int, "k": int }
        """
        q_idx = payload["query_idx"]
        k     = payload.get("k", 10)

        q_obj = queries.get_query(q_idx)
        docs_before, ranks_before, recall_before, hits = run_retrieval_only(
            q_obj, k,
            rel_field=cfg.relevant_documents_field, id_field=cfg.id_field
        )
        return {
            "docs_before": docs_before,
            "ranks_before": ranks_before,
            "recall_before": recall_before,
            "hits_before": hits,

        }

    @app.post("/annotation/reindex", tags=["save_annotations"])
    async def rebuild_index():
        _build_annotation_index(cfg.annotation_dir)
        return {"ok": True}



    @app.post("/annotation/save", tags=["save_annotations"])
    async def save_annotation(payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Saves three sibling files:
        annotations/q{Q}_d{D}/{TIMESTAMP}.html / .txt / .json
        and (re)builds annotations/index.html with links to every snapshot.
        """
        try:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            q   = payload["query_idx"]
            d   = payload["doc_idx"]

            # ---------- write individual files ----------
            stem_dir: Path = cfg.annotation_dir / f"q{q}_d{d}"
            stem_dir.mkdir(parents=True, exist_ok=True)
            stem = stem_dir / ts

            # HTML snapshot
            html_bytes = base64.b64decode(payload["html"])
            html_path  = stem.with_suffix(".html")
            html_path.write_bytes(html_bytes)

            # free‑text note
            stem.with_suffix(".txt").write_text(payload["annotation"], encoding="utf-8")

            # raw JSON (strip big fields)
            meta = {k: v for k, v in payload.items() if k not in {"html"}}
            stem.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

            _build_annotation_index(cfg.annotation_dir)

            return {"ok": True, "path_base": str(stem)}
        except Exception as exc:
            # surface any filesystem/base64 errors to the client
            raise HTTPException(500, f"Annotation save failed: {exc}") from exc

    @app.get("/prompts", tags=["assets"])
    def list_prompts():
        return [Path(p).stem for p in glob.glob(prompt_glob)]

    @app.get("/extractors", tags=["assets"])
    def list_extractors():
        return [Path(p).stem for p in glob.glob(extractor_glob)]

    @app.put("/prompt/{name}", tags=["assets"], response_class=PlainTextResponse)
    async def save_prompt(
        name: str,
        body: str = Body(..., media_type="text/plain")
    ):
        (cfg.prompt_dir / f"{name}.txt").write_text(body, encoding="utf-8")
        return "OK"

    @app.get("/prompt/{name}", tags=["assets"], response_class=PlainTextResponse)
    def load_prompt(name: str):
        with open( cfg.prompt_dir / f"{name}.txt", 'rt') as f:
            return f.read()

    @app.get("/config", tags=["debug"])
    def get_config() -> Dict[str, Any]:
        return _settings_to_json(cfg)

    @app.get("/query/{idx}", tags=["data"])
    def get_query(idx: int) -> Dict[str, Any]:
        try:
            return queries.get_query(idx)
        except IndexError:
            raise HTTPException(404, "query idx out of range")

    @app.get("/ui_config", tags=["debug"])
    def get_ui_cfg():
        """Return the raw UI‑defaults file (JSON inside YAML is also fine)."""
        txt = cfg.ui_config.read_text()
        if cfg.ui_config.suffix.lower() in {".yaml", ".yml"}:
            return yaml.safe_load(txt)
        return json.loads(txt)

    @app.post("/prompt/run", tags=["core"])
    async def run_prompt(payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Required keys in payload:
            query_idx, doc_idx, prompt_name, extractor_name, k, llm_cfg
        """
        q_idx   = payload["query_idx"]
        doc_idx = payload["doc_idx"]
        k       = payload.get("k", 10)
        print('Running [get query]')
        q_obj   = queries.get_query(q_idx)
        rel_ids = q_obj[cfg.relevant_documents_field]
        rel_ids_set = set(rel_ids)
        if not (0 <= doc_idx < len(rel_ids)):
            raise HTTPException(400, "doc_idx out of bounds")
        print(json.dumps(q_obj, indent=2))

        # Always resolve by ID (no get_by_index)
        target_id   = rel_ids[doc_idx]
        try:
            relevant_doc = corpus.get_by_id(target_id)
        except KeyError:
            raise HTTPException(400, f"relevant doc id {target_id} not in corpus")


        # ---------------- prompt construction -------------------------- #
        print('Running [prompt construction]')
        prompt_tpl = (cfg.prompt_dir / f"{payload['prompt_name']}.txt").read_text()
        filled_prompt = prompt_tpl.format(**relevant_doc)     # expand only doc fields
        print(filled_prompt)
        # ---------------- LLM call + extraction ------------------------ #
        print('Running [llm call]')
        llm   = LLMClient(payload["llm_cfg"])
        raw   = llm.complete(filled_prompt)
        extractor = load_extractor(cfg.extractor_dir / f"{payload['extractor_name']}.py")
        rh_keys: List[str] = extractor(raw)
        print(rh_keys)

        docs_before, ranks_before, recall_before, hits = run_retrieval_only(q_obj, k, rel_field=cfg.relevant_documents_field, id_field=cfg.id_field)
        
        # ---------------- augment ranking ------------------------------ #
        print('Running [augmentation ranking]')
        print('Running [original ranking] subroutine')
        key_dists = []
        if rh_keys:
            key_vecs = np.vstack([embed(k) for k in rh_keys])
            q_vec    = embed(q_obj["query"], type='query')
            key_dists = (1- np.matmul(key_vecs, q_vec)).tolist()

        augmented = {row: dist for row, dist in hits}
        for i, dist in enumerate(key_dists, 1):
            augmented[-i] = dist                              

        augmented_hits = sorted(augmented.items(), key=lambda x: x[1])
        ranks_after = {doc_id: 9999999999 for doc_id in rel_ids}
        docs_after = []

        # ensure every relevant doc has an entry
        for r, (row, _) in enumerate(augmented_hits):
                if row < 0:
                    i = -row-1
                    ranks_after[target_id] = min(ranks_after.get(target_id, 9999999999), r + 1)
                    docs_after.append({'title': rh_keys[i], 'corpusid': i, 'reverse_hyde': True})
                else:
                    doc = corpus.get_by_index(row)
                    doc_id = doc[cfg.id_field]
                    if doc_id in rel_ids_set:
                        ranks_after[doc_id] = min(ranks_after.get(doc_id, 9999999999), r + 1)
                    docs_after.append(doc)
        print(ranks_after)
        for r, (doc, (row, d)) in enumerate(zip(docs_after, augmented_hits)):
            if row > 0:
                print(f"rank {r+1} ({d}): {doc['title']}")
            if row < 0:
                print(f"rank {r+1} ({d}): [new] {doc['title']}")

        # ---------------- recall --------------------------------------- #
        recall_before = {}
        recall_after = {}
        for kk in (1, 5, 20):
            tp_before = sum(1 for doc_id in rel_ids if ranks_before[doc_id] <= kk)
            tp_after = sum(1 for doc_id in rel_ids if ranks_after[doc_id] <= kk)
            recall_before[kk] = tp_before / len(rel_ids)
            recall_after[kk]  = tp_after / len(rel_ids)

        return {
            "raw_response":  raw,
            "reverse_keys":  rh_keys,
            "ranks_before":  ranks_before,
            "ranks_after":   ranks_after,
            "recall_before": recall_before,
            "recall_after":  recall_after,
            "relevant_doc":  relevant_doc,
            "docs_before": docs_before,
            "hits_before": hits,
            "docs_after": docs_after,
            "hits_after": augmented_hits,
        }

    frontend_dir = Path(__file__).parent.parent / "frontend"
    app.mount(
        "/annotations",
        StaticFiles(directory=cfg.annotation_dir, html=True),
        name="annotations",
    )
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
    return app