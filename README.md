# reverse_hyde
Ongoing research for developing a reverse hyde workflow to improve retrieval of scientific literature. Plan is to use RL with IR metric as reward to train a LLM to generate reverse-hyde keys (e.g. interesting details or ideas from the paper, what other problems can it be used to solve, etc) given a scientific paper in context. We would measure the effectiveness of those keys by how well they improve recall of literature search queries.

# /experiments
Jupyter notebooks and/or web interfaces for prompt development & error analysis on a literature retrieval task ([LitSearch](https://github.com/princeton-nlp/LitSearch)). Will be moving to RL after validating effectiveness of having good reverse hyde keys with more prompt engineering results.
