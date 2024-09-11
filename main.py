import numpy as np
import pandas as pd
from pathlib import Path
import os.path
from data import ensure_preprocessed_sentences, load_docs
from tqdm.auto import tqdm
import torch
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from time import time
from multiprocessing import Pool

def gen_embeddings(model, docs):
    with torch.no_grad():
        return model.encode(
            [txt.strip() for txt in docs.text.tolist()],
            batch_size=512,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

def load_embeddings(docs, model_name='paraphrase-MiniLM-L6-v2', path_prefix="data/doc_embeddings"):
    model = SentenceTransformer(model_name)
    path = f"{path_prefix}.{model_name}.npz"
    fp16_path = f"{path_prefix}.{model_name}.fp16.npz"
    if not os.path.exists(path):
        X = gen_embeddings(model, docs)
        np.savez(path, X)
        np.savez(fp16_path, X.astype('float16'))
    else:
        return np.load(path)['arr_0']

@dataclass
class ScoreResult:
    query: str
    results: pd.DataFrame

def topk(qembed, pembed, docs, k=None, thresh=None):
    assert k or thresh, "Need to pass k or thresh."
    assert k is None or thresh is None, "cannot pass both k and thresh"

    scores = qembed @ pembed.T
    if k is not None:
        scores *= -1
        result_ids = scores.argpartition(k, axis=1)[:, :(k+1)]
        # partition isn't sorted, so sort the results
        rid_ordering = np.take_along_axis(scores, result_ids, axis=1).argsort(axis=1)
        result_ids = np.take_along_axis(result_ids, rid_ordering, axis=1)
        scores *= -1
    elif thresh is not None:
        result_ids = [
            np.where(score_row >= thresh)[0] for score_row in scores
        ]
        # Partition isn't sorted, so sort the results
        # Note the raggedness of the above array.
        result_ids = [
            rids[(-score_row[rids]).argsort()]
            for rids, score_row in zip(result_ids, scores)
        ]

    return [
        pd.DataFrame({
            "match_id": ids,
            "text": docs.iloc[ids].text,
            "score": scores[i, ids],
        }).set_index("match_id")
        for i, ids in enumerate(result_ids)
    ]

def by_indices(docs, batch_size=1000):
    return [docs.index[i:i+batch_size] for i in range(0, len(docs), batch_size)]

def topk_results(X, path, docs, **kwargs):
    """calculate topk embeddings for all batches"""
    if not os.path.exists(path):
        print("Calculating all results ...")
        results = []
        for indices in tqdm(by_indices(docs, batch_size=1000)):
            matches = topk(X[indices], X, docs=docs, **kwargs)
            for idx, m in zip(indices, matches):
                m['src_id'] = idx
                m = m.reset_index().set_index(["src_id", "match_id"])
                results.append(m)
            tqdm.write(str(m))
        results = pd.concat(results, axis=0)
        results[['score']].to_parquet(path)
    return pd.read_parquet(path)

if __name__ == "__main__":
    pd.options.display.max_colwidth = 80

    print("loading docs...")
    _DOCS = load_docs()
    print("_DOCS =\n",_DOCS)

    # https://github.com/Ankur3107/transformers-on-macbook-m1-gpu
    # this ensures that the current MacOS version is at least 12.3+
    print("MPS is available:", torch.backends.mps.is_available())
    # this ensures that the current current PyTorch installation was built with MPS activated.
    print("MPS is built:", torch.backends.mps.is_built())
    X = load_embeddings(_DOCS)
    print("load embeddings...")
    print("X =\n", X)

    print("dispatch work ...")
    results = topk_results(_DOCS, path="data/results.parquet", thresh=0.7)

    #import IPython; IPython.embed()