import pandas as pd
import os
from tqdm.auto import tqdm
from spacy.lang.en import English

# for sentence splitting
sentencizer = English()
sentencizer.add_pipe('sentencizer')

def load_docs(path="data/data.parquet"):
    #if not os.path.exists(out_path):
    #    print("Preprocessing docs ...")
    #    pd.read_csv(path, usecols=['text']).to_parquet(out_path)
    return pd.read_parquet(path)


def load_sentences(path="data/data.csv", ndocs=None):
    """Returns a DataFrame of sentences."""
    csv = load_docs(path=path, ndocs=ndocs)
    return pd.DataFrame([(doc_id, sent_id, sent.text.strip())
        for doc_id, doc in enumerate(tqdm(csv.text.tolist(), desc="sentencizer"))
        for sent_id, sent in enumerate(sentencizer(doc).sents)
    ], columns=('doc_id','sent_id','text')).set_index(['doc_id','sent_id'])

def ensure_preprocessed_sentences(out_path="data/sentences-preprocessed.csv.zstd", nrows=None):
    if not os.path.exists(out_path):
        print("Preprocessing sentences...")
        df = load_sentences().to_csv(out_path)
    return pd.read_csv(out_path, nrows=nrows)


if __name__ == "__main__":
    ensure_preprocessed_sentences()