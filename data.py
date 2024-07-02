import pandas as pd
import dataclasses
from tqdm.auto import tqdm
from spacy.lang.en import English

# for sentence splitting
sentencizer = English()
sentencizer.add_pipe('sentencizer')

def load_sentences(path="data/data.csv", ndocs=None):
    csv = pd.read_csv(path, nrows=ndocs, usecols=['text'])
    return pd.DataFrame([(doc_id, sent_id, sent.text.strip())
        for doc_id, doc in enumerate(tqdm(csv.text.tolist(), desc="sentencizer"))
        for sent_id, sent in enumerate(sentencizer(doc).sents)
    ], columns=('doc_id','sent_id','text')).set_index(['doc_id','sent_id'])


if __name__ == "__main__":
    print("Saving sentences to CSV:")
    df = load_sentences()
    print(df)
    df.to_csv("data/sentences.csv.xz")