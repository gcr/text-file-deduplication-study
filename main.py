import numpy as np
import pandas as pd
from data import all_sentences
from tqdm.auto import tqdm
import torch
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

print("starting load...")
_SENTENCES = all_sentences(ndocs=25000)
print(_SENTENCES)

# https://github.com/Ankur3107/transformers-on-macbook-m1-gpu
# this ensures that the current MacOS version is at least 12.3+
print(torch.backends.mps.is_available())
# this ensures that the current current PyTorch installation was built with MPS activated.
print(torch.backends.mps.is_built())



X = model.encode(
    _SENTENCES.text.tolist(),
    batch_size=1024,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
)
print(X.shape)
