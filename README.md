# [Chip Huyen's ML interview practice book](https://huyenchip.com/ml-interviews-book/), problem 6.1.13

By Kimmy and Amelia!

## Problem statement.
> You have 1 million text files, each is a news article scraped from various news sites. Since news sites often report the same news, even the same articles, many of the files have content very similar to each other. Write a program to filter out these files so that the end result contains only files that are sufficiently different from each other in the language of your choice. You’re free to choose a metric to define the “similarity” of content between files.

## Dataset.
1. Download the "Humans vs LLM" corpus here: https://www.kaggle.com/datasets/starblasters8/human-vs-llm-text-corpus
2. Verify file integrity:
```
$ sha256sum data/data.csv
9a957196ee1dd1f67ab4b2fa55a48be2c7218683e6dfa1fb51145f68a521d8d5  data.csv
```

Data contains:
- **789,449** documents, with
- **21,788,331** sentences.

## Approaches considered.
1. Split documents into sentences. Documents are considered identical if >X% of the sentences are found in another document.
2. Cosine distance using sentence embedding. Model is `paraphrase-MiniLM-L6-v2` from huggingface's `SentenceTransformers` package.