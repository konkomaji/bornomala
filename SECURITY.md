# Security Policy

## Scope

The Bengali tokenizer (`bengali-tokenizer/`, package `bntok`) is a CPU-only
library. It reads text you give it and, when asked to build a training corpus,
downloads public datasets from official sources (the Hugging Face Hub:
AI4Bharat Sangraha, Wikimedia Wikisource/Wikipedia, XL-Sum, and optionally
CC-100). It does not execute untrusted code and does not transmit your input
anywhere.

## Reporting a vulnerability

Please report security issues privately to **work.konkomaji@gmail.com** rather
than opening a public issue. Include:

- a description of the issue and its impact,
- steps to reproduce,
- affected version or commit.

You can expect an acknowledgement within a reasonable time. Please give us a
chance to address the issue before public disclosure.

## Things to be aware of

- **Corpus and model downloads.** Training data and the published tokenizer
  artifact are fetched from the Hugging Face Hub. Set `HF_TOKEN` only with a
  token you trust; gated sources require accepting their owner's terms.
- **Untrusted training corpora.** `bntok train` reads arbitrary text files you
  point it at; treat corpus files from unknown sources the way you would any
  other untrusted input to a data pipeline.
