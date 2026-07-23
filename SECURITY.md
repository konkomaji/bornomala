# Security Policy

## Scope

MotherTongueIndex is a CPU-only analysis tool. It reads text you give it and
downloads public tokenizer files from official sources (OpenAI tiktoken, the
Hugging Face Hub). It does not execute untrusted code and does not transmit your
input anywhere unless you run the optional web server or the optional
`eval/reasoning_probe.py` with your own API endpoint.

## Reporting a vulnerability

Please report security issues privately to **work.konkomaji@gmail.com** rather
than opening a public issue. Include:

- a description of the issue and its impact,
- steps to reproduce,
- affected version or commit.

You can expect an acknowledgement within a reasonable time. Please give us a
chance to address the issue before public disclosure.

## Things to be aware of

- **Tokenizer downloads.** Model tokenizers are fetched from the Hugging Face
  Hub and OpenAI. Set `HF_TOKEN` only with a token you trust. Gated models
  require accepting the model owner's terms.
- **The web server** (`web/`) is intended for local or trusted deployment. If
  you expose it publicly, put it behind your own authentication and rate
  limiting; it runs tokenizers on arbitrary input.
- **API keys** for the reasoning probe are supplied by you and never stored by
  this project.
