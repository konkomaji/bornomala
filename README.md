<p align="center">
  <img src="mothertongueindex/docs/assets/logo.svg" width="132" height="132" alt="Project Bornomala"/>
</p>

<h1 align="center">Project Bornomala &nbsp;বর্ণমালা</h1>

<p align="center">
  <b>Building a Bengali-first, dialect-aware large language model.</b><br/>
  A Bengali language model trained for Bengali, including the dialects of West Bengal, not English with Bengali bolted on.
</p>

<p align="center">
  <img alt="status" src="https://img.shields.io/badge/status-in%20planning-4A46E0"/>
  <img alt="license" src="https://img.shields.io/badge/license-Apache--2.0-00A9A5"/>
  <img alt="goal" src="https://img.shields.io/badge/goal-Bengali--first%20LLM-FF6F5B"/>
  <img alt="focus" src="https://img.shields.io/badge/dialect-aware-7A2BE2"/>
</p>

---

## The project

**Project Bornomala is a programme to build a Bengali-first large language
model** that understands Bengali the way Bengali is actually written and spoken,
including its literary register and the dialects of West Bengal that no existing
model covers.

It rests on one claim, stated so it can be attacked:

> The binding constraint on Bengali language modelling is not compute or
> architecture. It is the absence of a large, clean, high-register Bengali
> corpus (it exists only as page images), and the total absence of any
> computational resource for the Bengali dialects of West Bengal.

So the work runs in the causal order **OCR into corpus into model**, and dialect
documentation is treated as a first-class data asset, not a footnote.

The full plan, the tracks, the roadmap, the phase gates, and the risk register
are in the technical whitepaper:

**[PROJECT_BORNOMALA_Technical_Specification.md](PROJECT_BORNOMALA_Technical_Specification.md)**

## The programme (what Bornomala builds)

These five tracks are Project Bornomala itself: they produce the Bengali-first
model and the data it needs. They are built in this main repository.

| Track | Deliverable | Purpose | Status |
|---|---|---|---|
| **A** | Bengali tokenizer | Grapheme-cluster-aware, literary-weighted Bengali vocabulary | Planned |
| **B** | Bengali document OCR | Recover the 1850 to 1950 Bengali literary corpus trapped in page images | Planned |
| **C** | West Bengal dialect corpus | First computational resource for the five West Bengal dialect groups | Planned |
| **D** | Speech (ASR and TTS) | Dialect-aware Bengali speech | Planned |
| **E** | Foundation model | A 2 to 4B Bengali-first, dialect-aware, on-device model | Planned |

## Subprojects (separate tools in this repository)

Bornomala also hosts standalone tools that support the mission but are their own,
distinct things. They are not the Bengali model, and not one of the tracks above.

| Subproject | What it is | Relation to Bornomala | Status |
|---|---|---|---|
| **[MotherTongueIndex](mothertongueindex/)** | A multilingual tokenizer efficiency analyzer: measures how efficiently mainstream LLM tokenizers encode any language, versus English. | A separate tool. It studies the tokenization inequity Bornomala exists to fix, and can benchmark Bornomala's Bengali tokenizer once Track A ships. It does not build any Bornomala model. | Active |

> **Two different things.** Project Bornomala is the Bengali-first, dialect-aware
> LLM programme described in the whitepaper. MotherTongueIndex is a separate
> subproject tool that happens to live in the same repository. They should not be
> conflated.

## Repository layout

```
bornomala/
├── PROJECT_BORNOMALA_Technical_Specification.md   the whitepaper (the LLM programme)
├── mothertongueindex/                             subproject: a separate tokenizer-analysis tool
├── LICENSE · CONTRIBUTING · CODE_OF_CONDUCT · SECURITY · CITATION.cff
└── .github/                                        CI, issue and PR templates
```

## Licensing

Code is **Apache-2.0**. Corpora and datasets produced by the programme are
released **CC BY 4.0** or **CC BY-SA 4.0** (whitepaper section 17.3).

## Citation

See [CITATION.cff](CITATION.cff). Principal investigator: Konko Maji
(work.konkomaji@gmail.com).
