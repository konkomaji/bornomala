# Reading Bengali on Its Own Terms

### Toward a Grammar-First, Featural Akshara Tokenizer for the Bengali Script

**Author:** Konko Maji ([@konkomaji](https://github.com/konkomaji), work.konkomaji@gmail.com)
**Programme:** Bornomala — Bengali-first Language Technology Research
**Status:** Position paper / working draft (v2, literature-grounded)
**Date:** 24 July 2026

---

## Abstract

Every large language model in wide use today reads Bengali through a lens that was ground for English. The tokenizer, the very first stage of the pipeline, was built on assumptions that happen to be true for the Latin script and happen to be false for Bengali. The result is a quiet, structural penalty that the published literature now measures precisely: the same content costs a Bengali speaker several times more tokens than its English equivalent, roughly five times more under mainstream GPT-era tokenizers, which means more money, less usable context, slower inference, and a model that has only ever seen the language as a shower of fragments (Ahia et al., 2023; Petrov et al., 2023; Arnett et al., 2024).

The usual response is to retrain Byte Pair Encoding (BPE) on Bengali text and call the problem solved. In this paper I argue that this response is not wrong so much as shallow. It fixes the vocabulary while keeping the frame, and the frame is the actual problem. BPE is a compression scheme that discovers structure it does not understand, and it is known to fragment morphemes even in English (Bostrom and Durrett, 2020). Bengali does not need its structure discovered. Bengali is a structure, an abugida, a generative system with explicit and finite rules that Unicode itself has already standardized. The right move is not to teach an English-shaped tool to tolerate Bengali. It is to build a tool whose atomic unit is the unit Bengali actually uses, and whose logic is Bengali's own grammar.

This version of the paper is grounded in a survey of roughly thirty relevant works. That survey confirms three things and complicates a fourth. It confirms the token tax is real and quantified. It confirms that grammar-first and morphology-aware tokenization is an established, working direction, not a fringe idea. It confirms that the orthographic syllable (akshara) is a recognized atomic unit for Indic scripts. And it complicates the novelty claim in a way I want to state honestly up front: every individual ingredient of the design proposed here already exists somewhere in the literature. What does not yet exist, across five independent literature searches, is the specific synthesis, a featural, factored akshara emitted as the tokenizer's own output rather than as an embedding add-on, organized grammar-first with statistics as a fallback, lossless and reversible end to end, and built for Bengali. That combination is the contribution, and this paper is its founding design record.

---

## 1. Introduction

I want to start with a claim that sounds obvious and turns out not to be: a computer has no native language. It does not find English easier than Bengali the way a person raised in one might. To a machine, both are just bytes. So when we observe, as we plainly do, that today's models handle English with ease and Bengali with visible strain, we are not looking at a fact about the languages. We are looking at a fact about the tools, and behind the tools, a fact about the data and the people who built them.

That distinction matters, because it tells us where the fix has to live. If Bengali were genuinely harder for a computer, there would be little to do but wait for bigger models. But it is not harder. Bengali is fully regular where it counts, its script is a finite and rule-governed system, and its grammar is more systematic than English orthography by a wide margin. The strain we see is engineered in, not inherent. And anything engineered in can be engineered out.

This paper traces that strain to its source, the tokenizer, and argues for rebuilding it from the script up rather than adapting it from English down. The argument moved through several stages as I worked it out, and I have kept that shape here because I think the order is the point. We begin by understanding why the English tools work at all. Only then can we see precisely which of their assumptions Bengali breaks, and only then does the native design become obvious rather than merely appealing. Then, because good ideas are cheap and I would rather know early if mine is common, I place the whole design against the existing literature and state plainly what is already done and what is genuinely open.

---

## 2. Why English Is Easy (and Why It Has Nothing to Do With English)

BPE, the tokenization algorithm underneath most modern models, is at heart a compression trick. It starts with the raw bytes of the text and repeatedly glues together the pair of symbols that co-occurs most often, building up from bytes to subwords to whole words. It carries no knowledge of grammar, of morphemes, of meaning. It only counts.

The remarkable thing is that this works so well for English, and it is worth being precise about why, because the reason is almost an accident. English, written in ASCII, has a property so convenient we forget it is a property at all:

```
A   = U+0041 = one byte (0x41)
cat = U+0063 U+0061 U+0074 = three bytes
```

One letter is one codepoint is one byte. The character boundary and the byte boundary are the same boundary. There are no marks that combine with their neighbours, nothing is stored in one order and read in another, and a word is simply its letters laid end to end. When BPE merges bytes, it is merging letters directly, and when it stumbles onto a frequent sequence of letters, that sequence very often turns out to be a real morpheme, `-ing`, `un-`, `-tion`, because English morphology sits close to the surface of its spelling.

So BPE is a discovery algorithm that discovers English structure without understanding it, and gets away with it because English spelling is shallow and linear. This is the crucial framing for everything that follows. BPE does not succeed because it is clever about language. It succeeds because English, at the byte level, is almost trivially well-behaved. The competence belongs to the writing system, not the algorithm. And even for English the fit is imperfect: Bostrom and Durrett (2020) showed that BPE's greedy merges align poorly with morphology, and that a Unigram language-model tokenizer recovers morpheme-like units more faithfully. If the frame leaks for English, it is no surprise it bursts for Bengali.

---

## 3. The Token Tax

Now hand that same algorithm a Bengali sentence.

Bengali lives in the Unicode block U+0980 to U+09FF, and each of its characters occupies three bytes in UTF-8:

```
ক = U+0995 = E0 A6 95   (three bytes)
```

Before any cleverness at all, before a single merge, Bengali already carries three times the byte weight of English for the same number of characters. Arnett, Chang, and Bergen (2024) formalized exactly this as the byte premium, the ratio of UTF-8 bytes needed to encode content-matched text across languages, and computed it for 1,155 languages; scripts in the Bengali-Burmese neighbourhood sit around three to four times a low-premium baseline. This is the floor, the cost that exists before the tokenizer even runs.

On top of the floor sits the deeper injury. Because the model was trained overwhelmingly on English, BPE learned English merges. It never saw enough Bengali to learn that a given run of bytes forms a word, so it falls back to tiny fragments, sometimes down to individual bytes. The published measurements are stark and consistent:

- **Ahia et al. (2023), "Do All Languages Cost the Same?" (EMNLP 2023)** studied OpenAI's API across twenty-two typologically diverse languages and found that speakers of many languages are overcharged while receiving poorer results, and that the overcharged speakers disproportionately come from regions where the service is already least affordable. The most expensive languages cost more than twelve times what English costs for the same content.
- **Petrov et al. (2023), "Language Model Tokenizers Introduce Unfairness Between Languages" (NeurIPS 2023)** measured, on the FLORES-200 parallel corpus across seventeen tokenizers, that the same text can differ in tokenized length by up to roughly fifteen times, and named the consequences explicitly: higher API cost, higher latency, and shorter usable context. This is precisely the chain of harms this paper is concerned with, established independently.
- **Bengali specifically** is measured at roughly five times the English token count under the GPT-3.5 and GPT-4 era `cl100k_base` tokenizer (Jun, 2023, on the MASSIVE parallel dataset). Bengali sits at the top of the three-to-five-times band, not the bottom.

The downstream effects compound:

| Effect | Consequence for a Bengali user |
| --- | --- |
| More tokens per sentence | The same meaning costs three to five times more tokens |
| Higher API cost | You pay more for identical content |
| Shorter effective context | The context window fills faster; less room to think |
| Weaker learning | The model sees shards, not words; semantics are harder to acquire |
| Slower inference | More tokens means more compute per sentence |

One honest nuance, because it matters for how we frame the pitch. The newest tokenizers have narrowed the gap. OpenAI's `o200k_base` (GPT-4o) roughly halved Indic token counts versus `cl100k`, and purpose-built Indic tokenizers do better still: Bengali fertility falls to about 2.05 tokens per word with Sarvam-1 and lower with SUTRA and IndicSuperTokenizer. Arnett et al. (2025) further decompose the remaining inequity into whitespace pre-tokenization, vocabulary allocation, and the inherent bytes-per-character of the script. So the strongest defensible framing is not "Bengali always costs five times more," it is "Bengali costs three to five times more under mainstream tokenizers, and still materially more than one times even under the best current ones, and every one of those best ones is still frequency-driven BPE with the script bolted on." The tax has been reduced. It has not been abolished, and it has not been addressed at the level of the script's actual structure.

None of this is a fact about Bengali. It is a fact about a compressor that was fitted to someone else's language and then asked to work on ours. The tokenizer is a compression scheme shaped by its training data; train it on an English-dominated corpus and you get an English-optimal compressor; everyone else pays the tax. That is the whole mechanism, and naming it plainly is the first step to refusing it.

---

## 4. The Mistake We Keep Making

Here is the turn in the argument, and it is the reason I wanted to write this down rather than simply patch the code.

The instinct, once you see the token tax, is to retrain BPE on a large Bengali corpus. Do that and the numbers improve. Fertility drops, fewer tokens per word, and it feels like progress. But step back and look at what you have actually built. You have taken an algorithm whose entire premise is discover the structure of the language by counting bytes and pointed it at a language whose structure is already written down, explicitly, in the script itself. You are paying an algorithm to rediscover, imperfectly and from data, facts that Bengali states outright.

This is the mistake in its general form: we keep trying to understand Bengali the way the English tools understand English. Even a thoughtful improvement, segmenting into proper orthographic units before running BPE, is still BPE at its heart. It is still frequency-merge compression, still the English idea wearing Bengali clothes. The atom changes; the mind does not.

The survey behind this paper makes the point sharper than I could have on my own, because it shows the whole industry making exactly this move. Every shipping Indic tokenizer I could find, AI4Bharat's IndicBERT and IndicTrans, Sarvam-1, SUTRA, Krutrim, IndicSuperTokenizer, Google's MuRIL, BanglaBERT, is a frequency-driven subword model, BPE or SentencePiece Unigram or WordPiece, trained on Indic data. The differentiation between them is corpus balance, vocabulary size, and pre-tokenization regular expressions plus Unicode normalization. Not one of them makes the script's structure primary. IndicSuperTokenizer's own authors report that they experimented with full morphological segmentation and abandoned it on latency grounds, choosing language-specific pre-tokenization instead. The frame is not questioned anywhere in production. It is only ever tuned.

If we want a tokenizer that is fundamentally Bengali, we cannot port the English frame onto better atoms. We have to ask what Bengali's own frame is, and build from there. BPE is a discovery algorithm for languages whose structure is hidden. Bengali's structure is not hidden. So the native question is not "how do we compress Bengali?" It is "how do we parse it?"

---

## 5. What Bengali Actually Is

Bengali is an abugida, and this single fact reorganizes everything.

In an alphabet like the Latin one, letters are inert beads on a string; meaning accretes only through sequence. An abugida is different. Its organizing unit is not the letter but the akshara, the orthographic syllable, and the akshara is not laid down bead by bead. It is generated by a small productive system:

```
akshara  =  onset  ×  vowel  ×  modifiers

onset    =  a consonant, or a conjunct of consonants joined by hasanta
vowel    =  the inherent vowel অ, or a dependent vowel sign (matra)
modifier =  nasalization, anusvara, visarga, and the like
```

Every valid akshara in the language can be produced from these rules. The system is finite and fully specified. This is the deep contrast with English: English spelling is chaos that must be memorized letter by irregular letter (`cough`, `though`, `through`, `thought` share five letters and agree on nothing), whereas Bengali's script is compositional by design. Give me the consonants, the vowel, and the marks, and the rules tell me the akshara. There is no comparable generative rule for English orthography, and there never can be, because English orthography is a historical accretion, not a system.

This is not a private intuition. The orthographic syllable as a rule-defined atomic unit has a clear precedent: Kunchukuttan and Bhattacharyya (2016) used exactly this unit, the variable-length consonant-cluster-plus-vowel derived by rule, as the atom for statistical machine translation between related Indic languages, and it beat word, morpheme, and character units on small corpora. The idea that the akshara is the right atom is a decade old. What has not happened is carrying it, in factored form, all the way into a modern generative tokenizer for Bengali.

A second, independent line of evidence for the same three-way factorization comes from handwriting recognition rather than tokenization. Alam et al. (2020) built a large Bengali handwritten-grapheme dataset (the corpus behind the Bengali.AI Kaggle grapheme-classification competition) and found that naive character-level classification does not work for Bengali, because the akshara's combinatorics, roughly 168 grapheme roots times 11 vowel diacritics times 7 consonant diacritics, yield on the order of 13,000 distinct surface forms, against roughly 250 character-variant forms for English. Their fix was to stop treating the grapheme as one opaque class and instead label it on three independent axes: root, vowel diacritic, consonant diacritic. That is exactly the onset/vowel/modifier factorization proposed above, arrived at from a completely different problem (OCR-style classification, not tokenization) and a completely different community. Two fields converging on the same three-axis decomposition of the akshara is evidence the factorization is a property of the script, not an artifact of either paper's design choices.

The implication is direct and, once seen, hard to unsee. We should not be compressing Bengali. We should be parsing it, reading each word back into the compositional structure that produced it, and emitting that structure as our tokens. The abugida is not an obstacle to tokenization. The abugida is a tokenizer, one that Bengali scribes have been running by hand for a thousand years. Our job is to implement it.

---

## 6. How Unicode Sees the Two Scripts

To build the parser we need to look carefully at how Bengali is actually encoded, and to hold it against English so the divergence is exact rather than impressionistic.

### 6.1 English: one-to-one-to-one

As we saw, ASCII gives English a clean identity between letter, codepoint, and byte. The character boundary is the byte boundary. This is why byte-level tools work: the bytes are the letters.

### 6.2 Bengali: logical order and the virama model

Bengali is stored in what Unicode calls logical order, which is roughly typing order, and crucially not always visual order. Each character has its own codepoint at three bytes each:

```
ক = U+0995        (consonant ka)
ত = U+09A4        (consonant ta)
র = U+09B0        (consonant ra)
স = U+09B8        (consonant sa)
্  = U+09CD        (HASANTA / virama, the conjunct-former)
ি  = U+09BF        (i-matra, stored AFTER its consonant, READ before it)
ী  = U+09C0        (ii-matra)
```

One visual akshara, the thing a reader perceives as a single indivisible unit, is very often several codepoints in sequence. Consider স্ত্রী (*strī*, "wife"), a single visual blob:

```
স   U+09B8
্    U+09CD    (hasanta)
ত   U+09A4
্    U+09CD    (hasanta)
র   U+09B0
ী    U+09C0    (ii-matra)
────────────────────────────
6 codepoints · 18 bytes · 1 akshara · 1 perceived unit
```

Two features here are exactly the ones that break the English assumptions. First, the matras ি, ে, ৈ and the two-part signs ো, ৌ are stored after their consonant but rendered before or around it; storage order and reading order come apart. Second, the hasanta ্ combines its neighbours into a single conjunct; characters are not inert beads but active combiners. A byte-level tool, blind to both, will happily cut স্ত্রী in the middle of a cluster and split a vowel from the consonant it belongs to. That is the tokenizer disease in one image, and it is exactly what the Indic-tokenizer benchmark literature reports standard BPE doing: shattering conjuncts and matras into meaningless sub-akshara pieces.

### 6.3 The question of the long words

The natural worry at this point is scale. Bengali is famous for its conjuncts, যুক্তাক্ষর, and some stack three or four consonants deep:

```
ক্ষ্ম   (as in লক্ষ্মী)   =  ক U+0995 · ্ U+09CD · ষ U+09B7 · ্ U+09CD · ম U+09AE
ঙ্ক্ষ   (as in আকাঙ্ক্ষা) =  ঙ U+0999 · ্ U+09CD · ক U+0995 · ্ U+09CD · ষ U+09B7
```

If every such cluster had to be enumerated in a lookup table, the task would be enormous and probably unfinishable. But it does not, and this is the finding I most want to record clearly, because it inverts the intuition.

**Multi-conjunct words are not a scaling problem. They are a single regular rule applied more times.**

Look again at the shape of every akshara: a consonant, then any number of (hasanta plus consonant) pairs, then an optional vowel sign, then optional modifiers. That is a regular grammar, and Unicode has already standardized it as the Indic virama model, and at the text-segmentation level in the Extended Grapheme Cluster rules of Unicode Standard Annex 29. Written as a pattern:

```
Akshara  =  Consonant  (Virama Consonant)*  Matra?  Modifier*
         |  Vowel  Modifier*

    Virama   =  U+09CD  (hasanta)
    Modifier =  ং U+0982  |  ঃ U+0983  |  ঁ U+0981
```

The Kleene star on `(Virama Consonant)*` is the whole answer. Two consonants, three, four, however many, are the same rule matched a few more times. A finite-state machine parses this in linear time regardless of cluster depth, and because it only ever groups the codepoints without altering them, the parse is losslessly reversible: rejoin the aksharas and you recover the original bytes exactly. The machinery to do this is not hypothetical. Google's Nisaba library provides finite-state normalization and reversible operations for ten Brahmic scripts (Johny, Gutkin, and Roark, 2021), the AI4Bharat Indic NLP Library ships a working rule-based `orthographic_syllabify` function, and UAX 29 grapheme-cluster segmentation is a deterministic finite-state process. We would be assembling proven components, not inventing segmentation.

Set the two scripts side by side and the feasibility picture actually flips in Bengali's favour on the point that matters most:

| Property | English | Bengali |
| --- | --- | --- |
| character to codepoint | one to one | one to one |
| akshara to codepoints | not applicable | one to many, but rule-governed |
| segmentation rule | byte boundary | regular grammar (finite-state) |
| bounded in practice | yes | yes, rarely beyond four consonants |
| explicit generative rule exists | no (spelling is memorized) | yes (the virama model) |
| lossless round-trip | trivial | yes, codepoints preserved intact |

English has no generative spelling rule at all. Bengali has one, and it is standardized. For the specific job of segmentation, Bengali is the more tractable script, not the less. The long words we were afraid of are the strongest evidence that the grammar-first approach is correct.

---

## 7. A Grammar-First, Featural Architecture

With the diagnosis settled, the design follows. I want to describe it in terms of what it rejects, because the rejections are what make it native rather than derivative.

**It rejects the flat token ID.** BPE emits opaque integers; ক্ষ becomes token number four-thousand-something, and the model has no way to know it is related to ক or to ষ. The native alternative is a featural, factored representation. Each akshara is emitted as its structure, not as a blob:

```
স্ত্রী   →   ( onset = [স, ত, র],  vowel = ী,  modifiers = ∅ )
কি      →   ( onset = [ক],        vowel = ি,  modifiers = ∅ )
```

A model fed this can see, for free, that স্ত্রী and স্ত share an onset, and it generalizes across the entire script without having to learn each combination from scratch. This is phonology-style representation, and it mirrors how the script actually works in a reader's head. It is worth being clear that factored representations of writing systems are known to help: SCRIPT decomposes Korean syllable-blocks into their Jamo consonant and vowel components and reports better capture of grammatical regularity (2026); the Factorizer represents each subword through summed factored embeddings (Samuel and Øvrelid, 2023); and Mersha and Wu (2020) show that decomposing Amharic, itself an alphasyllabary in the same broad family, into consonant-vowel structure helps morphology. The distinction this paper draws, and it is the crux of the novelty, is that all three of those do the decomposition at the embedding layer while leaving the tokenizer untouched. Here the factored akshara is the tokenizer's own output.

**It rejects the word-as-frequency-blob.** Bengali morphology is agglutinative and strikingly regular. A noun is a root with a stack of suffixes, each contributing one grammatical fact:

```
ছেলেটিকে   =  ছেলে (boy)  +  টি (classifier)  +  কে (dative)
বইগুলোতে   =  বই (book)   +  গুলো (plural)     +  তে (locative)
```

The native tokenizer treats these boundaries as real and emits morphemes as units. Its vocabulary is not a frequency-ranked list but a set of roots, a set of functional morphemes, and the rules that compose them. The suffix গুলো is learned once and reused across every noun in the language, which is exactly the kind of generalization BPE can only approximate, and which the morphology-aware literature (Section 9) shows pays off.

**The full pipeline, with no BPE in its core, is then:**

```
raw text
  → normalize          (canonical Bengali; collapse duplicate encodings)
  → akshara parse      (the finite-state virama grammar of Section 6)
  → featural encode    (each akshara becomes its structured tuple)
  → morphological parse (root plus suffix decomposition)
  → structured token stream
  → lossless decode     (regenerate the exact surface form)
```

### 7.1 The one honest concession

A purely rule-based system is brittle in exactly the places the real world is messy: loanwords and code-mixing (the everyday Banglish of "আমি busy আছি"), spelling variation, dialect, typos, and genuinely novel roots. BPE, precisely because it is dumb and statistical, handles all of this without complaint.

So the design does not abolish statistics. It demotes them. The industry default is statistics first, with Bengali bolted on top. The native design inverts the hierarchy:

> **Bengali grammar is primary. Statistics are the fallback.**

The grammar parses the large regular core of the language, which is most of it. A small statistical layer, a conventional BPE trained on Bengali, catches only what the grammar cannot: the loanwords, the noise, the unknowns. Structure leads; statistics serve. This exact control flow is not speculative either. The Turkish "Tokens with Meaning" system (Bayram et al., 2026) runs a rule-based morphological analyzer first and falls back to BPE only when analysis fails, and it is explicitly lossless and reversible. TransLIST (Sandhan et al., 2022) bakes Sanskrit sandhi knowledge into the architecture with neural flexibility for the residue. The pattern works. What no one has done is instantiate it for Bengali with featural akshara tokens.

### 7.2 Formal guarantees

A tokenizer that we intend to trust has to be more than well-motivated; it has to satisfy properties we can state precisely and, ideally, prove. I set out four, because a serious tokenizer must be all four at once, and each is a formal predicate rather than an adjective. The full development, with the token algebra and the property-test contract, is given in the companion document `FORMAL_SPEC.md`; here I state the properties and the single design law that makes them hold.

Let Σ be the set of bytes and let any input be a byte string s ∈ Σ*. The tokenizer is a pair of functions, `encode : Σ* → T*` and `decode : T* → Σ*`.

**Lossless** is the identity `decode(encode(s)) = s` for every s, which is to say `encode` is injective. The way to get this almost for free is a discipline I take as the founding rule: the tokenizer is a segmentation, never a rewriting. If `encode(s)` only ever cuts s into contiguous chunks c₁ through cₖ whose concatenation is exactly s, then decoding is concatenation and losslessness holds by construction, with no theorem to discharge. The tension is normalization, which we want for efficiency but which is many-to-one and therefore destroys injectivity on its own. The resolution is to factor each token into two channels, a `logical_id` that is normalized and model-facing, and a `residual` that carries the small difference needed to restore the exact original bytes. The residual is empty for canonical text, which is almost all text, and non-empty only for the rare variant spelling. Information that normalization would have discarded is not discarded; it is relocated to a channel the model may ignore and the decoder may not.

**Reliable** is totality: `encode(s)` is defined for every s in Σ*, with no input that raises, hangs, or falls through. This is guaranteed by a coverage argument. The layered fallback ends in a byte-level layer whose alphabet is Σ itself, so in the worst case every byte is its own token; a partition therefore always exists, and `encode` is total. Reliability also requires determinism, the same input yielding the same output on every call, which the finite-state construction provides.

**Fast** is a complexity bound: `encode` runs in O(n) time in the length of the input, in a single left-to-right pass with only a small constant of lookahead, and can stream. This follows from the akshara grammar being regular, hence recognizable by a finite-state machine, hence linear and backtracking-free. This is the strongest speed guarantee available, and it is the direct answer to the latency objection that led the IndicSuperTokenizer team to abandon morphological segmentation; the grammar layer is cheap, and the morphological layer must be kept finite-state as well so the bound survives.

**Optimized** is a constrained objective, not a free one. We minimize expected fertility, tokens per word over a corpus, subject to the other three properties and the structural constraint that no token may split an akshara or a morpheme. The statistical fallback does the minimizing, but only within the feasible region the constraints permit. This is where the honest tension of Section 12 lives, stated formally: constraining the merges can only shrink the feasible set, so constrained fertility is greater than or equal to unconstrained BPE fertility. The claim the programme must defend by measurement is therefore not that token counts fall unconditionally, but that the featural representation buys back more in structure and per-token quality than the constraint costs in count.

These reduce to one design law, and I state it as the invariant the implementation must never violate: the tokenizer segments and never rewrites; normalization lives in a residual channel; a byte-level fallback guarantees totality; a regular grammar guarantees linear time; and statistics optimize only inside the constraint box. Held to that law, three of the four properties hold by construction, and the fourth, losslessness under normalization, is enforced by the residual channel and checked exhaustively by a round-trip fuzzer rather than argued.

---

## 8. Complete Language Understanding as a Foundation

None of the above works better than the linguistic knowledge underneath it. A grammar-first tokenizer is only ever as good as its grammar. So before the parser, before any code of consequence, comes the real foundational task: capturing Bengali completely, as machine-readable data rather than prose, so the parser simply reads it.

This is the knowledge base at the heart of the programme. It is organized as small, individually verifiable modules:

| Module | Contents |
| --- | --- |
| Unicode substrate | Every codepoint U+0980 to U+09FF and its role |
| Vowels | 11 independent vowels: অ আ ই ঈ উ ঊ ঋ এ ঐ ও ঔ |
| Consonants | 39 consonants: ক through ৎ, plus ং ঃ ঁ |
| Matras | The dependent vowel signs, each with its position and reorder flag |
| Conjuncts | The যুক্তাক্ষর inventory, with the opaque ligatures (ক্ষ, জ্ঞ) marked and decomposed |
| Numerals | ০ ১ ২ ৩ ৪ ৫ ৬ ৭ ৮ ৯, and word-to-number formation |
| Symbols | Khanda-ta ৎ, anusvara, visarga, chandrabindu, the daṛi ।, and control characters |
| Noun morphology | Classifier, plural, and case marker tables |
| Verb morphology | The conjugation paradigms: root by tense, aspect, person, honorific |
| Pronouns | The honorific-by-case grid across তুই / তুমি / আপনি and their kin |
| Sandhi | The junction rules, needed for reversible morpheme splitting |
| Register | সাধু and চলিত forms, and orthographic variation |

There is a welcome consequence of Section 6 here. Because the akshara grammar is regular and handles conjuncts of any depth structurally, the conjunct module shrinks dramatically. We do not need to enumerate every possible cluster. The finite-state parser produces the members of a transparent conjunct like স্ত্র for free; the table is needed only for the handful of opaque ligatures whose members are not visually recoverable, and for the special positional forms (the ref র্, the ra-phala ্র, the ya-phala ্য). The grind we feared is a fraction of what it looked like.

The character-level modules, vowels through symbols and the Unicode substrate, are finite, tractable, and buildable now. Morphology is larger and, in the case of sandhi, genuinely hard, but the verb paradigm for all its size is highly regular and yields to patient tabulation. Assembling this knowledge base is, I suspect, a contribution in its own right; a complete, correct, machine-readable account of the Bengali writing system does not currently exist in a form a tokenizer can consume, and building one may make Bornomala the authoritative source.

---

## 9. Related Work

I surveyed roughly thirty works across five lines of research. The honest summary is that this is a crowded neighbourhood, and any claim to novelty has to be made against specific, named prior art rather than against a vacuum. I group the work by the line it belongs to and, for each, state how the design here relates to it.

### 9.1 The token tax is established

The premise of Section 3 is not mine to prove; it is proven. Ahia et al. (2023, EMNLP) and Petrov et al. (2023, NeurIPS) are the canonical citations, documenting cost, latency, and context disparities of up to twelve to fifteen times. Arnett, Chang, and Bergen (2024) formalize the byte premium across 1,155 languages, and Arnett et al. (2025) decompose the remaining inequity into pre-tokenization, vocabulary allocation, and encoding. Jun (2023) gives the widely cited five-times figure for Bengali. This line motivates the work; it does not compete with it.

### 9.2 BPE fragments morphemes

Bostrom and Durrett (2020) is the anchor: BPE's greedy merges align poorly with morphology, and Unigram LM does better. This grounds the critique in Section 4 and licenses the whole morphology-aware direction.

### 9.3 Morphology-aware and grammar-first tokenizers

This is the line closest to the architecture in Section 7, and it is active. MorphPiece (Jabbar, 2023) injects a morpheme table and reports large downstream gains, though English-only and preprint. MorphBPE (Asgari et al., 2025) prohibits merges across morpheme boundaries and lowers loss across English, Russian, Hungarian, and Arabic. FLOTA (Hofmann et al., 2022) re-segments with an existing vocabulary to respect morphology. Crucially for the "grammar-first, statistics-fallback" control flow, the Turkish "Tokens with Meaning" system (Bayram et al., 2026) runs morphological analysis first and BPE only on failure, and is explicitly lossless; and TransLIST (Sandhan et al., 2022) does grammar-first Sanskrit segmentation with a neural fallback. My design shares their organizing principle. It differs in unit (featural akshara, not morpheme-over-Latin-subword) and in target (Bengali).

### 9.4 Akshara and grapheme-aware Indic tokenizers

This is the line closest to the atom in Section 5, and it is where Bengali is most conspicuously under-served. SGPE, the "Separate Before You Compress" architecture (Darshana, 2026), is philosophically nearest: it separates the script's rules from statistical compression and guarantees no valid syllable is split, but it is tested on Sinhala and Hindi, still runs a statistical merge phase, and does not claim losslessness. Grapheme Pair Encoding (Velayuthan and Sarveswaran, COLING 2025) seeds BPE on grapheme clusters for Tamil, Sinhala, and Hindi, but keeps the akshara an opaque atom. MorphTok (Brahma et al., ICML TokShop 2025) contributes Constrained BPE, which forbids merges that detach a matra from its consonant, exactly the abugida constraint one wants, but for Hindi and Marathi. And the single most directly comparable work, BengaliBPE (Patwary and Noman, 2025), is Bengali-specific with grapheme-level initialization and morphology-aware merges, but it is BPE with a grapheme seed: the atom is a learned subword, not a factored akshara, and reversibility is not claimed. Its reported win is interpretability, not a decisive downstream accuracy gain. Kunchukuttan and Bhattacharyya (2016) remain the rule-defined orthographic-syllable precedent, for SMT rather than neural tokenization. The production Indic tokenizers, IndicSuperTokenizer (Krutrim, 2025), Krutrim's base tokenizer (2024), Sarvam-1, SUTRA (TWO AI, 2024), Google MuRIL, and AI4Bharat's IndicBERT and IndicTrans, are all frequency-driven subword models; IndicSuperTokenizer's team explicitly rejected full morphological segmentation on latency grounds. Grapheme-atomic Bengali tools that do exist, BnGraphemizer (2024) and GraDeT-HTR (2025), are built for handwriting recognition, not language modeling.

### 9.5 Featural and factored representations

SCRIPT (2026) for Korean Jamo, the Factorizer (Samuel and Øvrelid, 2023), and Amharic alphasyllabary embeddings (Mersha and Wu, 2020) all decompose characters into structured sub-units. Every one of them does so at the embedding layer and leaves the tokenizer unchanged, or uses learned non-linguistic factors. This is the sharpest point of differentiation: the design here makes the factored akshara the tokenizer's output, not an embedding enrichment applied after an unchanged tokenizer.

### 9.6 Tokenizer-free and learned-boundary models

The opposite philosophical pole is worth naming because it shares the critique but draws the reverse conclusion. ByT5 (Xue et al., 2022) and CANINE (Clark et al., 2022) remove the vocabulary and work on bytes or codepoints; MEGABYTE (Yu et al., 2023) and the Byte Latent Transformer (Pagnoni et al., 2024) group bytes into patches, the latter by next-byte entropy; Charformer (Tay et al., 2022) and MANTa (Godey et al., 2022) learn soft segmentation end to end. All of these say: since fixed vocabularies are unfair to non-Latin scripts, remove or learn the boundaries. The grammar-first position is the exact inverse: the boundaries are neither arbitrary nor to be learned from entropy; they are given by the script's grammar, and we should honour them.

### 9.7 Reversibility and canonicalization

Nisaba (Johny, Gutkin, and Roark, 2021) provides finite-state, reversible normalization for Brahmic scripts; the "Language Models over Canonical Byte-Pair Encodings" line (ICML 2025) argues for deterministic canonical encodings. These support the losslessness requirement in Section 7 and provide reusable machinery.

---

## 10. Novelty and Positioning

I will state the novelty the way I would want a reviewer to, which is skeptically.

**Every individual ingredient of this design already exists.** The akshara as atom exists (GPE, MorphTok, BengaliBPE, Kunchukuttan 2016). The matra-binding constraint exists (MorphTok's Constrained BPE). Grammar-first with statistical fallback exists (Turkish "Tokens with Meaning," TransLIST). Reversibility exists (the Turkish hybrid, Nisaba). Featural decomposition into consonant, vowel, and modifier exists (SCRIPT, Factorizer, Amharic embeddings). Finite-state akshara segmentation exists (Nisaba, the Indic NLP Library, UAX 29). If the claim were "no one has thought about grammar-aware Indic tokenization," it would simply be false, and I would deserve to be rejected for making it.

**The synthesis does not exist.** Across five independent searches, no published system simultaneously has all four of the following properties:

| Property | Present in prior art? | Where it is missing |
| --- | --- | --- |
| Featural akshara as the tokenizer's own output | Only as embeddings (SCRIPT, Factorizer) | Indic BPE work keeps the akshara opaque |
| Grammar primary, statistics only as fallback | Yes, for Turkish and Sanskrit | Not for Brahmic featural tokenization |
| Lossless and reversible end to end | Yes, in a few places | Not combined with featural akshara output |
| Built and benchmarked for Bengali | Rarely | The akshara wave skips Bengali; BengaliBPE is opaque-atom BPE |

The defensible contribution, then, is precise: a tokenizer that unifies the featural-embedding line (SCRIPT), the constrained-Indic-BPE line (MorphTok and GPE), and the grammar-first-hybrid line (TransLIST and the Turkish system) into a single, reversible, featural-akshara tokenizer for Bengali, with the script's grammar as the primary generator of tokens and statistics demoted to a fallback for the irregular residue. The positioning sentence I will use, and defend, is: we do not claim to have invented grammar-aware Indic tokenization; we claim to be the first to carry the factored akshara all the way into the tokenizer's output for Bengali, grammar-first and losslessly, and to measure it.

---

## 11. Roadmap

The dependency order is clear, and it lets us prove the thesis early with a small, honest artifact rather than a large act of faith.

1. **Unicode substrate.** The ground truth every later stage sits on.
2. **The atoms.** Vowels, consonants, matras, with codepoints, positions, and reorder flags.
3. **The akshara finite-state parser.** Roughly 150 lines. It reads the codepoint stream, groups it into aksharas by the virama grammar, handles matra reordering and unbounded conjuncts, and round-trips losslessly. This is the foundation stone, and it is testable immediately on the hard words: স্ত্রী, ক্ষ্ম, আকাঙ্ক্ষা, ঋত্বিক.
4. **Measurement.** Benchmark the parser's atoms against the current BPE-based tokenizer and against the published baselines, Sarvam-1 (Bengali fertility about 2.05), SUTRA, IndicSuperTokenizer, and BengaliBPE, on the MotherTongueIndex metrics: fertility, byte premium, and downstream task score. Prove the native atom is competitive before building anything on top of it.
5. **Featural encoding and morphology.** Only once steps 1 through 4 stand.

The discipline in this order is deliberate. Each stage is verifiable on its own, and the whole approach is falsifiable at step 4. If the native atom does not beat the strong baselines on the numbers, we will know before we have invested in the morphology, and we can say so.

---

## 12. Risks and Open Problems

I would rather state the difficulties plainly than have them discovered later. Two of these are risks the literature actively warns about, and I take them seriously.

**The fertility tension is the make-or-break risk.** The entire pitch of this work is reducing tokens per word. But morphology-aware tokenizers frequently increase it: the Turkish "Tokens with Meaning" system trades roughly 1.5 times more tokens per word for its linguistic fidelity. If splitting বইগুলোতে into three morphemes produces more tokens than a good BPE that keeps it whole, I have made the token tax worse in the name of fixing it. The featural-akshara layer must be shown to lower fertility while preserving structure, and this is not guaranteed by anything in the design. It is the first number I must produce, and I must be willing to report it if it goes the wrong way.

**Latency is a real, documented obstacle.** IndicSuperTokenizer's authors tried full morphological segmentation and dropped it because it was too slow for production. A grammar-first tokenizer whose parser and morphological analyzer cannot keep up with a training or serving pipeline is a research toy, not a tool. The finite-state akshara layer is cheap; the morphological analyzer is where cost accumulates, and it must be engineered for speed from the start.

**BengaliBPE already occupies part of the ground.** A Bengali-specific, grapheme-initialized, morphology-aware tokenizer was published in November 2025. I must differentiate against it concretely, on the factored-output and reversibility axes, and ideally beat it on measured downstream tasks rather than on interpretability alone. Being second to a weaker version of an idea is a real danger; the answer is to be decisively different and decisively better on the numbers.

**Reversibility must be treated as sacred.** Normalization and morpheme splitting both change the surface form, and sandhi changes spelling at morpheme joins. Encode followed by decode must be the identity function, including for non-canonical and misspelled input, or the tokenizer is not trustworthy no matter how good its other numbers look. This is an engineering discipline as much as a research question, and it constrains every earlier stage.

**Downstream models must accept structured input.** A featural token stream is non-standard, and any model trained on it needs its input and output layers designed to match. That is the price of leaving the English stack, and in a real sense it is also the point, but it is a cost and I record it as one.

None of these is a reason not to proceed. They are the shape of the work, and the first four are exactly what step 4 of the roadmap is designed to expose early.

---

## 13. Conclusion

The problem was never that computers find Bengali hard. It is that we handed Bengali to a tool built on the quiet conveniences of English and then read the tool's struggle as the language's fault. BPE works for English by a happy accident of ASCII and shallow orthography; it has no such luck with an abugida, and the published measurements of the token tax, three to five times for Bengali under mainstream tokenizers, show the cost of pretending otherwise. No amount of retraining changes what BPE fundamentally is, a compressor that guesses at structure from counts.

Bengali does not need to be guessed at. It is a generative system with finite, explicit, Unicode-standardized rules, and its notorious multi-conjunct words turn out to be the easiest evidence for this: a single regular grammar, parsed in linear time, reversible by construction. The literature confirms the direction, grammar-first and morphology-aware tokenization works, and it also confirms the gap, no one has carried the factored akshara into the tokenizer's own output for Bengali, grammar-first and losslessly. The path forward is not to make an English tool tolerate Bengali but to build a Bengali tool, resting on a complete and machine-readable understanding of the language itself, and to measure it honestly against strong baselines and against the fertility risk that could sink it.

That is the tokenizer Bornomala is here to build, and reading the script on its own terms is where it begins.

---

## Acknowledgements

This work is authored by **Konko Maji** as part of the **Bornomala** programme for Bengali-first language technology. The line of argument, from the diagnosis of the token tax through the abugida-as-grammar reframing to the grammar-first featural architecture, its knowledge base, and the honest novelty positioning against the surveyed literature, is his. The paper documents the reasoning behind the tokenizer at the centre of that programme and is intended as its founding design record.

Correspondence: work.konkomaji@gmail.com · [github.com/konkomaji](https://github.com/konkomaji)

---

## References

### Tokenization inequity and the byte premium
- Ahia, O., Kumar, S., Gonen, H., Kasai, J., Mortensen, D. R., Smith, N. A., Tsvetkov, Y. (2023). *Do All Languages Cost the Same? Tokenization in the Era of Commercial Language Models.* EMNLP 2023. https://arxiv.org/abs/2305.13707 · https://aclanthology.org/2023.emnlp-main.614/
- Petrov, A., La Malfa, E., Torr, P. H. S., Bibi, A. (2023). *Language Model Tokenizers Introduce Unfairness Between Languages.* NeurIPS 2023. https://arxiv.org/abs/2305.15425 · Tool: https://aleksandarpetrov.github.io/tokenization-fairness/
- Arnett, C., Chang, T. A., Bergen, B. K. (2024). *A Bit of a Problem: Measurement Disparities in Dataset Sizes Across Languages.* https://arxiv.org/abs/2403.00686
- Arnett, C., Chang, T. A., Biderman, S., Bergen, B. K. (2025). *Explaining and Mitigating Crosslingual Tokenizer Inequities.* https://arxiv.org/abs/2510.21909
- Jun, Y. (2023). *All Languages Are Not Created (Tokenized) Equal.* https://www.topbots.com/all-languages-are-not-tokenized-equal/

### BPE critique and morphology-aware tokenization
- Bostrom, K., Durrett, G. (2020). *Byte Pair Encoding is Suboptimal for Language Model Pretraining.* Findings of EMNLP 2020. https://arxiv.org/abs/2004.03720 · https://aclanthology.org/2020.findings-emnlp.414/
- Jabbar, H. (2023). *MorphPiece: A Linguistic Tokenizer for Large Language Models.* https://arxiv.org/abs/2307.07262
- Asgari, E., El Kheir, Y., Sadraei Javaheri, M. A. (2025). *MorphBPE: A Morpho-Aware Tokenizer Bridging Linguistic Complexity for Efficient LLM Training Across Morphologies.* https://arxiv.org/abs/2502.00894
- Hofmann, V., Schütze, H., Pierrehumbert, J. (2022). *An Embarrassingly Simple Method to Mitigate Undesirable Properties of Pretrained Language Model Tokenizers (FLOTA).* ACL 2022. https://aclanthology.org/2022.acl-short.43/
- Bayram, M. A., et al. (2026). *Tokens with Meaning: A Hybrid Tokenization Approach for Turkish.* https://arxiv.org/html/2508.14292
- Pan, Y., Li, X., et al. (2020). *Morphological Word Segmentation on Agglutinative Languages for Neural Machine Translation.* https://arxiv.org/abs/2001.01589
- *Unsupervised Morphological Tree Tokenizer* (2024). https://arxiv.org/abs/2406.15245
- *The Importance of Morphology-Aware Subword Tokenization for NLP Tasks in Slovak* (2026). Expert Systems with Applications. https://www.sciencedirect.com/science/article/pii/S0957417426004057

### Akshara / grapheme-aware and Indic tokenizers
- Darshana, K. (2026). *Separate Before You Compress: The WWHO Tokenization Architecture (SGPE).* https://arxiv.org/abs/2603.25309
- Velayuthan, M., Sarveswaran, K. (2025). *Egalitarian Language Representation in Language Models: It All Begins with Tokenizers (Grapheme Pair Encoding).* COLING 2025. https://arxiv.org/abs/2409.11501 · https://aclanthology.org/2025.coling-main.400.pdf
- Brahma, M., Karthika, N. J., Singh, A., et al. (2025). *MorphTok: Morphologically Grounded Tokenization for Indian Languages (Constrained BPE).* TokShop @ ICML 2025. https://arxiv.org/abs/2504.10335
- Patwary, F. A., Al Noman, A. (2025). *Evaluating Subword Tokenization Techniques for Bengali: A Benchmark Study with BengaliBPE.* https://arxiv.org/abs/2511.05324
- Kunchukuttan, A., Bhattacharyya, P. (2016). *Orthographic Syllable as Basic Unit for SMT between Related Languages.* EMNLP 2016. https://aclanthology.org/D16-1196/ · https://arxiv.org/abs/1610.00634
- Rana, S., Menezes, et al. (Krutrim AI) (2025). *IndicSuperTokenizer: An Optimized Tokenizer for Indic Multilingual LLMs.* https://arxiv.org/abs/2511.03237 · https://openreview.net/forum?id=CSrGFB070m
- Krutrim AI (2024). *Krutrim LLM / Tokenizer.* https://arxiv.org/abs/2407.12481
- Shravan, R. (2026). *BrahmicTokenizer-131K.* https://arxiv.org/html/2605.29379
- *Multilingual Tokenization through the Lens of Indian Languages* (2025). https://arxiv.org/abs/2506.17789
- *Evaluating Tokenizer Performance of Large Language Models Across Official Indian Languages* (2024). https://arxiv.org/abs/2411.12240
- BnGraphemizer (2024). IEEE. https://ieeexplore.ieee.org/document/10456463/
- GraDeT-HTR (2025). EMNLP 2025 Demos. https://aclanthology.org/2025.emnlp-demos.52/
- AI4Bharat. *IndicBERT / IndicTrans2 / Indic NLP Catalog.* https://github.com/AI4Bharat/IndicBERT · https://github.com/AI4Bharat/IndicTrans2 · https://ai4bharat.github.io/indicnlp_catalog/

### Grammar-first, finite-state, and Sanskrit/Brahmic infrastructure
- Sandhan, J., Singha, A., Rao, N., Samanta, S., Behera, L., Goyal, P. (2022). *TransLIST: A Transformer-Based Linguistically Informed Sanskrit Tokenizer.* Findings of EMNLP 2022. https://arxiv.org/abs/2210.11753
- Johny, C., Gutkin, A., Roark, B. (2021). *Finite-State Script Normalization and Processing Utilities: The Nisaba Brahmic Library.* EACL 2021 Demos. https://aclanthology.org/2021.eacl-demos.3/
- *A Finite State and Rule-based Akshara-to-Prosodeme (A2P) Converter in Hindi* (2017). https://arxiv.org/abs/1705.01833
- The Unicode Consortium. *Unicode Standard Annex 29: Unicode Text Segmentation (Extended Grapheme Clusters).* https://unicode.org/reports/tr29/
- The Unicode Standard. *Bengali block (U+0980–U+09FF) and the Indic virama model.*

### Featural / factored / compositional representations
- Alam, S., Reasat, T., Sushmit, A. S., Siddiquee, S. M., Rahman, F., Hasan, M., Humayun, A. I. (2020). *A Large Multi-Target Dataset of Common Bengali Handwritten Graphemes.* ICDAR 2021 (Springer LNCS vol. 12917, ch. 26); preprint https://arxiv.org/abs/2010.00170
- *SCRIPT: Subcharacter Compositional Representation Injection for Korean* (2026). https://arxiv.org/html/2604.12377
- Samuel, D., Øvrelid, L. (2023). *Tokenization with Factorized Subword Encoding (Factorizer).* Findings of ACL 2023. https://arxiv.org/abs/2306.07764
- Mersha, M., Wu, J. (2020). *Morphology-rich Alphasyllabary Embeddings (Amharic).* LREC 2020. https://aclanthology.org/2020.lrec-1.315/
- *Language Models over Canonical Byte-Pair Encodings* (2025). ICML 2025. https://icml.cc/virtual/2025/poster/44596

### Tokenizer-free and learned-boundary models
- Xue, L., Barua, A., Constant, N., Al-Rfou, R., Narang, S., Kale, M., Roberts, A., Raffel, C. (2022). *ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models.* TACL. https://arxiv.org/abs/2105.13626
- Clark, J. H., Garrette, D., Turc, I., Wieting, J. (2022). *CANINE: Pre-training an Efficient Tokenization-Free Encoder.* TACL. https://arxiv.org/abs/2103.06874
- Yu, L., Simig, D., Flaharty, C., Shleifer, S., Zettlemoyer, L., et al. (2023). *MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers.* NeurIPS 2023. https://arxiv.org/abs/2305.07185
- Pagnoni, A., et al. (2024). *Byte Latent Transformer: Patches Scale Better Than Tokens.* https://arxiv.org/abs/2412.09871
- Tay, Y., Tran, V. Q., Ruder, S., et al. (2022). *Charformer: Fast Character Transformers via Gradient-based Subword Tokenization.* ICLR 2022. https://arxiv.org/abs/2106.12672
- Godey, N., et al. (2022). *MANTa: Efficient Gradient-Based Tokenization for End-to-End Robust Language Modeling.* Findings of EMNLP 2022. https://arxiv.org/abs/2212.07284

### Bengali and Indic LLM ecosystem (models and tokenizers referenced)
- Sarvam AI (2024). *Sarvam-1.* https://www.sarvam.ai/blogs/sarvam-1 · https://huggingface.co/sarvamai/sarvam-1
- TWO AI (2024). *SUTRA multilingual tokenizer.* https://www.two.ai/blog/understanding-sutra-s-multilingual-tokenizer · https://arxiv.org/abs/2411.12240
- Google Research (2021). *MuRIL: Multilingual Representations for Indian Languages.*
- CSE BUET (2021–2022). *BanglaBERT.* https://github.com/sagorbrur/bangla-bert · https://www.emergentmind.com/topics/banglabert-model
- OpenAI (2024). *tiktoken (cl100k_base, o200k_base).* https://github.com/openai/tiktoken

---

*© 2026 Konko Maji / Bornomala. Working draft, v2 (literature-grounded); circulated for discussion.*
