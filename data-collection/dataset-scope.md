# Bengali text corpus: era + copyright scoping

Not legal advice. This is a working scope, sourced from public information as
of 2026-08, to be verified per-work before any large-scale ingestion. Rule E4
applies here same as everywhere else in this project: never present an
estimate as measured, never present a legal read as a legal opinion.

## 1. The rule that actually matters

**Indian Copyright Act 1957: literary works are protected for the author's
life plus 60 years, counted from 1 January of the year after death.** So a
work is public domain in India in year `Y` if `death_year <= Y - 61`.

> **For 2026: an author who died in 1965 or earlier is public domain now.**
> An author who died in 1966 becomes public domain on 2027-01-01 - track
> those separately, they're one year from clearing.

Exceptions to hold in mind, not exceptions to the headline rule:
- Government works, photographs, films, sound recordings: 60 years from
  **publication**, not death - a different clock.
- Anonymous/pseudonymous works: 60 years from publication, unless the
  author is identified within that period, at which point the life+60 rule
  re-applies.
- Joint authorship: clock starts at the **last** surviving author's death.
- **A public-domain original text does not make every edition of it public
  domain.** A modern critical edition, a variorum apparatus, a fresh
  translation, a typeset "complete works" volume, or a scan with added
  OCR/metadata can carry its own separate copyright on top of a PD
  underlying text. Bichitra (below) is the concrete example of this in my
  own source list, not a hypothetical.

## 2. Era taxonomy

**Start point: the Charyapada, the earliest documented Bengali writing.**
47 mystical verses composed by Buddhist mahasiddhas, dated by scholarship
to roughly the 8th-12th century CE (no fully agreed single date), written
in Old Bengali / *sandhya bhasha* on palm leaf. The manuscript itself was
lost for centuries and rediscovered by Haraprasad Shastri in the Nepal
royal court library in 1907, first published by Vangiya Sahitya Parishad
(Kolkata) in 1916 as *Hajar Bachharer Purana Bangala Bhasay Bauddhagan O
Doha*. Flag honestly: linguists also treat Old Bengali/Charyapada-era
Abahatta as a shared ancestor of Bengali, Assamese, and Odia, so its
classification as "Bengali" specifically is itself part of an ongoing
scholarly debate, not a settled fact - document that debate rather than
picking a side. Copyright status: the 8th-12th century verses are PD by
a margin of a millennium; the 1916 Vangiya Sahitya Parishad *edition* is
old enough to be PD too, but as always, check the specific edition used.

Five buckets, chosen to line up with the project's own framing (spec
section 4.3, 10.3) and with what's actually separately sourceable:

| Era | Range | What it is |
|---|---|---|
| **Old/Middle Bengali** | 8th-12th c. CE (Charyapada) through ~1800 | Charyapada, Vaishnav padavali, Mangalkavya, Chaitanya-era hagiography, other manuscript-tradition literature before print |
| **Pre-Vidyasagar** | ~1800-1850 | Early Fort William College prose, Rammohan Roy's reform tracts - print era begins, prose standardisation not yet done |
| **Colonial / Bengal Renaissance** | 1850-1947 | Vidyasagar's prose standardisation onward through independence - the spec's own "literary corpus 1850-1950" gap (section 4.3), the 30%-share letterpress category in Track B (section 10.3) |
| **Post-independence** | 1947-1991 | Partition, land-reform era, Naxalbari, through pre-liberalisation |
| **Modern** | 1991-present | Liberalisation onward; almost entirely still in copyright |

The first three buckets are firmly **public domain by original-authorship
date**; the fourth is mixed and must be checked author-by-author; the fifth
is **not** public domain as a rule - treat every post-1991 text as
copyrighted unless proven otherwise.

## 3. Genre checklist (what "all kind of cultural literature" breaks into)

Not just novels. Per era, look for:

- Literary: novel, short story, poetry, drama
- Religious/devotional: Vaishnav padavali, Mangalkavya, hagiography, Pir
  sahitya (Muslim-Bengali devotional narrative, see Battala below)
- Folk/oral, published: rupkatha collections (Thakurmar Jhuli-type),
  proverb and riddle collections, ballad (gitika) collections, regional
  song-form transcriptions (Bhadu, Tusu, Jhumur, Bhawaiya, Gambhira -
  section 8 below has these per-dialect)
- Essay, polemic, reform tract
- Periodical/journalism: literary magazines (Bangadarshan, Sabuj Patra,
  Bharati), newspapers (Amrita Bazar Patrika, Jugantar)
- Letters, diaries, autobiography (Rassundari Devi's *Amar Jiban* is the
  first published Bengali woman's autobiography, 1876 - squarely PD)
- Administrative/documentary: land and revenue records, municipal
  documents (already `forms_tables` in Track B's OCR category set)
- Popular/commercial print culture: advertisements, almanacs (panjika),
  primers - CSSSC's Heidelberg collection explicitly includes this stratum
- **Battala literature**: cheap, mass-market Kolkata popular print, 19th c.
  onward - distinct register and readership from the "Renaissance" literary
  canon, includes Musalmani Bangla (Perso-Arabic-inflected register) Pir
  sahitya like the Johuranama tradition below. Worth its own `category`
  value, not folded into generic "literary" - it is a different linguistic
  register (closer to spoken/colloquial than sadhu bhasha) and currently
  absent from every published corpus per the spec's own gap analysis
- Pala/kissa performance texts: narrative verse written for oral
  performance (panchali/pala form) - Bonbibir Johuranama tradition is the
  concrete Sundarban example, section 8
- Grammar/lexicon as primary source: dialect dictionaries and linguistic
  survey specimens are themselves a genre worth collecting, not just a
  research aid - see section 8, Grierson's specimens are literally
  transcribed dialect text
- **Medieval verse-epic translation**: Krittibas Ojha's *Krittivasi
  Ramayan* (c. 1381-1461, Ramayana pancali form) and Kashiram Das's
  *Kashidasi Mahabharat* (17th c., first four of eighteen parvas) - not
  minor texts, contemporary sources call the Krittibasi Ramayan "by far the
  most popular book in Bengal" (1911). Bigger single-author volume than
  the Charyapada, squarely PD, already digitised on DLI/archive.org -
  this is a priority pull, not a footnote
- Early science writing / science fiction: Jagadish Chandra Bose's
  *Palatak Tufan* (1896, expanded 1921) is credited as the founding text
  of Bangla science fiction - Bose (d. 1937) is PD, opens a genre almost
  nobody's corpus covers
- Detective and popular genre fiction: colonial-era Bengali detective
  fiction (Panchkori Dey, Dinendra Kumar Roy) and historical fiction
  (Bankim's historical novels are already in section 4's roster) -
  distinct register from literary-canon prose, worth its own `category`
  value rather than folding into generic "novel"
- Children's literature: Sukumar Ray's nonsense verse (*Abol Tabol*,
  1923), Upendrakishore Ray Chowdhury's tales - Sukumar Ray (d. 1923) and
  Upendrakishore (d. 1915) both PD; distinct child-register vocabulary and
  syntax from adult literary prose, worth keeping separate for a tokenizer
  corpus that wants register diversity
- Travelogue (ভ্রমণকাহিনী): a real, named 19th-20th c. Bengali genre,
  e.g. Rabindranath's own travel writing - distinct narrative register
  from fiction, worth its own value
- Genealogical/religious record: kulapanji (Kulin Brahmin lineage
  records) - a real written-document genre, structurally closer to the
  spec's `forms_tables` OCR category than to prose
- Culinary/domestic manual: e.g. Bipradas Mukhopadhyay's *Mistanna Pak*
  (1906, sweets/confectionery) - a concrete, dateable, PD example of a
  genre (recipe/how-to prose) otherwise absent from every table in this
  document so far

## 4. Representative public-domain figure roster

**Representative, not exhaustive** - a true exhaustive list is thousands of
names and is exactly the kind of thing that should be generated
mechanically (section 6), not hand-typed. These are sourced, death-dated
examples across categories, all clearing the death-year-<=-1965 rule as of
2026 unless flagged otherwise.

| Name | Died | PD in India | Category | What's documented |
|---|---|---|---|---|
| Rammohan Roy | 1833 | yes | Reformer | Reform tracts, correspondence |
| Ishwar Chandra Vidyasagar | 1891 | yes | Reformer/prose | Prose standardisation, essays, primers |
| Ramakrishna Paramahamsa | 1886 | yes | Religious | Recorded sayings (Kathamrita) |
| Michael Madhusudan Dutt | 1873 | yes | Poet/playwright | Poetry, drama |
| Bankim Chandra Chattopadhyay | 1894 | yes | Novelist | Novels, essays, Bangadarshan editorship |
| Swami Vivekananda | 1902 | yes | Religious/writer | Lectures, letters, essays |
| Rassundari Devi | c.1899 | yes | Autobiography | *Amar Jiban* (1876), first published Bengali woman's autobiography |
| Khudiram Bose | 1908 | yes | Freedom fighter | Final letters (held at West Bengal State Archives) |
| Bagha Jatin (Jatindranath Mukherjee) | 1915 | yes | Freedom fighter | Correspondence, movement records |
| Chittaranjan Das | 1925 | yes | Freedom fighter/writer | Speeches, poetry, political writing |
| Swarnakumari Devi | 1932 | yes | Novelist/editor | Novels (*Chinnamukul*, *Kahake?*), edited *Bharati* |
| Begum Rokeya Sakhawat Hossain | 1932 | yes | Writer/reformer | *Sultana's Dream* (1908), *Matichur*, *Padmarag*, *Abarodhbasini* |
| Surya Sen | 1934 | yes | Freedom fighter | Movement correspondence and records |
| Rabindranath Tagore | 1941 | yes (since 2001/2002) | Novelist/poet/essayist | Full corpus - novels, poetry, essays, songs, letters |
| Sarat Chandra Chattopadhyay | 1938 | yes | Novelist | Novels, short stories |
| Sri Aurobindo (Aurobindo Ghosh) | 1950 | yes | Freedom fighter/philosopher | Political writing, philosophy, poetry |
| Subhas Chandra Bose | 1945 (disputed) | yes under either account | Freedom fighter | Speeches, letters, *The Indian Struggle* |
| Bibhutibhushan Bandyopadhyay | 1950 | yes | Novelist | Novels (*Pather Panchali*), short stories |
| Jibanananda Das | 1954 | yes | Poet | Poetry, essays |
| Manik Bandopadhyay | 1956 | yes | Novelist | Novels, short stories |

**Flagged, do not treat as clear:**
- **Kazi Nazrul Islam** (d.1976) - some individual works carry PD-India /
  PD-Bangladesh tags on Wikimedia Commons and at least one 1964 story
  collection is marked public domain by the Digital Library of India, but
  his 1972 move to Dhaka and Bangladesh's separate nationalisation of his
  literary rights make the general status genuinely mixed across his
  corpus. **Check per-work, don't assume the whole corpus clears.**
- **Tarashankar Bandyopadhyay** (d.1971), **Ashapurna Devi** (d.1995),
  and any author who died 1966 or later: **not yet public domain in
  India**, full stop, until their own death-year+61 date arrives.

## 5. Concrete source catalog

| Source | Coverage | Format | License/access note |
|---|---|---|---|
| **Bengali Wikisource** (bn.wikisource.org) | Renaissance-era PD texts onward, ongoing volunteer transcription | Already-digital plain text | PD text under Wikisource's own PD-verification norms; safest single source since transcription (not just scan) already exists |
| **Digital Library of India / archive.org** (`in.ernet.dli.*` prefix) | Large 19th-20th c. Bengali book scans, catalogued by DLI | Page-image scans | Per-item; many explicitly marked "In Public Domain" in DLI metadata (confirmed for at least one Nazrul volume) - **check each item's own marking, don't assume the whole DLI Bengali set is PD** |
| **CSSSC / Heidelberg CrossAsia collection** | Early printed Bengali literature and periodicals, 1800-1950, plus Assamese and visual/commercial print culture | Page-image scans | Described as unrestricted online access, joint CSSSC + Univ. of Heidelberg + British Library Endangered Archives Programme + Center for Research Libraries |
| **Bichitra** (bichitra.jdvu.ac.in, Jadavpur Univ.) | Tagore's complete works, manuscript + printed, ~140,000 pages | Page-image scans + transcription apparatus | Underlying Tagore text is PD, but the site states **"All rights reserved, School of Cultural Texts and Records, Jadavpur University"** on its own variorum/apparatus - use as a Track B image source with care, don't redistribute their transcription layer as if it were ours |
| **National Digital Library of India** (ndl.gov.in) | Aggregator, 40M+ books incl. Bengali | Mixed | Per-item license, no blanket status - treat as a discovery index, verify each hit individually |
| **West Bengal State Archives** | Freedom-movement correspondence (e.g. Khudiram Bose's letters) | Archival documents | Access/reproduction terms are institutional, not stated online - contact before assuming reuse rights |

## 6. What needs explicit permission or licensing (not scrapeable)

- Every author who died 1966 or later, until their own PD date
- All living authors
- Modern periodicals and news archives (Anandabazar Patrika, Bartaman,
  etc.) - copyrighted as compiled works even when individual old articles
  might otherwise be old enough, because the archive itself is a
  separately-rights-held compilation
- Bichitra's own transcription/critical-apparatus layer (see above) -
  Track B should re-derive transcripts from the page images via my own
  OCR pipeline, not copy theirs
- Any modern "complete works" (Rachanabali) typeset edition - the
  typesetting and editorial apparatus is a separate copyright from the
  underlying PD text

## 8. Dialect-specific written sources (text, not field speech - that's deferred per current instruction)

Field elicitation/speech collection (spec section 11.3, Track C) is explicitly
**deferred for now**. This section is the different, narrower question: what
*already-written* material exists per West Bengal dialect group, usable as
text right now. Findings, honestly stated - most of this region's dialect
material is oral-performance tradition, not a written-text tradition, and
that gap is itself worth recording, not glossed over.

| Dialect group | Written/text-fixed material found | Era / date | PD status | Source |
|---|---|---|---|---|
| **Rāṛhī** | *Is* the literary standard (sadhu/chalit Bengali derives from it - spec section 5) - the entire colonial/Renaissance canon in section 5's roster is de facto Rarhi-substrate text. Separately, **Baul-Fakir song tradition** (Birbhum, Murshidabad, Nadia, Bankura) has real published collections, e.g. Upendranath Bhattacharya's *Banglar Baul O Baul Gaan* (monograph + song texts) | Baul song collections mid-20th c. onward | Standard literary canon: yes (see section 4). Baul monographs: check per-edition, mid-20th-c. compilations may still be in copyright | Standard sources in section 5; Baul monographs via library/NDLI search |
| **Manbhumi/Jharkhandi** | **Bhadu, Tusu, Jhumur** song lyrics - transcribed and anthologized in academic folklore studies (Shodhganga theses, journal articles cited above), not a standalone published lyric-book tradition as far as this pass found. Most individual songs are **anonymous, composer unknown, transmitted orally** - i.e. genuinely folk, not single-authored | Songs undated/traditional; academic transcriptions 20th-21st c. | Traditional song text itself: PD as anonymous/undated folk material. Academic transcription/anthology layer: check per-publication, recent theses/journals are in copyright | Shodhganga (INFLIBNET) theses, folklore journal articles - discovery layer, verify each anthology's own rights before reuse |
| **Varendrī** | **Gambhira** song-theatre - has an actual named early print precedent: *Addyer Gambhira*, published by Krishnacharan Sarkar, Malda, 1913. **Alkap** folk theatre (Malda/Murshidabad) is explicitly **unscripted** - sources describe it as having "no written script," improvised dialogue around a known story - not a text source, only relevant later for speech/performance capture | Gambhira: 1913 print. Alkap: n/a, oral only | *Addyer Gambhira* (1913): PD by age. Alkap: not applicable, no text exists to collect | 1913 Gambhira volume - locate via NDLI/library catalogue search, not yet confirmed digitised |
| **Sundarbanī** | **Bonbibir Johuranama** tradition - three named, dated texts: Boinuddin's *Bonbibi Jahuranama* (1877-78), Munshi Mohammad Khater's version (1881), Mohammad Munshi's version (1899), all adapting Krishnaram Das's 1686 *Ray-mangal*. Written in **Musalmani Bangla** (Perso-Arabic-inflected register), Battala print tradition, panchali/kissa verse form - a genuinely distinct written register, not just Standard Bengali with local color | 1877-1899 (adapting a 1686 source) | Firmly PD, 125+ years old, named authors long deceased | Battala-literature archives/catalogues (National Library Kolkata, Bangiya Sahitya Parishat); not yet confirmed on Wikisource/DLI - to locate |
| **Kāmrūpī/Rangpuri** | **Bhawaiya** song lyrics - the one dialect group with a real, old, named written specimen: **Grierson personally collected two Bhawaiya (Rangpuri) lyrics in 1898 and published them in the *Linguistic Survey of India*, Vol. V Part I (1903)** - actual transcribed dialect text, not a paraphrase | 1898 collection, 1903 publication | Firmly PD (120+ years, Grierson d. 1941, both clocks cleared) | *Linguistic Survey of India* Vol. V Part I, 1903 - already located on archive.org (section 5's LSI entries), confirmed accessible now |

**Cross-cutting find**: the **Linguistic Survey of India** (Grierson, 1898-1928,
all volumes on archive.org under the `in.ernet.dli.*` prefix) is the single
best *written, dated, per-dialect* text source this pass found - it
predates every dialect corpus in Appendix B.2 of the spec by a century, is
unambiguously PD, and Vol. V Part I specifically covers Bengali/Assamese
dialect specimens including the Rangpuri material above. Worth pulling in
full, not just as a citation - **recommended as the actual first
`ocr_ground_truth` / `general_text` source to process for dialect-adjacent
written text**, ahead of any new field collection.

## 9. The mechanical roster: done, not just proposed

Section 4's roster is representative by construction - I never intended to
hand-type an exhaustive "every historical Bengali figure whose writing is
PD" list, because that isn't something I can do honestly by hand. Wikidata
carries structured death-date + occupation + language-of-work data and is
queryable by SPARQL, so I wrote the query and ran it: `data-collection/
scripts/fetch_pd_bengali_writers.sh` filters occupation in {writer, poet,
novelist, playwright, journalist, essayist}, native or spoken/written
language = Bangla, death year <= 1965 (2026's cutoff, section 1's rule),
and writes the result to `data-collection/pd-bengali-writers.csv`. Same
"mechanical over manual" principle I already apply via `bntok.normalize`
and `schema.py`'s validator elsewhere in this repo.

**Result: 867 named, dated, sourced people**, each with a Wikidata QID and
permalink for traceability, ranging from Ramai Pandit (d. c. 1200) through
a cluster of writers who died in 1965 itself. I did not spot-check every
row - that's still on me to do before treating any single name as
confirmed rather than "Wikidata says so."

**Honest limits, stated plainly, not smoothed over:**
- This is a **floor, not a ceiling**. Wikidata only returns people who (a)
  have a Wikidata item at all, (b) have `date of death` filled in, (c) have
  `native language` or `languages spoken/written` set to Bangla specifically
  (not just "born in Bengal" - plenty of real historical Bengali writers
  are missing one of these three and simply don't appear), and (d) have an
  occupation tagged from my exact six-item list. Freedom fighters, reformers,
  and religious figures whose primary Wikidata occupation isn't
  literary are excluded here by design - that's what section 4's separate
  hand-curated table is for.
  A person absent from this CSV is not thereby "not public domain" -
  it just means Wikidata's metadata didn't catch them. This script is a
  real, reproducible starting point, not the final word on who's PD.
- Re-run with a later `cutoff_year` argument each subsequent year to pick
  up authors who died in 1966, 1967, ... as their own PD dates arrive -
  the script takes the year as its one argument for exactly this reason.
- Every row still needs the section-1 edition caveat applied before use:
  the *person's* death date being old enough doesn't clear a specific
  modern reprint or annotated edition of their work - find and use an old
  edition, or the original periodical printing, not a 2020s "collected
  works" volume.

## 10. Confirmed sources: real archive.org identifiers, verified, ready to pull

Sections 3 and 8 flagged the medieval epics and the *Linguistic Survey of
India* as priority pulls. I went and found them - not just cited them as
"should exist somewhere." Every identifier below I checked myself against
archive.org's metadata API and, for the epics and the LSI volume, actually
downloaded the OCR text and confirmed it's genuinely the right content
(not just a title match) before listing it here.

| Work | Edition | Archive.org identifier | Year | Language tag | OCR text already available |
|---|---|---|---|---|---|
| Krittibasi Ramayan | ed. Ashutosh Bhattacharjya | `in.ernet.dli.2015.302459` | 1940 | ben | Yes - `_djvu.txt`, 3.3MB, confirmed real Bengali text on download (OCR noise present, 1940s letterpress, expected) |
| Krittibasi Ramayan | ed. Birendrakrishna Bhadra | `in.ernet.dli.2015.302458` | - | ben | Yes - `_djvu.txt`, not yet content-verified |
| Krittibasi Ramayan | Mahakabi Krittibas | `in.ernet.dli.2015.464248` | 1953 | **hin (mislabeled)** | Yes - `_djvu.txt`, language tag is wrong (this is Krittibas's Bengali Ramayan, not Hindi) - DLI metadata language tags aren't reliable, verify content not just the tag |
| Kashidasi Mahabharat | ed. Sudeb Chandra Chattopadhyaya | `dli.bengal.10689.753` | 1925 | ben | Yes - `_djvu.txt` |
| Kashidasi Mahabharat, Vol. 1 | attributed directly to Kashiram Das | `in.ernet.dli.2015.302409` | 1960 | ben | Yes - `_djvu.txt` - Vol. 1 is apt anyway, since only the first four of eighteen parvas are actually Kashiram Das's own work (section 3) |
| **Linguistic Survey of India, Vol. V, Part I** | Grierson | `in.ernet.dli.2015.61745` | 1903 | eng (survey text; specimens embedded in Bengali/Assamese script) | Yes - `_djvu.txt`, 750KB, **confirmed by download**: title page reads "INDO-ARYAN FAMILY, Eastern Group. Part I. SPECIMENS OF THE BENGALI AND ASSAMESE LANGUAGES" - exactly the volume section 8 needed, 421 Bengali/Assamese/dialect-name hits in the text |

All six: firmly PD by the section-1 rule (authors centuries to 120+ years
dead; DLI's own scanning/OCR work doesn't create fresh rights the way
Bichitra's variorum apparatus does - this is a straight scan-and-OCR
repository, not an annotated critical edition). Direct download pattern:
`https://archive.org/download/<identifier>/<filename>` - filenames are in
the table implicitly via each item's own metadata, fetch
`https://archive.org/metadata/<identifier>` first to get the exact
`_djvu.txt` filename before pulling it.

**Not yet located, still open**: the 1913 *Addyer Gambhira* volume
(section 8, Varendrī) - searched archive.org's advanced-search API the
same way as the epics and didn't get a confident hit this pass, still
"to locate," not confirmed absent.

**Next real step, not done in this pass**: the raw OCR text pulled here is
letterpress-quality, not clean - genuine 1920s-1960s scan noise throughout
(spot-checked in the Bhattacharjya Ramayan download). Turning this into
actual `general_text` / `ocr_ground_truth` schema rows needs real cleanup
work, not a straight dump - that's downstream of this discovery pass.
