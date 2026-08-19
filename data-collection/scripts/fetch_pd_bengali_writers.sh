#!/usr/bin/env bash
# I generate data-collection/pd-bengali-writers.csv from here - a mechanical,
# re-runnable pull of every Wikidata person recorded as a Bengali-language
# writer/poet/novelist/playwright/journalist/essayist who died in or before
# CURRENT_YEAR - 61, i.e. public domain in India under the Copyright Act
# 1957's life-plus-60-years rule (see dataset-scope.md section 1 for the
# rule itself). This is section 9 of dataset-scope.md done, not proposed -
# the roster there was explicitly representative by construction; this is
# the real, complete, mechanically reproducible version.
#
# Usage: bash fetch_pd_bengali_writers.sh [cutoff_year]
#   cutoff_year defaults to 1965 (2026 - 61). Pass a different year to
#   regenerate the list in a future year without hand-editing the query.
#
# Occupations covered (Wikidata QIDs): writer (Q36180), poet (Q49757),
# novelist (Q6625963), playwright (Q214917), journalist (Q1930187),
# essayist (Q11774202). Language match: native language (P103) OR
# languages spoken/written (P1412) = Bangla (Q9610) - I confirmed this QID
# via wbsearchentities; Q25268 (a guess I tried first) does not exist as
# the Bengali-language item and returns zero results silently, so don't
# reuse that number.
#
# What this is not: a list of freedom fighters, reformers, or anyone whose
# significance is historical rather than literary-output - that's section 4
# of dataset-scope.md, kept as a small hand-curated table on purpose since
# "occupation = writer" on Wikidata doesn't capture "wrote letters that
# matter historically." This script only pulls people Wikidata itself
# tags as literary-occupation writers.

set -euo pipefail

CUTOFF_YEAR="${1:-1965}"
OUT_CSV="$(dirname "$0")/../pd-bengali-writers.csv"
TMP_JSON="$(mktemp)"

QUERY="SELECT ?person ?personLabel ?dod ?occLabel WHERE {
  ?person wdt:P106 ?occ .
  VALUES ?occ { wd:Q36180 wd:Q49757 wd:Q6625963 wd:Q214917 wd:Q1930187 wd:Q11774202 }
  ?person wdt:P570 ?dod .
  FILTER(YEAR(?dod) <= ${CUTOFF_YEAR})
  { ?person wdt:P103 wd:Q9610 } UNION { ?person wdt:P1412 wd:Q9610 }
  SERVICE wikibase:label { bd:serviceParam wikibase:language \"en\". }
}
ORDER BY ?dod"

curl -s -G "https://query.wikidata.org/sparql" \
  -H "Accept: application/sparql-results+json" \
  -H "User-Agent: bornomala-research/1.0 (work.konkomaji@gmail.com)" \
  --data-urlencode "query=${QUERY}" \
  -o "${TMP_JSON}"

python - "${TMP_JSON}" "${OUT_CSV}" << 'PYEOF'
import json, collections, csv, sys

in_path, out_path = sys.argv[1], sys.argv[2]
data = json.load(open(in_path, encoding="utf-8"))
by_person = collections.OrderedDict()
for row in data["results"]["bindings"]:
    qid = row["person"]["value"].rsplit("/", 1)[-1]
    name = row["personLabel"]["value"]
    if name.startswith("Q") and name[1:].isdigit():
        continue  # label didn't resolve - skip rather than record a bare QID as a name
    dod = row["dod"]["value"][:4]
    occ = row["occLabel"]["value"]
    entry = by_person.setdefault(qid, {"name": name, "dod": dod, "occ": set()})
    entry["occ"].add(occ)

items = sorted(by_person.items(), key=lambda kv: (kv[1]["dod"], kv[1]["name"]))
with open(out_path, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["name", "death_year", "occupation", "wikidata_id", "wikidata_url"])
    for qid, info in items:
        w.writerow([info["name"], info["dod"], "|".join(sorted(info["occ"])), qid,
                    f"https://www.wikidata.org/wiki/{qid}"])

print(f"{len(items)} people written to {out_path}")
PYEOF

rm -f "${TMP_JSON}"
