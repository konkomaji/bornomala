"""
Language metadata - the "all languages" layer.

The engine itself is script-agnostic: it will tokenize and score text in ANY of
the world's languages, because it works from Unicode grapheme clusters and the
model's real tokenizer, not from a fixed language list. This module adds
*metadata* so the tool can label, group, and present languages nicely: name,
autonym (endonym), ISO codes, writing system, family, region, and an approximate
speaker count.

Facts here are conservative and verifiable (script/family/ISO code). Speaker
figures are rounded order-of-magnitude approximations and are labelled as such;
they are for sorting and context only, never presented as precise statistics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Language:
    code: str          # ISO 639-3
    name: str          # English name
    autonym: str       # endonym (how speakers write its name); "" if unset
    script: str        # Unicode script name (matches segment.script_of labels)
    family: str        # language family
    region: str        # primary region
    speakers_m: int    # approximate speakers, millions (order-of-magnitude)

    def as_dict(self) -> dict:
        return asdict(self)


# A broad, deliberately diverse set spanning scripts and families. Not
# exhaustive - the tool works on languages absent from this list too; these are
# the ones we give first-class labels and presets to.
_ROWS: list[Language] = [
    Language("eng", "English",     "English",    "Latin",       "Indo-European (Germanic)", "Global",        1500),
    Language("cmn", "Chinese (Mandarin)", "中文",  "Han",         "Sino-Tibetan",             "China",         1100),
    Language("hin", "Hindi",       "हिन्दी",       "Devanagari",  "Indo-European (Indo-Aryan)","India",         610),
    Language("spa", "Spanish",     "Español",    "Latin",       "Indo-European (Romance)",  "Americas/Spain",560),
    Language("arb", "Arabic",      "العربية",     "Arabic",      "Afro-Asiatic (Semitic)",   "MENA",          420),
    Language("ben", "Bengali",     "বাংলা",       "Bengali",     "Indo-European (Indo-Aryan)","Bengal",        280),
    Language("por", "Portuguese",  "Português",  "Latin",       "Indo-European (Romance)",  "Brazil/Portugal",260),
    Language("rus", "Russian",     "Русский",    "Cyrillic",    "Indo-European (Slavic)",   "Russia",        255),
    Language("jpn", "Japanese",    "日本語",       "Han",         "Japonic",                  "Japan",         125),
    Language("pnb", "Punjabi",     "ਪੰਜਾਬੀ",       "Gurmukhi",    "Indo-European (Indo-Aryan)","Punjab",        150),
    Language("mar", "Marathi",     "मराठी",       "Devanagari",  "Indo-European (Indo-Aryan)","India",         100),
    Language("tel", "Telugu",      "తెలుగు",       "Telugu",      "Dravidian",                "India",          95),
    Language("tur", "Turkish",     "Türkçe",     "Latin",       "Turkic",                   "Türkiye",         90),
    Language("tam", "Tamil",       "தமிழ்",        "Tamil",       "Dravidian",                "India/Lanka",     85),
    Language("kor", "Korean",      "한국어",       "Hangul",      "Koreanic",                 "Korea",           82),
    Language("vie", "Vietnamese",  "Tiếng Việt", "Latin",       "Austroasiatic",            "Vietnam",         85),
    Language("urd", "Urdu",        "اردو",        "Arabic",      "Indo-European (Indo-Aryan)","Pakistan/India",  70),
    Language("fra", "French",      "Français",   "Latin",       "Indo-European (Romance)",  "France/Africa",  310),
    Language("deu", "German",      "Deutsch",    "Latin",       "Indo-European (Germanic)", "C. Europe",      135),
    Language("guj", "Gujarati",    "ગુજરાતી",      "Gujarati",    "Indo-European (Indo-Aryan)","India",          60),
    Language("kan", "Kannada",     "ಕನ್ನಡ",       "Kannada",     "Dravidian",                "India",           45),
    Language("mal", "Malayalam",   "മലയാളം",      "Malayalam",   "Dravidian",                "India",           38),
    Language("ori", "Odia",        "ଓଡ଼ିଆ",        "Oriya",       "Indo-European (Indo-Aryan)","India",          38),
    Language("pes", "Persian",     "فارسی",       "Arabic",      "Indo-European (Iranian)",  "Iran",            80),
    Language("tha", "Thai",        "ไทย",         "Thai",        "Kra-Dai",                  "Thailand",        60),
    Language("ita", "Italian",     "Italiano",   "Latin",       "Indo-European (Romance)",  "Italy",           65),
    Language("ell", "Greek",       "Ελληνικά",   "Greek",       "Indo-European (Hellenic)", "Greece",          13),
    Language("heb", "Hebrew",      "עברית",       "Hebrew",      "Afro-Asiatic (Semitic)",   "Israel",           9),
    Language("hye", "Armenian",    "Հայերեն",     "Armenian",    "Indo-European (Armenian)", "Armenia",           7),
    Language("ind", "Indonesian",  "Bahasa Indonesia","Latin",  "Austronesian",             "Indonesia",      200),
    Language("swa", "Swahili",     "Kiswahili",  "Latin",       "Niger-Congo (Bantu)",      "E. Africa",       80),
    Language("amh", "Amharic",     "አማርኛ",       "Ethiopic",    "Afro-Asiatic (Semitic)",   "Ethiopia",        35),
    Language("mya", "Burmese",     "မြန်မာ",       "Myanmar",     "Sino-Tibetan",             "Myanmar",         33),
    Language("sin", "Sinhala",     "සිංහල",        "Sinhala",     "Indo-European (Indo-Aryan)","Sri Lanka",       17),
    Language("khm", "Khmer",       "ខ្មែរ",         "Khmer",       "Austroasiatic",            "Cambodia",        16),
]

LANGUAGES: dict[str, Language] = {lang.code: lang for lang in _ROWS}


def all_languages() -> list[Language]:
    return list(_ROWS)


def by_script(script: str) -> list[Language]:
    return [lang for lang in _ROWS if lang.script == script]


def get(code: str) -> Language | None:
    return LANGUAGES.get(code)


# Scripts with well-known tokenization pain (non-Latin, especially Indic/complex
# scripts) - useful for the tool's "who is most disadvantaged vs English" view.
COMPLEX_SCRIPTS = {
    "Bengali", "Devanagari", "Tamil", "Telugu", "Kannada", "Malayalam",
    "Gujarati", "Gurmukhi", "Oriya", "Sinhala", "Myanmar", "Khmer",
    "Thai", "Ethiopic", "Arabic",
}
