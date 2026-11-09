"""Configurable text-preprocessing pipeline.

Steps (each switchable): markup stripping, lowercasing, tokenization (regex),
number / punctuation removal, stop-word removal, stemming OR lemmatization,
minimum token length, and n-gram generation.

NLTK is used only for the Porter stemmer, the WordNet lemmatizer and its
stop-word list; everything has an offline fallback so the app runs even if
NLTK data cannot be downloaded (e.g. restricted lab machines).
"""
from __future__ import annotations

import html
import re
from collections import Counter
from dataclasses import dataclass, asdict, field

# ---------------------------------------------------------------- resources
_FALLBACK_STOPWORDS = set("""
a about above after again against all am an and any are aren't as at be because been before being below
between both but by can can't cannot could couldn't did didn't do does doesn't doing don't down during each
few for from further had hadn't has hasn't have haven't having he he'd he'll he's her here here's hers herself
him himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself let's me more most mustn't
my myself no nor not of off on once only or other ought our ours ourselves out over own same shan't she she'd
she'll she's should shouldn't so some such than that that's the their theirs them themselves then there there's
these they they'd they'll they're they've this those through to too under until up very was wasn't we we'd we'll
we're we've were weren't what what's when when's where where's which while who who's whom why why's with won't
would wouldn't you you'd you'll you're you've your yours yourself yourselves also said says say will would
""".split())


def _load_stopwords() -> set[str]:
    try:
        from nltk.corpus import stopwords
        return set(stopwords.words("english")) | {"also", "said", "says", "would", "could", "mr", "mrs", "ms"}
    except Exception:
        try:
            import nltk
            nltk.download("stopwords", quiet=True)
            from nltk.corpus import stopwords
            return set(stopwords.words("english")) | {"also", "said", "says", "would", "could", "mr", "mrs", "ms"}
        except Exception:
            return _FALLBACK_STOPWORDS | {"mr", "mrs", "ms"}


STOPWORDS = _load_stopwords()

try:
    from nltk.stem import PorterStemmer
    _stemmer = PorterStemmer()
except Exception:  # pragma: no cover
    _stemmer = None

_lemmatizer = None


def _get_lemmatizer():
    global _lemmatizer
    if _lemmatizer is not None:
        return _lemmatizer
    try:
        from nltk.stem import WordNetLemmatizer
        lem = WordNetLemmatizer()
        lem.lemmatize("tests")
    except Exception:
        try:
            import nltk
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
            from nltk.stem import WordNetLemmatizer
            lem = WordNetLemmatizer()
            lem.lemmatize("tests")
        except Exception:  # offline lab machine: rule-based plural lemmatizer
            class _Fallback:
                def lemmatize(self, t):
                    if t.endswith("ies") and len(t) > 4:
                        return t[:-3] + "y"
                    if t.endswith("s") and not t.endswith("ss") and len(t) > 3:
                        return t[:-1]
                    return t
            lem = _Fallback()
    _lemmatizer = lem
    return lem


def _simple_stem(tok: str) -> str:  # fallback suffix stripper
    for suf in ("ational", "ization", "fulness", "ousness", "iveness", "ing", "edly", "ed", "ies", "es", "ly", "s"):
        if tok.endswith(suf) and len(tok) - len(suf) >= 3:
            return tok[: -len(suf)]
    return tok


# ---------------------------------------------------------------- config
@dataclass
class PreprocessConfig:
    strip_markup: bool = True
    lowercase: bool = True
    remove_numbers: bool = True
    remove_punctuation: bool = True
    remove_stopwords: bool = True
    normalizer: str = "stem"          # "none" | "stem" | "lemma"
    min_token_len: int = 2
    ngram_min: int = 1
    ngram_max: int = 1
    custom_stopwords: list[str] = field(default_factory=list)

    def as_dict(self):
        return asdict(self)

    def label(self) -> str:
        parts = []
        parts.append("lower" if self.lowercase else "case")
        parts.append("stop" if self.remove_stopwords else "nostop")
        parts.append(self.normalizer)
        parts.append(f"ng{self.ngram_min}-{self.ngram_max}")
        return "|".join(parts)


_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['\-][A-Za-z0-9]+)*|[^\sA-Za-z0-9]")


def raw_tokens(text: str) -> list[str]:
    """Whitespace/punctuation tokenization with no normalisation (the 'before' view)."""
    return [t for t in _TOKEN_RE.findall(text)]


def preprocess(text: str, cfg: PreprocessConfig) -> list[str]:
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    if cfg.strip_markup:
        text = html.unescape(text)
        text = _TAG_RE.sub(" ", text)
        text = _URL_RE.sub(" ", text)
    if cfg.lowercase:
        text = text.lower()
    toks = _TOKEN_RE.findall(text)
    out = []
    stops = STOPWORDS | {s.lower() for s in cfg.custom_stopwords}
    lem = _get_lemmatizer() if cfg.normalizer == "lemma" else None
    for t in toks:
        if cfg.remove_punctuation:
            if not any(c.isalnum() for c in t):
                continue
            t = t.replace("'", "").replace("-", "")
        if cfg.remove_numbers and any(c.isdigit() for c in t):
            continue
        if len(t) < cfg.min_token_len:
            continue
        if cfg.remove_stopwords and t.lower() in stops:
            continue
        if cfg.normalizer == "stem":
            t = _stemmer.stem(t) if _stemmer else _simple_stem(t)
        elif cfg.normalizer == "lemma":
            t = lem.lemmatize(t)
        if t:
            out.append(t)
    if cfg.ngram_max > 1 or cfg.ngram_min > 1:
        return make_ngrams(out, cfg.ngram_min, cfg.ngram_max)
    return out


def make_ngrams(tokens: list[str], nmin: int, nmax: int) -> list[str]:
    grams = []
    for n in range(nmin, nmax + 1):
        if n == 1:
            grams.extend(tokens)
        else:
            grams.extend("_".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1))
    return grams


def preprocess_corpus(texts, cfg: PreprocessConfig) -> list[list[str]]:
    return [preprocess(t, cfg) for t in texts]


def corpus_stats(token_lists: list[list[str]]) -> dict:
    tf = Counter()
    for toks in token_lists:
        tf.update(toks)
    lengths = [len(t) for t in token_lists]
    n = max(len(lengths), 1)
    total = sum(lengths)
    hapax = sum(1 for c in tf.values() if c == 1)
    return {
        "documents": len(token_lists),
        "total_tokens": total,
        "vocabulary_size": len(tf),
        "avg_tokens_per_doc": round(total / n, 1),
        "min_tokens": min(lengths) if lengths else 0,
        "max_tokens": max(lengths) if lengths else 0,
        "type_token_ratio": round(len(tf) / total, 4) if total else 0,
        "hapax_legomena": hapax,
        "term_freq": tf,
        "lengths": lengths,
    }
