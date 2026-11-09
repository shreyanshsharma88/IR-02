"""Unit tests for textmining.preprocessing - every switch of PreprocessConfig."""
from __future__ import annotations

import pytest

from textmining import preprocessing as P
from textmining.preprocessing import (PreprocessConfig, corpus_stats, make_ngrams, preprocess,
                                      preprocess_corpus, raw_tokens)

# a config with every optional step switched OFF - the "identity" baseline
OFF = dict(strip_markup=False, lowercase=False, remove_numbers=False, remove_punctuation=False,
           remove_stopwords=False, normalizer="none", min_token_len=1)


def cfg(**kw) -> PreprocessConfig:
    return PreprocessConfig(**{**OFF, **kw})


def test_default_config_values():
    c = PreprocessConfig()
    assert (c.strip_markup, c.lowercase, c.remove_numbers, c.remove_punctuation, c.remove_stopwords) == (True,) * 5
    assert c.normalizer == "stem" and c.min_token_len == 2
    assert (c.ngram_min, c.ngram_max) == (1, 1) and c.custom_stopwords == []
    assert c.label() == "lower|stop|stem|ng1-1"
    assert c.as_dict()["normalizer"] == "stem"


def test_raw_tokens_keeps_everything():
    assert raw_tokens("Hello, World! It's 2005.") == ["Hello", ",", "World", "!", "It's", "2005", "."]


def test_non_string_input():
    assert preprocess(None, PreprocessConfig()) == []
    assert preprocess(12345, cfg()) == ["12345"]


def test_lowercase_toggle():
    assert preprocess("Apple Banana", cfg(lowercase=True)) == ["apple", "banana"]
    assert preprocess("Apple Banana", cfg(lowercase=False)) == ["Apple", "Banana"]


def test_porter_stemmer_lowercases_even_when_lowercase_off():
    """Documented behaviour (report section 9): NLTK's Porter stemmer lowercases its output,
    so 'No lowercasing' + stem == baseline. Lowercasing only matters with normalizer none/lemma."""
    assert preprocess("London", cfg(lowercase=False, normalizer="stem")) == ["london"]


def test_markup_stripping():
    text = "<p>Hello <b>world</b></p> visit https://bbc.co.uk/news or www.example.com &amp; more"
    on = preprocess(text, cfg(strip_markup=True, remove_punctuation=True))
    assert on == ["Hello", "world", "visit", "or", "more"]
    off = preprocess(text, cfg(strip_markup=False, remove_punctuation=True))
    assert "p" in off and "b" in off and "bbc" in off and "amp" in off


def test_html_entities_unescaped_then_tags_removed():
    assert preprocess("&lt;i&gt;quoted&lt;/i&gt;", cfg(strip_markup=True, remove_punctuation=True)) == ["quoted"]


def test_number_removal():
    text = "Profits rose 25% to 3.5bn in 2004"
    on = preprocess(text, cfg(remove_numbers=True, remove_punctuation=True))
    assert on == ["Profits", "rose", "to", "in"]          # '3', '5bn', '25', '2004' gone
    off = preprocess(text, cfg(remove_numbers=False, remove_punctuation=True))
    assert "2004" in off and "25" in off


def test_punctuation_removal():
    text = "Wait... it's a well-known fact!"
    on = preprocess(text, cfg(remove_punctuation=True))
    assert on == ["Wait", "its", "a", "wellknown", "fact"]
    off = preprocess(text, cfg(remove_punctuation=False))
    assert "!" in off and "." in off and "it's" in off and "well-known" in off


def test_stopword_removal():
    text = "the cat sat on the mat"
    assert preprocess(text, cfg(remove_stopwords=True)) == ["cat", "sat", "mat"]
    assert preprocess(text, cfg(remove_stopwords=False)) == text.split()


def test_stopwords_case_insensitive():
    assert preprocess("The Cat", cfg(remove_stopwords=True, lowercase=False)) == ["Cat"]


def test_domain_stopwords_included():
    for w in ("said", "also", "mr"):
        assert w in P.STOPWORDS


def test_custom_stopwords():
    c = cfg(remove_stopwords=True, custom_stopwords=["Cat", "MAT"])
    assert preprocess("the cat sat on the mat", c) == ["sat"]


def test_custom_stopwords_ignored_when_stopword_removal_off():
    assert preprocess("cat sat", cfg(remove_stopwords=False, custom_stopwords=["cat"])) == ["cat", "sat"]


@pytest.mark.parametrize("norm,expected", [
    ("none", ["running", "studies", "cars"]),
    ("stem", ["run", "studi", "car"]),
    ("lemma", ["running", "study", "car"]),
])
def test_normalizers(norm, expected):
    assert preprocess("running studies cars", cfg(normalizer=norm)) == expected


def test_stem_vs_lemma_vs_none_vocab_sizes():
    text = "connect connected connecting connection connections"
    sizes = {n: len(set(preprocess(text, cfg(normalizer=n)))) for n in ("none", "lemma", "stem")}
    assert sizes["stem"] < sizes["lemma"] < sizes["none"]
    assert sizes["stem"] == 1


def test_lemma_fallback_when_wordnet_unavailable(monkeypatch):
    """Simulate a locked-down lab machine: nltk WordNet import + download both fail."""
    import nltk
    import nltk.stem

    class BrokenLemmatizer:
        def lemmatize(self, *_a, **_k):
            raise LookupError("Resource wordnet not found")

    monkeypatch.setattr(P, "_lemmatizer", None)
    monkeypatch.setattr(nltk.stem, "WordNetLemmatizer", BrokenLemmatizer)
    monkeypatch.setattr(nltk, "download", lambda *a, **k: False)
    lem = P._get_lemmatizer()
    assert type(lem).__name__ == "_Fallback"
    assert lem.lemmatize("studies") == "study"
    assert lem.lemmatize("cars") == "car"
    assert lem.lemmatize("class") == "class"     # -ss is not a plural
    assert lem.lemmatize("bus") == "bus"         # too short to strip
    assert preprocess("studies cars", cfg(normalizer="lemma")) == ["study", "car"]


def test_simple_stem_fallback(monkeypatch):
    monkeypatch.setattr(P, "_stemmer", None)
    assert P._simple_stem("jumping") == "jump"
    assert P._simple_stem("cats") == "cat"
    assert P._simple_stem("is") == "is"
    assert preprocess("jumping cats", cfg(normalizer="stem")) == ["jump", "cat"]


def test_min_token_length():
    text = "a an ox cat horse"
    assert preprocess(text, cfg(min_token_len=1)) == text.split()
    assert preprocess(text, cfg(min_token_len=3)) == ["cat", "horse"]
    assert preprocess(text, cfg(min_token_len=5)) == ["horse"]


def test_make_ngrams():
    toks = ["a", "b", "c"]
    assert make_ngrams(toks, 1, 1) == toks
    assert make_ngrams(toks, 2, 2) == ["a_b", "b_c"]
    assert make_ngrams(toks, 1, 2) == ["a", "b", "c", "a_b", "b_c"]
    assert make_ngrams(toks, 1, 3) == ["a", "b", "c", "a_b", "b_c", "a_b_c"]
    assert make_ngrams(["a"], 2, 2) == []


def test_bigram_format_in_pipeline():
    out = preprocess("prime minister tony blair", cfg(ngram_min=1, ngram_max=2))
    assert out == ["prime", "minister", "tony", "blair", "prime_minister", "minister_tony", "tony_blair"]
    only_bi = preprocess("prime minister tony blair", cfg(ngram_min=2, ngram_max=2))
    assert only_bi == ["prime_minister", "minister_tony", "tony_blair"]


def test_bigrams_formed_after_stopword_removal():
    out = preprocess("the prime minister of britain", cfg(remove_stopwords=True, ngram_min=2, ngram_max=2))
    assert out == ["prime_minister", "minister_britain"]


def test_full_default_pipeline():
    text = "<p>The Bank of England raised rates by 0.25% on Thursday, said Mr King.</p>"
    assert preprocess(text, PreprocessConfig()) == ["bank", "england", "rais", "rate", "thursday", "king"]


def test_corpus_stats_values():
    tl = [["a", "b", "a"], ["b", "c"], []]
    st = corpus_stats(tl)
    assert st["documents"] == 3
    assert st["total_tokens"] == 5
    assert st["vocabulary_size"] == 3
    assert st["avg_tokens_per_doc"] == round(5 / 3, 1)
    assert st["min_tokens"] == 0 and st["max_tokens"] == 3
    assert st["type_token_ratio"] == round(3 / 5, 4)
    assert st["hapax_legomena"] == 1                 # only 'c' occurs once
    assert st["term_freq"] == {"a": 2, "b": 2, "c": 1}
    assert st["lengths"] == [3, 2, 0]


def test_corpus_stats_empty():
    st = corpus_stats([])
    assert st["documents"] == 0 and st["total_tokens"] == 0 and st["type_token_ratio"] == 0


def test_preprocess_corpus_matches_single():
    texts = ["Shares rose sharply.", "Football match tonight!"]
    c = PreprocessConfig()
    assert preprocess_corpus(texts, c) == [preprocess(t, c) for t in texts]


def test_baseline_on_real_corpus(baseline_tokens):
    st = corpus_stats(baseline_tokens)
    assert st["documents"] == 2225
    assert st["vocabulary_size"] == 21238
    assert st["avg_tokens_per_doc"] == 203.0
