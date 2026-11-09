"""Multinomial Naive Bayes against Manning, Raghavan & Schuetze, IIR Example 13.1."""
from __future__ import annotations

import math
from collections import Counter

import pytest

from textmining.naive_bayes import MultinomialNaiveBayes, stratified_split

TRAIN = [
    ("Chinese Beijing Chinese", "yes"),
    ("Chinese Chinese Shanghai", "yes"),
    ("Chinese Macao", "yes"),
    ("Tokyo Japan Chinese", "no"),
]
D5 = "Chinese Chinese Chinese Tokyo Japan".split()


@pytest.fixture
def iir():
    return MultinomialNaiveBayes(alpha=1.0).fit([d.split() for d, _ in TRAIN], [c for _, c in TRAIN])


def p(model, term, c):
    return math.exp(model.log_likelihood[c].get(term, model.log_unseen[c]))


def test_vocabulary_and_priors(iir):
    assert iir.vocab == {"Chinese", "Beijing", "Shanghai", "Macao", "Tokyo", "Japan"}
    assert iir.classes == ["no", "yes"]
    assert math.exp(iir.log_prior["yes"]) == pytest.approx(3 / 4)
    assert math.exp(iir.log_prior["no"]) == pytest.approx(1 / 4)


def test_conditional_probabilities_example_13_1(iir):
    assert p(iir, "Chinese", "yes") == pytest.approx(3 / 7)
    assert p(iir, "Tokyo", "yes") == pytest.approx(1 / 14)
    assert p(iir, "Japan", "yes") == pytest.approx(1 / 14)
    assert p(iir, "Chinese", "no") == pytest.approx(2 / 9)
    assert p(iir, "Tokyo", "no") == pytest.approx(2 / 9)
    assert p(iir, "Japan", "no") == pytest.approx(2 / 9)


def test_likelihoods_sum_to_one_over_vocab(iir):
    for c in iir.classes:
        assert sum(p(iir, t, c) for t in iir.vocab) == pytest.approx(1.0)


def test_unnormalised_posteriors_example_13_1(iir):
    s = iir.class_scores(D5)
    yes, no = math.exp(s["yes"]), math.exp(s["no"])
    assert yes == pytest.approx(3 / 4 * (3 / 7) ** 3 * (1 / 14) ** 2)
    assert no == pytest.approx(1 / 4 * (2 / 9) ** 3 * (2 / 9) ** 2)
    assert round(yes, 4) == 0.0003
    assert round(no, 4) == 0.0001
    assert iir.predict_one(D5) == "yes"
    assert iir.predict([D5, ["Tokyo", "Japan"]]) == ["yes", "no"]


def test_predict_proba_sums_to_one(iir):
    pr = iir.predict_proba_one(D5)
    assert sum(pr.values()) == pytest.approx(1.0)
    yes = 3 / 4 * (3 / 7) ** 3 * (1 / 14) ** 2
    no = 1 / 4 * (2 / 9) ** 5
    assert pr["yes"] == pytest.approx(yes / (yes + no))
    assert max(pr, key=pr.get) == iir.predict_one(D5)


def test_unknown_terms_ignored(iir):
    assert iir.class_scores(D5 + ["Kyoto", "Paris"]) == iir.class_scores(D5)
    # only unknown terms -> priors decide
    assert iir.predict_one(["unseen"]) == "yes"


def test_alpha_parameter():
    m = MultinomialNaiveBayes(alpha=0.5).fit([d.split() for d, _ in TRAIN], [c for _, c in TRAIN])
    assert p(m, "Chinese", "yes") == pytest.approx((5 + 0.5) / (8 + 0.5 * 6))


def test_top_features():
    m = MultinomialNaiveBayes().fit([["goal"] * 3 + ["match"] * 3, ["bank"] * 3 + ["shares"] * 3],
                                    ["sport", "business"])
    tf = m.top_features(2)
    assert {t for t, _ in tf["sport"]} == {"goal", "match"}
    assert {t for t, _ in tf["business"]} == {"bank", "shares"}


LABELS = ["a"] * 50 + ["b"] * 30 + ["c"] * 20


def test_stratified_split_proportions_and_disjoint():
    tr, te = stratified_split(LABELS, 0.2, seed=1)
    assert set(tr).isdisjoint(te)
    assert sorted(tr + te) == list(range(len(LABELS)))
    assert Counter(LABELS[i] for i in te) == {"a": 10, "b": 6, "c": 4}
    assert tr == sorted(tr) and te == sorted(te)


def test_stratified_split_deterministic_by_seed():
    assert stratified_split(LABELS, 0.2, 7) == stratified_split(LABELS, 0.2, 7)
    assert stratified_split(LABELS, 0.2, 7) != stratified_split(LABELS, 0.2, 8)


def test_stratified_split_tiny_class_gets_one_test_doc():
    tr, te = stratified_split(["x", "x", "y", "y", "y", "y", "y", "y", "y", "y"], 0.1, 0)
    assert Counter(["x", "x", "y", "y", "y", "y", "y", "y", "y", "y"][i] for i in te) == {"x": 1, "y": 1}


def test_real_split_sizes(labels):
    tr, te = stratified_split(labels, 0.2, 42)
    assert (len(tr), len(te)) == (1781, 444)


def test_vocab_built_from_training_data_only(baseline_tokens, labels):
    tr, te = stratified_split(labels, 0.2, 42)
    m = MultinomialNaiveBayes().fit([baseline_tokens[i] for i in tr], [labels[i] for i in tr])
    train_terms = set().union(*(baseline_tokens[i] for i in tr))
    test_only = set().union(*(baseline_tokens[i] for i in te)) - train_terms
    assert m.vocab == train_terms
    assert test_only and m.vocab.isdisjoint(test_only)
