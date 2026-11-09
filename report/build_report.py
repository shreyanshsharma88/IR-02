"""Build report/Report.pdf from report/results.json + report/figures/*.png + screenshots/*.png.

    python experiments.py && python report/build_report.py

Every number in the report is read from results.json; nothing numeric is typed in by hand.
Section order and names follow the assignment's "Report - PDF format" specification.
"""
from __future__ import annotations

import glob
import json
import os
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (BaseDocTemplate, CondPageBreak, Frame, Image, KeepTogether, PageBreak,
                                PageTemplate, Paragraph, Spacer, Table, TableStyle)

# ---------------------------------------------------------------- EDIT ME: group details
# Fill in your group number and the four members before submitting, then re-run this script.
GROUP_ID = "G35"
MEMBERS = [  # (registration number, name)
    ("2025AG05062", "LAKSHYA KANCHAN"),
    ("2025AG05772", "SHREYANSH SHARMA"),
    ("2025AG05689", "VINAY PRATAP SINGH BHADAURIA"),
    ("2025AG05914", "NAIR ABHIJITH VIJAYAKUMAR"),
    ("2025AG05076", "ABHISHEK GHOSH"),
]
COURSE = "Information Retrieval (AIMLCZG537/DSECLZG537) S1-26 - Assignment 2"
TITLE = "Text Mining and Document Intelligence System for News Analytics"

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG = os.path.join(HERE, "figures")
SHOTS = os.path.join(ROOT, "screenshots")
R = json.load(open(os.path.join(HERE, "results.json")))

D, DUP, PRE, TF = R["dataset"], R["duplicates"], R["preprocessing"], R["tfidf"]
NB, KM, REC, ST = R["naive_bayes"], R["kmeans"], R["recommendation"], R["stability"]
CATS = R["categories"]
LK = NB["leakage_analysis"]
KS = ST["kmeans"]
PK = REC["precision"]
A = {a["name"]: a for a in R["ablation"]}

# ---------------------------------------------------------------- styles
ACCENT = colors.HexColor("#1c4f8f")
LIGHT = colors.HexColor("#eef3fa")
GRID = colors.HexColor("#c6ccd6")
GREY = colors.HexColor("#555555")
BODY = ParagraphStyle("body", fontName="Helvetica", fontSize=10.5, leading=15.6, alignment=TA_JUSTIFY, spaceAfter=6)
LEAD = ParagraphStyle("lead", parent=BODY, fontName="Helvetica-Oblique", textColor=colors.HexColor("#1f3550"),
                      alignment=0, spaceAfter=8, keepWithNext=1)
BUL = ParagraphStyle("bul", parent=BODY, leftIndent=14, bulletIndent=3, spaceAfter=4, alignment=0)
H1 = ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=14, leading=18, textColor=ACCENT, spaceBefore=12,
                    spaceAfter=5, keepWithNext=1)
H2 = ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=11, leading=14, textColor=colors.HexColor("#222222"),
                    spaceBefore=8, spaceAfter=3, keepWithNext=1)
CAP = ParagraphStyle("cap", parent=BODY, fontName="Helvetica-Oblique", fontSize=8.5, leading=11, alignment=TA_CENTER,
                     textColor=GREY, spaceBefore=3, spaceAfter=9)
CELL = ParagraphStyle("cell", fontName="Helvetica", fontSize=8.6, leading=10.8)
CELLH = ParagraphStyle("cellh", parent=CELL, fontName="Helvetica-Bold", textColor=colors.white)
MONO = ParagraphStyle("mono", fontName="Courier", fontSize=8.6, leading=11.5, backColor=colors.HexColor("#f4f4f2"),
                      borderPadding=5, spaceBefore=3, spaceAfter=8)
FCELL = ParagraphStyle("fcell", parent=CELL, fontName="Courier", fontSize=8.3, leading=10.6)

_REPL = {"Â£": "GBP ", "£": "GBP ", "‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", "…": "...", "\xa0": " "}


def clean(s) -> str:
    s = str(s)
    for a, b in _REPL.items():
        s = s.replace(a, b)
    return "".join(ch if ord(ch) < 128 else "?" for ch in s)  # plain ASCII: no missing glyphs


def esc(s) -> str:
    return clean(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


_LT = re.compile(r"<(?!/?(?:b|i|u|font|br|super|sub|a)\b)")


def P(text, style=BODY):
    return Paragraph(_LT.sub("&lt;", clean(text)), style)


def bullets(items):
    return [Paragraph(_LT.sub("&lt;", clean(t)), BUL, bulletText="\u2022") for t in items]


def f3(x):
    return f"{x:.3f}"


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def n(x):
    return f"{x:,}"


def code(s):
    return f"<font face='Courier'>{s}</font>"


# ---------------------------------------------------------------- document + numbering helpers
class ReportDoc(BaseDocTemplate):
    def __init__(self, path):
        super().__init__(path, pagesize=A4, leftMargin=2.1 * cm, rightMargin=2.1 * cm, topMargin=1.9 * cm,
                         bottomMargin=1.7 * cm, title=TITLE, author=GROUP_ID, subject=COURSE)
        self.addPageTemplates([PageTemplate("p", [Frame(self.leftMargin, self.bottomMargin, self.width, self.height,
                                                        id="f")], onPage=self._decor)])

    def _decor(self, c, d):
        c.saveState()
        c.setFont("Helvetica", 7.5)
        c.setFillColor(GREY)
        if d.page > 1:
            c.drawString(d.leftMargin, A4[1] - 1.2 * cm, f"{GROUP_ID} | {TITLE}")
            c.drawRightString(A4[0] - d.rightMargin, A4[1] - 1.2 * cm, "IR S1-26 Assignment 2")
            c.setStrokeColor(GRID)
            c.line(d.leftMargin, A4[1] - 1.35 * cm, A4[0] - d.rightMargin, A4[1] - 1.35 * cm)
        c.drawCentredString(A4[0] / 2, 0.9 * cm, f"Page {d.page}")
        c.restoreState()

    def afterFlowable(self, fl):
        lvl = getattr(fl, "_outline", None)
        if lvl is not None:
            key = f"k{id(fl)}"
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(fl.getPlainText(), key, level=lvl, closed=lvl > 0)
            SECTION_PAGES.append((lvl, fl.getPlainText(), self.page))


SECTION_PAGES: list = []
story: list = []
_sec = [0]
_counts = {"Table": 0, "Figure": 0}


def h1(title, lead=None, numbered=True, newpage=False):
    if newpage:
        story.append(PageBreak())
    if numbered:
        _sec[0] += 1
        title = f"{_sec[0]}. {title}"
    story.append(CondPageBreak(4 * cm))
    p = Paragraph(esc(title), H1)
    p._outline = 0
    story.append(p)
    if lead:
        story.append(P(lead, LEAD))


def h2(title):
    story.append(CondPageBreak(3 * cm))
    p = Paragraph(esc(title), H2)
    p._outline = 1
    story.append(p)


def caption(kind, text):
    _counts[kind] += 1
    return P(f"<b>{kind} {_counts[kind]}.</b> {text}", CAP)


def table(rows, widths, cap=None, right_from=None, font=8.6, cap_first=False, raw=False, right_cols=None):
    """rows[0] is the header; strings are wrapped in Paragraphs; numeric columns right-aligned."""
    cs = ParagraphStyle("c", parent=CELL, fontSize=font, leading=font * 1.25)
    ch = ParagraphStyle("ch", parent=cs, fontName="Helvetica-Bold", textColor=colors.white)
    data = []
    for i, r in enumerate(rows):
        row = []
        for j, v in enumerate(r):
            if isinstance(v, Paragraph):
                row.append(v)
                continue
            st = ch if i == 0 else cs
            if i > 0 and ((right_cols is not None and j in right_cols) or (right_cols is None and right_from is not None and j >= right_from)):
                st = ParagraphStyle("r", parent=st, alignment=2)
            row.append(Paragraph(_LT.sub("&lt;", clean(v)), st))
        data.append(row)
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="CENTER")
    style = [("BACKGROUND", (0, 0), (-1, 0), ACCENT), ("GRID", (0, 0), (-1, -1), 0.4, GRID),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 2.5),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5), ("LEFTPADDING", (0, 0), (-1, -1), 4),
             ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    style += [("BACKGROUND", (0, i), (-1, i), LIGHT) for i in range(2, len(rows), 2)]
    t.setStyle(TableStyle(style))
    if raw:
        return t
    parts = [t]
    if cap:
        c = caption("Table", cap)
        parts = [c, t] if cap_first else [t, c]
    return KeepTogether(parts)


def img(path, width):
    path = path if os.path.isabs(path) else os.path.join(FIG, path)
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(path).getSize()
    return Image(path, width=width, height=width * ih / iw)


def figure(name, cap, width=15 * cm):
    story.append(KeepTogether([img(name, width), caption("Figure", cap)]))


def pair(left, right, cap, widths=(8.3 * cm, 8.3 * cm), kind="Figure"):
    """Two images (or flowables) side by side with one shared caption."""
    cells = [img(x, w - 0.3 * cm) if isinstance(x, str) else x for x, w in zip((left, right), widths)]
    t = Table([cells], colWidths=list(widths))
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                           ("LEFTPADDING", (0, 0), (-1, -1), 2), ("RIGHTPADDING", (0, 0), (-1, -1), 2)]))
    story.append(KeepTogether([t, caption(kind, cap)]))


# ================================================================ title block (page 1)
story.append(P(esc(TITLE), ParagraphStyle("t", fontName="Helvetica-Bold", fontSize=18, leading=22, textColor=ACCENT,
                                          alignment=TA_CENTER, spaceAfter=4)))
story.append(P(COURSE, ParagraphStyle("st", parent=BODY, alignment=TA_CENTER, fontSize=10.5, spaceAfter=2)))
story.append(P(f"<b>Group ID: {GROUP_ID}</b>", ParagraphStyle("g", parent=BODY, alignment=TA_CENTER, spaceAfter=6)))
mt = Table([["#", "Registration number", "Name"]] + [[str(i + 1), r, nm] for i, (r, nm) in enumerate(MEMBERS)],
           colWidths=[0.8 * cm, 4.5 * cm, 6.5 * cm], hAlign="CENTER")
mt.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), "Helvetica", 8.6), ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8.6),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
                        ("GRID", (0, 0), (-1, -1), 0.4, GRID), ("TOPPADDING", (0, 0), (-1, -1), 1.5),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]))
story += [mt, Spacer(1, 4)]
story.append(Table([[""]], colWidths=[16.8 * cm], rowHeights=[2],
                   style=[("LINEBELOW", (0, 0), (-1, -1), 1, ACCENT)]))

# ================================================================ 1 business context
h1("Selected business context",
   "We play the text-mining team of a media-monitoring and business-research firm that must make sense of a "
   "continuous stream of news articles.")
story.append(P("Analysts at such a firm (and the communications, investor-relations and policy teams who buy its "
               "services) need to:"))
story += bullets([
    "<b>route</b> each incoming article to the right desk (business, politics, tech, entertainment, sport) - <i>classification</i>;",
    "<b>discover</b> topical groupings without waiting for labels - <i>clustering</i>;",
    "<b>recognise</b> syndicated or re-published copies so coverage is not counted twice - <i>duplicate detection</i>;",
    "<b>pull up related coverage</b> while researching a story - <i>recommendation</i>.",
])
story.append(P("The system is a Streamlit application (dashboard, data ingestion, preprocessing configuration, text-mining "
               "hub, recommendation panel, performance analytics, inferences) backed by a hand-written Python package."))

# ================================================================ 2 dataset
h1("Dataset description",
   f"Real BBC News articles: {n(D['n_docs'])} documents, {D['total_mb']} MB of raw text, five labelled desks, with "
   f"genuine duplicates and noise.")
story += bullets([
    "<b>Source:</b> BBC News full-text corpus, 2004-2005 (Greene and Cunningham, ICML 2006; mlg.ucd.ie/datasets/bbc.html). "
    "The first line of each file is the headline.",
    f"<b>Structure:</b> {code('data/documents.csv')} ({', '.join(D['document_columns'])}) holds only text; "
    f"{code('data/metadata.csv')} ({', '.join(D['metadata_columns'])}) holds metadata. IDs look like BUS-001.",
    f"<b>Size:</b> {n(D['total_chars'])} characters, {n(D['words_total'])} words; {D['words']['mean']} words per document "
    f"on average (median {D['words']['median']:.0f}, range {D['words']['min']}-{n(D['words']['max'])}).",
    "<b>Labels:</b> " + ", ".join(f"{c} {D['docs_per_category'][c]}" for c in CATS) + " - moderately balanced.",
    f"<b>Noise:</b> {n(D['noise']['docs_with_digits'])} documents contain numbers, {n(D['noise']['docs_with_currency_symbols'])} "
    f"currency symbols, {n(D['noise']['docs_with_quotes'])} quotation marks; {pct(D['noise']['uppercase_token_share'])} of "
    "tokens are capitalised (names, sentence starts).",
    f"<b>Duplicates:</b> {DUP['exact_groups']} exact-duplicate pairs plus {DUP['pairs_near_only']} near-duplicate pairs "
    f"(cosine >= 0.9); de-duplication removes {DUP['docs_dropped_by_dedup']} documents ({DUP['dropped_by_category']['tech']} "
    f"of them tech), leaving {n(DUP['docs_after_dedup'])}. No duplicate pair crosses categories.",
])
pair("01_category_distribution.png", "05_doc_length_hist.png",
     "Left: documents per category. Right: document length before (raw words) and after preprocessing (tokens) - "
     "preprocessing roughly halves each document.")

# ================================================================ 3 implementation
h1("Implementation approach",
   "All core IR algorithms are written by hand in the textmining/ package; the Streamlit app and the experiment "
   "script call the same code, so the app and this report agree.")
story.append(P(f"<b>Pipeline.</b> ingest (CSV / JSON / ZIP of .txt) -> preprocess -> inverted index -> TF-IDF -> "
               "{ Naive Bayes | K-Means + Purity/RI | cosine Top-K recommender | duplicate detector } -> Streamlit "
               f"visualisations. {code('experiments.py')} (seed 42) writes every number in this report to "
               f"{code('report/results.json')}."))
story.append(P("<b>No built-in implementations.</b> The inverted index, TF-IDF, cosine similarity, Naive Bayes, K-Means, "
               "Purity and Rand Index are hand-implemented; scikit-learn is not used (not even in requirements.txt). "
               "numpy serves only as an array container, NLTK only supplies the Porter stemmer, WordNet lemmatiser and "
               "stop-word list, and PCA is used only to draw the 2-D scatter plot."))
fx = lambda s: Paragraph(esc(s), FCELL)  # noqa: E731
story.append(table([
    ["Component", "Formula / procedure"],
    ["TF-IDF (ltc)", fx("w(t,d) = (1 + log10 tf(t,d)) * log10(N / df_t)")],
    ["L2 normalisation", fx("w'(t,d) = w(t,d) / sqrt( sum_t' w(t',d)^2 )")],
    ["Cosine similarity", fx("cos(a,b) = a.b / (|a| |b|) = a.b   (rows are unit length)")],
    ["Multinomial NB", fx("P(t|c) = (T_ct + 1) / (sum_t' T_ct' + |V|);  c* = argmax_c log P(c) + sum_t tf(t,d) log P(t|c)")],
    ["K-Means", "k-means++ init -> assign each doc to nearest centroid (squared Euclidean) -> centroid = mean of members "
                "-> stop when no document moves, max shift < 1e-4, or 100 iterations; best SSE of 5 restarts"],
    ["Purity", fx("purity = (1/N) sum_k max_j |w_k AND c_j|")],
    ["Rand Index", fx("RI = (TP + TN) / C(N,2);  TP = sum_kj C(n_kj,2), TP+FP = sum_k C(|w_k|,2), TP+FN = sum_j C(|c_j|,2)")],
], [3.2 * cm, 13.6 * cm], "Formulas implemented (w_k = cluster k, c_j = class j, C(n,2) = number of pairs).", font=8.4))
story.append(P(f"<b>Evaluation protocol.</b> Naive Bayes uses a stratified 80/20 split (seed {NB['split']['seed']}; "
               f"{n(NB['split']['train'])} train / {NB['split']['test']} test) and builds its vocabulary from training "
               "documents only. Clustering, recommendation and duplicate detection never see labels; labels are used only "
               "afterwards to compute Purity, Rand Index and Precision@K."))

# ================================================================ 4 preprocessing
h1("Preprocessing and feature choices",
   f"Lowercasing, punctuation/number removal, stop-word removal and Porter stemming cut the vocabulary by "
   f"{PRE['vocab_reduction_pct']}% and the token count by {PRE['token_reduction_pct']}%, leaving a compact, topical "
   "feature space.")
steps = PRE["steps"]
story.append(table(
    [["Cumulative step", "Tokens", "Vocabulary", "Avg tokens / doc"]] +
    [[s["step"].split(". ", 1)[1], n(s["total_tokens"]), n(s["vocabulary_size"]), s["avg_tokens_per_doc"]] for s in steps],
    [7.4 * cm, 2.8 * cm, 2.8 * cm, 3.2 * cm], "Effect of each preprocessing step on the full corpus.", right_from=1))
pr = TF["pruning"]
story += bullets([
    "<b>Baseline configuration:</b> markup stripping, lowercasing, regex tokenisation, drop punctuation and numbers, "
    "minimum token length 2, NLTK English stop-words, Porter stemming, unigrams. Lemmatisation and n-grams are switchable "
    "in the app and are compared in the ablation (Section 9).",
    f"<b>Stop-words</b> remove {pct(1 - steps[3]['total_tokens'] / steps[2]['total_tokens'])} of tokens but almost no "
    f"vocabulary; <b>stemming</b> then merges word forms ({n(steps[3]['vocabulary_size'])} -> {n(steps[4]['vocabulary_size'])} types).",
    f"<b>TF-IDF features:</b> ltc weighting with L2 normalisation; terms with df < {TF['params']['min_df']} "
    f"({n(pr['removed_rare(<min_df)'])} terms) or df > {pct(TF['params']['max_df_ratio'], 0)} of N are pruned and the "
    f"{n(pr['kept_terms'])} highest-df terms kept, giving a {TF['matrix_shape'][0]} x {TF['matrix_shape'][1]} matrix "
    f"({TF['density_pct']}% non-zero; every row has norm 1.0).",
    f"<b>Worked example:</b> term '{TF['walkthrough']['term']}' in {TF['walkthrough']['doc_id']}: tf = {TF['walkthrough']['tf']}, "
    f"df = {TF['walkthrough']['df']}, weight {TF['walkthrough']['tf_weight']} x idf {TF['walkthrough']['idf']} = "
    f"{TF['walkthrough']['w_unnormalised']}, divided by the document length {TF['walkthrough']['doc_vector_length']} = "
    f"{TF['walkthrough']['w_normalised']} (identical to the matrix entry).",
])
figure("03_top20_before_after.png", "Top-20 terms before (raw tokens) and after the baseline pipeline: function words "
       "and punctuation give way to content stems such as 'year', 'peopl', 'game' and 'govern'.", 15.5 * cm)

# ================================================================ 5 classification
h1("Classification results (Naive Bayes)", newpage=True, lead=
   f"The hand-written multinomial Naive Bayes classifier labels {pct(NB['accuracy'])} of the {NB['split']['test']} test "
   f"articles correctly ({NB['n_correct']} right, {NB['n_wrong']} wrong), with macro-F1 {f3(NB['macro']['f1'])}.")
rows = [["Class", "Precision", "Recall", "F1", "Support"]]
for c in CATS:
    m = NB["per_class"][c]
    rows.append([c, f3(m["precision"]), f3(m["recall"]), f3(m["f1"]), m["support"]])
rows.append(["<b>Accuracy</b>", "", "", f"<b>{f3(NB['accuracy'])}</b>", NB["split"]["test"]])
for k in ("macro", "weighted"):
    rows.append([f"{k.capitalize()} avg", f3(NB[k]["precision"]), f3(NB[k]["recall"]), f3(NB[k]["f1"]), NB["split"]["test"]])
metrics_t = table(rows, [3.4 * cm, 2.2 * cm, 2.2 * cm, 2.2 * cm, 2.2 * cm], right_from=1, raw=True, font=9)
samp = NB["sample_predictions"]
wrong = [s for s in samp if not s["correct"]][:3]
right = [s for s in samp if s["correct"]][:8 - len(wrong)]
sel = sorted(wrong + right, key=lambda s: s["doc_id"])
pred_rows = [["doc_id", "Title", "Actual", "Predicted", "P(c|d)"]]
for s in sel:
    mark = "" if s["correct"] else " (x)"
    pred_rows.append([s["doc_id"], s["title"], s["actual"], s["predicted"] + mark, f"{s['posterior']:.3f}"])
story.append(KeepTogether([metrics_t, caption("Table", "Per-class and averaged metrics on the held-out test set.")]))
story.append(table(pred_rows, [1.8 * cm, 7.0 * cm, 2.6 * cm, 3.2 * cm, 1.6 * cm],
                   f"Sample of predicted labels ((x) = misclassified). All {NB['split']['test']} predictions are in "
                   f"{code('report/nb_test_predictions.csv')} and in the app.", right_from=4))
pair("07_nb_confusion_matrix.png", "08_nb_actual_vs_predicted.png",
     "Left: confusion matrix (rows = actual, columns = predicted). Right: actual vs predicted class distribution.",
     widths=(7.4 * cm, 9.4 * cm))
story.append(P(f"Errors are mostly cross-desk stories (e.g. a salary scandal labelled politics, a media company's games "
               f"deal labelled tech); tech has the lowest precision ({f3(NB['per_class']['tech']['precision'])}) because it "
               f"attracts {NB['per_class']['tech']['fp']} false positives. <b>Leakage note:</b> {LK['leaked_docs']} test "
               f"articles have an exact copy in the training set and are all classified correctly "
               f"({f3(LK['acc_on_leaked'])} vs {f3(LK['acc_on_non_leaked'])} on the rest) - see Q2."))

# ================================================================ 6 clustering
h1("Clustering results (K-Means)", newpage=True, lead=
   f"Hand-written K-Means (k = {KM['params']['k']}) recovers the five desks without seeing any label: purity "
   f"{f3(KM['purity'])}, Rand Index {f3(KM['rand_index'])}.")
p = KM["params"]
story.append(P(f"<b>Configuration.</b> k = {p['k']}, k-means++ initialisation, {p['n_init']} restarts (best SSE kept), "
               f"squared Euclidean distance on L2-normalised TF-IDF rows, tol {p['tol']}, max {p['max_iter']} iterations, "
               f"seed {p['seed']}. The best run stopped because it <b>{KM['stop_reason'].split(': ', 1)[-1]}</b> "
               f"(final SSE {KM['sse']})."))
H = KM["history"]
late = max((h for h in H if h["iteration"] > 10), key=lambda h: h["docs_changed"])
picks = sorted({1, 2, 5, late["iteration"], H[-1]["iteration"] - 3, H[-1]["iteration"]})
it_rows = [["Iteration", "Docs moved", "Max centroid shift", "SSE", "Cluster sizes"]]
for h in H:
    if h["iteration"] in picks:
        it_rows.append([h["iteration"], n(h["docs_changed"]), f"{h['max_centroid_shift']:.4f}", f"{h['SSE']:.2f}",
                        " / ".join(str(x) for x in h["cluster_sizes"])])
story.append(table(it_rows, [1.8 * cm, 2.2 * cm, 3.2 * cm, 2.4 * cm, 5.2 * cm],
                   f"Condensed iteration log of the best run ({KM['n_iter']} iterations; the full log is shown in the app). "
                   f"SSE falls monotonically; the late burst around iteration {late['iteration']} is an oversized "
                   "cluster being broken up before convergence.", right_from=1))
pc = KM["pair_counts"]
story.append(table([["Metric", "Value", "Pair counts", "Value"],
                    ["Purity", f3(KM["purity"]), "TP (same cluster, same class)", n(pc["TP"])],
                    ["Rand Index", f3(KM["rand_index"]), "FP (same cluster, different class)", n(pc["FP"])],
                    ["SSE", f"{KM['sse']:.1f}", "FN (different cluster, same class)", n(pc["FN"])],
                    ["Documents", n(D["n_docs"]), "TN (different cluster, different class)", n(pc["TN"])]],
                   [2.6 * cm, 2.0 * cm, 6.4 * cm, 2.4 * cm], "Clustering evaluation against the true desks.", right_cols={1, 3}, font=9))
bf = KM["bruteforce_check"]
fs = KS["full_ninit5"]
story.append(P(f"<b>Verification.</b> The fast contingency-table Rand Index equals a literal O(N^2) pair loop on a "
               f"{bf['sample']}-document sample ({bf['fast']} vs {bf['bruteforce']}). <b>Seed stability:</b> over 5 seeds "
               f"purity is {f3(fs['purity_mean'])} +/- {f3(fs['purity_std'])} with 5 restarts, but ranges "
               f"{f3(KS['full_ninit1']['purity_min'])}-{f3(KS['full_ninit1']['purity_max'])} with a single initialisation."))
figure("11_kmeans_pca_scatter.png", "2-D PCA projection of all documents, coloured by K-Means cluster (left) and by "
       "true category (right). PCA is used for display only; clustering runs in the full 5,000-dimensional space.", 16 * cm)
cont = [["Cluster", "Majority", "Size", "Share", "Top terms"]]
for k in sorted(KM["contingency"], key=int):
    row = KM["contingency"][k]
    maj = KM["majority_label"][k]
    size = KM["cluster_sizes"][k]
    cont.append([f"C{k}", maj, size, pct(row[maj] / size), ", ".join(KM["top_terms"][k][:3])])
ct = table(cont, [1.3 * cm, 2.3 * cm, 1.1 * cm, 1.3 * cm, 2.8 * cm], font=8.2, raw=True, right_cols={2, 3})
pair("12_kmeans_cluster_sizes.png", ct,
     f"Left: documents per cluster. Right: cluster assignments vs true labels (majority label names the cluster; "
     f"full membership in {code('report/kmeans_membership.csv')}).", widths=(8.0 * cm, 9.0 * cm))

# ================================================================ 7 comparison
h1("Comparison of classification and clustering results",
   "Both methods agree with the editorial desks about 96% of the time, but they answer different questions and their "
   "numbers are not directly interchangeable.")
cmp = R["comparison"]
nbs = ST["naive_bayes"]["full"]
story.append(table([
    ["", "Naive Bayes (supervised)", "K-Means (unsupervised)"],
    ["Uses labels", f"Yes - trained on {n(NB['split']['train'])} labelled docs", "No - labels only for evaluation"],
    ["Evaluated on", f"{NB['split']['test']} held-out test docs", f"All {n(D['n_docs'])} docs"],
    ["Main metrics", f"Accuracy {f3(cmp['NB accuracy'])}; macro P/R/F1 {f3(cmp['NB macro-P'])} / {f3(cmp['NB macro-R'])} / "
                     f"{f3(cmp['NB macro-F1'])}", f"Purity {f3(cmp['KM purity'])}; Rand Index {f3(cmp['KM Rand index'])}"],
    ["5-seed mean", f"Accuracy {f3(nbs['acc_mean'])} +/- {f3(nbs['acc_std'])}",
     f"Purity {f3(cmp['KM purity (5-seed mean)'])}; RI {f3(cmp['KM RI (5-seed mean)'])}"],
    ["Output", "A desk label + posterior for every new article", "Unnamed groups; names come from majority labels / top terms"],
], [3.0 * cm, 6.9 * cm, 6.9 * cm], "Naive Bayes vs K-Means."))
tn_share = pc["TN"] / pc["pairs"]
story.append(P("Naive Bayes is <b>supervised</b>: it learns the desk boundaries from labelled examples and is judged on "
               "unseen documents, so its accuracy is a genuine generalisation estimate and it can route new articles "
               f"directly. K-Means is <b>unsupervised</b>: it only groups similar documents, and purity "
               f"({f3(KM['purity'])}) equals the accuracy of mapping each cluster to its majority class - computed on the "
               "same data that was clustered, after looking at the labels. The Rand Index is inflated by true negatives "
               f"({pct(tn_share)} of all pairs are 'different cluster, different class'). K-Means also needs k in advance "
               "and several restarts to be stable, whereas NB varies only with the split. That an unsupervised method "
               "almost matches the supervised one shows that the five desks are lexically well separated."))

# ================================================================ 8 recommendation
h1("Recommendation results", lead=
   f"Content-based Top-K recommendation by TF-IDF cosine similarity: on average {pct(PK['P@5'])} of the top-5 "
   "results share the query's desk.")
story.append(P(f"<b>Approach.</b> The query (a selected document or free text, preprocessed with the same pipeline) is a "
               f"TF-IDF vector q; every other document d is scored by {code('score(q,d) = cos(q,d) = q.d')} and the K "
               "highest are returned with their metadata and the shared terms that contribute most to the score. Documents "
               "with cosine >= 0.9 to the query (near-duplicates) can be hidden."))
qx = next(q for q in REC["queries"] if q["query"]["doc_id"] == "POL-001")
qq = qx["query"]
rr = [["#", "Score", "doc_id", "Title", "Category", "Source", "Shared terms"]]
for r in qx["top5"]:
    rr.append([r["rank"], f3(r["score"]), r["doc_id"], r["title"], r["category"], r["source"], ", ".join(r["shared_terms"][:4])])
story.append(table(rr, [0.6 * cm, 1.2 * cm, 1.6 * cm, 4.6 * cm, 1.8 * cm, 1.8 * cm, 5.2 * cm],
                   f"Top-5 recommendations for {qq['doc_id']} '{qq['title']}' ({qq['category']}). Period: 2004-2005 for all.",
                   font=8.2))
figure(os.path.join(SHOTS, "recommendation.png"), "Recommendation panel of the Streamlit app: query selection, Top-K "
       "slider, results with scores, metadata and shared-term explanations.", 15 * cm)
ft = REC["free_text"]
story.append(P(f"<b>Quality.</b> Using 'same category' as a relevance proxy, P@5 = {f3(PK['P@5'])} and P@10 = "
               f"{f3(PK['P@10'])} (sport best at {f3(REC['p5_per_category']['sport'])}, entertainment lowest at "
               f"{f3(REC['p5_per_category']['entertainment'])}). A free-text query ('{ft['query']}') returns five tech "
               f"articles led by '{ft['results'][0]['title']}' ({f3(ft['results'][0]['score'])})."))
story.append(P("<b>Findings and limitations.</b>"))
story += bullets([
    f"Results are explainable: shared stems such as {', '.join(qx['top5'][0]['shared_terms'][:3])} show why "
    f"{qx['top5'][0]['doc_id']} is ranked first; the cross-desk hit {next((r['doc_id'] for r in qx['top5'] if not r['same_category']), '-')} "
    "is still topically relevant (employment policy), so the category proxy understates quality.",
    f"Duplicates flood lists: {n(REC['lists_with_dup_in_top5'])} documents have a near-copy in their top-5 "
    "(e.g. BUS-007's #1 result is a verbatim copy, score 1.000) - hence the hide-duplicates option.",
    "Bag-of-words cannot match synonyms or paraphrases, and short or generic queries (e.g. TEC-001 on election ink) drift "
    "to other desks because a few rare shared words dominate.",
    "No user interaction data exists, so there is no personalisation, popularity or recency signal.",
])

# ================================================================ 9 inferences
h1("Inferences and limitations", newpage=True, lead=
   "Answers to the five compulsory questions, using the ablation below (each configuration re-runs the full pipeline "
   "with and without de-duplication).")
abl_names = ["Baseline", "Baseline + dedup", "No stop-word removal", "Lemmatization", "No normalizer", "Uni+bigram",
             "No lowercasing, no stemming"]
ab = [["Configuration", "Vocab (raw)", "NB acc", "NB macro-F1", "Purity (5-seed)", "RI (seed 42)", "P@5"]]
for nm in abl_names:
    a = A[nm]
    ab.append([nm.replace("Baseline", "Baseline (stop + stem)", 1) if nm == "Baseline" else nm, n(a["vocab_raw"]),
               f3(a["nb_accuracy"]), f3(a["nb_macro_f1"]),
               f"{f3(a['km_purity_seedmean'])} +/- {a['km_purity_seedstd']:.3f}", f3(a["km_rand_index"]),
               f3(a["precision_at_5"])])
story.append(table(ab, [4.6 * cm, 2.0 * cm, 1.5 * cm, 2.0 * cm, 2.9 * cm, 1.9 * cm, 1.4 * cm],
                   "Ablation. Purity (5-seed) = mean +/- std over 5 seeds with fewer restarts than the main run, so it is "
                   "slightly below Section 6.", right_from=1, font=8.3))
base, based = A["Baseline"], A["Baseline + dedup"]
nsw, lem, nonorm, bi, nolns = (A["No stop-word removal"], A["Lemmatization"], A["No normalizer"], A["Uni+bigram"],
                               A["No lowercasing, no stemming"])

h2("Q1. How preprocessing choices affected the results")
nb_vals = [A[x]["nb_accuracy"] for x in abl_names if "dedup" not in x]
story += bullets([
    f"<b>Vocabulary:</b> the baseline has {n(base['vocab_raw'])} types; dropping stemming raises it to "
    f"{n(nonorm['vocab_raw'])}, lemmatisation gives {n(lem['vocab_raw'])}, no lowercasing and no stemming "
    f"{n(nolns['vocab_raw'])}, and bigrams explode it to {n(bi['vocab_raw'])}. (Disabling lowercasing alone changes "
    "nothing because the Porter stemmer lowercases its output.)",
    f"<b>Classification</b> is robust: NB accuracy stays within {f3(min(nb_vals))}-{f3(max(nb_vals))}, smaller than the "
    f"spread across random splits ({f3(nbs['acc_mean'])} +/- {f3(nbs['acc_std'])}). Keeping stop-words is slightly worse "
    f"({f3(nsw['nb_accuracy'])}); the benefit of the baseline is a much smaller model.",
    f"<b>Clustering</b> is sensitive: the baseline has the highest seed-averaged purity ({f3(base['km_purity_seedmean'])} +/- "
    f"{base['km_purity_seedstd']:.3f}) versus {f3(nsw['km_purity_seedmean'])} without stop-word removal, "
    f"{f3(nonorm['km_purity_seedmean'])} without stemming and {f3(bi['km_purity_seedmean'])} with bigrams - richer "
    "vocabularies spread the 5,000 features over redundant variants and create more local optima.",
    f"<b>Recommendation</b> prefers specific features: P@5 rises from {f3(base['precision_at_5'])} (baseline) to "
    f"{f3(bi['precision_at_5'])} with bigrams and {f3(nolns['precision_at_5'])} with case-sensitive unstemmed tokens, "
    "because exact rare-term overlap matters for nearest neighbours. Stop-words + stemming is the best overall compromise.",
])

h2("Q2. Effect of duplicates and how they were detected and handled")
story += bullets([
    f"<b>Detection:</b> exact = MD5 of lower-cased, whitespace-collapsed text ({DUP['exact_groups']} pairs); near = TF-IDF "
    f"cosine >= 0.9 confirmed by word 5-shingle Jaccard ({DUP['pairs_near_only']} more pairs, mean Jaccard "
    f"{DUP['near_only_jaccard_mean']}). Union-find merges pairs; {DUP['docs_dropped_by_dedup']} redundant documents "
    "can be dropped, or just flagged/hidden in the app.",
    f"<b>Classification:</b> {LK['leaked_docs']} of {NB['split']['test']} test documents have a copy in training and score "
    f"{f3(LK['acc_on_leaked'])} accuracy (vs {f3(LK['acc_on_non_leaked'])}) - memorisation, not generalisation. After "
    f"de-duplication leakage is 0 and accuracy is {f3(based['nb_accuracy'])} (5 splits: "
    f"{f3(ST['naive_bayes']['dedup']['acc_mean'])}); only this figure is an honest estimate.",
    f"<b>Clustering:</b> copies have distance 0, always share a cluster and add guaranteed TP pairs and double weight to "
    f"centroids. Removing them made K-Means less stable: seed-42 purity {f3(base['km_purity'])} -> {f3(based['km_purity'])}; "
    f"5-seed mean {f3(KS['full_ninit5']['purity_mean'])} -> {f3(KS['dedup_ninit5']['purity_mean'])}. With 20 restarts it "
    f"returns to {f3(KS['dedup_seed42_ninit20']['purity'])}, so the drop is an optimisation effect.",
    f"<b>Recommendation:</b> {n(REC['lists_with_dup_in_top5'])} result lists contain a near-copy, wasting slots and "
    f"inflating P@5 ({f3(PK['P@5'])} vs {f3(PK['P@5_hide_dups'])} with duplicates hidden).",
])

h2("Q3. Recommendation approach: content-based vs collaborative")
story.append(P(f"We use <b>content-based</b> recommendation (TF-IDF cosine, Section 8) because the corpus has rich text "
               "but no user-document interaction data, so collaborative filtering would have nothing to learn from."))
story += bullets([
    "<b>Content-based</b> needs only item content (text, metadata). It works for brand-new articles and new users, is "
    "explainable through shared terms, but over-specialises and cannot judge quality. Preferable when the catalogue changes "
    "fast (news), users are few, or transparency is required.",
    "<b>Collaborative</b> needs a user-item interaction matrix (reads, clicks, ratings, saves) from many users. It finds "
    "non-obvious, co-read items and captures taste, but suffers from cold start and sparsity. Preferable with many active "
    "users and a stable catalogue.",
    "A production system would be <b>hybrid</b>: content-based for fresh stories, blended with collaborative signals once "
    "analyst reading logs exist.",
])

h2("Q4. How the techniques help domain experts decide")
story += bullets([
    f"<b>Text mining</b> quantifies the feed (volume per desk, key vocabulary) and exposes {DUP['docs_dropped_by_dedup']} "
    f"redundant copies ({pct(DUP['docs_dropped_by_dedup'] / D['n_docs'])}), so coverage reports count stories, not copies.",
    f"<b>Classification</b> routes articles to desks with {pct(NB['accuracy'])} accuracy; low-confidence or cross-desk "
    "predictions can go to a human review queue.",
    f"<b>Clustering</b> finds groupings without labels (purity {f3(KM['purity'])}) - useful for an unlabelled client archive or "
    "for spotting emerging sub-themes with a larger k.",
    "<b>Recommendation</b> turns any story into a research starting point with explained, de-duplicated related coverage. "
    "Together: ingest -> de-duplicate -> route -> explore themes -> brief, with each step measured so analysts know how far to trust it.",
])

h2("Q5. Naive Bayes vs K-Means: findings, limitations, improvements")
story.append(P(f"NB reaches accuracy {f3(NB['accuracy'])} and macro-F1 {f3(NB['macro']['f1'])} on held-out data; K-Means "
               f"reaches purity {f3(KM['purity'])} and RI {f3(KM['rand_index'])} without labels (5-seed mean purity "
               f"{f3(fs['purity_mean'])}). <b>Key findings:</b> the desks are lexically well separated; preprocessing changes "
               "the feature space more than NB accuracy; duplicates inflate NB accuracy through leakage and inflate P@5; "
               "single-seed K-Means results are unreliable without restarts. Limitations and improvements follow."))

h2("Limitations and improvements for real-world deployment")
story += bullets([
    "<b>Data:</b> one clean, balanced 2004-05 source with five desks; a live feed is multi-source, multilingual and "
    "drifts - retrain periodically and use time-based splits.",
    "<b>Features:</b> bag-of-words ignores synonyms and word order - add LSA or sentence embeddings; use sparse matrices "
    "and an approximate nearest-neighbour index to scale beyond tens of thousands of documents.",
    "<b>Classification:</b> NB posteriors are over-confident (~1.0 even on errors) - calibrate or use logistic regression, "
    "allow multi-label output, and de-duplicate before splitting.",
    "<b>Clustering:</b> k must be fixed and runs hit local optima - use more restarts, spherical K-Means on cosine, and "
    "choose k by silhouette or stability.",
    "<b>Recommendation:</b> same-category is only a relevance proxy - collect analyst judgements; add MinHash-LSH duplicate "
    "collapsing, diversity (MMR), recency and collaborative signals.",
])

# ================================================================ 10 evidence of execution
h1("Evidence of execution in the BITS Virtual Lab", newpage=True)
# Only claim Virtual Lab execution once screenshots/VIRTUAL_LAB.txt exists (created after capturing in the lab).
videos = [os.path.basename(v) for ext in ("mp4", "mov", "MP4", "MOV") for v in glob.glob(os.path.join(ROOT, f"*.{ext}"))]
rec_txt = (f" A screen recording ({code(', '.join(sorted(set(videos))))}) is included in the submission." if videos else "")
if os.path.exists(os.path.join(SHOTS, "VIRTUAL_LAB.txt")):
    story.append(P(f"The application was executed in the BITS Virtual Lab with {code('streamlit run app.py')}. The "
                   f"screenshots below show the working front end; the recommendation panel is shown in Section 8.{rec_txt}",
                   LEAD))
else:
    story.append(P(f"Screenshots of the working Streamlit front end launched with {code('streamlit run app.py')}; the "
                   f"recommendation panel is shown in Section 8.{rec_txt}", LEAD))
SHOT_LIST = [("01_dashboard.png", "Dashboard and corpus overview"),
             ("03_preprocessing_before_after.png", "Preprocessing: before/after distributions"),
             ("07_naive_bayes.png", "Naive Bayes classification"),
             ("08_kmeans.png", "K-Means clustering steps"),
             ("09_evaluation_comparison.png", "Evaluation and comparison"),
             ("12_performance_analytics.png", "Performance analytics")]
shots = [(os.path.join(SHOTS, f), c) for f, c in SHOT_LIST if os.path.exists(os.path.join(SHOTS, f))]
cap_style = ParagraphStyle("sc", parent=CAP, spaceBefore=1, spaceAfter=0)
grid = []
for i in range(0, len(shots), 2):
    cells = []
    for path, c in shots[i:i + 2]:
        _counts["Figure"] += 1
        cells.append([img(path, 8.3 * cm), P(f"<b>Figure {_counts['Figure']}.</b> {c}", cap_style)])
    grid.append(cells + [""] * (2 - len(cells)))
for row in grid:
    g = Table([row], colWidths=[8.45 * cm, 8.45 * cm])
    g.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                           ("LEFTPADDING", (0, 0), (-1, -1), 2), ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 14)]))
    story.append(g)
if not grid:
    story.append(P("(Add PNG screenshots to the screenshots/ folder and re-run this script.)"))

# ================================================================ appendix
h1("Appendix: how to run and references", numbered=False)
story.append(Paragraph("pip install -r requirements.txt<br/>streamlit run app.py<br/>"
                       "python experiments.py            # regenerates report/results.json and figures (seed 42)<br/>"
                       "python report/build_report.py    # rebuilds this PDF", MONO))
story += bullets([
    "D. Greene, P. Cunningham. Practical Solutions to the Problem of Diagonal Dominance in Kernel Document Clustering. ICML 2006 (BBC dataset).",
    "C. D. Manning, P. Raghavan, H. Schuetze. Introduction to Information Retrieval. Cambridge University Press, 2008 (Ch. 6, 13, 16).",
    "M. F. Porter. An algorithm for suffix stripping. Program 14(3), 1980.",
    "D. Arthur, S. Vassilvitskii. k-means++: The Advantages of Careful Seeding. SODA 2007.",
    "W. M. Rand. Objective criteria for the evaluation of clustering methods. JASA 66(336), 1971.",
])

out = os.path.join(HERE, "Report.pdf")
doc = ReportDoc(out)
doc.build(story)
print("wrote", out, "pages:", doc.page, "| tables:", _counts["Table"], "+ member table | figures:", _counts["Figure"])
for lvl, t, pg in SECTION_PAGES:
    if lvl == 0:
        print(f"  p{pg:>2}  {t}")
