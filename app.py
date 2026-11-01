"""Streamlit demo: phishing website detection with 5 classifiers.

Run app:        streamlit run app.py
Self-check:     python app.py --test
"""
import html
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
TARGET = "status"        # 1 = phishing, 0 = legitimate
LABELS = ["Legitimate", "Phishing"]
MODEL_NAMES = ["Logistic Regression", "Decision Tree", "kNN", "Naive Bayes", "SVM"]

CSS = """
<style>
/* ---------- dark theme (self-contained, no config file needed) ---------- */
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {background: #0f1115 !important;}
.stApp {color: #e5e7eb !important;}
[data-testid="stHeader"], header[data-testid="stHeader"] {display: none !important; height: 0 !important;
    pointer-events: none !important;}
[data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stSkipLink"],
footer, #MainMenu {display: none !important;}
.block-container {padding-top: 1.5rem !important; max-width: 1150px;}
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p {color: #e5e7eb !important;}
code {background: #222a3b !important; color: #e5e7eb !important; border-radius: 4px; padding: 1px 6px;}
[data-testid="stVerticalBlockBorderWrapper"] {border-color: #2a2f3a !important;}

/* file uploader */
[data-testid="stFileUploaderDropzone"] {background: #171a21 !important; border: 1px dashed #3a4256 !important;}
[data-testid="stFileUploader"] * {color: #cbd5e1 !important;}
[data-testid="stFileUploader"] button {background: #222a3b !important; border: 1px solid #3a4256 !important;}
[data-testid="stFileUploader"] svg {fill: #cbd5e1 !important;}
[data-testid="stFileChip"], [data-testid="stFileChip"] > div,
[data-testid="stFileUploaderFile"] {background: #222a3b !important; border-color: #3a4256 !important;}
[data-testid="stFileChip"] *, [data-testid="stFileUploaderFile"] * {color: #e5e7eb !important;}

/* buttons */
.stButton > button {background: #2f6fed !important; color: #ffffff !important; border: 1px solid #2f6fed !important;
                    border-radius: 8px; font-weight: 600; padding: 0.5rem 1.6rem;}
.stButton > button:hover {background: #4380ff !important; border-color: #4380ff !important;}
.stButton > button:disabled {background: #222a3b !important; color: #6b7686 !important; border-color: #2a2f3a !important;}
.stDownloadButton > button {background: #222a3b !important; color: #e5e7eb !important;
                            border: 1px solid #3a4256 !important; border-radius: 8px;}
.stDownloadButton > button:hover {border-color: #4f8cff !important;}

/* select box and dropdown (works on older and newer Streamlit) */
[data-testid="stSelectbox"] div, [data-testid="stSelectbox"] input {
    background-color: #171a21 !important; color: #e5e7eb !important; -webkit-text-fill-color: #e5e7eb !important;}
[data-testid="stSelectbox"] [role="group"], [data-testid="stSelectbox"] [data-baseweb="select"] > div {
    border: 1px solid #3a4256 !important; border-radius: 8px !important;}
[data-testid="stSelectbox"] button {background: transparent !important;}
[data-testid="stSelectbox"] [data-testid="stWidgetLabel"],
[data-testid="stSelectbox"] [data-testid="stWidgetLabel"] * {background: transparent !important;}
[data-testid="stSelectbox"] svg {fill: #cbd5e1 !important;}
[data-baseweb="popover"], [data-baseweb="popover"] *, [data-baseweb="menu"],
[role="listbox"], [role="listbox"] *, [role="option"],
[data-testid="stSelectboxVirtualDropdown"], [data-testid="stSelectboxVirtualDropdown"] * {
    background-color: #171a21 !important; color: #e5e7eb !important;}
[role="option"]:hover, [role="option"][aria-selected="true"], [role="option"][data-focused="true"],
[role="option"]:hover * , [role="option"][aria-selected="true"] * {background-color: #26304a !important;}

/* larger, resolution-friendly select box (rem/clamp units scale with the screen) */
[data-testid="stSelectbox"] [data-testid="stWidgetLabel"] p {
    font-size: clamp(1.05rem, 0.9rem + 0.5vw, 1.35rem) !important; font-weight: 600 !important; color: #f1f5f9 !important;}
[data-testid="stSelectbox"] [data-baseweb="select"] > div, [data-testid="stSelectbox"] [role="group"] {
    min-height: 3rem;}
[data-testid="stSelectbox"] [data-baseweb="select"] div, [data-testid="stSelectbox"] input {
    font-size: clamp(1rem, 0.9rem + 0.35vw, 1.2rem) !important;}
[role="option"] {font-size: clamp(1rem, 0.9rem + 0.35vw, 1.2rem) !important;}

/* expander */
[data-testid="stExpander"], [data-testid="stExpander"] details {background: #171a21 !important;
    border-color: #2a2f3a !important; border-radius: 10px;}
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary *,
[data-testid="stExpander"] p, [data-testid="stExpander"] li {color: #e5e7eb !important;}
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary:hover,
[data-testid="stExpander"] summary > div {background: #171a21 !important;}
[data-testid="stExpander"] summary svg {fill: #cbd5e1 !important;}

/* tabs (older: baseweb buttons; newer: react-aria divs) */
[role="tablist"] {gap: 6px; border-bottom: 1px solid #2a2f3a;}
[role="tab"] {height: 60px; padding: 0 30px; background: transparent !important;}
[role="tab"] p {font-size: 1.35rem !important; font-weight: 600 !important; color: #94a3b8 !important;}
[role="tab"][aria-selected="true"] {background: #1a2236 !important;}
[role="tab"][aria-selected="true"] p {color: #ffffff !important;}
[data-baseweb="tab-highlight"], [role="tab"] > div:not([data-testid="stMarkdownContainer"]) {
    background-color: #4f8cff !important; height: 3px;}
[data-baseweb="tab-border"] {background-color: #2a2f3a !important;}

/* ---------- custom components ---------- */
.hero {background: linear-gradient(135deg, #131a2b 0%, #1b2438 100%); border: 1px solid #2a3350;
       border-left: 6px solid #4f8cff; border-radius: 14px; padding: 26px 32px; margin-bottom: 18px;}
.hero h1 {color: #f8fafc; margin: 0 0 4px 0; font-size: 2rem; padding: 0;}
.hero .sub {color: #93a4c3; font-size: .95rem; margin-bottom: 14px;}
.hero .lbl {color: #7fa6ff; font-size: .75rem; letter-spacing: .12em; text-transform: uppercase; font-weight: 700;}
.hero p {color: #d6deed; margin: 4px 0 12px 0; font-size: 1rem; line-height: 1.55;}
.pill {display: inline-block; background: #222c44; color: #c7d4ee; border: 1px solid #33405f;
       border-radius: 6px; padding: 3px 12px; margin: 4px 8px 0 0; font-size: .82rem;}
.sect {color: #f1f5f9; font-size: 1.1rem; font-weight: 700; margin: 18px 0 2px 0;}
.note {color: #94a3b8; font-size: .9rem; margin-bottom: 10px;}
.alert {border-radius: 10px; padding: 12px 16px; margin: 12px 0; font-size: .95rem;}
.alert.info {background: #15213a; border: 1px solid #28406b; color: #cfe0ff;}
.alert.err {background: #3a1a1f; border: 1px solid #7a2c35; color: #fcd0d4;}
.kpi {border: 1px solid #2a2f3a; background: #171a21; border-radius: 10px; padding: 14px 18px; color: #e5e7eb;}
.kpi .l {font-size: .74rem; color: #94a3b8; text-transform: uppercase; letter-spacing: .05em;}
.kpi .v {font-size: 1.65rem; font-weight: 700; line-height: 1.3; color: #f8fafc;}
.kpi .d {font-size: .78rem; color: #7b8799;}
.final {background: linear-gradient(135deg, #0f2a22 0%, #12352a 100%); border: 1px solid #1f6f50;
        border-left: 6px solid #34c38f; border-radius: 14px; padding: 24px 30px; margin: 20px 0 22px 0;}
.final .tag {font-size: .78rem; letter-spacing: .14em; text-transform: uppercase; color: #6ee7b7; font-weight: 700;}
.final .name {font-size: 2.1rem; font-weight: 800; color: #f8fafc; margin: 2px 0 12px 0;}
.final .chips span {display: inline-block; background: #0b1f19; border: 1px solid #25664b; color: #d1fae5;
                    border-radius: 6px; padding: 5px 13px; margin: 0 8px 8px 0; font-size: .9rem;}
.final .chips b {color: #fff;}
.final h4 {color: #a7f3d0; font-size: .85rem; letter-spacing: .08em; text-transform: uppercase; margin: 14px 0 6px 0;}
.final p {margin: 0 0 6px 0; font-size: .96rem; color: #d1e8de; line-height: 1.55;}
.final b {color: #ffffff;}
.rank {width: 100%; border-collapse: collapse; margin-top: 8px; font-size: .92rem;}
.rank th {text-align: left; color: #86d4b4; font-weight: 600; border-bottom: 1px solid #25664b; padding: 6px 10px;}
.rank td {padding: 6px 10px; color: #d1e8de; border-bottom: 1px solid #1c4a38;}
.rank tr.top td {color: #fff; font-weight: 700; background: rgba(52,195,143,.14);}
.icard {border: 1px solid #2a2f3a; border-left: 6px solid var(--c); border-radius: 10px;
        padding: 14px 18px; margin-bottom: 10px; background: #171a21; color: #e5e7eb;}
.icard .top {display: flex; justify-content: space-between; align-items: center; font-size: 1.08rem;}
.badge {background: var(--c); color: #fff; border-radius: 6px; padding: 3px 12px; font-size: .8rem; font-weight: 700;}
.icard .prof {margin-top: 6px; font-weight: 600; color: #f1f5f9; font-size: .95rem;}
.icard .txt {margin-top: 3px; font-size: .92rem; color: #cbd5e1; line-height: 1.5;}
.icard .sub {color: #94a3b8;}
.legend span {display: inline-block; border-radius: 5px; padding: 2px 10px; margin-right: 10px; font-size: .82rem; color: #fff;}
.tbl-wrap {overflow-x: auto; border: 1px solid #2a2f3a; border-radius: 10px; margin: 8px 0 14px 0;}
.dtable {border-collapse: collapse; width: 100%; font-size: .9rem;}
.dtable th {background: #1a2030; color: #9fb0cc; text-align: left !important; padding: 9px 14px;
            font-weight: 600; border-bottom: 1px solid #2a2f3a; white-space: nowrap;}
.dtable td {padding: 8px 14px; border-bottom: 1px solid #1f2430; color: #dbe2ee; white-space: nowrap;}
.dtable td.num {text-align: right;}
.rec {background: #1f6f4a; color: #fff; border-radius: 4px; padding: 1px 8px; font-size: .72rem; margin-left: 8px;}
</style>
"""


def model_path(name):
    return MODEL_DIR / (name.replace(" ", "_").lower() + ".pkl")


def load_model(name):
    return joblib.load(model_path(name))


def score(model, X, y):
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    metrics = {
        "Accuracy": accuracy_score(y, pred),
        "AUC": roc_auc_score(y, proba),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }
    return metrics, pred


def split_xy(df):
    if TARGET not in df.columns:
        raise ValueError(f"CSV must contain a '{TARGET}' column (1 = phishing, 0 = legitimate).")
    return df.drop(columns=TARGET), df[TARGET]


def kpi(label, value, note=""):
    extra = f'<div class="d">{note}</div>' if note else ""
    return f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div>{extra}</div>'


def notice(text, kind="info"):
    return f'<div class="alert {kind}">{text}</div>'


def plural(n, word):
    return f"{n:,} {word}" if n == 1 else f"{n:,} {word}s"


def html_table(df, index=False):
    return '<div class="tbl-wrap">' + df.to_html(index=index, border=0, classes="dtable") + "</div>"


def cell_style(v, metric=""):
    strong, ok = (0.85, 0.70) if metric == "MCC" else (0.92, 0.80)
    bg = "#1f6f4a" if v >= strong else "#8a6a14" if v >= ok else "#9b3a42"
    return f"background-color: {bg}; color: #ffffff; font-weight: 600"


def metrics_table(table, best):
    head = "".join(f"<th>{c}</th>" for c in ["Model", *table.columns])
    rows = ""
    for name, row in table.iterrows():
        cells = "".join(f'<td class="num" style="{cell_style(v, c)}">{v:.4f}</td>'
                        for c, v in row.items())
        tag = '<span class="rec">Recommended</span>' if name == best else ""
        rows += f"<tr><td><b>{name}</b>{tag}</td>{cells}</tr>"
    return f'<div class="tbl-wrap"><table class="dtable"><tr>{head}</tr>{rows}</table></div>'


def draw_confusion(y, pred):
    cm = confusion_matrix(y, pred, labels=[0, 1])
    names = [["True Negative", "False Positive"], ["False Negative", "True Positive"]]
    fills = [["#1f7a54", "#a8434b"], ["#a8434b", "#1f7a54"]]
    bg, fg = "#0f1115", "#cbd5e1"
    fig, ax = plt.subplots(figsize=(5, 4.2))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    for i in range(2):
        for j in range(2):
            ax.add_patch(plt.Rectangle((j, 1 - i), 1, 1, color=fills[i][j], ec=bg, lw=4))
            ax.text(j + .5, 1.6 - i, f"{cm[i, j]:,}", ha="center", va="center",
                    fontsize=22, fontweight="bold", color="white")
            ax.text(j + .5, 1.3 - i, names[i][j], ha="center", va="center",
                    fontsize=10, color="#f1f5f9")
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([.5, 1.5], [f"Predicted\n{LABELS[0]}", f"Predicted\n{LABELS[1]}"], color=fg)
    ax.set_yticks([1.5, .5], [f"Actual\n{LABELS[0]}", f"Actual\n{LABELS[1]}"], color=fg)
    ax.tick_params(length=0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return fig


def interpret(m, y, pred):
    """Return (rating, colour, profile, explanation, error summary) for one model."""
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    mcc, prec, rec = m["MCC"], m["Precision"], m["Recall"]
    rating, colour = (("Strong", "#2f9e6e") if mcc >= 0.88 else
                      ("Good", "#3b82c4") if mcc >= 0.80 else
                      ("Moderate", "#c58a1b") if mcc >= 0.60 else ("Weak", "#b8454f"))
    if rec - prec > 0.2:
        profile = "High recall, low precision"
        text = (f"Detects {rec:.0%} of actual phishing sites, but only {prec:.0%} of the sites it "
                "flags are truly phishing, so it generates a large number of false positives "
                "(legitimate sites blocked). Not suitable as a standalone classifier.")
    elif prec - rec > 0.02:
        profile = "High precision, lower recall"
        text = (f"{prec:.0%} of the sites it flags are truly phishing, but it detects only "
                f"{rec:.0%} of actual phishing sites, so some phishing sites go undetected.")
    elif rec - prec > 0.02:
        profile = "High recall, lower precision"
        text = (f"Detects {rec:.0%} of actual phishing sites, but only {prec:.0%} of the sites "
                "it flags are truly phishing, so some legitimate sites are blocked.")
    else:
        profile = "Balanced precision and recall"
        text = (f"Precision ({prec:.0%}) and recall ({rec:.0%}) are close, so false positives "
                "and false negatives occur at a similar rate.")
    errors = (f"{plural(fn, 'phishing site')} missed (false negatives); "
              f"{plural(fp, 'legitimate site')} wrongly flagged (false positives); "
              f"{fn + fp:,} errors in {tn + fp + fn + tp:,} records.")
    return rating, colour, profile, text, errors


def interp_card(name, m, y, pred, crown=False):
    rating, colour, profile, text, errors = interpret(m, y, pred)
    mark = " &nbsp;(Recommended)" if crown else ""
    return (f'<div class="icard" style="--c:{colour}"><div class="top"><b>{name}{mark}</b>'
            f'<span class="badge">{rating} &nbsp;|&nbsp; MCC {m["MCC"]:.4f}</span></div>'
            f'<div class="prof">{profile}</div><div class="txt">{text}</div>'
            f'<div class="txt sub">{errors}</div></div>')


def final_card(table, y):
    metrics = list(table.columns)
    ranked = table.sort_values(["MCC", "F1"], ascending=False)
    best, second = ranked.index[0], ranked.index[1]
    b = table.loc[best]
    leads = [c for c in metrics if table[c].idxmax() == best]
    margin = table.loc[best, "MCC"] - table.loc[second, "MCC"]
    chips = "".join(f"<span>{c} <b>{b[c]:.4f}</b></span>" for c in metrics)
    rows = "".join(
        f'<tr class="{"top" if i == 0 else ""}"><td>{i + 1}</td><td>{n}</td>'
        f'<td>{r["MCC"]:.4f}</td><td>{r["F1"]:.4f}</td><td>{r["Accuracy"]:.4f}</td></tr>'
        for i, (n, r) in enumerate(ranked.iterrows()))
    narrow = (" The margin is narrow, so the top models perform comparably; "
              f"{best} is selected on the stated criterion." if margin < 0.01 else "")
    return (
        f'<div class="final"><div class="tag">Final result: recommended model</div>'
        f'<div class="name">{best}</div><div class="chips">{chips}</div>'
        f"<h4>How the best model is decided</h4>"
        f"<p>Models are ranked by the <b>Matthews Correlation Coefficient (MCC)</b>, with F1 as the "
        f"tie-breaker. MCC ranges from -1 to +1 and uses all four confusion-matrix counts "
        f"(true/false positives and negatives), so unlike accuracy it is not inflated when the "
        f"classes are unevenly sized (here {y.mean():.0%} phishing, {1 - y.mean():.0%} legitimate).</p>"
        f"<p>{best} ranks first with MCC {b['MCC']:.4f}, ahead of {second} by {margin:.4f}. "
        f"It leads on {len(leads)} of {len(metrics)} metrics ({', '.join(leads)}).{narrow}</p>"
        f"<h4>Ranking</h4><table class='rank'><tr><th>Rank</th><th>Model</th><th>MCC</th>"
        f"<th>F1</th><th>Accuracy</th></tr>{rows}</table></div>")


def run_app():
    import altair as alt
    import streamlit as st

    st.set_page_config(page_title="Phishing Website Detection", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="hero"><h1>Phishing Website Detection</h1>'
        '<div class="sub">Machine Learning Assignment 2: classification model comparison</div>'
        '<div class="lbl">Problem statement</div>'
        "<p>Classify whether a website URL is a legitimate page or a phishing attempt designed to "
        "steal user credentials. Blacklists cannot keep up with newly generated URLs, so the models "
        "here learn from 81 URL, page-content and domain attributes (for example URL length, number "
        "of dots and hyphens, use of an IP address, and domain age) to flag fraudulent links.</p>"
        '<div class="lbl">Objective</div>'
        "<p>Train five classifiers on labelled website data, then evaluate them on unseen test data "
        "using Accuracy, AUC, Precision, Recall, F1 and MCC to identify the most reliable model.</p>"
        '<span class="pill">Logistic Regression</span><span class="pill">Decision Tree</span>'
        '<span class="pill">kNN</span><span class="pill">Naive Bayes</span>'
        '<span class="pill">SVM</span></div>',
        unsafe_allow_html=True)

    with st.container(border=True):
        upload = st.file_uploader("Upload test_data.csv (81 feature columns plus a `status` column: "
                                  "1 = phishing, 0 = legitimate)", type="csv")
        run = st.button("Evaluate all models", type="primary", disabled=upload is None)

    if upload is None:
        return

    try:
        df = pd.read_csv(upload)
        X, y = split_xy(df)
    except Exception as err:
        st.markdown(notice("Could not read the file: " + html.escape(str(err)), "err"),
                    unsafe_allow_html=True)
        return

    st.markdown('<div class="sect">Test data summary</div>', unsafe_allow_html=True)
    stats = [("Test records", f"{len(df):,}", "websites (rows)"),
             ("Input features", X.shape[1], "attributes per website"),
             ("Actual phishing", f"{int((y == 1).sum()):,}", "records with status = 1"),
             ("Actual legitimate", f"{int((y == 0).sum()):,}", "records with status = 0")]
    for col, (label, value, note) in zip(st.columns(4), stats):
        col.markdown(kpi(label, value, note), unsafe_allow_html=True)

    key = (upload.name, upload.size)
    if run:
        try:
            with st.spinner("Scoring all models..."):
                outputs = {n: score(load_model(n), X, y) for n in MODEL_NAMES}
        except Exception as err:
            st.markdown(notice("Prediction failed (check the columns match the training data): "
                               + html.escape(str(err)), "err"), unsafe_allow_html=True)
            return
        st.session_state["key"] = key
        st.session_state["metrics"] = {n: o[0] for n, o in outputs.items()}
        st.session_state["preds"] = {n: o[1] for n, o in outputs.items()}

    ready = st.session_state.get("key") == key
    if not ready:
        st.markdown(notice("File loaded. Click <b>Evaluate all models</b> to see the results."),
                    unsafe_allow_html=True)
    else:
        table = pd.DataFrame(st.session_state["metrics"]).T.round(4)
        table.index.name = "Model"
        best = table.sort_values(["MCC", "F1"], ascending=False).index[0]

    tab_cmp, tab_det, tab_final, tab_data = st.tabs(
        ["Comparison", "Model details", "Final result", "Data preview"])

    with tab_data:
        st.markdown('<div class="note">First 15 rows of the uploaded file '
                    "(scroll sideways to see all columns).</div>", unsafe_allow_html=True)
        st.markdown(html_table(df.head(15)), unsafe_allow_html=True)

    if not ready:
        with tab_cmp:
            st.markdown(notice("Run the evaluation to see the comparison."), unsafe_allow_html=True)
        with tab_det:
            st.markdown(notice("Run the evaluation to inspect a model."), unsafe_allow_html=True)
        with tab_final:
            st.markdown(notice("Run the evaluation to see the final result."), unsafe_allow_html=True)
        return

    with tab_final:
        st.markdown(final_card(table, y), unsafe_allow_html=True)

    with tab_cmp:
        st.markdown('<div class="sect">Evaluation metrics (all models, same test data)</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="legend"><span style="background:#1f6f4a">Strong: 0.92 and above (MCC: 0.85 and above)</span>'
                    '<span style="background:#8a6a14">Acceptable: 0.80 to 0.92 (MCC: 0.70 to 0.85)</span>'
                    '<span style="background:#9b3a42">Weak: below 0.80 (MCC: below 0.70)</span></div>',
                    unsafe_allow_html=True)
        st.markdown(metrics_table(table, best), unsafe_allow_html=True)
        with st.expander("How to read these metrics"):
            st.markdown(
                "- **Accuracy**: share of all websites classified correctly.\n"
                "- **AUC**: ability to rank phishing sites above legitimate ones (1.0 is perfect).\n"
                "- **Precision**: of the sites flagged as phishing, the share that truly are.\n"
                "- **Recall**: of the actual phishing sites, the share that were caught.\n"
                "- **F1**: harmonic mean of precision and recall.\n"
                "- **MCC**: correlation between predicted and true labels (-1 to +1); "
                "the primary ranking criterion here.")
        long = table.reset_index().melt("Model", var_name="Metric", value_name="Score")
        palette = ["#4e79a7", "#76b7b2", "#59a14f", "#edc948", "#f28e2b", "#bab0ac"]
        chart = (alt.Chart(long).mark_bar().encode(
            x=alt.X("Model:N", sort=MODEL_NAMES, axis=alt.Axis(labelAngle=0, title=None)),
            xOffset="Metric:N",
            y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Metric:N", sort=list(table.columns),
                            scale=alt.Scale(domain=list(table.columns), range=palette),
                            legend=alt.Legend(orient="top", title=None)),
            tooltip=["Model", "Metric", alt.Tooltip("Score:Q", format=".4f")])
            .properties(height=340)
            .configure(background="#0f1115")
            .configure_axis(labelColor="#cbd5e1", titleColor="#cbd5e1", gridColor="#252b38",
                            domainColor="#3a4256", tickColor="#3a4256", labelFontSize=12)
            .configure_legend(labelColor="#cbd5e1", titleColor="#cbd5e1", labelFontSize=12)
            .configure_view(strokeWidth=0))
        st.altair_chart(chart, theme=None)

        st.markdown('<div class="sect">Model-by-model interpretation (ranked by MCC)</div>',
                    unsafe_allow_html=True)
        for name in table.sort_values(["MCC", "F1"], ascending=False).index:
            st.markdown(interp_card(name, st.session_state["metrics"][name], y,
                                    st.session_state["preds"][name], name == best),
                        unsafe_allow_html=True)
        st.download_button("Download results (CSV)", table.to_csv().encode(),
                           "model_results.csv", "text/csv")

    with tab_det:
        choice = st.selectbox("Select a model to inspect", MODEL_NAMES,
                              index=MODEL_NAMES.index(best))
        m, pred = st.session_state["metrics"][choice], st.session_state["preds"][choice]
        for col, (name, value) in zip(st.columns(6), m.items()):
            col.markdown(kpi(name, f"{value:.4f}"), unsafe_allow_html=True)
        st.write("")
        st.markdown(interp_card(choice, m, y, pred, choice == best), unsafe_allow_html=True)
        left, right = st.columns(2)
        with left:
            st.markdown("**Confusion matrix** (green = correct, red = misclassified)")
            st.pyplot(draw_confusion(y, pred))
        with right:
            st.markdown("**Classification report**")
            report = pd.DataFrame(classification_report(y, pred, target_names=LABELS,
                                                        output_dict=True)).T
            if "accuracy" in report.index:
                report.loc["accuracy", "support"] = len(y)
            report["support"] = report["support"].astype(int)
            report = report.round(3)
            report.index.name = "Class"
            st.markdown(html_table(report.reset_index()), unsafe_allow_html=True)


def self_test():
    """Checks that models load and metrics compute (uses a slice of the training file)."""
    df = pd.read_csv(MODEL_DIR / "train_data.csv").head(500)
    X, y = split_xy(df)
    for name in MODEL_NAMES:
        metrics, pred = score(load_model(name), X, y)
        assert len(pred) == len(y), f"{name}: wrong number of predictions"
        assert all(0 <= v <= 1 for k, v in metrics.items() if k != "MCC"), f"{name}: bad metric"
    try:
        split_xy(df.drop(columns=TARGET))
        raise AssertionError("missing target column was not detected")
    except ValueError:
        pass
    print("app self-test: all checks passed")


if "--test" in sys.argv:
    self_test()
else:
    run_app()
