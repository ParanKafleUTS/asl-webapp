"""
ASL Model Comparison Suite
===========================
Evaluates all 7 trained models on a shared test set and produces:

  Figure 1  — Model Overview (architecture summary table)
  Figure 2  — Accuracy & Loss bar chart (all models)
  Figure 3  — Confusion matrices (one per model, 7 subplots)
  Figure 4  — Per-class accuracy heatmap (models × classes)
  Figure 5  — Confidence distribution (violin plot per model)
  Figure 6  — Top-1 vs Top-3 accuracy comparison
  Figure 7  — Speed benchmark (inference time per model)
  Figure 8  — Precision / Recall / F1 radar chart
  Figure 9  — ROC-AUC per class (one model highlighted)
  Figure 10 — Confidence calibration curves
  Figure 11 — Hardest / Easiest classes (bar chart)
  Figure 12 — Model size vs accuracy scatter

Run:
    python compare_models.py

Outputs → ./model_comparison_results/
"""

import os, sys, time, pickle, warnings, itertools
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import cv2
import mediapipe as mp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_fscore_support, roc_curve, auc,
    top_k_accuracy_score,
)
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras

# ── Config ───────────────────────────────────────────────────────────────────

MODEL_DIR   = r"C:\Users\kafle\Documents\VS_AI\Hand_sign_to_voice"
# Point this to ANY folder containing class sub-folders of images to evaluate on.
# Uses the Kaggle dataset structure: DATA_DIR/A/img1.jpg, DATA_DIR/B/img2.jpg …
DATA_DIR    = r"C:\Users\kafle\Documents\VS_AI\Hand_sign_to_voice\asl_alphabet_train\asl_alphabet_train"
OUT_DIR     = os.path.join(os.path.dirname(__file__), "model_comparison_results")
N_SAMPLES   = 50          # images PER CLASS for evaluation (keep low for speed)
IMG_SIZE    = 224
RANDOM_SEED = 42
DPI         = 150

os.makedirs(OUT_DIR, exist_ok=True)

# ── Colour palette ────────────────────────────────────────────────────────────

MODEL_COLORS = {
    "efficientnetb0_skeleton" : "#7c3aed",
    "mobilenetv2_skeleton"    : "#a78bfa",
    "efficientnetb0_crop"     : "#0284c7",
    "mobilenetv2_crop"        : "#38bdf8",
    "efficientnetb0_fast"     : "#059669",
    "mobilenetv2_raw"         : "#f59e0b",
    "landmark_mlp"            : "#ef4444",
}

STYLE = {
    "figure.facecolor"   : "#0f1520",
    "axes.facecolor"     : "#141c2e",
    "axes.edgecolor"     : "#334155",
    "axes.labelcolor"    : "#e2e8f0",
    "text.color"         : "#e2e8f0",
    "xtick.color"        : "#94a3b8",
    "ytick.color"        : "#94a3b8",
    "grid.color"         : "#1e293b",
    "grid.linewidth"     : 0.6,
    "axes.titlesize"     : 13,
    "axes.labelsize"     : 11,
    "legend.fontsize"    : 9,
    "legend.facecolor"   : "#0f1520",
    "legend.edgecolor"   : "#334155",
}
plt.rcParams.update(STYLE)
sns.set_theme(style="dark", rc=STYLE)

# ── Label sets ────────────────────────────────────────────────────────────────

ASL_29 = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "nothing", "space"]
ASL_28 = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "space"]
ASL_26 = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

def get_labels(model_name, model):
    n = model.output_shape[-1]
    if "mlp" in model_name or "landmark" in model_name:
        enc = os.path.join(MODEL_DIR, "landmark_label_encoder.pkl")
        if os.path.isfile(enc):
            with open(enc, "rb") as f:
                le = pickle.load(f)
            return list(le.classes_)
    if n == 29: return ASL_29
    if n == 28: return ASL_28
    return ASL_26

# ── MediaPipe ─────────────────────────────────────────────────────────────────

mp_hands = mp.solutions.hands
_hands   = None

def hands():
    global _hands
    if _hands is None:
        _hands = mp_hands.Hands(
            static_image_mode=True, max_num_hands=1,
            min_detection_confidence=0.4,
        )
    return _hands

def get_landmarks(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res = hands().process(rgb)
    if not res.multi_hand_landmarks:
        return None, None
    lm   = res.multi_hand_landmarks[0]
    h, w = img_bgr.shape[:2]
    xs   = [p.x for p in lm.landmark]
    ys   = [p.y for p in lm.landmark]
    m    = 0.15
    bbox = (
        max(0, int((min(xs)-m)*w)), max(0, int((min(ys)-m)*h)),
        min(w, int((max(xs)+m)*w)), min(h, int((max(ys)+m)*h)),
    )
    return lm, bbox

# ── Preprocessing (matches training exactly) ──────────────────────────────────

def preprocess(img_bgr, model_name, model):
    mtype = (
        "skeleton" if "skeleton" in model_name else
        "landmark" if ("landmark" in model_name or "mlp" in model_name) else
        "crop"     if "crop"     in model_name else
        "raw"
    )
    H, W = (model.input_shape[1], model.input_shape[2]) if mtype != "landmark" else (None, None)

    if mtype in ("crop", "raw"):
        lm, bbox = get_landmarks(img_bgr)
        if lm is None:
            # fallback: whole image
            crop = img_bgr
        else:
            x1,y1,x2,y2 = bbox
            crop = img_bgr[y1:y2, x1:x2] if x2>x1 and y2>y1 else img_bgr
        resized = cv2.resize(crop, (W, H))
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        # raw: external /255; crop/fast: Rescaling baked in → [0-255]
        inp = rgb/255.0 if mtype=="raw" else rgb
        return np.expand_dims(inp, 0)

    if mtype == "skeleton":
        lm, _ = get_landmarks(img_bgr)
        if lm is None:
            return None
        canvas = np.ones((H, W, 3), np.uint8)*255
        CONN = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),
                (10,11),(11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),
                (18,19),(19,20),(5,9),(9,13),(13,17)]
        pts  = [(int(p.x*W), int(p.y*H)) for p in lm.landmark]
        for a,b in CONN:
            cv2.line(canvas, pts[a], pts[b], (0,0,0), 2, cv2.LINE_AA)
        for x,y in pts:
            cv2.circle(canvas, (x,y), 3, (0,0,255), -1, cv2.LINE_AA)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32)
        return np.expand_dims(rgb, 0)     # Rescaling baked in

    if mtype == "landmark":
        lm, _ = get_landmarks(img_bgr)
        if lm is None:
            return None
        coords = []
        for p in lm.landmark:
            coords.extend([p.x, p.y, p.z])
        return np.array(coords, dtype=np.float32).reshape(1, 63)

# ── Dataset loader ────────────────────────────────────────────────────────────

def load_dataset(n_per_class=N_SAMPLES):
    """
    Returns list of (img_bgr, class_str).
    Scans DATA_DIR for class sub-folders.
    Falls back to a synthetic dataset if DATA_DIR does not exist.
    """
    if not os.path.isdir(DATA_DIR):
        print(f"\n⚠  DATA_DIR not found: {DATA_DIR}")
        print("   Using synthetic random dataset for demonstration.")
        print("   Set DATA_DIR to your test image folder for real results.\n")
        rng = np.random.default_rng(RANDOM_SEED)
        data = []
        for cls in ASL_29:
            for _ in range(n_per_class):
                img = rng.integers(0, 256, (200,200,3), dtype=np.uint8)
                data.append((img, cls))
        return data

    rng   = np.random.default_rng(RANDOM_SEED)
    data  = []
    classes = sorted([d for d in os.listdir(DATA_DIR)
                      if os.path.isdir(os.path.join(DATA_DIR, d))])
    print(f"Found {len(classes)} classes in {DATA_DIR}")
    for cls in classes:
        folder = os.path.join(DATA_DIR, cls)
        files  = [f for f in os.listdir(folder)
                  if f.lower().endswith((".jpg",".jpeg",".png"))]
        chosen = rng.choice(files, min(n_per_class, len(files)), replace=False)
        for f in chosen:
            img = cv2.imread(os.path.join(folder, f))
            if img is not None:
                data.append((img, cls))
    rng.shuffle(data)
    print(f"Loaded {len(data)} images across {len(classes)} classes.")
    return data

# ── Evaluate one model ────────────────────────────────────────────────────────

def evaluate_model(model_name, model, dataset, labels):
    y_true, y_pred, y_prob, times = [], [], [], []
    n_labels = len(labels)
    label2idx = {l: i for i, l in enumerate(labels)}

    for img_bgr, cls in dataset:
        if cls not in label2idx:
            continue
        inp = preprocess(img_bgr, model_name, model)
        if inp is None:
            continue

        t0    = time.perf_counter()
        probs = model.predict(inp, verbose=0)[0]
        times.append(time.perf_counter() - t0)

        pred_idx = int(np.argmax(probs))
        y_true.append(label2idx[cls])
        y_pred.append(pred_idx)
        y_prob.append(probs)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    acc    = float(np.mean(y_true == y_pred))
    top3   = float(top_k_accuracy_score(y_true, y_prob, k=3,
                                        labels=list(range(n_labels))))
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    cm     = confusion_matrix(y_true, y_pred, labels=list(range(n_labels)))
    speed  = float(np.mean(times)) * 1000   # ms

    return dict(
        name=model_name, labels=labels,
        y_true=y_true, y_pred=y_pred, y_prob=y_prob,
        acc=acc, top3=top3, precision=p, recall=r, f1=f,
        cm=cm, speed_ms=speed,
        n_params=model.count_params(),
        size_mb=os.path.getsize(os.path.join(MODEL_DIR, model_name)) / 1e6,
    )

# ── Plotting helpers ──────────────────────────────────────────────────────────

def short(name):
    return (name.replace("efficientnetb0","EffB0")
                .replace("mobilenetv2","MNV2")
                .replace("_skeleton","_Skel")
                .replace("_crop","_Crop")
                .replace("_fast","_Fast")
                .replace("_raw","_Raw")
                .replace("landmark_mlp","LMK_MLP")
                .replace(".h5",""))

def color_of(name):
    key = name.replace(".h5","")
    return MODEL_COLORS.get(key, "#94a3b8")

def save(fig, fname):
    path = os.path.join(OUT_DIR, fname)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  ✓ Saved {fname}")
    plt.close(fig)

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Model Overview Table
# ═════════════════════════════════════════════════════════════════════════════

def fig_overview(results):
    fig, ax = plt.subplots(figsize=(14, 4))
    fig.patch.set_facecolor("#0f1520")
    ax.axis("off")

    headers = ["Model", "Type", "Input Shape", "Classes",
               "Params (M)", "Size (MB)", "Preprocessing"]
    type_map = {
        "skeleton":"Skeleton Image","crop":"Hand Crop",
        "raw":"Raw Crop","landmark":"Landmark MLP",
    }
    pre_map = {
        "skeleton":"White canvas + bones [0-255]",
        "crop"    :"MediaPipe crop [0-255] (baked rescale)",
        "raw"     :"MediaPipe crop ÷255 → [0,1]",
        "landmark":"Raw (x,y,z)×21 coords",
    }

    rows = []
    for r in results:
        n  = r["name"].replace(".h5","")
        mt = ("skeleton" if "skeleton" in n else
              "landmark" if "mlp" in n or "landmark" in n else
              "crop"     if "crop" in n else "raw")
        rows.append([
            short(r["name"]),
            type_map[mt],
            str(r.get("input_shape","(None,224,224,3)")),
            str(len(r["labels"])),
            f"{r['n_params']/1e6:.2f}",
            f"{r['size_mb']:.1f}",
            pre_map[mt],
        ])

    col_widths = [0.13, 0.12, 0.13, 0.07, 0.10, 0.09, 0.30]
    table = ax.table(
        cellText=rows, colLabels=headers,
        cellLoc="center", loc="center",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)

    for (row, col), cell in table.get_celld().items():
        cell.set_facecolor("#1e293b" if row == 0 else ("#141c2e" if row%2 else "#0f1520"))
        cell.set_text_props(color="#e2e8f0")
        cell.set_edgecolor("#334155")
        if row == 0:
            cell.set_text_props(color=color_of(results[col-1]["name"]) if col > 0 else "#00e5ff",
                                fontweight="bold")

    ax.set_title("Figure 1 — Model Overview", color="#00e5ff",
                 fontsize=14, fontweight="bold", pad=20)
    save(fig, "fig01_model_overview.png")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Accuracy & Metrics Bar Chart
# ═════════════════════════════════════════════════════════════════════════════

def fig_metrics(results):
    names   = [short(r["name"]) for r in results]
    colors  = [color_of(r["name"]) for r in results]
    metrics = ["acc", "top3", "precision", "recall", "f1"]
    labels  = ["Top-1 Acc", "Top-3 Acc", "Precision", "Recall", "F1 Score"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 6), sharey=False)
    fig.suptitle("Figure 2 — Model Metrics Comparison", color="#00e5ff",
                 fontsize=15, fontweight="bold", y=1.01)

    for ax, metric, label in zip(axes, metrics, labels):
        vals = [r[metric]*100 for r in results]
        bars = ax.bar(names, vals, color=colors, edgecolor="#334155", linewidth=0.7, width=0.6)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 110)
        ax.yaxis.set_major_formatter(PercentFormatter(100))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.axhline(y=np.mean(vals), color="#00e5ff", linestyle="--",
                   linewidth=1, alpha=0.5, label=f"avg {np.mean(vals):.1f}%")
        ax.legend(fontsize=7)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=7.5,
                    color="#e2e8f0", fontweight="bold")

    plt.tight_layout()
    save(fig, "fig02_metrics_comparison.png")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Confusion Matrices
# ═════════════════════════════════════════════════════════════════════════════

def fig_confusion(results):
    n  = len(results)
    nc = int(np.ceil(n / 2))
    fig, axes = plt.subplots(2, nc, figsize=(nc*6, 13))
    axes = axes.flatten()
    fig.suptitle("Figure 3 — Confusion Matrices (normalised)", color="#00e5ff",
                 fontsize=15, fontweight="bold")

    cmap = LinearSegmentedColormap.from_list(
        "asl", ["#0f1520", "#1e293b", "#0284c7", "#00e5ff"])

    for i, r in enumerate(results):
        ax   = axes[i]
        cm   = r["cm"].astype(float)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_n = np.where(row_sums > 0, cm / row_sums, 0)
        lbls = r["labels"]

        im = ax.imshow(cm_n, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

        tick_step = max(1, len(lbls)//10)
        ticks = list(range(0, len(lbls), tick_step))
        ax.set_xticks(ticks); ax.set_xticklabels([lbls[t] for t in ticks],
                              rotation=45, fontsize=7)
        ax.set_yticks(ticks); ax.set_yticklabels([lbls[t] for t in ticks],
                              fontsize=7)
        ax.set_title(f"{short(r['name'])}\nTop-1: {r['acc']*100:.1f}%",
                     color=color_of(r["name"]), fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")

        # Diagonal line overlay
        ax.plot([0, len(lbls)-1], [0, len(lbls)-1],
                color="#00e5ff", linewidth=0.5, alpha=0.3)

    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    save(fig, "fig03_confusion_matrices.png")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Per-Class Accuracy Heatmap
# ═════════════════════════════════════════════════════════════════════════════

def fig_per_class_heatmap(results):
    # Build shared label set (union of all models)
    all_labels = sorted(set(itertools.chain.from_iterable(r["labels"] for r in results)))
    n_models   = len(results)
    n_labels   = len(all_labels)
    lbl2i      = {l: i for i, l in enumerate(all_labels)}

    matrix = np.full((n_models, n_labels), np.nan)

    for mi, r in enumerate(results):
        lbls = r["labels"]
        for li, lbl in enumerate(lbls):
            mask = r["y_true"] == li
            if mask.sum() == 0:
                continue
            matrix[mi, lbl2i[lbl]] = float(np.mean(r["y_pred"][mask] == li))

    fig, ax = plt.subplots(figsize=(max(16, n_labels*0.55), n_models*1.0 + 2))
    fig.suptitle("Figure 4 — Per-Class Accuracy Heatmap (models × letters)",
                 color="#00e5ff", fontsize=14, fontweight="bold")

    cmap = LinearSegmentedColormap.from_list(
        "acc", ["#ef4444", "#f59e0b", "#10b981", "#00e5ff"])
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Per-class accuracy", fraction=0.02, pad=0.01)

    ax.set_xticks(range(n_labels)); ax.set_xticklabels(all_labels, fontsize=9)
    ax.set_yticks(range(n_models))
    ax.set_yticklabels([short(r["name"]) for r in results], fontsize=9)
    ax.set_xlabel("ASL Class"); ax.set_ylabel("Model")

    # Annotate cells
    for mi in range(n_models):
        for li in range(n_labels):
            v = matrix[mi, li]
            if not np.isnan(v):
                ax.text(li, mi, f"{v:.0%}", ha="center", va="center",
                        fontsize=6.5, color="white" if v < 0.6 else "#0f1520",
                        fontweight="bold")

    plt.tight_layout()
    save(fig, "fig04_perclass_heatmap.png")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Confidence Distribution (Violin)
# ═════════════════════════════════════════════════════════════════════════════

def fig_confidence_violin(results):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Figure 5 — Confidence Distribution", color="#00e5ff",
                 fontsize=14, fontweight="bold")

    for ax, correct_only, title in zip(
        axes,
        [True, False],
        ["Confidence on CORRECT predictions", "Confidence on WRONG predictions"]
    ):
        data, xlabels, clrs = [], [], []
        for r in results:
            mask  = (r["y_true"] == r["y_pred"]) if correct_only else (r["y_true"] != r["y_pred"])
            confs = r["y_prob"][mask].max(axis=1) * 100 if mask.sum() > 0 else np.array([0])
            data.append(confs)
            xlabels.append(short(r["name"]))
            clrs.append(color_of(r["name"]))

        parts = ax.violinplot(data, showmedians=True, showextrema=True)
        for i, (pc, c) in enumerate(zip(parts["bodies"], clrs)):
            pc.set_facecolor(c); pc.set_alpha(0.6); pc.set_edgecolor("white")
        for partname in ("cbars","cmins","cmaxes","cmedians"):
            parts[partname].set_color("#00e5ff"); parts[partname].set_linewidth(1.5)

        ax.set_xticks(range(1, len(xlabels)+1))
        ax.set_xticklabels(xlabels, rotation=40, ha="right", fontsize=8)
        ax.set_ylabel("Confidence (%)")
        ax.set_ylim(0, 105)
        ax.set_title(title, fontsize=11)
        ax.yaxis.set_major_formatter(PercentFormatter(100))
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save(fig, "fig05_confidence_violin.png")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Top-1 vs Top-3 Accuracy
# ═════════════════════════════════════════════════════════════════════════════

def fig_top1_top3(results):
    names  = [short(r["name"]) for r in results]
    top1   = [r["acc"]*100  for r in results]
    top3   = [r["top3"]*100 for r in results]
    gain   = [t3-t1 for t1, t3 in zip(top1, top3)]
    colors = [color_of(r["name"]) for r in results]
    x      = np.arange(len(names))
    w      = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Figure 6 — Top-1 vs Top-3 Accuracy", color="#00e5ff",
                 fontsize=14, fontweight="bold")

    # Grouped bars
    bars1 = ax1.bar(x-w/2, top1, w, color=colors, alpha=0.9,
                    label="Top-1", edgecolor="#334155")
    bars3 = ax1.bar(x+w/2, top3, w, color=colors, alpha=0.5,
                    label="Top-3", edgecolor="#334155", hatch="//")
    ax1.set_xticks(x); ax1.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
    ax1.set_ylabel("Accuracy (%)"); ax1.set_ylim(0, 115)
    ax1.yaxis.set_major_formatter(PercentFormatter(100))
    ax1.legend(); ax1.grid(axis="y", alpha=0.3)
    ax1.set_title("Top-1 vs Top-3 Accuracy", fontsize=11)
    for b, v in zip(bars1, top1):
        ax1.text(b.get_x()+b.get_width()/2, b.get_height()+1,
                 f"{v:.1f}%", ha="center", fontsize=7, color="#e2e8f0")
    for b, v in zip(bars3, top3):
        ax1.text(b.get_x()+b.get_width()/2, b.get_height()+1,
                 f"{v:.1f}%", ha="center", fontsize=7, color="#e2e8f0")

    # Gain chart
    bars = ax2.barh(names, gain, color=colors, edgecolor="#334155")
    ax2.set_xlabel("Top-3 gain over Top-1 (%)")
    ax2.set_title("Top-3 improvement over Top-1", fontsize=11)
    ax2.grid(axis="x", alpha=0.3)
    for b, v in zip(bars, gain):
        ax2.text(v+0.3, b.get_y()+b.get_height()/2,
                 f"+{v:.1f}%", va="center", fontsize=8, color="#e2e8f0")

    plt.tight_layout()
    save(fig, "fig06_top1_top3.png")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 7 — Inference Speed Benchmark
# ═════════════════════════════════════════════════════════════════════════════

def fig_speed(results):
    names   = [short(r["name"]) for r in results]
    speeds  = [r["speed_ms"] for r in results]
    colors  = [color_of(r["name"]) for r in results]
    fps_est = [1000/s for s in speeds]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Figure 7 — Inference Speed Benchmark", color="#00e5ff",
                 fontsize=14, fontweight="bold")

    bars = ax1.barh(names, speeds, color=colors, edgecolor="#334155")
    ax1.set_xlabel("Avg inference time (ms/frame)")
    ax1.set_title("Latency per frame (lower = faster)", fontsize=11)
    ax1.grid(axis="x", alpha=0.3)
    ax1.axvline(33, color="#f59e0b", linestyle="--", alpha=0.7, label="30 FPS threshold")
    ax1.legend()
    for b, v in zip(bars, speeds):
        ax1.text(v+0.3, b.get_y()+b.get_height()/2,
                 f"{v:.1f} ms", va="center", fontsize=8, color="#e2e8f0")

    bars2 = ax2.barh(names, fps_est, color=colors, edgecolor="#334155")
    ax2.set_xlabel("Theoretical max FPS")
    ax2.set_title("Throughput (frames per second)", fontsize=11)
    ax2.grid(axis="x", alpha=0.3)
    ax2.axvline(30, color="#f59e0b", linestyle="--", alpha=0.7, label="30 FPS target")
    ax2.legend()
    for b, v in zip(bars2, fps_est):
        ax2.text(v+0.3, b.get_y()+b.get_height()/2,
                 f"{v:.0f} fps", va="center", fontsize=8, color="#e2e8f0")

    plt.tight_layout()
    save(fig, "fig07_speed_benchmark.png")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 8 — Precision / Recall / F1 Radar Chart
# ═════════════════════════════════════════════════════════════════════════════

def fig_radar(results):
    categories = ["Top-1 Acc", "Top-3 Acc", "Precision", "Recall", "F1", "Speed\n(inv-norm)"]
    N = len(categories)
    angles = [n / N * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    # Normalise speed: higher is better → invert
    max_speed = max(r["speed_ms"] for r in results)

    fig = plt.figure(figsize=(14, 8))
    fig.suptitle("Figure 8 — Model Radar Chart (Precision / Recall / F1 / Speed)",
                 color="#00e5ff", fontsize=14, fontweight="bold")

    n  = len(results)
    nc = min(4, n); nr = int(np.ceil(n / nc))
    for i, r in enumerate(results):
        ax = fig.add_subplot(nr, nc, i+1, polar=True)
        speed_norm = 1 - (r["speed_ms"] / max_speed)
        vals = [r["acc"], r["top3"], r["precision"], r["recall"], r["f1"], speed_norm]
        vals += vals[:1]

        ax.plot(angles, vals, color=color_of(r["name"]), linewidth=2)
        ax.fill(angles, vals, color=color_of(r["name"]), alpha=0.25)
        ax.set_xticks(angles[:-1]); ax.set_xticklabels(categories, fontsize=7.5)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["25%","50%","75%","100%"], fontsize=6)
        ax.tick_params(colors="#94a3b8")
        ax.set_facecolor("#141c2e")
        ax.spines["polar"].set_color("#334155")
        ax.grid(color="#334155", linewidth=0.6)
        ax.set_title(short(r["name"]), color=color_of(r["name"]),
                     fontsize=8.5, fontweight="bold", pad=12)

    plt.tight_layout()
    save(fig, "fig08_radar_chart.png")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 9 — ROC Curves (best model highlighted)
# ═════════════════════════════════════════════════════════════════════════════

def fig_roc(results):
    best = max(results, key=lambda r: r["acc"])
    r    = best
    lbls = r["labels"]
    n_cls = len(lbls)

    # Binarize
    y_bin = label_binarize(r["y_true"], classes=list(range(n_cls)))
    if y_bin.shape[1] == 1:
        return  # skip binary edge case

    fig, axes = plt.subplots(4, 8, figsize=(22, 11))
    axes = axes.flatten()
    fig.suptitle(
        f"Figure 9 — ROC Curves per Class [{short(best['name'])}  "
        f"— Best Model  Top-1: {best['acc']*100:.1f}%]",
        color="#00e5ff", fontsize=13, fontweight="bold"
    )

    for i, (ax, lbl) in enumerate(zip(axes, lbls)):
        if i >= y_bin.shape[1]:
            ax.axis("off"); continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], r["y_prob"][:, i])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color_of(best["name"]), lw=1.5)
        ax.plot([0,1],[0,1], color="#334155", lw=0.8, linestyle="--")
        ax.fill_between(fpr, tpr, alpha=0.15, color=color_of(best["name"]))
        ax.set_title(f"{lbl}  AUC={roc_auc:.2f}", fontsize=8, color="#e2e8f0")
        ax.set_xlim(0,1); ax.set_ylim(0,1.02)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor("#0f1520")

    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    save(fig, "fig09_roc_curves.png")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 10 — Calibration Curves
# ═════════════════════════════════════════════════════════════════════════════

def fig_calibration(results):
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle("Figure 10 — Confidence Calibration Curves\n"
                 "(ideal model = diagonal; above = overconfident)",
                 color="#00e5ff", fontsize=13, fontweight="bold")

    ax.plot([0,1],[0,1], "w--", linewidth=1.2, label="Perfect calibration", alpha=0.5)
    ax.fill_between([0,1],[0,1],[0,0], alpha=0.05, color="white")

    n_bins = 10
    for r in results:
        probs   = r["y_prob"].max(axis=1)
        correct = (r["y_true"] == r["y_pred"]).astype(float)
        bin_edges = np.linspace(0, 1, n_bins+1)
        bin_acc, bin_conf, bin_n = [], [], []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (probs >= lo) & (probs < hi)
            if mask.sum() > 0:
                bin_acc.append(correct[mask].mean())
                bin_conf.append(probs[mask].mean())
                bin_n.append(mask.sum())
        ax.plot(bin_conf, bin_acc, "o-", color=color_of(r["name"]),
                linewidth=2, markersize=5, label=short(r["name"]))

    ax.set_xlabel("Mean predicted confidence"); ax.set_ylabel("Fraction correct")
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.legend(loc="upper left"); ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    save(fig, "fig10_calibration.png")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 11 — Hardest & Easiest Classes
# ═════════════════════════════════════════════════════════════════════════════

def fig_hard_easy(results):
    # Average per-class accuracy across all models
    all_labels = sorted(set(itertools.chain.from_iterable(r["labels"] for r in results)))
    avg_acc    = {}
    for lbl in all_labels:
        scores = []
        for r in results:
            if lbl not in r["labels"]:
                continue
            li   = r["labels"].index(lbl)
            mask = r["y_true"] == li
            if mask.sum() > 0:
                scores.append(np.mean(r["y_pred"][mask] == li))
        avg_acc[lbl] = np.mean(scores) if scores else 0.0

    sorted_lbls = sorted(avg_acc, key=avg_acc.get)
    accs        = [avg_acc[l] for l in sorted_lbls]
    colors      = ["#ef4444" if a < 0.5 else "#f59e0b" if a < 0.75 else "#10b981"
                   for a in accs]

    fig, ax = plt.subplots(figsize=(max(12, len(sorted_lbls)*0.55), 6))
    fig.suptitle("Figure 11 — Average Per-Class Accuracy Across All Models\n"
                 "(red = hardest, green = easiest)",
                 color="#00e5ff", fontsize=13, fontweight="bold")

    bars = ax.bar(sorted_lbls, [a*100 for a in accs],
                  color=colors, edgecolor="#334155", width=0.7)
    ax.set_xlabel("ASL Class"); ax.set_ylabel("Average Accuracy (%)")
    ax.set_ylim(0, 115)
    ax.axhline(np.mean(accs)*100, color="#00e5ff", linestyle="--",
               linewidth=1.2, label=f"Mean {np.mean(accs)*100:.1f}%")
    ax.yaxis.set_major_formatter(PercentFormatter(100))
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, accs):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+1,
                f"{v:.0%}", ha="center", fontsize=7.5, color="#e2e8f0")

    plt.tight_layout()
    save(fig, "fig11_hard_easy_classes.png")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 12 — Size vs Accuracy Scatter
# ═════════════════════════════════════════════════════════════════════════════

def fig_size_accuracy(results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Figure 12 — Model Size vs Accuracy Trade-off",
                 color="#00e5ff", fontsize=14, fontweight="bold")

    for ax, xkey, xlabel in zip(
        axes,
        ["size_mb", "n_params"],
        ["Model file size (MB)", "Parameter count (M)"],
    ):
        for r in results:
            x = r[xkey] / (1e6 if xkey=="n_params" else 1)
            y = r["acc"]*100
            c = color_of(r["name"])
            ax.scatter(x, y, s=200, color=c, edgecolors="white",
                       linewidths=0.8, zorder=5)
            ax.annotate(short(r["name"]), (x, y),
                        textcoords="offset points", xytext=(6, 4),
                        fontsize=7.5, color=c)
        ax.set_xlabel(xlabel); ax.set_ylabel("Top-1 Accuracy (%)")
        ax.yaxis.set_major_formatter(PercentFormatter(100))
        ax.grid(alpha=0.3); ax.set_title(xlabel, fontsize=10)

    plt.tight_layout()
    save(fig, "fig12_size_vs_accuracy.png")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 13 — Summary Leaderboard
# ═════════════════════════════════════════════════════════════════════════════

def fig_leaderboard(results):
    # Composite score: 50% accuracy + 20% F1 + 15% top3 + 15% speed_norm
    max_speed = max(r["speed_ms"] for r in results)
    scored = []
    for r in results:
        s = (0.50*r["acc"] + 0.20*r["f1"] + 0.15*r["top3"]
             + 0.15*(1 - r["speed_ms"]/max_speed))
        scored.append((r, s))
    scored.sort(key=lambda x: x[1], reverse=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle("Figure 13 — Overall Leaderboard\n"
                 "(composite: 50% Top-1 + 20% F1 + 15% Top-3 + 15% Speed)",
                 color="#00e5ff", fontsize=13, fontweight="bold")

    names  = [short(r["name"]) for r, _ in scored]
    scores = [s*100 for _, s in scored]
    colors = [color_of(r["name"]) for r, _ in scored]

    bars = ax.barh(names[::-1], scores[::-1], color=colors[::-1],
                   edgecolor="#334155", height=0.6)
    ax.set_xlabel("Composite Score (%)")
    ax.set_xlim(0, 115)
    ax.grid(axis="x", alpha=0.3)
    ax.xaxis.set_major_formatter(PercentFormatter(100))

    medals = ["🥇","🥈","🥉"] + [""] * 10
    for i, (b, v, (r, _)) in enumerate(zip(bars[::-1], scores[::-1], scored)):
        ax.text(v+0.5, b.get_y()+b.get_height()/2,
                f"{v:.1f}%  {medals[i]}", va="center", fontsize=9,
                color="#e2e8f0", fontweight="bold")

    plt.tight_layout()
    save(fig, "fig13_leaderboard.png")

# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "="*60)
    print("  ASL MODEL COMPARISON SUITE")
    print("="*60 + "\n")

    # ── Load dataset ─────────────────────────────────────────────
    dataset = load_dataset()

    # ── Load & evaluate models ───────────────────────────────────
    results = []
    model_files = [m for m in [
        "efficientnetb0_skeleton.h5",
        "mobilenetv2_skeleton.h5",
        "efficientnetb0_crop.h5",
        "mobilenetv2_crop.h5",
        "efficientnetb0_fast.h5",
        "mobilenetv2_raw.h5",
        "landmark_mlp.h5",
    ] if os.path.isfile(os.path.join(MODEL_DIR, m))]

    if not model_files:
        print(f"❌  No model files found in {MODEL_DIR}")
        sys.exit(1)

    for mfile in model_files:
        print(f"\n── Evaluating {mfile} ──")
        try:
            model = keras.models.load_model(
                os.path.join(MODEL_DIR, mfile), compile=False
            )
            labels = get_labels(mfile, model)
            r = evaluate_model(mfile, model, tqdm(dataset, desc="  Inference"), labels)
            r["input_shape"] = str(model.input_shape)
            results.append(r)
            print(f"   Top-1: {r['acc']*100:.1f}%  Top-3: {r['top3']*100:.1f}%"
                  f"  F1: {r['f1']*100:.1f}%  Speed: {r['speed_ms']:.1f}ms")
            del model
        except Exception as e:
            print(f"   ⚠  Skipped {mfile}: {e}")

    if not results:
        print("No models evaluated successfully.")
        sys.exit(1)

    # ── Generate all figures ─────────────────────────────────────
    print(f"\n── Generating {13} figures → {OUT_DIR}\n")
    fig_overview(results)
    fig_metrics(results)
    fig_confusion(results)
    fig_per_class_heatmap(results)
    fig_confidence_violin(results)
    fig_top1_top3(results)
    fig_speed(results)
    fig_radar(results)
    fig_roc(results)
    fig_calibration(results)
    fig_hard_easy(results)
    fig_size_accuracy(results)
    fig_leaderboard(results)

    # ── Print summary table ──────────────────────────────────────
    print("\n" + "="*60)
    print("  FINAL SUMMARY")
    print("="*60)
    print(f"{'Model':<30} {'Top-1':>6} {'Top-3':>6} {'F1':>6} {'Speed':>8}")
    print("-"*60)
    for r in sorted(results, key=lambda x: x["acc"], reverse=True):
        print(f"{short(r['name']):<30} "
              f"{r['acc']*100:>5.1f}% "
              f"{r['top3']*100:>5.1f}% "
              f"{r['f1']*100:>5.1f}% "
              f"{r['speed_ms']:>6.1f}ms")

    print(f"\n✅  All figures saved to: {OUT_DIR}")
    print("    Open the folder to view all 13 PNG files.\n")

if __name__ == "__main__":
    main()
