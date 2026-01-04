import os
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from pathlib import Path
from sklearn.metrics import roc_curve, auc
from matplotlib.cm import get_cmap
import pandas as pd

# --- CONFIGURATION ---
base_dir = "ijb_test_result"
head_dirs = ["facelivtv2_xs","facelivtv2_s", "facelivtv2_m", "facelivtv2_l"]  # 2head → 6head
score_filenames = ["ijbb.npy", "ijbc.npy"]
label_paths = {
    "ijbb.npy": "/home/ndr/Container/ImageDataset/face_dataset/ijb/IJBB/meta/ijbb_template_pair_label.txt",
    "ijbc.npy": "/home/ndr/Container/ImageDataset/face_dataset/ijb/IJBC/meta/ijbc_template_pair_label.txt",
}
save_path = "./plots"
os.makedirs(save_path, exist_ok=True)

# --- COLOR SAMPLER ---
def sample_colours_from_colourmap(n, cmap_name='Set2'):
    cmap = get_cmap(cmap_name)
    return [cmap(i / n) for i in range(n)]

# --- READ TEMPLATE PAIR LIST ---
def read_template_pair_list(path):
    pairs = pd.read_csv(path, sep=' ', header=None).values
    t1 = pairs[:, 0].astype(int)
    t2 = pairs[:, 1].astype(int)
    label = pairs[:, 2].astype(int)
    return t1, t2, label

# --- LOOP FOR EACH TARGET (IJBB & IJBC) ---
for score_file in score_filenames:
    target = Path(score_file).stem.upper()  # IJBB / IJBC
    label_path = label_paths[score_file]

    if not os.path.exists(label_path):
        print(f"❌ Label file not found: {label_path}")
        continue

    print(f"\n📖 Reading labels from: {label_path}")
    _, _, label = read_template_pair_list(label_path)

    # --- COLLECT SCORE FILES ---
    files = []
    for head in head_dirs:
        file_path = os.path.join(base_dir, head, score_file)
        if os.path.exists(file_path):
            files.append(file_path)
        else:
            print(f"⚠️ Missing: {file_path}")

    if not files:
        print(f"❌ No score files found for {target}, skipping.")
        continue

    # --- LOAD SCORES ---
    methods = []
    scores = []
    for file in files:
        methods.append(Path(file).parent.name)  # e.g., "2head"
        scores.append(np.load(file))

    methods = np.array(methods)
    scores = dict(zip(methods, scores))
    colours = dict(zip(methods, sample_colours_from_colourmap(methods.shape[0], 'Set2')))

    # --- ROC COMPUTATION ---
    x_labels = [10 ** -6, 10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1]
    tpr_fpr_table = PrettyTable(['Methods'] + [str(x) for x in x_labels])
    fig = plt.figure(figsize=(7, 6))

    for method in methods:
        fpr, tpr, _ = roc_curve(label, scores[method])
        roc_auc = auc(fpr, tpr)
        fpr = np.flipud(fpr)
        tpr = np.flipud(tpr)

        plt.plot(
            fpr,
            tpr,
            color=colours[method],
            lw=1.5,
            label=f"{method}"
            # label=f"{method} (AUC = {roc_auc * 100:.2f}%)"
        )

        # Add to summary table
        tpr_fpr_row = [f"{method}-{target}"]
        for x in x_labels:
            _, min_index = min(list(zip(abs(fpr - x), range(len(fpr)))))
            tpr_fpr_row.append(f"{tpr[min_index] * 100:.2f}")
        tpr_fpr_table.add_row(tpr_fpr_row)

    # --- FINAL PLOT SETTINGS ---
    plt.xlim([10 ** -6, 0.1])
    plt.ylim([0.3, 1.0])
    plt.grid(linestyle='--', linewidth=1)
    plt.xticks(x_labels, fontsize=12)
    plt.yticks(np.linspace(0.3, 1.0, 8), fontsize=12)
    plt.xscale('log')
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title(f'ROC on {target}', fontsize=16)
    plt.legend(loc="lower right")
    plt.legend(fontsize=16)
    plt.tight_layout()

    # --- SAVE PLOT ---
    output_file = os.path.join(save_path, f"{target.lower()}_roc.pdf")
    fig.savefig(output_file)
    print(f"✅ ROC curve saved to {output_file}")
    print(tpr_fpr_table)
