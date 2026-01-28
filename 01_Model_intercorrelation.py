#!/usr/bin/env python3
"""
Compute and plot a Spearman correlation matrix between multiple RDMs.

Usage:
    python rdm_corr_matrix.py

Notes:
- All RDMs are assumed square (N x N).
- Reordering is done from an RDM's original stimulus order to a target order.
- Correlations are computed on the upper-triangle vector (excluding diagonal).
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr


# -----------------------------
# Stimulus orderings
# -----------------------------
EEG_LIST_ACTION_SORTED = [
    'outdoor_manmade_0147', 'outdoor_manmade_0148', 'outdoor_natural_0246',
    'outdoor_natural_0062', 'outdoor_natural_0160', 'outdoor_natural_0255',
    'outdoor_natural_0128', 'indoor_0156', 'outdoor_manmade_0173',
    'outdoor_manmade_0089', 'outdoor_natural_0104', 'outdoor_natural_0273',
    'outdoor_natural_0079', 'outdoor_manmade_0175', 'outdoor_natural_0042',
    'outdoor_natural_0198', 'outdoor_manmade_0131', 'outdoor_natural_0091',
    'outdoor_manmade_0152', 'outdoor_natural_0200', 'outdoor_manmade_0157',
    'outdoor_manmade_0155', 'indoor_0282', 'outdoor_manmade_0256',
    'outdoor_manmade_0257', 'outdoor_natural_0011', 'indoor_0066',
    'outdoor_manmade_0119', 'outdoor_manmade_0220', 'outdoor_manmade_0068',
    'outdoor_manmade_0133', 'outdoor_manmade_0258', 'outdoor_manmade_0040',
    'outdoor_natural_0132', 'outdoor_manmade_0064', 'outdoor_manmade_0032',
    'outdoor_manmade_0063', 'outdoor_manmade_0015', 'outdoor_manmade_0110',
    'outdoor_manmade_0167', 'outdoor_manmade_0117', 'outdoor_manmade_0030',
    'outdoor_natural_0207', 'outdoor_natural_0053', 'outdoor_natural_0261',
    'outdoor_natural_0097', 'outdoor_natural_0004', 'outdoor_manmade_0149',
    'outdoor_natural_0034', 'outdoor_manmade_0161', 'indoor_0033',
    'indoor_0163', 'indoor_0235', 'indoor_0100', 'indoor_0058',
    'indoor_0145', 'indoor_0271', 'indoor_0266', 'indoor_0130',
    'outdoor_manmade_0276', 'indoor_0025', 'indoor_0021',
    'outdoor_manmade_0165', 'indoor_0283', 'indoor_0136', 'indoor_0249',
    'indoor_0279', 'indoor_0215', 'indoor_0221', 'indoor_0216',
    'indoor_0214', 'indoor_0080', 'indoor_0103', 'indoor_0146',
    'indoor_0055', 'indoor_0212', 'indoor_0281', 'outdoor_manmade_0154',
    'indoor_0270', 'outdoor_natural_0049', 'outdoor_natural_0009',
    'outdoor_natural_0010', 'indoor_0272', 'outdoor_natural_0008',
    'outdoor_natural_0052', 'outdoor_natural_0023', 'outdoor_natural_0250',
    'outdoor_natural_0050', 'outdoor_natural_0017', 'outdoor_natural_0252'
]

FMRI_STIM_ORDERING = [
    'indoor_0021', 'indoor_0025', 'indoor_0033', 'indoor_0055',
    'indoor_0058', 'indoor_0066', 'indoor_0080', 'indoor_0100',
    'indoor_0103', 'indoor_0130', 'indoor_0136', 'indoor_0145',
    'indoor_0146', 'indoor_0156', 'indoor_0163', 'indoor_0212',
    'indoor_0214', 'indoor_0215', 'indoor_0216', 'indoor_0221',
    'indoor_0235', 'indoor_0249', 'indoor_0266', 'indoor_0270',
    'indoor_0271', 'indoor_0272', 'indoor_0279', 'indoor_0281',
    'indoor_0282', 'indoor_0283', 'outdoor_manmade_0015',
    'outdoor_manmade_0030', 'outdoor_manmade_0032', 'outdoor_manmade_0040',
    'outdoor_manmade_0063', 'outdoor_manmade_0064', 'outdoor_manmade_0068',
    'outdoor_manmade_0089', 'outdoor_manmade_0110', 'outdoor_manmade_0117',
    'outdoor_manmade_0119', 'outdoor_manmade_0131', 'outdoor_manmade_0133',
    'outdoor_manmade_0147', 'outdoor_manmade_0148', 'outdoor_manmade_0149',
    'outdoor_manmade_0152', 'outdoor_manmade_0154', 'outdoor_manmade_0155',
    'outdoor_manmade_0157', 'outdoor_manmade_0161', 'outdoor_manmade_0165',
    'outdoor_manmade_0167', 'outdoor_manmade_0173', 'outdoor_manmade_0175',
    'outdoor_manmade_0220', 'outdoor_manmade_0256', 'outdoor_manmade_0257',
    'outdoor_manmade_0258', 'outdoor_manmade_0276', 'outdoor_natural_0004',
    'outdoor_natural_0008', 'outdoor_natural_0009', 'outdoor_natural_0010',
    'outdoor_natural_0011', 'outdoor_natural_0017', 'outdoor_natural_0023',
    'outdoor_natural_0034', 'outdoor_natural_0042', 'outdoor_natural_0049',
    'outdoor_natural_0050', 'outdoor_natural_0052', 'outdoor_natural_0053',
    'outdoor_natural_0062', 'outdoor_natural_0079', 'outdoor_natural_0091',
    'outdoor_natural_0097', 'outdoor_natural_0104', 'outdoor_natural_0128',
    'outdoor_natural_0132', 'outdoor_natural_0160', 'outdoor_natural_0198',
    'outdoor_natural_0200', 'outdoor_natural_0207', 'outdoor_natural_0246',
    'outdoor_natural_0250', 'outdoor_natural_0252', 'outdoor_natural_0255',
    'outdoor_natural_0261', 'outdoor_natural_0273'
]


# -----------------------------
# Paths (edit these if needed)
# -----------------------------
PATH_FMRI_BEHAVIOR_ACTION_NPZ = (
    "/home/clemens-uva/Github_repos/Visact_fMRI/code/VISACT_behavior/"
    "VISACT_fmri_behavior/fmri_behavior_action_rdms.npz"
)

RDM_FILES = {
    "Affordance(EEG)": "/home/clemens-uva/Github_repos/EEG/DATA/Behavioral_annotations/RDMs/action_average_RDM_euclidean.npy",
    "GIST": "/home/clemens-uva/Desktop/XAI---locomotive-affordance-perception/RDMs/GIST_265_EEG_euclidean.npy",
    "Start Point": "/home/clemens-uva/Desktop/XAI---locomotive-affordance-perception/RDMs/start_point_EEG_euc.npy",
    "Goal Point": "/home/clemens-uva/Desktop/XAI---locomotive-affordance-perception/RDMs/end_point_EEG_euc.npy",
    "Mean path": "/home/clemens-uva/Desktop/XAI---locomotive-affordance-perception/RDMs/mean_path_20_20_tiles_euclidean_EEG_sorted.npy",
    "3 Bins": "/home/clemens-uva/Desktop/XAI---locomotive-affordance-perception/RDMs/three_region_EEG_euc.npy",
    "8 Bins": "/home/clemens-uva/Desktop/XAI---locomotive-affordance-perception/RDMs/angular_bins_8_EEG_euc.npy",
    "180 Bins": "/home/clemens-uva/Desktop/XAI---locomotive-affordance-perception/RDMs/angular_bins_180_EEG_euc.npy",
    "Floor": "/home/clemens-uva/Desktop/XAI---locomotive-affordance-perception/RDMs/floor_20_20_tiles_euclidean_EEG_sorted.npy",
}


# -----------------------------
# Helpers
# -----------------------------
def load_rdm(path: str) -> np.ndarray:
    """Load an RDM from .npy or from an .npz containing arr_0."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    if path.endswith(".npz"):
        return np.load(path)["arr_0"]
    return np.load(path)


def reorder_rdm(rdm: np.ndarray, original_order: list[str], target_order: list[str]) -> np.ndarray:
    """Reorder RDM from original_order to target_order (rows/cols)."""
    idx = [original_order.index(stim) for stim in target_order]
    return rdm[np.ix_(idx, idx)]


def clean_rdm(rdm: np.ndarray) -> np.ndarray:
    """Force symmetry and zero diagonal."""
    rdm = (rdm + rdm.T) / 2.0
    np.fill_diagonal(rdm, 0.0)
    return rdm


def upper_triangle_vector(rdm: np.ndarray) -> np.ndarray:
    """Return condensed upper-triangle vector (excluding diagonal)."""
    return squareform(rdm, checks=False)


def spearman_corr_matrix(vectors: list[np.ndarray]) -> np.ndarray:
    """Compute Spearman correlation matrix across a list of vectors."""
    n = len(vectors)
    out = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            out[i, j] = spearmanr(vectors[i], vectors[j]).correlation
    return out


def plot_corr_matrix(
    corr: np.ndarray,
    labels: list[str],
    figsize=(10, 8),
    filename: str = "Supplementary_crosscorrelation"
) -> None:
    """Plot and save correlation matrix in ./Figures/"""

    n = corr.shape[0]

    # Determine save directory relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.path.join(script_dir, "Figures")
    os.makedirs(fig_dir, exist_ok=True)

    # 1. Calculate the maximum absolute value for a symmetric range
    masked = corr[~np.eye(n, dtype=bool)]
    abs_max = float(np.max(np.abs(masked)))
    
    # 2. Set the symmetrical limits
    vmax_symm = abs_max
    vmin_symm = -abs_max
    
    # 3. Use the symmetrical limits for the colormap normalization
    # TwoSlopeNorm is not strictly necessary anymore since vmin and vmax are symmetric around vcenter=0
    # but it works perfectly fine and is explicit about the center point.
    norm = mcolors.TwoSlopeNorm(vmin=vmin_symm, vcenter=0.0, vmax=vmax_symm)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corr, cmap="coolwarm", norm=norm)

    # Annotations and black diagonal
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.add_patch(
                    plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color="black", linewidth=0)
                )
            else:
                ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", color="black")

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=15)
    ax.set_yticklabels(labels, fontsize=15)

    # White grid lines
    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Spearman's Rho", rotation=90, labelpad=5, fontsize=20)
    cbar.ax.tick_params(labelsize=15)

    plt.tight_layout()

    # Save figures
    png_path = os.path.join(fig_dir, f"{filename}.png")
    svg_path = os.path.join(fig_dir, f"{filename}.svg")

    plt.savefig(png_path, dpi=300, transparent=True)
    plt.savefig(svg_path, dpi=300, transparent=True)

    print(f"Figure saved to:\n  {png_path}\n  {svg_path}")

    plt.show()

# -----------------------------
# Main
# -----------------------------
def main() -> None:
    # Load fMRI behavioral action RDMs and average across subjects
    fmri_action_rdms = load_rdm(PATH_FMRI_BEHAVIOR_ACTION_NPZ)  # (subj, N, N) presumably
    mean_action = np.mean(fmri_action_rdms, axis=0)

    behavior_rdms: dict[str, np.ndarray] = {
        "Affordance(fMRI)": mean_action,
    }

    # Load + reorder all EEG/model RDMs from EEG ordering -> fMRI ordering
    for name, path in RDM_FILES.items():
        rdm = load_rdm(path)
        rdm = reorder_rdm(rdm, EEG_LIST_ACTION_SORTED, FMRI_STIM_ORDERING)
        behavior_rdms[name] = rdm

    # Clean (symmetrize + zero diagonal)
    for k in list(behavior_rdms.keys()):
        behavior_rdms[k] = clean_rdm(behavior_rdms[k])

    # Vectorize and correlate
    labels = list(behavior_rdms.keys())
    vectors = [upper_triangle_vector(behavior_rdms[name]) for name in labels]
    corr = spearman_corr_matrix(vectors)

    # Plot
    plot_corr_matrix(corr, labels)


if __name__ == "__main__":
    main()
