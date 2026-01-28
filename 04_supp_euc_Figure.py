#!/usr/bin/env python3
"""
RSA correlation + noise ceiling + plot (stand-alone)

- Loads behavioral/model RDMs, reorders to fMRI stimulus order
- Loads ROI subject RDMs, computes subject-wise Spearman correlations
- One-sample t-tests against 0 for each ROI x behavior
- Plots bars + scatter + noise ceiling bands

Dependencies: numpy, pandas, matplotlib, scipy
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr, ttest_1samp


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "Figures")
os.makedirs(FIG_DIR, exist_ok=True)

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
    'indoor_0163', 'indoor_0235', 'indoor_0100',
    'indoor_0058', 'indoor_0145', 'indoor_0271',
    'indoor_0266', 'indoor_0130', 'outdoor_manmade_0276',
    'indoor_0025', 'indoor_0021', 'outdoor_manmade_0165',
    'indoor_0283', 'indoor_0136', 'indoor_0249',
    'indoor_0279', 'indoor_0215', 'indoor_0221',
    'indoor_0216', 'indoor_0214', 'indoor_0080',
    'indoor_0103', 'indoor_0146', 'indoor_0055',
    'indoor_0212', 'indoor_0281', 'outdoor_manmade_0154',
    'indoor_0270', 'outdoor_natural_0049', 'outdoor_natural_0009',
    'outdoor_natural_0010', 'indoor_0272', 'outdoor_natural_0008',
    'outdoor_natural_0052', 'outdoor_natural_0023', 'outdoor_natural_0250',
    'outdoor_natural_0050', 'outdoor_natural_0017', 'outdoor_natural_0252'
]


# ----------------------------
# Config (edit these paths)
# ----------------------------

@dataclass(frozen=True)
class Paths:
    fmri_behavior_action_npz: str = "/home/clemens-uva/Github_repos/Visact_fMRI/code/VISACT_behavior/VISACT_fmri_behavior/fmri_behavior_action_rdms.npz"
    fmri_behavior_object_npz: str = "/home/clemens-uva/Github_repos/Visact_fMRI/code/VISACT_behavior/VISACT_fmri_behavior/fmri_behavior_object_rdms.npz"

    eeg_action_avg_rdm: str = "/home/clemens-uva/Github_repos/EEG/DATA/Behavioral_annotations/RDMs/action_average_RDM_euclidean.npy"
    gist_rdm: str = "/home/clemens-uva/Desktop/XAI---locomotive-affordance-perception/RDMs/GIST_265_EEG_euclidean.npy"
    start_point_rdm: str = "/home/clemens-uva/Desktop/XAI---locomotive-affordance-perception/RDMs/start_point_EEG_euc.npy"
    goal_point_rdm: str = "/home/clemens-uva/Desktop/XAI---locomotive-affordance-perception/RDMs/end_point_EEG_euc.npy"
    mean_path_rdm: str = "/home/clemens-uva/Desktop/XAI---locomotive-affordance-perception/RDMs/mean_path_20_20_tiles_euclidean_EEG_sorted.npy"
    three_bins_rdm: str = "/home/clemens-uva/Desktop/XAI---locomotive-affordance-perception/RDMs/three_region_EEG_euc.npy"
    eight_bins_rdm: str = "/home/clemens-uva/Desktop/XAI---locomotive-affordance-perception/RDMs/angular_bins_8_EEG_euc.npy"
    oneeighty_bins_rdm: str = "/home/clemens-uva/Desktop/XAI---locomotive-affordance-perception/RDMs/angular_bins_180_EEG_euc.npy"
    floor_rdm: str = "/home/clemens-uva/Desktop/XAI---locomotive-affordance-perception/RDMs/floor_20_20_tiles_euclidean_EEG_sorted.npy"

    roi_dir: str = "/home/clemens-uva/Desktop/XAI---locomotive-affordance-perception/ROI_RDMs"
    # expected filenames inside roi_dir:
    # mean_all_sub_rdm_PPA_metric-euclidean.npy
    # mean_all_sub_rdm_OPA_metric-euclidean.npy
    # mean_all_sub_rdm_RSC_metric-euclidean.npy
    # mean_all_sub_rdm_V1_metric-euclidean.npy


# ----------------------------
# Helpers
# ----------------------------

def _check_square_rdm(rdm: np.ndarray, name: str) -> None:
    if rdm.ndim != 2 or rdm.shape[0] != rdm.shape[1]:
        raise ValueError(f"{name} must be square (N x N). Got shape={rdm.shape}.")


def reorder_rdm(rdm: np.ndarray, original_order: List[str], target_order: List[str]) -> np.ndarray:
    """Reorder square RDM from original_order -> target_order."""
    _check_square_rdm(rdm, "RDM")
    idx = {stim: i for i, stim in enumerate(original_order)}
    try:
        index_map = [idx[stim] for stim in target_order]
    except KeyError as e:
        missing = e.args[0]
        raise KeyError(f"Stimulus '{missing}' not found in original_order.") from e
    return rdm[np.ix_(index_map, index_map)]


def load_reordered_rdm(path: str, original_order: List[str], target_order: List[str], round_decimals: int = 5) -> np.ndarray:
    rdm = np.load(path)
    _check_square_rdm(rdm, os.path.basename(path))
    return reorder_rdm(rdm.round(round_decimals), original_order, target_order)


def compute_noise_ceilings(subject_rdms: np.ndarray) -> Tuple[float, float]:
    """
    Lower/upper noise ceilings (Spearman) for a stack of subject RDMs: shape (S, N, N).
    Upper: corr(subject, group_mean)
    Lower: corr(subject, mean(others))
    """
    if subject_rdms.ndim != 3:
        raise ValueError(f"subject_rdms must be (S,N,N). Got {subject_rdms.shape}")

    n_sub = subject_rdms.shape[0]
    group_rdm = subject_rdms.mean(axis=0)

    group_vec = squareform(group_rdm, checks=False)
    subj_vecs = [squareform(subject_rdms[i], checks=False) for i in range(n_sub)]

    upper = np.array([spearmanr(group_vec, v).correlation for v in subj_vecs], dtype=float)

    lower = np.empty(n_sub, dtype=float)
    for i in range(n_sub):
        other_mean = subject_rdms[np.arange(n_sub) != i].mean(axis=0)
        lower[i] = spearmanr(squareform(other_mean, checks=False), subj_vecs[i]).correlation

    return float(np.nanmean(lower)), float(np.nanmean(upper))


def rdv_from_rdm(rdm: np.ndarray, round_decimals: int = 5) -> np.ndarray:
    _check_square_rdm(rdm, "RDM")
    return squareform(rdm.round(round_decimals), checks=False)


# ----------------------------
# Main analysis
# ----------------------------

def build_behavior_rdms(paths: Paths) -> Dict[str, np.ndarray]:
    behavior_action_rdms = np.load(paths.fmri_behavior_action_npz)["arr_0"]
    mean_action = behavior_action_rdms.mean(axis=0)

    behavior_rdms = {
        "Loc. Affordance(fMRI)": mean_action,
        "Loc. Affordance(EEG)": load_reordered_rdm(paths.eeg_action_avg_rdm, EEG_LIST_ACTION_SORTED, FMRI_STIM_ORDERING),
        "GIST": load_reordered_rdm(paths.gist_rdm, EEG_LIST_ACTION_SORTED, FMRI_STIM_ORDERING),
        "Start Point": load_reordered_rdm(paths.start_point_rdm, EEG_LIST_ACTION_SORTED, FMRI_STIM_ORDERING),
        "Goal Point": load_reordered_rdm(paths.goal_point_rdm, EEG_LIST_ACTION_SORTED, FMRI_STIM_ORDERING),
        "Mean path": load_reordered_rdm(paths.mean_path_rdm, EEG_LIST_ACTION_SORTED, FMRI_STIM_ORDERING),
        "3 Bins": load_reordered_rdm(paths.three_bins_rdm, EEG_LIST_ACTION_SORTED, FMRI_STIM_ORDERING),
        "8 Bins": load_reordered_rdm(paths.eight_bins_rdm, EEG_LIST_ACTION_SORTED, FMRI_STIM_ORDERING),
        "180 Bins": load_reordered_rdm(paths.oneeighty_bins_rdm, EEG_LIST_ACTION_SORTED, FMRI_STIM_ORDERING),
        "Floor": load_reordered_rdm(paths.floor_rdm, EEG_LIST_ACTION_SORTED, FMRI_STIM_ORDERING),
    }
    return behavior_rdms


def load_roi_subject_rdms(paths: Paths, roi_name: str) -> np.ndarray:
    # Your naming logic: PPA_mean -> PPA key, etc. V1 is special case.
    if roi_name == "V1":
        fname = "mean_all_sub_rdm_V1_metric-euclidean.npy"
    else:
        roi_key = roi_name.split("_")[0]  # "PPA_mean" -> "PPA"
        fname = f"mean_all_sub_rdm_{roi_key}_metric-euclidean.npy"
    full = os.path.join(paths.roi_dir, fname)
    arr = np.load(full)
    if arr.ndim != 3:
        raise ValueError(f"{roi_name} file must be (S,N,N). Got {arr.shape} from {full}")
    return arr


def compute_correlations(
    behavior_rdms: Dict[str, np.ndarray],
    roi_subject_rdms: Dict[str, np.ndarray],
    round_decimals: int = 5
) -> pd.DataFrame:
    behavior_rdvs = {name: rdv_from_rdm(rdm, round_decimals) for name, rdm in behavior_rdms.items()}

    rows = []
    for roi_name, subj_rdms in roi_subject_rdms.items():
        subj_rdms = subj_rdms.round(round_decimals)

        lower_nc, upper_nc = compute_noise_ceilings(subj_rdms)

        for subj in range(subj_rdms.shape[0]):
            sub_name = f"sub_{subj + 1:02d}"
            brain_rdv = squareform(subj_rdms[subj], checks=False)

            for behavior_name, behavior_rdv in behavior_rdvs.items():
                sp = spearmanr(brain_rdv, behavior_rdv)
                rows.append({
                    "behavior": behavior_name,
                    "roi": roi_name,
                    "subject": sub_name,
                    "correlation": float(sp.correlation),
                    "p_value": float(sp.pvalue),
                    "lower_nc": lower_nc,
                    "upper_nc": upper_nc,
                })

    return pd.DataFrame(rows)


def ttests_against_zero(corr_df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (roi, behavior), g in corr_df.groupby(["roi", "behavior"]):
        vals = g["correlation"].astype(float).dropna().values
        if len(vals) < 2:
            continue
        t_stat, p_val = ttest_1samp(vals, 0.0)
        out.append({
            "roi": roi,
            "behavior": behavior,
            "n": int(len(vals)),
            "mean_corr": float(np.mean(vals)),
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
        })
    return pd.DataFrame(out)


def plot_results(corr: pd.DataFrame, sig_df: pd.DataFrame) -> None:
    fontsize = 15
    scatter_color = "gray"
    scatter_alpha = 0.5
    bar_width = 0.10
    roi_spacing = 1.5

    # Your original correction: 0.05/40
    sig_alpha = 0.05 / 40

    behavior_colors = ['#ff2c55', '#ff2c55', '#ee9b00', '#03c042', '#7e1db6',
                       '#052f77', "#7b91b8", "#0dc0d4", "#38fff5", "#42078f"]

    rois = ["PPA_mean", "OPA_mean", "RSC_mean", "V1"]
    roi_labels = ["PPA", "OPA", "MPA", "V1"]

    behaviors = corr["behavior"].unique()
    n_behaviors = len(behaviors)

    x_base = np.arange(len(rois)) * roi_spacing
    fig, ax = plt.subplots(figsize=(1.8 * len(rois) + 2, 5))

    for j, behavior in enumerate(behaviors):
        offset = (j - n_behaviors / 2) * bar_width + bar_width / 2
        color = behavior_colors[j % len(behavior_colors)]

        for i, roi in enumerate(rois):
            subset = corr[(corr["roi"] == roi) & (corr["behavior"] == behavior)]
            mean_corr = subset["correlation"].mean()
            sem_corr = subset["correlation"].sem()

            ax.bar(
                x_base[i] + offset, mean_corr, bar_width,
                color=color, label=behavior if i == 0 else "",
                yerr=sem_corr
            )

            ax.scatter(
                np.repeat(x_base[i] + offset, len(subset)),
                subset["correlation"],
                color=scatter_color, alpha=scatter_alpha, s=10, zorder=1
            )

            # significance star
            sig_row = sig_df[(sig_df["roi"] == roi) & (sig_df["behavior"] == behavior)]
            if not sig_row.empty and float(sig_row.iloc[0]["p_value"]) < sig_alpha:
                ax.text(
                    x_base[i] + offset, 0.15, "*",
                    ha="center", va="bottom",
                    fontsize=fontsize + 4, fontweight="bold", color="black"
                )

    # noise ceiling bands per ROI
    for i, roi in enumerate(rois):
        roi_rows = corr[corr["roi"] == roi]
        if roi_rows.empty:
            continue
        lower_nc = float(roi_rows["lower_nc"].iloc[0])
        upper_nc = float(roi_rows["upper_nc"].iloc[0])

        center = x_base[i]
        total_width = bar_width * n_behaviors
        left = center - total_width / 2
        right = center + total_width / 2

        ax.fill_between([left, right], lower_nc, upper_nc, color="gray", alpha=0.2, zorder=0)
        ax.hlines([lower_nc, upper_nc], xmin=left, xmax=right, colors="grey",
                  linestyles="dashed", lw=1.2, zorder=1)

    ax.set_xticks(x_base)
    ax.set_xticklabels(roi_labels, fontsize=fontsize)
    ax.set_ylabel("Spearman's Rho Correlation", fontsize=fontsize)
    ax.set_ylim(-0.25, 0.45)
    ax.tick_params(axis="y", labelsize=fontsize - 1)

    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=fontsize - 2)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.axhline(0, color="black", lw=1)

    plt.tight_layout()

    out_png = os.path.join(FIG_DIR, "supp_euc_behavior_roi_bars.png")
    out_svg = os.path.join(FIG_DIR, "supp_euc_behavior_roi_bars.svg")
    plt.savefig(out_png, dpi=300, transparent=True, bbox_inches="tight")
    plt.savefig(out_svg, dpi=300, transparent=True, bbox_inches="tight")
    print(f"Saved:\n  {out_png}\n  {out_svg}")

    plt.show()


def main() -> None:
    paths = Paths()

    behavior_rdms = build_behavior_rdms(paths)

    rois = ["PPA_mean", "OPA_mean", "RSC_mean", "V1"]
    roi_subject_rdms = {roi: load_roi_subject_rdms(paths, roi) for roi in rois}

    corr = compute_correlations(behavior_rdms, roi_subject_rdms)
    sig_df = ttests_against_zero(corr)

    # If you want to save tables:
    # corr.to_csv("rsa_correlations.csv", index=False)
    # sig_df.to_csv("ttests_against_zero.csv", index=False)

    plot_results(corr, sig_df)


if __name__ == "__main__":
    main()
