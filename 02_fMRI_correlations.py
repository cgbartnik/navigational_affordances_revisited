#!/usr/bin/env python3
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr, ttest_1samp


# ==========================================================
# Figures directory (next to this script)
# ==========================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "Figures")
os.makedirs(FIG_DIR, exist_ok=True)


# ==========================================================
# Stimulus orderings
# ==========================================================
EEG_LIST_ACTION_SORTED = [
    'outdoor_manmade_0147','outdoor_manmade_0148','outdoor_natural_0246',
    'outdoor_natural_0062','outdoor_natural_0160','outdoor_natural_0255',
    'outdoor_natural_0128','indoor_0156','outdoor_manmade_0173',
    'outdoor_manmade_0089','outdoor_natural_0104','outdoor_natural_0273',
    'outdoor_natural_0079','outdoor_manmade_0175','outdoor_natural_0042',
    'outdoor_natural_0198','outdoor_manmade_0131','outdoor_natural_0091',
    'outdoor_manmade_0152','outdoor_natural_0200','outdoor_manmade_0157',
    'outdoor_manmade_0155','indoor_0282','outdoor_manmade_0256',
    'outdoor_manmade_0257','outdoor_natural_0011','indoor_0066',
    'outdoor_manmade_0119','outdoor_manmade_0220','outdoor_manmade_0068',
    'outdoor_manmade_0133','outdoor_manmade_0258','outdoor_manmade_0040',
    'outdoor_natural_0132','outdoor_manmade_0064','outdoor_manmade_0032',
    'outdoor_manmade_0063','outdoor_manmade_0015','outdoor_manmade_0110',
    'outdoor_manmade_0167','outdoor_manmade_0117','outdoor_manmade_0030',
    'outdoor_natural_0207','outdoor_natural_0053','outdoor_natural_0261',
    'outdoor_natural_0097','outdoor_natural_0004','outdoor_manmade_0149',
    'outdoor_natural_0034','outdoor_manmade_0161','indoor_0033',
    'indoor_0163','indoor_0235','indoor_0100','indoor_0058','indoor_0145',
    'indoor_0271','indoor_0266','indoor_0130','outdoor_manmade_0276',
    'indoor_0025','indoor_0021','outdoor_manmade_0165','indoor_0283',
    'indoor_0136','indoor_0249','indoor_0279','indoor_0215','indoor_0221',
    'indoor_0216','indoor_0214','indoor_0080','indoor_0103','indoor_0146',
    'indoor_0055','indoor_0212','indoor_0281','outdoor_manmade_0154',
    'indoor_0270','outdoor_natural_0049','outdoor_natural_0009',
    'outdoor_natural_0010','indoor_0272','outdoor_natural_0008',
    'outdoor_natural_0052','outdoor_natural_0023','outdoor_natural_0250',
    'outdoor_natural_0050','outdoor_natural_0017','outdoor_natural_0252'
]

FMRI_STIM_ORDERING = [
    'indoor_0021','indoor_0025','indoor_0033','indoor_0055','indoor_0058',
    'indoor_0066','indoor_0080','indoor_0100','indoor_0103','indoor_0130',
    'indoor_0136','indoor_0145','indoor_0146','indoor_0156','indoor_0163',
    'indoor_0212','indoor_0214','indoor_0215','indoor_0216','indoor_0221',
    'indoor_0235','indoor_0249','indoor_0266','indoor_0270','indoor_0271',
    'indoor_0272','indoor_0279','indoor_0281','indoor_0282','indoor_0283',
    'outdoor_manmade_0015','outdoor_manmade_0030','outdoor_manmade_0032',
    'outdoor_manmade_0040','outdoor_manmade_0063','outdoor_manmade_0064',
    'outdoor_manmade_0068','outdoor_manmade_0089','outdoor_manmade_0110',
    'outdoor_manmade_0117','outdoor_manmade_0119','outdoor_manmade_0131',
    'outdoor_manmade_0133','outdoor_manmade_0147','outdoor_manmade_0148',
    'outdoor_manmade_0149','outdoor_manmade_0152','outdoor_manmade_0154',
    'outdoor_manmade_0155','outdoor_manmade_0157','outdoor_manmade_0161',
    'outdoor_manmade_0165','outdoor_manmade_0167','outdoor_manmade_0173',
    'outdoor_manmade_0175','outdoor_manmade_0220','outdoor_manmade_0256',
    'outdoor_manmade_0257','outdoor_manmade_0258','outdoor_manmade_0276',
    'outdoor_natural_0004','outdoor_natural_0008','outdoor_natural_0009',
    'outdoor_natural_0010','outdoor_natural_0011','outdoor_natural_0017',
    'outdoor_natural_0023','outdoor_natural_0034','outdoor_natural_0042',
    'outdoor_natural_0049','outdoor_natural_0050','outdoor_natural_0052',
    'outdoor_natural_0053','outdoor_natural_0062','outdoor_natural_0079',
    'outdoor_natural_0091','outdoor_natural_0097','outdoor_natural_0104',
    'outdoor_natural_0128','outdoor_natural_0132','outdoor_natural_0160',
    'outdoor_natural_0198','outdoor_natural_0200','outdoor_natural_0207',
    'outdoor_natural_0246','outdoor_natural_0250','outdoor_natural_0252',
    'outdoor_natural_0255','outdoor_natural_0261','outdoor_natural_0273'
]


# ==========================================================
# File paths (your originals, centralized)
# ==========================================================
PATH_BEHAVIOR_ACTION_RDMs = (
    "/home/clemens-uva/Github_repos/Visact_fMRI/code/VISACT_behavior/"
    "VISACT_fmri_behavior/fmri_behavior_action_rdms.npz"
)

PATH_EEG_ACTION_AVG = (
    "/home/clemens-uva/Github_repos/EEG/DATA/Behavioral_annotations/RDMs/"
    "action_average_RDM_euclidean.npy"
)

BEHAVIOR_MODEL_RDM_PATHS = {
    "Loc. Affordance(EEG)": PATH_EEG_ACTION_AVG,
    "GIST": "/home/clemens-uva/Desktop/XAI---locomotive-affordance-perception/RDMs/GIST_265_EEG_euclidean.npy",
    "Start Point": "/home/clemens-uva/Desktop/XAI---locomotive-affordance-perception/RDMs/start_point_EEG_euc.npy",
    "Goal Point": "/home/clemens-uva/Desktop/XAI---locomotive-affordance-perception/RDMs/end_point_EEG_euc.npy",
    "Mean path": "/home/clemens-uva/Desktop/XAI---locomotive-affordance-perception/RDMs/mean_path_20_20_tiles_euclidean_EEG_sorted.npy",
    "3 Bins": "/home/clemens-uva/Desktop/XAI---locomotive-affordance-perception/RDMs/three_region_EEG_euc.npy",
    "8 Bins": "/home/clemens-uva/Desktop/XAI---locomotive-affordance-perception/RDMs/angular_bins_8_EEG_euc.npy",
    "180 Bins": "/home/clemens-uva/Desktop/XAI---locomotive-affordance-perception/RDMs/angular_bins_180_EEG_euc.npy",
    "Floor": "/home/clemens-uva/Desktop/XAI---locomotive-affordance-perception/RDMs/floor_20_20_tiles_euclidean_EEG_sorted.npy",
}

ROI_ORDER = ["PPA_mean", "OPA_mean", "RSC_mean", "V1"]
ROI_FMRI_AVG_DIR = "/home/clemens-uva/Github_repos/Visact_fMRI/code/VISACT_brain_data/average"
V1_RDM_PATH = "/home/clemens-uva/Desktop/XAI---locomotive-affordance-perception/ROI_RDMs/mean_all_sub_rdm_V1_metric-correlation.npy"


# ==========================================================
# Helpers
# ==========================================================
def load_npz_arr0(path: str) -> np.ndarray:
    return np.load(path)["arr_0"]

def reorder_rdm(rdm: np.ndarray, original_order: list[str], target_order: list[str]) -> np.ndarray:
    idx = [original_order.index(s) for s in target_order]
    return rdm[np.ix_(idx, idx)]

def subset_rdm(rdm: np.ndarray, original_list: list[str], subset_list: list[str]) -> np.ndarray:
    idx = [original_list.index(s) for s in subset_list]
    return rdm[np.ix_(idx, idx)]

def clean_rdm(rdm: np.ndarray) -> np.ndarray:
    rdm = (rdm + rdm.T) / 2.0
    np.fill_diagonal(rdm, 0.0)
    return rdm

def compute_noise_ceilings(subject_rdms: np.ndarray) -> tuple[float, float]:
    n = subject_rdms.shape[0]
    group_rdm = subject_rdms.mean(axis=0)
    upper = [spearmanr(squareform(group_rdm), squareform(rdm)).correlation for rdm in subject_rdms]

    lower = []
    for i in range(n):
        loo_mean = np.mean(np.delete(subject_rdms, i, axis=0), axis=0)
        lower.append(spearmanr(squareform(loo_mean), squareform(subject_rdms[i])).correlation)

    return float(np.mean(lower)), float(np.mean(upper))


# ==========================================================
# Core analysis steps
# ==========================================================
def compute_behavior_brain_correlations(
    behavior_rdms: dict[str, np.ndarray],
    brain_rdms: np.ndarray,
    roi_name: str,
) -> pd.DataFrame:

    lower_nc, upper_nc = compute_noise_ceilings(brain_rdms)

    rows = []
    for subj, brain_rdm in enumerate(brain_rdms):
        brain_rdv = squareform(brain_rdm.round(5), checks=False)
        subj_name = f"sub_{subj+1:02d}" if roi_name != "V1" else "mean_V1"

        for beh_name, beh_rdm in behavior_rdms.items():
            beh_rdv = squareform(beh_rdm.round(5), checks=False)
            rho, p = spearmanr(brain_rdv, beh_rdv)

            rows.append({
                "roi": roi_name,
                "behavior": beh_name,
                "subject": subj_name,
                "correlation": rho,
                "p-value": p,
                "lower_nc": lower_nc,
                "higher_nc": upper_nc,
            })

    return pd.DataFrame(rows)


def test_against_zero(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (roi, beh), g in df.groupby(["roi", "behavior"]):
        vals = g["correlation"].astype(float)
        if len(vals) > 1:
            t, p = ttest_1samp(vals, 0.0)
            out.append({
                "roi": roi,
                "behavior": beh,
                "n": len(vals),
                "mean_corr": vals.mean(),
                "std_corr": vals.std(),
                "t-statistic": t,
                "p-value": p,
            })
    return pd.DataFrame(out)


def plot_roi_bars(
    corr_df: pd.DataFrame,
    sig_df: pd.DataFrame,
    title: str,
    filename: str,
):
    fontsize = 15
    bar_width = 0.10
    roi_spacing = 1.5
    sig_alpha = 0.05 / 40  # your threshold

    behaviors = corr_df["behavior"].unique()
    rois = ROI_ORDER
    n_beh = len(behaviors)

    # your color list
    behavior_colors = ['#ff2c55','#ff2c55', '#ee9b00', '#03c042', '#7e1db6',
                       '#052f77', '#7b91b8', '#0dc0d4', '#38fff5', '#42078f']

    x_base = np.arange(len(rois)) * roi_spacing
    fig, ax = plt.subplots(figsize=(6, 5))

    for j, beh in enumerate(behaviors):
        offset = (j - n_beh / 2) * bar_width + bar_width / 2
        color = behavior_colors[j % len(behavior_colors)]

        for i, roi in enumerate(rois):
            subset = corr_df[(corr_df["roi"] == roi) & (corr_df["behavior"] == beh)]
            mean_corr = subset["correlation"].mean()
            sem_corr = subset["correlation"].sem()

            ax.bar(x_base[i] + offset, mean_corr, bar_width, color=color, yerr=sem_corr)
            ax.scatter(np.repeat(x_base[i] + offset, len(subset)), subset["correlation"],
                       color="gray", alpha=0.5, s=10, zorder=1)

            sig_row = sig_df[(sig_df["roi"] == roi) & (sig_df["behavior"] == beh)]
            if not sig_row.empty and sig_row.iloc[0]["p-value"] < sig_alpha:
                ax.text(x_base[i] + offset, 0.42, "*", ha="center", va="bottom",
                        fontsize=12, fontweight="bold", color="black")

    # noise ceiling bands per ROI (assuming constant per ROI in corr_df)
    for i, roi in enumerate(rois):
        roi_rows = corr_df[corr_df["roi"] == roi]
        lower_nc = roi_rows["lower_nc"].iloc[0]
        upper_nc = roi_rows["higher_nc"].iloc[0]

        center = x_base[i]
        total_width = bar_width * n_beh
        left = center - total_width / 2
        right = center + total_width / 2

        ax.fill_between([left, right], lower_nc, upper_nc, color="gray", alpha=0.2, zorder=0)
        ax.hlines([lower_nc, upper_nc], xmin=left, xmax=right,
                  colors="grey", linestyles="dashed", lw=1.2, zorder=1)

    ax.set_xticks(x_base)
    ax.set_xticklabels(["PPA", "OPA", "MPA", "V1"], fontsize=fontsize)
    ax.set_ylabel("Spearman's Rho Correlation", fontsize=fontsize)
    ax.set_ylim(-0.25, 0.45)
    ax.tick_params(axis="y", labelsize=fontsize - 1)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.axhline(0, color="black", lw=1)
    ax.set_title(title)

    plt.tight_layout()

    png_path = os.path.join(FIG_DIR, f"{filename}.png")
    svg_path = os.path.join(FIG_DIR, f"{filename}.svg")
    plt.savefig(png_path, dpi=300, transparent=True)
    plt.savefig(svg_path, dpi=300, transparent=True)
    print(f"Saved:\n  {png_path}\n  {svg_path}")

    plt.show()


# ==========================================================
# Data loading: behavior + ROI RDMs
# ==========================================================
def build_behavior_rdms_full() -> dict[str, np.ndarray]:
    # fMRI behavioral affordance (subject-level -> mean)
    behavior_action_rdms = load_npz_arr0(PATH_BEHAVIOR_ACTION_RDMs)
    mean_action = behavior_action_rdms.mean(axis=0)

    behavior_rdms = {"Loc. Affordance(fMRI)": clean_rdm(mean_action)}

    # EEG/model rdms must be reordered EEG_order -> fMRI_order
    for name, path in BEHAVIOR_MODEL_RDM_PATHS.items():
        rdm = np.load(path)
        rdm = reorder_rdm(rdm, EEG_LIST_ACTION_SORTED, FMRI_STIM_ORDERING)
        behavior_rdms[name] = clean_rdm(rdm)

    return behavior_rdms


def load_roi_subject_rdms_full(roi_name: str) -> np.ndarray:
    if roi_name == "V1":
        # Your V1 file seems to be (subjects, N, N)
        v1 = np.load(V1_RDM_PATH)
        v1 = np.array(v1)
        return np.array([clean_rdm(v1[i]) for i in range(v1.shape[0])])

    path = os.path.join(ROI_FMRI_AVG_DIR, f"fmri_{roi_name}.npz")
    r = load_npz_arr0(path)  # (subjects, N, N)
    return np.array([clean_rdm(r[i]) for i in range(r.shape[0])])


# ==========================================================
# MAIN: run full + indoor subset
# ==========================================================
def run_analysis(behavior_rdms: dict[str, np.ndarray], subset_images: list[str] | None, tag: str) -> None:
    """
    If subset_images is None -> full analysis.
    Else -> subset both behavior and brain RDMs using subset_images, assuming ordering is EEG_LIST_ACTION_SORTED.
    """
    # Subset behavior rdms if requested
    if subset_images is not None:
        behavior_rdms = {
            name: subset_rdm(rdm, EEG_LIST_ACTION_SORTED, subset_images)
            for name, rdm in behavior_rdms.items()
        }

    all_corr = []
    for roi in ROI_ORDER:
        brain_rdms = load_roi_subject_rdms_full(roi)

        if subset_images is not None:
            brain_rdms = np.array([
                subset_rdm(brain_rdms[i], EEG_LIST_ACTION_SORTED, subset_images)
                for i in range(brain_rdms.shape[0])
            ])

        corr_df = compute_behavior_brain_correlations(behavior_rdms, brain_rdms, roi)
        all_corr.append(corr_df)

    corr_df = pd.concat(all_corr, ignore_index=True)
    sig_df = test_against_zero(corr_df)

    plot_roi_bars(
        corr_df=corr_df,
        sig_df=sig_df,
        title=f"Behavior ROI correlations ({tag})",
        filename=f"behavior_roi_bars_{tag.replace(' ', '_').lower()}",
    )


def main() -> None:
    print(f"Figures will be saved in: {FIG_DIR}")

    # Build full behavior/model RDMs (all 90 stimuli, aligned to fMRI ordering)
    behavior_rdms_full = build_behavior_rdms_full()

    # 1) FULL dataset
    run_analysis(
        behavior_rdms=behavior_rdms_full,
        subset_images=None,
        tag="full",
    )

    # 2) INDOOR subset
    indoor_subset = [img for img in EEG_LIST_ACTION_SORTED if img.startswith("indoor")]
    run_analysis(
        behavior_rdms=behavior_rdms_full,
        subset_images=indoor_subset,
        tag="indoor_subset",
    )

    # 2) OUTDOOR natural subset
    indoor_subset = [img for img in EEG_LIST_ACTION_SORTED if img.startswith("outdoor_natural")]
    run_analysis(
        behavior_rdms=behavior_rdms_full,
        subset_images=indoor_subset,
        tag="natural_subset",
    )

    # 2) OUTDOOR manmade subset
    indoor_subset = [img for img in EEG_LIST_ACTION_SORTED if img.startswith("outdoor_manmade")]
    run_analysis(
        behavior_rdms=behavior_rdms_full,
        subset_images=indoor_subset,
        tag="manmade_subset",
    )

if __name__ == "__main__":
    main()
