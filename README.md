# Navigational Affordance Representations

Code and scripts accompanying the analyses reported in:

**Navigational Affordances Revisited: Contrasting Path, Surface, and Locomotive Action Representations in the Human Visual System**  
*Bartnik, C. G. & Groen, I. I. A.*  

---

## Overview

Successful interaction with the environment requires the visual system to transform sensory input into action-relevant representations, commonly referred to as *affordances*. While prior work has focused primarily on navigational affordances, these have been operationalized in diverse ways (e.g., path annotations, surface layout, locomotive actions), often without systematic comparison.

This repository contains code for representational similarity analyses (RSA) evaluating multiple navigational affordance operationalizations using fMRI and EEG data collected for a diverse set of 90 real-world scenes. By comparing different affordance operationalizations within a unified framework, this work aims to clarify which aspects of navigational affordances are robustly reflected in neural representations.

---

## Key Findings

- Different operationalizations capture distinct dimensions of navigational affordances.
- High-level locomotive action representations show robust correspondence with neural activity across both spatial (fMRI) and temporal (EEG) domains.
- Path- and ground-surface–based representations show weak or negative correspondence with brain responses in our diverse image set.
- These results highlight the importance of affordance operationalization and stimulus composition in shaping observed neural representations.

---


## Repository layout

- **01_Model_intercorrelation.py** Intercorrelations of navigational affordance operationalizations   
- **02_fMRI_correlations.py** — region-of-interest (ROI) RSA with all spaces for all images and indoor subset.  
- **03_EEG_correlations.py** — time resolved RSA with all spaces for all images and indoor subset.  
- **04_supp_euc_Figure.py** — Supplementary Figure.  


## Data 

Data files used in this study are publicly available on OSF: [https://osf.io/v3rcq/overview](https://osf.io/v3rcq/overview)
