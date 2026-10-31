# DS-Research-Paper

**Do tongue and nail-bed images add anything beyond the conjunctiva for smartphone anemia screening?**
A re-examination of modality contribution and redundancy on the SEWA Rural multimodal anemia dataset.

Working title: *Site-specific deep learning for smartphone anemia screening: re-examining whether tongue and nail-bed images add value beyond the conjunctiva.*

## Background

The SEWA Rural / Google study ([Desai et al., 2026](#base-paper)) collected conjunctiva, tongue, nail-bed (open and closed hand) photos and fingertip PPG from ~5,900 patients in rural Gujarat, with venous Hb from a Sysmex XN-550 analyser as ground truth. Their final model uses **only the conjunctiva** plus age, sex and pregnancy status, because in an ablation (their Appendix 8) "no single or combination of image modalities provided lower MAE than the baseline model", and adding tongue and nails "did not improve performance".

That conclusion rests on one appendix table of MAE values, with no confidence intervals or significance tests, a single fusion method (feature concatenation), and a preprocessing pipeline designed for the conjunctiva (white-paper crop, CIELAB, small ResNet from scratch). "Did not help" has not yet been shown to mean "redundant". This project tests that claim properly.

## Research questions

1. Which single site or combination of sites (conjunctiva, tongue, nail bed) gives the best Hb estimation and anemia classification?
2. When a site does not add value, why? Is it uninformative, truly redundant with the conjunctiva, failed by the model/fusion, or hidden by poor image quality?
3. Do site-specific models (preprocessing and training tailored to each body site) change the answer compared with a single conjunctiva-tuned pipeline?
4. Does the answer differ by subgroup: children vs adults, pregnant women, anemia severity, lighting, phone model?

## Novelties

### 1. Best single site or combination
Train and evaluate all 7 subsets of the three sites, keeping everything else identical (architecture family, training budget, hyperparameter search, patient folds):

`C`, `T`, `N`, `C+T`, `C+N`, `T+N`, `C+T+N` (C = conjunctiva, T = tongue, N = nail bed with open + closed views treated as one site)

Unlike the base paper, report confidence intervals and significance tests for each comparison, plus subgroup results.

### 2. Why: diagnosing the result
For every site that does not add value, decide which explanation holds:

| Explanation | Meaning | Test |
|---|---|---|
| Uninformative | The site carries little Hb signal | Single-site model performs poorly |
| Truly redundant | Signal exists but duplicates the conjunctiva | Single-site models work but **miss the same patients** (error overlap); low Shapley contribution |
| Model / fusion failed | Information exists but the model can't use it | Result changes with site-specific models (novelty 3) or a different fusion method (late averaging vs concatenation vs attention) |
| Bad images | Henna, blur, glare, cold fingers hide the signal | Re-run on good-quality images only |

### 3. Site-specific models
Instead of pushing every site through the conjunctiva pipeline, build a pipeline tuned to each site:

- **Conjunctiva:** reimplementation of the base paper (white-paper crop, 256×256, CIELAB, 2-stage ResNet from scratch, metadata + MLP, Huber loss with inverse-bin reweighting, validation bias correction).
- **Tongue:** tongue segmentation/crop, colour-space choice, saliva glare handling.
- **Nail bed:** nail-bed region detection, combining open- and closed-hand views, henna/glare handling.

To keep the comparison fair, each site gets the same parameter budget, training budget and hyperparameter search; only the site-specific parts (ROI, colour handling, augmentation) differ.

**Execution order:** build the site-specific models (3) first, then run the subset comparison (1) with both the shared pipeline and the site-specific pipelines, then the diagnosis (2). Comparing sites only through a conjunctiva-tuned pipeline would bias the ranking before the analysis starts.

## Evaluation

- **Splits:** 5-fold cross-validation **by patient**, never by image, with one fixed fold assignment for every experiment. If the dataset marks the base paper's field-representative validation group (n=516), hold it out as an external test.
- **Hb estimation:** MAE (g/dL) with bootstrap 95% CI, share within ±1 g/dL, Bland–Altman plot vs lab Hb.
- **Classification** (WHO age-, sex- and pregnancy-specific cutoffs applied to predicted Hb): AUC for any anemia, moderate-or-severe, and severe; sensitivity at fixed specificity.
- **Added value:** paired differences (e.g. `C+N` vs `C`) with DeLong tests / bootstrap CIs; leave-one-site-out from the full model; per-site Shapley values.
- **Redundancy:** overlap of misclassified patients between single-site models.
- **Reference points:** HemoCue point-of-care Hb and clinician examination, as in the base paper.
- **Multiple comparisons:** main comparisons fixed in advance; correction applied for the rest.

Compare against our own retrained copy of the base model on our folds, not against the base paper's published numbers, unless their exact split is obtained.

## Scope

- **PPG is excluded.** It is ~110 MB per patient (almost all of the 650 GB dataset), and the base paper found it gave comparable or lower accuracy. Listed as future work.
- Children's anemia labels must use WHO age-specific cutoffs computed from `cbc_hgb_g_dl`, not the dataset's `hemoglobin_category` (fixed adult cutoffs).

## Data

Dataset: [`sewa-rural-care/anemia-survey-dataset`](https://huggingface.co/datasets/sewa-rural-care/anemia-survey-dataset) on Hugging Face. Gated access; **CC BY-NC 4.0** plus a data use policy (non-commercial, no re-identification, no redistribution of raw files). Never commit anything under `data/`.

| Mode | Per patient | All ~5,900 patients |
|---|---|---|
| `tabular` | ~0.08 MB | ~0.5 GB |
| `images` (4 photos + tabular) | ~0.84 MB | ~5 GB |
| `full` (+ PPG frames) | ~110 MB | ~650 GB |

```bash
python fetch_subset.py --mode images --budget-gb 6   # everything except PPG
```

```python
from load_subset import load_subset
ds = load_subset("images")
```

## Relevant SEWA Rural papers

<a id="base-paper"></a>
**Base paper.** Desai S, Gadgil S, Gottala VA, Jesalpura K, Ramnath D, Gruzewska E, Purandare H, Liu Y, Telang U, Taylor JA, Sawhney R, Tiwari R, Webster DR, Shetty S, Hassidim A, Matias Y, Howell M, Venkatraman S, Daryani B, Chen C, Kakarmath S, Daswani M, Wilson M, Pullakhandam R, Vispute S, Thompson M. *Smartphone-based Non-Invasive Classification of Anemia Severity using Deep Learning: A Cross-Sectional Study in Rural India.* medRxiv preprint, posted 26 Oct 2026. https://doi.org/10.64898/2026.10.22.26366247
- Full text (model in *Methods → Algorithm development*): https://www.medrxiv.org/content/10.64898/2026.10.22.26366247v1.full
- Supplementary appendix (layer details in Appendices 2, 3, 5; modality ablation in Appendix 8): https://www.medrxiv.org/content/10.64898/2026.10.22.26366247v1.supplementary-material
- Key results (test n=777): Hb MAE 1.12 g/dL; any anemia sensitivity/specificity 70%/70% (AUC 0.80); severe anemia 90%/80% (AUC 0.93); HemoCue capillary MAE 0.52 g/dL.

**Study protocol.** *Screening for anemia using multi-modal machine learning models on smartphones: protocol for a comparative accuracy study in rural India.* medRxiv, April 2025. https://doi.org/10.1101/2025.04.10.25325591 (ethics SR/IEC/2023/03/01; CTRI/2023/07/055313)

**Dataset.** SEWA Rural. *Anemia Detection — Multi-Modal Clinical SEWA Rural Dataset*, v1.0, Hugging Face, 2026. https://huggingface.co/datasets/sewa-rural-care/anemia-survey-dataset

## Related work

- **BPANet.** Lin ET, Lu SC, Liu AS, Ko CH, Huang CH, Tsai CL, Fu LC. *Deep Learning-Based Model for Non-invasive Hemoglobin Estimation via Body Parts Images: A Retrospective Analysis and a Prospective Emergency Department Study.* J Imaging Inform Med. 2025;38(2):775–792. https://doi.org/10.1007/s10278-024-01209-4. One shared ResNet-50 with a body-part tag that takes one image at a time (conjunctiva, palm or nail); it does not fuse several sites per patient because no paired dataset existed.
- **Conjunctiva + nail late fusion.** Bhuvan Gowda DL, Supreetha Gowda HD. *Non-Invasive Anemia Detection Using Conjunctiva and Nailbed Image Analysis with a Fusion Convolutional Neural Network.* IJLRP. 2026;7(7). https://doi.org/10.70528/IJLRP.v7.i7.2292. Averaging two CNNs gave 82.5% accuracy vs 84.5% for conjunctiva alone; no patient-level split or Hb reference, so weak evidence. Useful only as motivation.
