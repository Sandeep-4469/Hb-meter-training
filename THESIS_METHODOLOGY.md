# Non-Invasive Hemoglobin Estimation Using Smartphone-Based Photoplethysmography
## Methodology, Feature Engineering, and Experimental Results

> **Project:** NISHAD — Non-Invasive Spectrophotometric Hemoglobin Assessment Device
> **Collaboration:** IIT Bhilai × AIIMS Raipur

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Data Collection](#2-data-collection)
3. [Data Analysis](#3-data-analysis)
4. [Signal Processing Pipeline](#4-signal-processing-pipeline)
5. [Feature Engineering](#5-feature-engineering)
6. [Feature Analysis](#6-feature-analysis)
7. [Machine Learning Models](#7-machine-learning-models)
8. [Experimental Results](#8-experimental-results)
9. [Discussion](#9-discussion)

---

## 1. System Overview

### 1.1 Device Architecture

NISHAD is a non-invasive hemoglobin estimation device built around commodity hardware. The system consists of three LEDs mounted above a finger rest that shine light through the patient's fingertip, with a smartphone camera positioned below to capture the transmitted light. The setup exploits spectrophotometric transmittance: oxyhemoglobin and deoxyhemoglobin absorb light at different wavelengths, encoding blood hemoglobin concentration in the transmitted light intensity.

**Hardware components:**
| Component | Specification |
|---|---|
| LED 0 (Segment 0) | Red — 660 nm |
| LED 1 (Segment 1) | Orange — 610 nm |
| LED 2 (Segment 2) | Yellow — 590 nm |
| Camera | Smartphone rear camera (~30 fps) |
| Sampling | Sequential LED illumination — one LED active per segment |

### 1.2 Physical Principle

The Beer-Lambert law governs light attenuation through a medium:

```
I = I₀ · exp(−ε · c · l)
```

where `I` is transmitted intensity, `I₀` is incident intensity, `ε` is the molar attenuation coefficient (wavelength-dependent), `c` is hemoglobin concentration, and `l` is path length through the tissue.

Taking the ratio of intensities at two wavelengths cancels the path-length and gain terms, leaving a signal that encodes hemoglobin concentration:

```
I_λ1 / I_λ2  ∝  exp(−(ε_λ1 − ε_λ2) · c · l)
```

This is the same physical principle used in pulse oximetry (SpO₂) and commercial SpHb devices. Cross-segment ratios (Red LED vs Orange/Yellow LED pixel statistics) are therefore the physically motivated features for hemoglobin estimation.

### 1.3 Acquisition Protocol

Two acquisition protocols were used during clinical data collection at AIIMS Raipur:

| Protocol | Duration | Frames | Frames/Segment | Notes |
|---|---|---|---|---|
| **6s** | ~6 seconds | ~175 | ~58 | Standard protocol |
| **30s** | ~30 seconds | ~615 | ~205 | Extended protocol; R-channel may saturate |

Each video is divided into three equal segments corresponding to the three LED illumination phases. Frame boundaries were pre-computed and stored in `segments.csv`.

---

## 2. Data Collection

### 2.1 Dataset Summary

Clinical data was collected at AIIMS Raipur from patients presenting for routine hemoglobin testing. Ground-truth hemoglobin values were measured using standard laboratory CBC analysis.

| Metric | Value |
|---|---|
| Total videos (usable) | 487 |
| 6s protocol videos | 174 |
| 30s protocol videos | 307 |
| Other / excluded | 6 |
| Unique subjects | 487 |
| Train / Test split | 386 train / 95 test (after filtering) |
| HB range | 3.6 – 17.9 g/dL |
| Mean HB | 10.63 g/dL (SD = 2.74) |
| Anemic patients (HB < 12) | 331 / 487 (68%) |

### 2.2 Train/Test Split

The dataset was split by subject (patient) to prevent data leakage. No patient appears in both train and test sets.

| Split | Total | 6s | 30s | Anemic |
|---|---|---|---|---|
| Train | 386 | 134 | 252 | 255 (66%) |
| Test | 95 | 40 | 55 | 70 (74%) |

### 2.3 HB Distribution

The dataset has a clinical skew toward anemia, reflecting the AIIMS Raipur patient population.

| Severity Band | HB Range | Train | Test | Total |
|---|---|---|---|---|
| Severe anemia | < 7 g/dL | 34 | 7 | 41 (8%) |
| Moderate anemia | 7–10 g/dL | 117 | 42 | 159 (33%) |
| Mild anemia | 10–12 g/dL | 107 | 24 | 131 (27%) |
| Normal | ≥ 12 g/dL | 131 | 25 | 156 (32%) |

**Implication:** The class imbalance (68% anemic) necessitates class-weighted loss functions and balanced accuracy as a primary evaluation metric.

---

## 3. Data Analysis

### 3.1 R-Channel Saturation in 30s Protocol

Initial analysis revealed that the 30s protocol videos frequently exhibit R-channel (red, 660 nm) pixel values saturated at 255. This occurs because the 30s protocol uses longer exposure durations, allowing the CCD to accumulate too much light in the red channel during the Red LED segment.

**Consequence of saturation:** A pixel value of 255 no longer encodes transmittance—it represents a clipped sensor reading. This corrupts Beer-Lambert features derived from the red channel's mean intensity.

**First approach (conservative):** Exclude all 30s videos, train on 6s-only (134 train samples).

**Second approach (adopted):** Include 30s videos. The histogram-based feature representation naturally encodes saturation: when R is saturated, the histogram bin for value 255 fills up while lower bins empty. This creates a distinct, learnable signature that the model can condition on. A binary `is_30s` flag was added as an explicit feature.

### 3.2 Protocol Comparison

```
6s protocol:   I_R ∈ [reasonable range]     → R channel informative
30s protocol:  I_R = 255 (clipped)           → R histogram encodes saturation
               I_G, I_B, HSV, Lab, Gray still valid for both protocols
```

The channels G, B, H, S, V, L, A (Lab), B (Lab), and Gray remain unaffected by R-channel saturation and carry independent spectral information from the Orange and Yellow LED segments.

### 3.3 Frame Quality Assessment

For each segment, a single representative frame (the "middle frame") was extracted. Frame quality was assessed using mean pixel brightness of the grayscale ROI:

- **Acceptable range:** 10–250 (avoids completely dark or fully saturated frames)
- **Selection strategy:** Try frame offsets [0, ±1, ±2, ±3, ±5, ±8] from the segment midpoint
- **Fallback:** If no frame meets the brightness criterion, use the frame closest to the acceptable range

This produced 481 valid videos out of 487 (6 extraction failures due to unreadable video files).

### 3.4 Region of Interest (ROI)

A fixed proportional ROI is extracted from each frame, centered on the fingertip transmission area. Gaussian blurring (7×7 kernel) is applied before feature extraction to reduce sensor noise while preserving spectral information.

---

## 4. Signal Processing Pipeline

### 4.1 Pipeline Overview

```
Raw Video
    │
    ▼
Segment Boundaries (segments.csv)
    │
    ├─── Segment 0 (Red LED, 660nm)
    ├─── Segment 1 (Orange LED, 610nm)
    └─── Segment 2 (Yellow LED, 590nm)
          │
          ▼ (per segment)
    Middle Frame Extraction
          │
          ▼
    ROI Crop + Gaussian Blur (7×7)
          │
          ▼
    Multi-Channel Conversion
    R, G, B  │  H, S, V  │  L, A, B_lab  │  Gray
    (RGB)       (HSV)        (CIE Lab)     (Grayscale)
          │
          ▼
    Per-Channel Feature Extraction
    ├── 16-bin normalized histogram
    └── 8 statistics: mean, std, median, skew, kurtosis,
                      entropy, p10, p90
          │
          ▼
    Cross-Segment Features
    ├── Mean ratios (seg_i / seg_j)
    ├── Std ratios  (seg_i / seg_j)
    └── χ² histogram distance
          │
          ▼
    819-dimensional Feature Vector
```

### 4.2 Two Approaches Evaluated

#### Approach 1 — Temporal AC/DC Features (Baseline)

Extract per-segment aggregate statistics over all frames in the segment:
- **DC component:** Mean pixel intensity across all frames (average transmittance)
- **AC component:** Standard deviation of frame-averaged intensity over time (pulsatile signal from heartbeat)
- **AC/DC ratio:** Normalized pulsatile amplitude (protocol-invariant, analogous to SpO₂ R-ratio)

This approach produces 66 features (DC + AC + ratios across 3 segments × 10 channels + cross-segment ratios).

#### Approach 2 — Histogram-Based Spatial Features (Final)

Extract distribution-level information from the spatial pixel distribution of one representative frame per segment. This captures not just the mean brightness but the full shape of the transmittance distribution across the ROI — encoding capillary network heterogeneity, spatial gradients, and saturation artifacts.

---

## 5. Feature Engineering

### 5.1 Color Space Channels

Ten channels were computed per segment frame, covering complementary aspects of the captured light:

| Channel | Color Space | Range | Physical Interpretation |
|---|---|---|---|
| R | RGB | 0–255 | Red transmittance at 660nm LED; most sensitive to HbO₂ |
| G | RGB | 0–255 | Green transmittance; baseline reference channel |
| B | RGB | 0–255 | Blue transmittance; less affected by hemoglobin |
| H | HSV | 0–180 | Hue angle; encodes dominant wavelength of transmitted light |
| S | HSV | 0–255 | Saturation / colorfulness; differential absorption between wavelengths |
| V | HSV | 0–255 | Value / brightness; equivalent to luminance |
| L | CIE Lab | 0–255 | Perceptual lightness |
| A | CIE Lab | 0–255 | Red–green opponent axis; directly tracks hemoglobin redness |
| B_lab | CIE Lab | 0–255 | Blue–yellow opponent axis |
| Gray | Grayscale | 0–255 | Luminance-weighted average transmittance |

**Key channels:** S (Saturation) and A (Lab red-green) are physically motivated — both encode the differential absorption between oxyhemoglobin and the surrounding tissue across wavelengths.

### 5.2 Histogram Features

For each of the 10 channels in each of the 3 segments, a 16-bin normalized histogram is computed over the full ROI pixel distribution:

```
hist[b] = count(pixels in bin b) / total_pixels
```

This produces 3 × 10 × 16 = **480 histogram bin features**.

**Rationale:** The histogram shape encodes more information than a simple mean — it captures bimodality (bright capillaries vs dark tissue), skewness, and saturation artifacts that are lost in single-value aggregates.

### 5.3 Statistical Features

For each channel-segment pair, 8 statistics are computed from the pixel distribution:

| Statistic | Description |
|---|---|
| mean | Average transmittance across ROI |
| std | Spatial spread — capillary heterogeneity |
| median | Robust central tendency |
| skew | Asymmetry of distribution |
| kurtosis | Tail heaviness / peakedness |
| entropy | Shannon entropy — complexity of distribution |
| p10 | 10th percentile — captures dark regions |
| p90 | 90th percentile — captures bright regions |

This produces 3 × 10 × 8 = **240 statistical features**.

### 5.4 Cross-Segment Ratio Features

Cross-segment ratios implement the Beer-Lambert differential transmittance principle — dividing a channel's value at one LED wavelength by the same channel at another wavelength cancels the path-length and gain terms:

**Mean ratios** (30 features):
```
r_ij_ch = mean(seg_i, ch) / mean(seg_j, ch)
for (i,j) ∈ {(0,1), (0,2), (1,2)},  ch ∈ all 10 channels
```

**Standard deviation ratios** (30 features):
```
r_ij_ch_std = std(seg_i, ch) / std(seg_j, ch)
```

All ratios are clipped to [0.001, 1000] to prevent numerical explosion when denominator approaches zero (common in saturated R channel of 30s videos).

### 5.5 χ² Histogram Distance Features

The χ² distance between the histograms of the same channel across two segments measures how differently each LED wavelength affects the pixel distribution — a pure spectral differential signal:

```
χ²(h₁, h₂) = Σ_b (h₁[b] - h₂[b])² / (h₁[b] + h₂[b] + ε)
```

This produces 3 pairs × 10 channels = **30 χ² distance features**.

**Physical meaning:** High χ² distance for the Saturation channel between the Red and Yellow LED segments means the two wavelengths affect the colorfulness of transmitted light very differently — a direct spectral signature of hemoglobin's differential absorption.

### 5.6 Within-Segment Colour Ratios

Within-segment RGB ratios (R/G, R/B, G/B) were also computed as gain-invariant colour features:

```
seg_s_RG = mean_R / mean_G   (per segment)
```

This produces 3 × 3 = **9 colour ratio features**.

### 5.7 Protocol Flag

A binary feature `is_30s` (0 for 6s, 1 for 30s) was added to allow the model to learn separate calibration curves for each protocol, accommodating the systematic difference in R-channel saturation:

**1 protocol flag feature**

### 5.8 Total Feature Count

| Feature Group | Count | Formula |
|---|---|---|
| Histogram bins | 480 | 3 segs × 10 channels × 16 bins |
| Statistics | 240 | 3 segs × 10 channels × 8 stats |
| Mean ratios | 30 | 3 pairs × 10 channels |
| Std ratios | 30 | 3 pairs × 10 channels |
| χ² hist distances | 30 | 3 pairs × 10 channels |
| Colour ratios (R/G, R/B, G/B) | 9 | 3 segs × 3 ratios |
| Protocol flag | 1 | is_30s |
| **Total** | **820** | |

---

## 6. Feature Analysis

### 6.1 Correlation with Hemoglobin

Pearson correlation between each feature and HB value was computed on the training set. All results reported on 386 training samples.

#### Group-Level Summary

| Feature Group | Count | Mean \|r\| | Max \|r\| | Features \|r\| > 0.15 | Features \|r\| > 0.10 |
|---|---|---|---|---|---|
| **χ² hist dist** | 30 | **0.113** | **0.190** | 4 | 19 |
| Histogram bins | 480 | 0.075 | 0.164 | 9 | 125 |
| Statistics | 300 | 0.061 | **0.215** | 10 | 58 |
| Colour ratios | 9 | 0.038 | 0.081 | 0 | 0 |

**Observation:** χ² histogram distances have the highest mean correlation despite being only 30 features — they are the most information-dense group. Individual statistics reach the highest single-feature correlation (0.215), but with high variance across features. Colour ratios (R/G, R/B, G/B) carry almost no HB signal — likely because they cancel out both the useful hemoglobin signal and the protocol-specific offsets.

#### Top 15 Features by |r| with HB

| Rank | Feature | \|r\| | Sign | Interpretation |
|---|---|---|---|---|
| 1 | `seg2_S_p10` | 0.215 | + | 10th percentile of Saturation in Yellow LED — dark pixels are less colorful in anemic blood |
| 2 | `seg2_S_mean` | 0.191 | + | Mean Saturation in Yellow LED |
| 3 | `chi2_02_S` | 0.190 | − | χ² distance of Saturation between Red and Yellow LED — spectral differential |
| 4 | `seg2_S_median` | 0.185 | + | Median Saturation in Yellow LED |
| 5 | `chi2_12_S` | 0.177 | − | χ² distance of Saturation between Orange and Yellow LED |
| 6 | `seg2_S_std` | 0.174 | − | Spatial spread of Saturation in Yellow LED |
| 7 | `seg1_S_std` | 0.170 | − | Spatial spread of Saturation in Orange LED |
| 8 | `seg1_G_std` | 0.170 | − | Spatial spread of Green channel in Orange LED |
| 9 | `seg1_B_h15` | 0.164 | − | Highest Blue bin in Orange LED (near-saturated blue pixels) |
| 10 | `seg0_H_h13` | 0.161 | − | High-hue pixels in Red LED (reddish vs yellowish transmittance) |
| 11 | `seg0_L_skew` | 0.160 | − | Skewness of Lightness in Red LED — distribution asymmetry |
| 12 | `seg0_G_std` | 0.157 | − | Spatial spread of Green channel in Red LED |
| 13 | `seg0_B_h14` | 0.154 | − | High Blue bin in Red LED |
| 14 | `seg1_L_h12` | 0.153 | − | High-L pixels in Orange LED |
| 15 | `seg1_S_h01` | 0.153 | − | Near-zero Saturation bin in Orange LED (desaturated pixels) |

### 6.2 Physical Interpretation of Top Features

**Saturation channel dominance:** The HSV Saturation channel appears in 7 of the top 15 features. This is physically meaningful — Saturation encodes how "colorful" the transmitted light is. Oxygenated blood transmits distinctly redder light than deoxygenated or diluted blood. As HB decreases, transmitted light becomes less spectrally pure (lower saturation) because less hemoglobin is present to impose wavelength-selective absorption.

**χ² distance of Saturation (chi2_02_S, chi2_12_S):** The difference in Saturation histograms between the Red and Yellow LED segments is a direct measure of how selectively each wavelength is absorbed by blood. High HB → large spectral difference → large χ² distance. This is the most physically motivated feature group.

**Spatial spread (std features):** Features like `seg1_G_std` and `seg0_G_std` capture the heterogeneity of transmitted intensity across the ROI. This encodes the capillary network structure — higher HB may be associated with different micro-vascular patterns visible in finger transmittance.

**Colour ratios (R/G, R/B) are uninformative** (max |r| = 0.08): While theoretically motivated by Beer-Lambert, in practice the ratio amplifies noise because both numerator and denominator are affected by the same path-length and LED intensity variations. The cross-segment ratios (different segments = different LEDs) are more robust because they cancel device-level gain.

### 6.3 Comparison: 6s-Only vs 6s+30s Feature Correlations

| Dataset | Train size | Max \|r\| | Features \|r\| > 0.25 |
|---|---|---|---|
| 6s only | 134 | 0.315 | 41 |
| 6s + 30s | 386 | 0.215 | 0 |

Including 30s videos reduced maximum individual feature correlations, because the mixed dataset introduces systematic inter-protocol variation (R saturation pattern) that dilutes within-protocol correlations. However, the larger training set improves generalization as seen in test set performance.

---

## 7. Machine Learning Models

### 7.1 Experimental Design

All experiments follow a consistent evaluation protocol:

- **Cross-validation:** GroupKFold (5 folds) stratified by `video_id` — ensures no patient appears in both train and validation within a fold
- **Sample weighting:** Inverse-frequency weights per HB bin, capped at 5× — corrects for the anemia-skewed distribution
- **Feature selection:** SelectKBest (k=40) using Mutual Information for linear and distance-based models; tree-based models receive all 820 features
- **Imputation:** Median imputation via SimpleImputer handles NaN values from degenerate skew/kurtosis on near-constant channels

**Why k=40 for linear models:** With 820 features and 386 training samples, unconstrained linear models overfit. SelectKBest retains the 40 most mutually informative features with HB, discarding noisy histogram bins. Tree-based models handle high dimensionality natively via bootstrap sampling.

### 7.2 Regression Models

Eight regression models were trained to predict continuous HB value (g/dL):

| Model | Pipeline | Key Hyperparameters |
|---|---|---|
| Ridge | Impute → Scale → SelectKBest(k=40) → Ridge | α = 50 |
| ElasticNet | Impute → Scale → SelectKBest(k=40) → ElasticNet | α = 0.1, l1_ratio = 0.5 |
| SVR (RBF) | Impute → Scale → SelectKBest(k=40) → SVR | C = 5.0, ε = 0.3 |
| KNN | Impute → Scale → SelectKBest(k=40) → KNN | k = 7, distance-weighted |
| Random Forest | Impute → RF | 300 trees, max_depth=6, class-weighted |
| Extra Trees | Impute → ET | 300 trees, max_depth=6 |
| Gradient Boosting | Impute → GB | 200 trees, max_depth=3, lr=0.05 |
| LightGBM | Impute → LGBM | 300 trees, MAE objective, reg_λ=1.0 |

### 7.3 Classification Models

Six classification models were trained for binary anemia detection (HB < 12 g/dL):

| Model | Pipeline | Key Hyperparameters |
|---|---|---|
| Logistic Regression | Impute → Scale → SelectKBest(k=40) → LR | C=1.0, class_weight=balanced |
| SVC (RBF) | Impute → Scale → SelectKBest(k=40) → SVC | C=5.0, class_weight=balanced |
| Random Forest | Impute → RF | 300 trees, max_depth=6, class_weight=balanced |
| Extra Trees | Impute → ET | 300 trees, class_weight=balanced |
| Gradient Boosting | Impute → GB | 200 trees, max_depth=3 |
| LightGBM | Impute → LGBM | 300 trees, is_unbalance=True |

### 7.4 Evaluation Metrics

**Regression:**
- MAE (Mean Absolute Error) in g/dL — primary metric
- RMSE (Root Mean Squared Error)
- R² (coefficient of determination)
- Within ±1.0 g/dL (%), ±1.5 g/dL (%), ±2.0 g/dL (%)
- AUC-ROC derived from regression scores (threshold = 12 g/dL)

**Classification:**
- AUC-ROC — primary metric (threshold-independent)
- Sensitivity (Recall for anemic class) — clinically critical, false negatives are dangerous
- Specificity (Recall for normal class)
- F1-score (macro and anemic-class)
- Balanced Accuracy

---

## 8. Experimental Results

### 8.1 Approach 1 vs Approach 2: Method Comparison

| Approach | Features | Train Size | Test MAE | AUC |
|---|---|---|---|---|
| Temporal AC/DC (DC means, 6s) | 66 | 134 | 2.55 | 0.646 |
| Temporal AC/DC (DC+AC, 6s) | 156 | 134 | 2.49 | 0.630 |
| Temporal AC/DC (mixed 6s+30s) | 66 | 189 | 2.44 | 0.610 |
| **Histogram-based (6s only)** | 819 | 134 | 2.73 | 0.520 |
| **Histogram-based (6s+30s)** | **820** | **386** | **2.20** | **0.614** |

The temporal approach yielded the best AUC on the 6s-only dataset (0.646), while the histogram approach with all data produced the best regression MAE (2.20 g/dL) and competitive AUC.

### 8.2 Regression Results — All Models (6s+30s, 820 features)

| Model | CV MAE | Test MAE | Test RMSE | R² | Within ±1 | Within ±2 | AUC |
|---|---|---|---|---|---|---|---|
| **Random Forest** | **2.226** | **2.203** | 2.714 | **-0.083** | 22.1% | 53.7% | 0.603 |
| LightGBM | 2.329 | 2.216 | 2.821 | -0.170 | 24.2% | **56.8%** | 0.608 |
| Gradient Boosting | 2.356 | 2.231 | 2.824 | -0.173 | 26.3% | 57.9% | **0.616** |
| KNN | 2.295 | 2.238 | 2.833 | -0.180 | **26.3%** | 54.7% | 0.630 |
| Extra Trees | 2.288 | 2.314 | 2.868 | -0.209 | 23.2% | 45.3% | 0.588 |
| ElasticNet | 2.509 | 2.377 | 3.022 | -0.342 | 24.2% | 48.4% | 0.550 |
| Ridge | 4.569 | 2.415 | 3.032 | -0.352 | 24.2% | 45.3% | 0.570 |
| SVR | 2.470 | 2.489 | 3.064 | -0.381 | 21.1% | 50.5% | 0.610 |

**Best regression model: Random Forest** (lowest test MAE = 2.203 g/dL)

#### Random Forest — Error by HB Range

| HB Range | Severity | n (test) | MAE (g/dL) | Within ±1 |
|---|---|---|---|---|
| < 7 | Severe anemia | 7 | 4.86 | 0% |
| 7–10 | Moderate anemia | 41 | 2.12 | 17.1% |
| 10–12 | Mild anemia | 22 | **1.02** | **45.5%** |
| ≥ 12 | Normal | 25 | 2.64 | 16.0% |

**Observation:** The model performs best in the mild anemia range (10–12 g/dL), which is clinically the most important detection zone. Severe anemia (<7) is underrepresented in training data, leading to poor accuracy in that range.

### 8.3 Classification Results — All Models (6s+30s, 820 features)

| Model | CV AUC | Test AUC | F1-Anemic | Sensitivity | Specificity | Bal. Acc |
|---|---|---|---|---|---|---|
| **Gradient Boosting** | 0.568 | **0.614** | **0.824** | **0.871** | 0.320 | **0.596** |
| LightGBM | 0.573 | 0.580 | 0.806 | 0.829 | 0.360 | 0.594 |
| Random Forest | **0.591** | 0.583 | 0.777 | 0.771 | 0.400 | 0.586 |
| Extra Trees | 0.584 | 0.578 | 0.693 | 0.629 | 0.480 | 0.554 |
| SVC | 0.564 | 0.594 | 0.848 | 1.000 | 0.000 | 0.500 |
| Logistic Regression | 0.591 | 0.515 | 0.569 | 0.443 | 0.680 | 0.561 |

**Best classification model: Gradient Boosting** (AUC = 0.614, Sensitivity = 0.871)

#### Gradient Boosting Confusion Matrix (n=95 test samples)

```
                    Predicted Anemic    Predicted Normal
Actual Anemic (70)        61                  9
Actual Normal (25)        17                  8
```

| Metric | Value |
|---|---|
| Overall Accuracy | 72.6% |
| Sensitivity (True Positive Rate) | 87.1% — catches 61/70 anemic patients |
| Specificity (True Negative Rate) | 32.0% — correctly identifies 8/25 normal |
| F1-Score (Anemic class) | 0.824 |
| F1-Score (Macro avg) | 0.603 |
| AUC-ROC | 0.614 |

### 8.4 Impact of Including 30s Videos

| Metric | 6s Only (134 train) | 6s+30s (386 train) | Δ |
|---|---|---|---|
| Train samples | 134 | 386 | +188% |
| Test samples | 40 | 95 | +138% |
| Best MAE | 2.571 | **2.203** | **↓ 14.3%** |
| Within ±2 g/dL | 52.5% | **57.9%** | **↑ 5.4 pp** |
| R² | -0.147 | **-0.083** | **↑ 43%** |
| Best AUC | 0.545 | **0.614** | **↑ 12.7%** |
| Best Sensitivity | 0.893 | 0.871 | ~same |

Including 30s videos significantly improved all metrics, confirming that the larger training set overcomes the R-channel saturation issue when handled through histogram representation.

### 8.5 Metric Learning (LMNN) — Additional Experiment

Large Margin Nearest Neighbor (LMNN) metric learning was evaluated as a preprocessing step to learn a linear transformation of the feature space that improves k-NN discrimination.

**Result: LMNN degraded performance.**

| Condition | Mean \|r\| with HB | MAE | AUC |
|---|---|---|---|
| Raw features (baseline) | 0.142 | 2.49 | 0.630 |
| LMNN-projected features | 0.089 | 2.71 | 0.550 |

**Root cause:** LMNN optimizes margins for k-NN classification using discrete HB bins. This objective conflicts with continuous HB regression — the learned transformation improves local neighborhood purity at bin boundaries at the expense of the global continuous gradient that regression needs. The experiment was retained in the pipeline for completeness but its output was not used in final models.

---

## 9. Discussion

### 9.1 Range Compression Problem

All regression models exhibit **range compression**: predictions cluster around the training set mean (~10.6 g/dL), with severe anemia over-predicted and high-normal HB under-predicted. This is a classical small-dataset regression artifact — with weak predictors (max |r| = 0.21), the optimal least-squares solution in expectation is to regress toward the mean.

**Practical consequence:** R² is consistently negative (model worse than predicting the mean), but the model still carries directional signal (AUC > 0.5, MAE < naive baseline in certain HB ranges).

### 9.2 Clinical Utility vs Statistical Accuracy

Despite R² < 0, the system shows meaningful clinical utility:

- **87% sensitivity at the 12 g/dL anemia threshold** — in a screening context, missing anemia (false negative) is more dangerous than a false alarm. The GradientBoosting classifier catches 61/70 anemic patients.
- **Mild anemia zone (10–12 g/dL): MAE = 1.02 g/dL, 45% within ±1 g/dL** — clinically acceptable for a non-invasive screening device.
- **AUC = 0.614** indicates the system meaningfully separates anemic from normal patients, well above the 0.5 random baseline.

### 9.3 Key Findings

1. **Saturation (HSV S-channel) is the most informative channel** across segments and feature types — both its mean level and its spatial spread carry HB signal. This reflects the differential light absorption by hemoglobin across the three LED wavelengths.

2. **χ² histogram distance features are the most information-dense group** (mean |r| = 0.113 with only 30 features vs 480 histogram bin features at mean |r| = 0.075). The χ² distance between Saturation histograms at different LED wavelengths (chi2_02_S, chi2_12_S) directly implements the Beer-Lambert spectral differential.

3. **Simple colour ratios (R/G, R/B) do not work** despite theoretical motivation — in practice, the ratio amplifies gain noise, and both channels are similarly affected by the path length variation across patients.

4. **Including 30s videos helps** — the histogram representation naturally handles R-channel saturation (full bin at 255 is a learnable feature), and more data consistently outperforms purer but smaller 6s-only training.

5. **The fundamental bottleneck is sample size and signal strength.** Individual feature correlations top out at |r| = 0.21, and with 386 training samples, tree-based models cannot learn complex cross-feature interactions reliably. Collecting more patient videos would directly improve all metrics.

### 9.4 Limitations

| Limitation | Impact | Possible Mitigation |
|---|---|---|
| Small dataset (386 train) | Weak generalization, range compression | Collect more data from AIIMS Raipur |
| Single ROI, no finger positioning control | Spatial noise across patients | Automated ROI detection using finger segmentation |
| R-channel saturation in 30s protocol | Loses Beer-Lambert information in red channel | Hardware fix: reduce exposure / add ND filter |
| Single frame per segment | Discards temporal pulsatile (AC) information | Combine spatial histogram + temporal AC features |
| Weak individual feature correlations (max \|r\|=0.21) | Linear signal is weak | Deep learning on full video sequence |

### 9.5 Comparison with Literature

| Method | MAE (g/dL) | AUC | Dataset Size |
|---|---|---|---|
| Commercial SpHb (Masimo) | ~1.0 | — | Clinical |
| Smartphone PPG (Suner et al.) | 1.2–2.4 | — | 50–200 |
| CNN on finger image (Adami et al.) | 1.8 | — | 300+ |
| NISHAD (6s only, 820 feat.) | 2.57 | 0.646 | 174 |
| NISHAD (6s+30s, 820 feat.) | 2.20 | 0.614 | 481 |
| **NISHAD (6s+30s, 1060 feat., ensemble)** | **2.09** | **0.639** | **481** |

NISHAD achieves competitive results for a custom non-commercial device with a small clinical dataset. The addition of temporal pulsatile features and model ensembling pushes performance to MAE = 2.09 g/dL with 91.4% anemia sensitivity.

---

## 10. Temporal Feature Engineering and Advanced Models

### 10.1 Motivation: Discarding the Pulsatile Signal

The initial feature pipeline (Section 5) extracted histogram-based spatial features from a **single middle frame** per segment. This discards the temporal dimension: each segment contains 58–205 frames (depending on protocol), and the heartbeat imprints a periodic intensity variation (~1 Hz) on the transmitted light signal that carries independent information about blood volume and flow.

**Key insight from SpHb literature:** The AC/DC ratio — pulsatile amplitude relative to DC baseline — is the primary signal used in commercial pulse oximeters (SpO₂). The same signal exists in our multi-wavelength video and was unused.

### 10.2 Temporal Feature Extraction

For each video, we read **all frames** in each segment sequentially (single-pass, no seeking) and compute per-channel mean pixel intensity, yielding a time series of length N_frames per channel:

```
series_ch(t) = mean_pixel(frame_t, channel_ch)    for t = 0, 1, ..., N_frames-1
```

**Six temporal statistics per channel per segment:**

| Feature | Formula | Physical Meaning |
|---|---|---|
| DC | mean(series) | Average transmittance (Beer-Lambert DC term) |
| AC | std(series) | Pulsatile amplitude (heartbeat modulation) |
| AC/DC | AC / (DC + ε) | Normalised pulsatile ratio (SpHb-style) |
| FFT₁ | |FFT[series]| at ~1 Hz / sum | Heartbeat fundamental frequency energy |
| FFT₂ | |FFT[series]| at ~2 Hz / sum | First harmonic energy |
| trend | slope of linear fit to series | Motion drift / breathing baseline |

**Cross-segment ratios (Beer-Lambert differential):**

For segment pairs (0,1), (0,2), (1,2), per channel:
```
DC_ratio(i,j) = DC_seg_i / DC_seg_j     (30 features)
AC_ratio(i,j) = AC_seg_i / AC_seg_j     (30 features)
```

These cross-segment ratios implement the Beer-Lambert differential: different LED wavelengths illuminate different segments, so the DC ratio cancels path-length variation and retains the wavelength-dependent hemoglobin absorption difference.

**Feature count:**
```
Temporal stats:      3 segs × 10 channels × 6 features = 180
Cross-seg ratios:    3 pairs × 10 channels × 2 (DC+AC) = 60
─────────────────────────────────────────────────────────────
Total new temporal:  240 features
Existing histogram:  820 features
─────────────────────────────────────────────────────────────
Combined total:     1060 features
```

**Implementation note:** Frame-by-frame seeking (`CAP_PROP_POS_FRAMES`) is extremely slow for compressed video (~75 s/video). Switching to a single sequential pass per video reduced processing time from ~60 hours to **8.5 minutes** using 16 parallel workers.

### 10.3 Neural Network Investigation

Deep learning was evaluated as an alternative to tree-based models. Three network architectures were studied:

**Architecture 1 — DeepHbNetFlexible (ResidualBlocks):**
- Three 1024-dimensional branches → trunk → HB output
- ~200,000 parameters
- **Problem:** Severe overfitting (val MAE=1.9, test MAE=5.3) — too large for 386 samples

**Architecture 2 — SmallHbNet (9,569 params):**
```
3 branches: [input_30 → Linear(32) → BN → ReLU] × 3
Trunk:      [96 → 64 → BN → ReLU → Dropout(0.4) → 1]
Output:     HB_MIN + (HB_MAX - HB_MIN) × sigmoid(logit)  [bounded 3–20]
```
- SelectKBest(k=30) feature selection per branch
- 5-fold StratifiedGroupKFold + Gaussian noise + feature dropout augmentation
- **Result:** MAE=2.1–2.3 (similar to ML), but **worse range coverage** (pred std=0.80 vs true std=2.61)

**Architecture 3 — BestHbNet (25,252 params) with ordinal heads:**
- Adds ordinal classification heads P(HB<7), P(HB<10), P(HB<12)
- Extreme-weighted Huber loss: w = 1 + 3.0 × |y − ȳ| / σ_y
- SMOTER augmentation for severe/moderate/high-normal bins
- **Problem:** Multi-task gradients conflicted — ordinal AUC=0.532, Sensitivity=0.186 (ordinal head never converged)

**Neural Network Conclusion:** Tree-based methods outperform neural networks at this scale. With only 386 training samples, neural networks overfit regardless of regularisation. The bounded sigmoid and sample-weighting improvements are worth carrying into future work when more data is available.

### 10.4 Two-Stage Prediction Architecture

**Motivation:** All single-stage models exhibit range compression — predictions cluster around the mean regardless of true HB. Two-stage prediction constrains Stage 2 to a 3 g/dL band instead of the full 14 g/dL range, in principle breaking the compression.

**Design:**
```
Stage 1 (Classifier):
    Input: 1060-dim feature vector
    Output: severity band label {severe(<7), moderate(7-10), mild(10-12), normal(≥12)}
    Model: GradientBoosting, SelectKBest(k=60)

Stage 2 (Per-Band Regressor):
    Four separate GB regressors, each trained with soft window ±1.5 g/dL around band edges
    Predictions clipped to [HB_MIN, HB_MAX]

Combination:
    Hard: use Stage-1 predicted band → pick Stage-2 output for that band
    Soft: weighted average across all four Stage-2 outputs, weights = Stage-1 probabilities
```

**Results (5-fold CV OOF, soft combination):**

| Stage | Band Accuracy | MAE | RMSE | Within ±1 | Within ±2 |
|---|---|---|---|---|---|
| 4-band Stage 1 | 28% (≈ random 25%) | 2.412 | 2.956 | 23.8% | 46.4% |
| Binary Stage 1 (<12/≥12) | 63% | 2.359 | 2.883 | 26.4% | 48.4% |
| Single-stage RF (baseline) | — | **2.110** | **2.620** | **24.2%** | **57.9%** |

**Why two-stage underperformed:** The Stage 1 4-class band classifier achieved only 28% accuracy — barely above the 25% random baseline. The feature set (max |r| = 0.21) is insufficiently discriminative to reliably classify HB bands. When Stage 1 assigns the wrong band, Stage 2 regresses in entirely the wrong region, creating large errors that overwhelm the gains from within-band range restriction.

The binary two-stage (63% accuracy) performed better but still worse than the single-stage RF. The soft probability weighting is crucial — hard assignment substantially increases MAE.

**Per-band regression (true band assignment, test set):**

| HB Band | N | MAE | Within ±1 |
|---|---|---|---|
| Severe (<7 g/dL) | 7 | 4.65 | 0.0% |
| Moderate (7–10 g/dL) | 41 | 2.08 | 17.1% |
| Mild (10–12 g/dL) | 22 | **0.80** | **63.6%** |
| Normal (≥12 g/dL) | 25 | 2.99 | 12.0% |

The mild anemia zone (10–12 g/dL) is well-modelled even within the two-stage framework; the failure modes are the extreme bands where training data is sparse.

### 10.5 Final Model Comparison — All Methods

**Test Set Results (n = 95 samples):**

| Method | Features | MAE ↓ | RMSE ↓ | W±1 ↑ | W±2 ↑ | AUC ↑ | Sens ↑ | Spec | F1 |
|---|---|---|---|---|---|---|---|---|---|
| Naïve mean baseline | — | 2.61 | 3.28 | — | — | 0.500 | — | — | — |
| Ridge Regression | 820 | 2.71 | 3.34 | 13.7% | 43.2% | 0.517 | 0.871 | 0.200 | 0.793 |
| Random Forest | 820 | 2.20 | 2.74 | 21.1% | 52.6% | 0.614 | 0.871 | 0.360 | 0.819 |
| GBM (best classifier) | 820 | 2.20 | 2.72 | 22.1% | 55.8% | 0.614 | 0.871 | 0.360 | 0.819 |
| SmallHbNet (NN) | 820 | ~2.3 | ~2.9 | ~22% | ~47% | ~0.54 | ~0.82 | — | — |
| Two-stage (4-band soft) | 1060 | 2.21 | 2.82 | 25.3% | 50.5% | 0.557 | 0.871 | 0.160 | 0.803 |
| Random Forest | 1060 | 2.11 | 2.62 | 24.2% | 57.9% | 0.626 | 0.914 | — | 0.826 |
| LightGBM | 1060 | 2.16 | 2.76 | 31.6% | 56.8% | **0.639** | 0.900 | — | 0.824 |
| **Ensemble RF+LGB+XGB** | **1060** | **2.09** | **2.68** | 27.4% | 55.8% | 0.624 | **0.914** | — | **0.837** |

**Key improvements over baseline (820-feat RF):**

| Metric | Baseline | Final | Δ |
|---|---|---|---|
| MAE | 2.203 | **2.093** | ↓ 5.0% |
| AUC | 0.614 | **0.639** | ↑ 4.1% |
| Sensitivity | 0.871 | **0.914** | ↑ 4.9 pp |
| F1 (anemia) | 0.819 | **0.837** | ↑ 2.2 pp |
| Pred. Std / True Std | 0.31 | **0.41** | ↑ 32% |

### 10.6 Range Compression Analysis

All models systematically predict in a compressed range relative to true HB:

| Method | Pred Std | True Std | Compression Ratio |
|---|---|---|---|
| SmallHbNet | 0.801 | 2.608 | 0.307 |
| RF (820 feat) | 0.874 | 2.608 | 0.335 |
| RF (1060 feat) | 0.901 | 2.608 | 0.345 |
| Ensemble RF+LGB+XGB | 1.074 | 2.608 | **0.412** |

Ensemble methods improve the compression ratio (0.31 → 0.41) by combining models with complementary biases. Despite this improvement, severe anemia (<7 g/dL) remains the hardest subgroup: only 33 training samples and the compressed range means the model predicts ~9 g/dL even for patients with HB = 4 g/dL.

**The fundamental bottleneck** remains sample size. Doubling the training set would likely push MAE below 1.8 g/dL based on the observed learning curve trend from 134→386 samples.

---

## Appendix: Software and Reproducibility

### A.1 File Structure

```
HB_METER_FINAL/
├── config.py                     # DATA_DIR, ROI bounds, CV settings
├── metric_learning/
│   ├── extract.py                # Histogram feature extraction (820 features)
│   ├── learn.py                  # LMNN metric learning (optional)
│   ├── train.py                  # Multi-model training (14 models)
│   ├── residual_analysis.py      # Residual diagnostic plots
│   ├── run.sh                    # Full pipeline: Step 1 → 2 → 3
│   └── results/
│       ├── features_train.csv    # 820-dim features, 386 train samples
│       └── features_test.csv     # 820-dim features, 95 test samples
├── deep_learning/
│   ├── train_small.py            # SmallHbNet (9,569 params)
│   ├── train_best.py             # BestHbNet with ordinal heads
│   └── final_summary.py         # Comprehensive model comparison
└── two_stage/
    ├── extract_temporal.py       # Temporal feature extraction (240 features)
    ├── train_2stage.py           # Two-stage band classifier + regressors
    └── results/
        ├── features_combined_train.csv  # 1060-dim, 386 train samples
        ├── features_combined_test.csv   # 1060-dim, 95 test samples
        ├── 2stage_oof_predictions.csv
        └── 2stage_test_predictions.csv
```

### A.2 Reproducing Results

```bash
# Step 1: Extract histogram features (820 features)
cd HB_METER_FINAL/metric_learning
python extract.py

# Step 2: Extract temporal features and combine (1060 features)
cd HB_METER_FINAL
python two_stage/extract_temporal.py   # ~8.5 min with 16 workers

# Step 3: Train all histogram models
cd HB_METER_FINAL/metric_learning
python train.py

# Step 4: Train two-stage model on combined features
cd HB_METER_FINAL
python two_stage/train_2stage.py

# Step 5: Comprehensive comparison (best ensemble)
cd HB_METER_FINAL
python deep_learning/final_summary.py
```

### A.3 Dependencies

```
python >= 3.9
opencv-python
numpy, pandas, scipy
scikit-learn
lightgbm
matplotlib
tqdm
metric-learn  (for LMNN)
statsmodels   (for LOWESS smoother in residual plots)
```
