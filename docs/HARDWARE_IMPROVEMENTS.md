# Hardware Improvements for the NISHAD Non-Invasive Haemoglobin Meter

> **Status:** design recommendations, evidence-based.
> **Scope:** the acquisition hardware and capture protocol — i.e. the parts of the
> system *upstream* of the training code. Software/pipeline fixes are tracked
> separately in the training-pipeline correctness PR.

## 1. Why hardware is the bottleneck

On the current paired data (lab `HB_Actual` vs device `HB_Meter`), the device's
readings **correlate only weakly with ground truth (Pearson r ≈ 0.35)** and have a
**MAE (~3.3 g/dL) larger than simply predicting the cohort mean (~1.9 g/dL)**.
Critically, every severe-anaemia case (lab Hb 5.6–6.4 g/dL) reads ≈10–12 g/dL — the
device has a "floor" and cannot detect the anaemia it is built to screen for.

This is the signature of a system whose **input signal does not carry haemoglobin
information**, which is primarily a *hardware / capture-physics* problem, not a
model-tuning problem. The published state of the art reaches RMSE ≈ 0.3–1.3 g/dL
[1, 2, 3] — but only with hardware choices the current device does not yet make.

## 2. Recommended hardware changes (in priority order)

### 2.1 Move from visible LEDs to near-infrared (NIR) wavelengths
**Current:** Red 660 nm, Orange 610 nm, Yellow 590 nm — all in the visible band.
**Problem:** melanin and other skin chromophores absorb strongly in the visible
range, so the signal is dominated by skin tone rather than blood Hb, and the device
is biased across skin colours [1, 2].
**Recommendation:** use a **dual-/multi-wavelength NIR design**. The JMIR
state-of-the-art review recommends **850 nm and 940 nm** (Hb-sensitive) plus
**~1070 nm** (plasma/water-sensitive) for a Beer-Lambert dual-wavelength ratio [1].
NIR also penetrates tissue far better (1–2 cm) than visible light [1].
HemaApp obtained **RMSE 1.26 g/dL** specifically when it added NIR + incandescent
illumination over the visible-only embodiment [3].

### 2.2 Capture linear intensity (RAW), not gamma-encoded video
**Problem:** the Beer-Lambert law (`I = I₀·e^(−εcl)`) requires **linear** light
intensity. Smartphone video is gamma/sRGB-encoded, auto-exposed,
auto-white-balanced and H.264-compressed, so a ratio of recorded pixels is **not**
proportional to the transmittance ratio — the physical model breaks before the
model ever sees the data.
**Recommendation:**
- Capture **RAW (Camera2 / DNG)** or, at minimum, **lock exposure, ISO, focus and
  white balance** for the whole recording.
- **Linearise** the response (undo gamma) before computing any ratio.
- The best PPG-based results use a **dedicated digital PPG front-end** (e.g.
  MAX30102, 18-bit ADC with on-chip ambient-light rejection) that outputs linear
  counts directly [2, 4, 7]. Consider a dedicated multi-wavelength PPG sensor
  instead of, or alongside, the camera.

### 2.3 Measure the incident intensity I₀ (reference channel)
**Problem:** transmittance is `I/I₀`, but I₀ is never measured; the pipeline divides
by a video-wide mean, which is not a per-wavelength reference.
**Recommendation:** record a **reference reading per wavelength** (no-finger or a
fixed reference target) at the start of each session so true transmittance is
defined and gain drift can be removed.

### 2.4 Seal out ambient light and fix finger geometry/pressure
**Problem:** ambient light and variable contact pressure are dominant PPG noise
sources; pressure changes the AC/DC ratio that encodes Hb.
**Recommendation:** a **light-tight finger enclosure** with a **fixed, repeatable
contact geometry and pressure** (a simple spring or pressure sensor). JMIR
explicitly recommends a fully-covered PPG device captured with minimal ambient
light [1]; the MAX30102-based reference design used a 3D-printed light-sealing
fingertip shell [2].

### 2.5 Eliminate sensor saturation
**Problem:** in the 30 s protocol the Red channel saturates at 255; a clipped pixel
encodes no transmittance and corrupts all red-channel Beer-Lambert features.
**Recommendation:** set exposure/gain so no channel clips (target well below 255);
validate per-session that max channel value stays in range.

### 2.6 Validate the optics on a tissue phantom before in-vivo use
**Recommendation:** before collecting more patient data, confirm on **liquid/solid
tissue phantoms of known Hb concentration** that `log(I_λ1/I_λ2)` tracks
concentration. If it does not on a phantom, no machine-learning model can recover
it in vivo [1]. Diffuse-reflectance/RGB comparative studies provide a template for
this optical validation [8].

## 3. Data collection improvements (to support the above)

- **Record demographics with every sample** (age, sex) — they are strong, free
  predictors and reduce error by ~14% in ablation [2]; WHO anaemia thresholds are
  sex/age-specific.
- **Recruit a wide Hb range and diverse skin tones** to avoid skin-tone bias and the
  current regression-to-the-mean behaviour [2, 5].
- **One reference standard:** prefer a lab CBC analyser over portable
  hemoglobinometers for ground truth, and log measurement time/conditions.
- **Public datasets for prototyping/pre-training** while the hardware is revised:
  - Abuzairi et al. PPG+Hb dataset (660 nm + 880 nm, 68 subjects) [7].
  - Dimauro et al. *Eyes-Defy-Anemia* conjunctival-image dataset [6].

## 4. Expected payoff

| Change | Evidence | Reported accuracy in source |
|---|---|---|
| NIR + dedicated PPG front-end | Wang et al. (HemaApp) [3] | RMSE 1.26 g/dL |
| Multi-wavelength PPG + simple features + demographics | Ni et al. [2] | RMSE 0.32 g/dL, R² 0.97 (n=68) |
| Smartphone imaging + deep learning (controlled capture) | Chen et al. [5] | clinically usable real-time Hb |

## References

1. Hasan MK, Aziz MH, Zarif MII, Hasan M, Hashem MMA, Guha S, Love RR, Ahamed S. *Noninvasive Hemoglobin Level Prediction in a Mobile Phone Environment: State of the Art Review and Recommendations.* JMIR mHealth uHealth 2021;9(4):e16806. doi:10.2196/16806
2. Ni B, Wang C, Yang Y, Ji X, Mayet AM, Pan X, Sun J, Miao X. *An approach to machine learning-based non-invasive hemoglobin estimation using multi-wavelength PPG signal features.* Front Physiol 2026;17:1637455. doi:10.3389/fphys.2026.1637455
3. Wang EJ, Li W, Hawkins D, Gernsheimer T, Norby-Slycord C, Patel SN. *HemaApp: noninvasive blood screening of hemoglobin using smartphone cameras.* Proc. ACM UbiComp 2016:593–604. doi:10.1145/2971648.2971653
4. Zhu J, Sun R, Liu H, Wang T, Cai L, Chen Z, et al. *A non-invasive hemoglobin detection device based on multispectral photoplethysmography.* Biosensors 2024;14(1):22. doi:10.3390/bios14010022
5. Chen Y, Hu X, Zhu Y, Liu X, Yi B. *Real-time non-invasive hemoglobin prediction using deep learning-enabled smartphone imaging.* BMC Med Inform Decis Mak 2024;24:187. doi:10.1186/s12911-024-02585-1
6. Dimauro G, Griseta ME, Camporeale MG, Clemente F, Guarini A, Maglietta R. *An intelligent non-invasive system for automated diagnosis of anemia exploiting a novel dataset.* Artif Intell Med 2023;136:102477. doi:10.1016/j.artmed.2022.102477
7. Abuzairi T, Vinia E, Yudhistira MA, Rizkinia M, Eriska W. *A dataset of hemoglobin blood value and photoplethysmography signal for machine learning-based non-invasive hemoglobin measurement.* Data Brief 2024;52:109823. doi:10.1016/j.dib.2023.109823
8. *Diffuse reflectance spectroscopy and RGB-imaging: a comparative study of non-invasive haemoglobin assessment.* Sci Rep 2024;14:73084. doi:10.1038/s41598-024-73084-6
9. Golap MA, Raju SMTU, Haque MR, Hashem MMA. *Hemoglobin and glucose level estimation from PPG characteristics features of fingertip video using MGGP-based model.* Biomed Signal Process Control 2021;67:102478. doi:10.1016/j.bspc.2021.102478
10. Kavsaoğlu AR, Polat K, Hariharan M. *Non-invasive prediction of hemoglobin level using machine learning techniques with the PPG signal's characteristics features.* Appl Soft Comput 2015;37:983–991.
