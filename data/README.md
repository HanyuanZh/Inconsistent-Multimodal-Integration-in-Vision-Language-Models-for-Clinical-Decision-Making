# Data

The three datasets are not redistributed here; obtain them from their providers
under their terms of use. `--data-root` must contain:

```
<data-root>/
  prostate/
    t2/<patient_id>_<study_id>.png
    dwi/<patient_id>_<study_id>.png
    adc/<patient_id>_<study_id>.png
    marksheet.csv
  skin/
    images/...
    meta/meta.csv
  amd/
    EyePaired_AMD_448/
      manifest.csv
      AMD/<eye_id>/{cfp,oct}/*.jpg
      Normal/<eye_id>/{cfp,oct}/*.jpg
```

## Prostate — PI-CAI

Public training data and marksheet of the PI-CAI challenge
(<https://pi-cai.grand-challenge.org/>; Saha et al., *Lancet Oncol* 2024): 1,500
biparametric MRI examinations from 1,476 patients.

* One 256 × 256 grayscale PNG per examination and sequence (T2W, high b-value DWI,
  ADC), named `<patient_id>_<study_id>.png`, in `t2/`, `dwi/` and `adc/`.
* `marksheet.csv` is the PI-CAI marksheet; the columns used are `patient_id`,
  `study_id`, `psa`, `prostate_volume` and `case_csPCa` (reference label, `YES`/`NO`).

> **To be added:** the script that extracts the 2-D PNGs from the PI-CAI volumes.

## Skin — Derm7pt

Seven-point checklist dataset (<https://derm.cs.sfu.ca/>; Kawahara et al., *IEEE JBHI*
2019): 1,011 lesions, each with a clinical and a dermoscopic image. Use the release
as distributed: `images/` and `meta/meta.csv`. A lesion is positive when
`management == "excision"`.

## AMD — MMC-AMD, eye-paired subset

MMC-AMD (Wang et al., MICCAI 2019; *IEEE JBHI* 2022). The study used the eyes that
have both a colour fundus photograph and OCT: 768 eyes from 605 subjects, 769 CFP
images and 1,217 OCT B-scans (195 normal, 58 dry AMD, 331 wet AMD, 184 PCV),
resized to 448 × 448.

`manifest.csv` has one row per eye with columns `eye_id`, `subject_id`,
`original_label` (`healthy`, `dry_amd`, `wet_amd`, `pcv`), `cfp_files` and `oct_files`
(`|`-separated paths relative to `EyePaired_AMD_448/`).

> **To be added:** the script that builds `EyePaired_AMD_448` from MMC-AMD.
