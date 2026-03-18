# 👁️ IDSC2026 — Glaucoma Detection AI

> **International Data Science Challenge 2026**  
> Theme: *Mathematics for Hope in Healthcare*  
> Hosted by UPM (Malaysia) · UNAIR, UNMUL, UB (Indonesia)
  
[![Streamlit App]( https://idsc2026-glaucoma-detection-rjdutnaqxlasdumwawhmci.streamlit.app/ )]

---

## 🎯 Problem Statement

Glaucomatous Optic Neuropathy (GON) is the **leading cause of irreversible blindness** worldwide, affecting 64.3 million people globally. Approximately **50% of cases remain undiagnosed** until advanced stages. Traditional diagnosis requires specialist ophthalmologists — a barrier in many regions.

**Our Solution:** An AI-powered fundus image classifier that detects GON with >99% AUC, enabling early screening at primary care level.

---

## 📊 Results

| Model | Test AUC | Sensitivity | Quality-Aware |
|---|---|---|---|
| MobileNetV2 (Baseline) | 0.9919 | 97.00% | ❌ |
| EfficientNet-B4 | 0.9967+ | 98.00% | ❌ |
| **QA-EfficientNet-B4 (Ours)** | **0.9991 (val)** | - | ✅ |

- 🎯 **False Negatives: only 2 / 132 test images**
- 🎯 **False Positives: only 2 / 132 test images**

---

## 🔬 Key Innovations

1. **Quality-Aware Loss** — Leverages FundusQ-Net image quality scores during training. High-quality images contribute more to model learning. Unique approach not found in existing HYGD literature.
2. **Patient-level Split** — Strict train/val/test split by patient ID to prevent data leakage (commonly overlooked).
3. **GradCAM Interpretability** — Visual explanation of model decisions highlighting optic disc region.
4. **Threshold Optimization** — Clinical-grade threshold tuning for maximum sensitivity (0.923 optimal).
5. **Ablation Study** — Systematic comparison: MobileNetV2 → EfficientNet-B4 → QA-EfficientNet.

---

## 🗂️ Repository Structure

```
IDSC2026-Glaucoma-Detection/
├── app.py                          ← Streamlit web app
├── final_model.pth                 ← Trained model weights (EfficientNet-B4)
├── requirements.txt                ← Python dependencies
├── IDSC2026_HYGD_Glaucoma_Detection.ipynb  ← Full training notebook
├── README.md
└── assets/
    ├── gradcam_glaucoma.png
    ├── gradcam_normal.png
    └── evaluation_results.png
```

---

## 🚀 Run Locally

```bash
git clone https://github.com/reinhart07/IDSC2026-Glaucoma-Detection
cd IDSC2026-Glaucoma-Detection
pip install -r requirements.txt
streamlit run app.py
```

---

## 📓 Notebook

The full pipeline is available in [`IDSC2026_HYGD_Glaucoma_Detection.ipynb`](./IDSC2026_HYGD_Glaucoma_Detection.ipynb):

1. ⚙️ Environment Setup
2. 📥 Dataset Download & Loading
3. 🔍 Exploratory Data Analysis
4. 🔧 Preprocessing Pipeline (patient-level split)
5. 📱 Baseline 1 — MobileNetV2
6. 🚀 Baseline 2 — EfficientNet-B4
7. 💡 Innovation — Quality-Aware EfficientNet-B4
8. 📊 Model Comparison & Ablation Study
9. 🔥 GradCAM Interpretability
10. 🔎 Error Analysis & Threshold Optimization
11. 📝 Summary & Clinical Relevance

---

## 📦 Dataset

**Hillel Yaffe Glaucoma Dataset (HYGD) v1.0.0**

- 747 Digital Fundus Images (JPG)
- 288 patients (ages 36–95)
- Gold-standard annotations based on full ophthalmic examination
- Source: [PhysioNet](https://physionet.org/content/hillel-yaffe-glaucoma-dataset/1.0.0/)

**Citation:**
```
Abramovich, O., et al. (2025). Hillel Yaffe Glaucoma Dataset (HYGD): 
A Gold-Standard Annotated Fundus Dataset for Glaucoma Detection (version 1.0.0). 
PhysioNet. https://doi.org/10.13026/z0ak-km33
```

---

## ⚠️ Disclaimer

This tool is **for research purposes only**. It is not a substitute for professional ophthalmological examination. Always consult a qualified healthcare professional for medical diagnosis.

---

## 🏆 Competition

**IDSC 2026** — International Data Science Challenge  
- Stage 1 Deadline: 25 March 2026
- Grand Final: 11 April 2026
- Theme: Mathematics for Hope in Healthcare