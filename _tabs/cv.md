---
layout: page
title: CV
icon: fas fa-file-alt
order: 4
permalink: /cv/
toc: true
---

<a href="/assets/pdf/JoonSeo_Oh_CV.pdf" download="JoonSeo_Oh_CV.pdf" class="btn btn-outline-primary mb-3"><i class="fas fa-download me-2" aria-hidden="true"></i>Download CV (PDF)</a>

*Last updated: November 6, 2026*

# JoonSeo Oh

Undergraduate Student in Artificial Intelligence Engineering, Inha University

[joons81511@gmail.com](mailto:joons81511@gmail.com) · [GitHub](https://github.com/JoonSeoOh) · [Blog](https://joonseooh.github.io)

## Research Interests

Computer Vision; Human Pose Estimation; Fairness and Robustness under Challenging Imaging Conditions.

## Education

**Inha University**, South Korea  
Undergraduate Program in Artificial Intelligence Engineering  
March 2022 – Present · Expected graduation: August 2027  
Cumulative GPA: **4.25 / 4.50**

**Freie Universität Berlin**, Germany  
Exchange Student · Spring 2025

## Research Experience

**Undergraduate Researcher — Trustworthy AI Lab, Inha University**  
March 2026 – Present  
Advisor: Prof. Sungho Park

**Exposure-Conditioned Fairness in Human Pose Estimation**  
Ongoing Research · Early Stage

- Working to reproduce UDAPose and evaluate human pose estimation performance on FHIBE under normal- and low-light conditions.
- Investigating whether the performance drop from normal to low light differs across skin-tone groups, with a focus on degradation disparities rather than static accuracy differences.
- The broader study aims to characterize skin-tone-specific degradation curves under controlled low-light conditions and, if systematic disparities are confirmed, explore mitigation while preserving average low-light performance.

## Selected Projects

### Automatic Photo Failure Detection Using Classical Machine Learning

Individual Course Project · Machine Learning · Spring 2026  
[GitHub Repository](https://github.com/JoonSeoOh/shot-failure-detection)

- Built a three-stage photo quality screening pipeline on KonIQ-10k using 17 handcrafted image features, including sharpness, exposure, contrast, noise, and color statistics.
- Compared rule-based classification, logistic regression, random forests, RBF-SVM, and tuned XGBoost; the XGBoost model achieved a test ROC-AUC of 0.869 and F1-score of 0.823 for MOS-threshold-based quality classification.
- Implemented context-aware heuristics to reconsider images flagged as failures when blur or backlighting may reflect intentional bokeh or silhouette effects.
- Integrated SHAP explanations with rule-based advice to summarize model-attributed quality issues and suggest photographic adjustments.

### YOLOv5-Based Screening of Handgun Components for 3D-Printing Safety

Course Project · Creative Design in Artificial Intelligence (AIE1004)  
Team Leader · Four-member team · Fall 2022

- Led a four-member team developing a prototype to screen 3D-slicer screenshots for handgun components and issue warnings for flagged inputs.
- Fine-tuned a pretrained YOLOv5s detector in PyTorch on a custom three-class dataset (frame, magazine, and slide), using Google Colab.
- Trained on a team-curated dataset of 900 source images combining 450 real-component photographs and 450 Cura screenshots, expanded to 1,652 images through augmentation.

## Competition Experience

**R.O.A.D. Barbados Historic Handwriting Challenge — Zindi**  
Participant · 2026  
[Competition Page](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge)

- Responsible for image preprocessing and fine-tuning Qwen-family vision-language models for transcription of historical handwritten documents.
- Proposed and experimented with image cropping and color transformations to address preprocessing challenges in degraded document images.
- Contributed to a team approach combining multiple fine-tuned models through ensembling.

## Honors and Scholarships

- **Academic Merit Scholarship**, Inha University · Spring 2026
- **Exchange Student Scholarship** · Spring 2025

## Technical Skills

- **Programming:** Python
- **Machine Learning:** PyTorch, scikit-learn, XGBoost, OpenCV, NumPy, SHAP, YOLOv5
- **Tools:** Git, GitHub, GitHub Actions, Jekyll, Markdown, LaTeX

## Languages

- **English:** TOEIC 895/990
