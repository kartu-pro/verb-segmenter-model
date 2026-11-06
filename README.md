# Georgian Verb Segmentation (NLP & Linguistics)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-Framework-red.svg)]()

A deep learning and natural language processing project focused on the morphological segmentation of Georgian verbs, using a **6-slot verb system**.

---

## 📌 Project Overview

This repository explores, implements, and benchmarks multiple machine learning architectures—ranging from lightweight BiLSTM taggers to Sequence-to-Sequence Transformers (`google/byt5-small`)—to break down conjugated verb surface forms into their constituent segments/morphemes.

### The 6-Slot Georgian Verb System Used
1. **Prefix (PRV):** Directional / Aspect
2. **Person Marker:** Subject/Object agreement
3. **Version Vowel:** Semantic orientation
4. **Verb Root:** Core semantics (Required)
5. **PFSF (Present Future Stem Formant):** Stem characteristics
6. **Suffix:** TAM (Tense-Aspect-Mood) / Inflection

---

## 📊 Dataset

- **Source:** [Open Georgian dataset](https://kartu-pro.github.io/resources/open-georgian/)
- **Volume:** 2,940 unique verbs and 40,768 conjugated forms (including explicit morpheme splits).
- **Split Strategy:** Data is split into training, validation, and test sets strictly **by verb** (preventing data leakage across splits).

---

## 🚀 Model Architectures Explored

1. **Sequence Tagging (BiLSTM & BiLSTM-CRF):** 
   - Character-level classification using BIO or Boundary slot schemes.
   - Enhanced with a Linear-Chain CRF layer to jointly model legal tag transitions.
2. **Sequence-to-Sequence Transformers (`google/byt5-small`):**
   - Evaluated under Single-word and Group-of-6 context strategies.

---

## 📈 Results Summary
| Model Configuration | Morpheme F1 (%) | Exact Match (%) | Slot Accuracy (%) | Morpheme Precision (%) | Morpheme Recall (%) | Training Time (min) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| BiLSTM-Meta (boundaries) | 91.57 | 81.36 | 91.61 | 91.36 | 91.78 | 2.9 |
| BiLSTM-Meta (bio) | 94.93 | 86.56 | 94.88 | 94.28 | 95.58 | 2.2 |
| BiLSTM-Meta-CRF (boundaries) | 93.28 | 82.99 | 93.38 | 92.74 | 93.83 | 10.9 |
| BiLSTM-Meta-CRF (bio) | 95.79 | 90.72 | 95.66 | 94.83 | 96.76 | 10.4 |
| google/byt5-small (single) | 97.51 | 94.83 | 97.23 | 96.62 | 98.42 | 344.9 |
| google/byt5-small (group6) | 96.79 | 88.13 | 96.51 | 95.79 | 97.82 | 228.8 |
