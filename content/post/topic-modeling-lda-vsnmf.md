---
title: "Topic Modeling in NLP: Comparing LDA vs NMF with Python"
date: 2026-09-15T20:50:00+01:00
draft: false
tags: ["Python", "NLP", "Machine Learning", "Data Science"]
categories: ["Artificial Intelligence"]
cover:
    image: "" # Passe methanata API generated image ekak damu
    alt: "LDA vs NMF Topic Modeling"
---

Topic modeling is a powerful Natural Language Processing (NLP) technique used to discover hidden thematic structures in large text collections. When building text-mining pipelines, the two most prominent algorithms are **Latent Dirichlet Allocation (LDA)** and **Non-negative Matrix Factorization (NMF)**. 

But which one performs better in a real-world scenario? In this technical guide, we will compare LDA and NMF side-by-side using real clinical trial data.

## 1. The Dataset & Environment Setup

To make this comparison practical, we are using a dataset of 559 clinical trial records fetched from ClinicalTrials.gov. 

For the execution environment, the models were evaluated in two separate setups to compare processing speeds:
* **Local Machine:** Windows environment using Jupyter Notebook.
* **Cloud Environment:** Google Colab.

## 2. Text Preprocessing: Bag-of-Words vs. TF-IDF

A common mistake in topic modeling is using the wrong vectorizer. 
* **For LDA:** We use `CountVectorizer` (Bag-of-Words) because LDA is a probabilistic model that relies on raw word frequencies.
* **For NMF:** We use `TfidfVectorizer` (Term Frequency-Inverse Document Frequency) because NMF is a linear-algebraic model that benefits from normalized term weights.

## 3. The Core Python Implementation

Here is the complete scikit-learn implementation to train both models efficiently.

*(Methanata api kalin hadapu Python code block eka danawa)*

## 4. Performance Evaluation & Metrics

Execution time is just one factor. To truly understand which model performed better on the 559 clinical records, we must look at the mathematical evaluation metrics:

### Topic Coherence (Cv)
Coherence measures how logically connected the top words in a generated topic are. A higher score means the topic makes more human sense.
* **LDA Coherence:** $C_v = 0.5203$
* **NMF Coherence:** $C_v = 0.3101$

### Topic Diversity
Diversity measures how unique the topics are from one another. A score closer to 1.0 means less overlap between topics.
* **LDA Diversity:** 0.950
* **NMF Diversity:** 0.867

## 5. Execution Speed: Local Jupyter vs. Google Colab

*(Methana api liyanawa Colab eke GPU/RAM ekka run weddi lda_time saha nmf_time local PC ekata wada kochchara wenas unada kiyala)*

## Conclusion

*(Anthimata summary ekak denawa LDA wada hoda unath, NMF wala execution speed/advantages monawada kiyala)*