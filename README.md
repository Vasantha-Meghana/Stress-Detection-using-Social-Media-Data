# Stress Detection and Emotion Classification Using Social Media Analytics

This project presents an AI-driven framework to detect stress and classify emotions from social media content using advanced Natural Language Processing (NLP) models. The system compares traditional machine learning models with state-of-the-art transformer-based language models on Reddit and Twitter datasets.

---

## Project Overview

The mental health of individuals is increasingly reflected in social media posts. This project uses data-driven techniques to:
- Detect stress-indicative language.
- Classify underlying emotions like sadness, anger, fear, etc.
- Compare performance of traditional ML models (SVM, Logistic Regression, XGBoost) with modern PLMs (BERT, RoBERTa, DistilBERT).
- Use topic modeling (LDA) for deeper insights.
- Address issues like sarcasm, informal language, and data imbalance.

---

## Key Components

| Notebook Title | Description |
|----------------|-------------|
| `Stress_Detection_using_ML_algorithms(Reddit_data).ipynb` | Traditional ML-based stress detection using Reddit data. |
| `Stress_Detection_using_ML_algorithms(Twitter_data).ipynb` | Traditional ML models applied on Twitter dataset. |
| `Stress_Detection_using_RoBERTa.ipynb` | Fine-tuning RoBERTa for contextual stress classification. |
| `Stress_Detection_using_rule_based_sentimental_analysis_tools.ipynb` | Rule-based emotion tagging and stress labeling. |

---

## Methodology

1. **Data Collection**: Reddit & Twitter datasets with stress-positive and stress-negative labels.
2. **Preprocessing**: Tokenization, stopword removal, stemming/lemmatization, handling imbalance using SMOTE.
3. **Modeling**:
   - ML: SVM, Logistic Regression, XGBoost using TF-IDF and ELMo.
   - PLM: BERT, RoBERTa, DistilBERT using HuggingFace Transformers.
4. **Emotion Detection**: Fine-tuned BERT models to identify emotional triggers indicative of stress.
5. **Evaluation**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix.
6. **Interpretability**: SHAP values and attention visualizations for model explainability.

---

## Results Summary

- **RoBERTa** achieved the highest performance with **97.2% accuracy**.
- Traditional ML models performed well as baselines but lacked contextual understanding.
- BERT-based emotion classification significantly improved interpretability.
- LDA uncovered stress themes like “academic pressure”, “relationship issues”, etc.

---

## Applications

- **Mental health surveillance systems**.
- **Early intervention platforms for therapists**.
- **Real-time analysis of stress trends on social media**.
- Can be extended to multimodal setups (text + voice/image).

---

## Setup Instructions

### Clone the repository
```bash
git clone https://github.com/your-username/stress-detection-social-media.git
cd stress-detection-social-media
```
### Create virtual environment (optional)
```bash
python -m venv venv
source venv/bin/activate  # for Linux/macOS
venv\Scripts\activate     # for Windows
```
### Install dependencies
```bash
pip install -r requirements.txt
```
