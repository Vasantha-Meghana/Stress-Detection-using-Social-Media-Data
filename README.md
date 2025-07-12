# Stress Detection and Emotion Classification Using Social Media Analytics

A robust and scalable AI-powered system developed using **Python**, **NLP**, and **Transformer-based models** to detect **stress** and classify **emotions** from social media platforms like Reddit and Twitter. This project includes multiple **Jupyter notebooks** demonstrating both traditional ML methods and modern deep learning techniques (BERT, RoBERTa).

## How It Works

The system detects stress and emotional cues in social media posts through these stages:

1. **Data Collection** – Reddit and Twitter posts using the [SenticNet stress-detection repository](https://github.com/SenticNet/stress-detection)
2. **Preprocessing** – Cleansing, tokenization, lemmatization, emoji removal, and vectorization
3. **Modeling**:
   - Traditional ML: TF-IDF + SVM / Logistic Regression / XGBoost
   - Transformer Models: BERT, RoBERTa, DistilBERT fine-tuned using Hugging Face
4. **Emotion Classification** – Using BERT to detect stress-indicative emotions like sadness, anger, fear
5. **Topic Modeling** – LDA to identify common themes in stress-positive posts

The result is a model that accurately identifies stress indicators and emotional context from user-generated content.

---

## Tech Stack

  - Python 3
  - Scikit-learn
  - TensorFlow / PyTorch
  - Hugging Face Transformers
  - NLTK, SpaCy, Gensim
  - Matplotlib / Seaborn for visualization

---

## Repository Structure

```plaintext
stress-detection-social-media/
├── notebooks/
│   ├── Stress_Detection_ML_Reddit.ipynb
│   ├── Stress_Detection_ML_Twitter.ipynb
│   ├── Stress_Detection_RoBERTa.ipynb
│   └── Stress_Detection_RuleBased.ipynb
│
├── data/
│   └── README.md                        # Dataset info and links to SenticNet repo
├── requirements.txt                     # All Python dependencies
├── documentation.pdf                    # Detailed report
├── stress_detection_research_paper.pdf  # Final research paper
└── README.md                            # Project documentation
```

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

## Key Components

| Notebook Title | Description |
|----------------|-------------|
| `Stress_Detection_using_ML_algorithms(Reddit_data).ipynb` | Traditional ML-based stress detection using Reddit data. |
| `Stress_Detection_using_ML_algorithms(Twitter_data).ipynb` | Traditional ML models applied on Twitter dataset. |
| `Stress_Detection_using_RoBERTa.ipynb` | Fine-tuning RoBERTa for contextual stress classification. |
| `Stress_Detection_using_rule_based_sentimental_analysis_tools.ipynb` | Rule-based emotion tagging and stress labeling. |

---

## Sample Output

<p align="center"><b>Prediction Result for Stress Detection</b></p>

<p align="center">
  <img width="590" height="395" alt="Stress Detection Output" src="https://github.com/user-attachments/assets/cd2c3a32-5039-4e0f-8d56-10ac6bc7f0f9" />
</p>

<p align="center"><b>Prediction Result for Emotion Classification</b></p>

<p align="center">
  <img width="647" height="395" alt="Emotion Detection Output" src="https://github.com/user-attachments/assets/8e26a909-2be1-4c4f-965a-f22822193e8f" />
</p>

---

## Results Summary

<p align="center"><b>Performance Comparison</b></p>

<p align="center">
  <img width="672" height="219" alt="Model Performance Table" src="https://github.com/user-attachments/assets/6bb64528-5e58-4ad2-98c3-fcb7d1b27396" />
</p>

- **RoBERTa** achieved the highest performance with **97.2% accuracy**.
- Traditional ML models performed well as baselines but lacked contextual understanding.
- BERT-based emotion classification significantly improved interpretability.
- Topic modeling (LDA) uncovered key themes such as “academic pressure”, “relationship issues”, etc.


