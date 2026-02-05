# 📊 Analisis Sentimen Game Honkai: Star Rail

> Sentiment analysis of Honkai: Star Rail game reviews from Google Play Store using Machine Learning & Deep Learning

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 **Project Overview**

This project analyzes **5,000 Indonesian reviews** of Honkai: Star Rail from Google Play Store to classify user sentiment into **3 categories**:
- 🟢 **Positif** (Positive)
- 🟡 **Netral** (Neutral)  
- 🔴 **Negatif** (Negative)

Using **Natural Language Processing (NLP)** and **Deep Learning**, this project achieves **92.16% accuracy** with a GRU model.

---

## 🚀 **Key Features**

✅ Automated web scraping from Google Play Store  
✅ Indonesian text preprocessing (cleaning, stemming, stopword removal)  
✅ Class balancing with upsampling technique  
✅ Two model comparison:
  - Gradient Boosting Classifier (91.07% accuracy)
  - Bidirectional GRU Neural Network (92.16% accuracy)  
✅ Real-time sentiment inference

---

## 📂 **Repository Structure**

```
Analisis-sentimen-game-Honkai-Star-Rail/
├── Scrapping_data.ipynb          # Data collection script
├── pelatihan_model.ipynb         # Model training & evaluation
├── ulasan_aplikasi.csv           # Dataset (5,000 reviews)
├── best_gru_model.h5             # Trained GRU model
└── README.md                     # Documentation
```

---

## 🛠️ **Tech Stack**

### **Data Collection**
- `google-play-scraper` - Web scraping from Play Store

### **Text Processing**
- `nltk` - Tokenization & stopwords
- `Sastrawi` - Indonesian stemming
- `pandas` - Data manipulation

### **Machine Learning**
- `scikit-learn` - Traditional ML (Gradient Boosting)
- `TensorFlow/Keras` - Deep Learning (GRU model)

### **Visualization**
- `matplotlib`, `seaborn` - Charts
- `wordcloud` - Text visualization

---

## 📊 **Methodology**

### **1. Data Collection**
- Scraped **5,000 reviews** from Honkai: Star Rail (Google Play Store)
- Language: Indonesian (`lang='id'`)
- Sort: Most Relevant
- Features: `content`, `score`, `userName`, `at`, `thumbsUpCount`

### **2. Data Preprocessing**
```python
# Text cleaning pipeline:
1. Cleaning     → Remove URLs, mentions, numbers, punctuation
2. Casefolding  → Convert to lowercase
3. Slang fixing → Normalize Indonesian slang words
4. Tokenizing   → Split into words
5. Filtering    → Remove stopwords
6. Lemmatizing  → Convert to base form
7. Stemming     → Indonesian root words (Sastrawi)
```

### **3. Labeling Strategy**
```python
Score 1-2  → Negatif
Score 3    → Netral  
Score 4-5  → Positif
```

**Original distribution:**
- Positif: 3,189 (63.8%)
- Negatif: 1,392 (27.8%)
- Netral: 419 (8.4%)

**After balancing (upsampling):**
- Each class: 3,189 samples
- Total: 9,567 samples

### **4. Model Training**

#### **A. Gradient Boosting Classifier**
```python
- TF-IDF vectorization (max_features=10000, ngram_range=(1,2))
- n_estimators=200
- learning_rate=0.1
- max_depth=5
```

**Results:**
```
              precision    recall  f1-score   support
     Negatif       0.86      0.94      0.90       638
      Netral       0.94      0.99      0.97       638
     Positif       0.94      0.81      0.87       638

    accuracy                           0.91      1914
```

#### **B. Bidirectional GRU Model** ⭐
```python
Architecture:
- Embedding Layer (20,000 vocab, 128 dim)
- SpatialDropout1D (0.3)
- Bidirectional GRU (128 units, dropout=0.3)
- GlobalMaxPooling1D
- Dense (128 units, ReLU)
- Dropout (0.5)
- Dense (3 units, Softmax)

Training:
- Optimizer: Adam (lr=0.0003)
- Loss: Sparse Categorical Crossentropy
- Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- Epochs: 20 (best: epoch 15)
```

**Results:**
```
              precision    recall  f1-score   support
     Negatif       0.90      0.92      0.91       638
      Netral       0.92      0.99      0.96       638
     Positif       0.94      0.85      0.89       638

    accuracy                           0.92      1914
```

---

## 🎯 **Model Performance**

| Model                       | Accuracy | Best For             |
|-----------------------------|----------|----------------------|
| Gradient Boosting           | 91.07%   | Speed & Interpretability |
| **GRU (Bidirectional)**     | **92.16%** | **Highest Accuracy** |

**Winner:** GRU model performs better on contextual understanding of Indonesian text.

---

## 💻 **How to Use**

### **1. Clone Repository**
```bash
git clone https://github.com/Alfan345/Analisis-sentimen-game-Honkai-Star-Rail.git
cd Analisis-sentimen-game-Honkai-Star-Rail
```

### **2. Install Dependencies**
```bash
pip install google-play-scraper nltk Sastrawi wordcloud seaborn tensorflow scikit-learn pandas numpy matplotlib
```

### **3. Run Notebooks**
Open in **Google Colab** or **Jupyter Notebook**:
- `Scrapping_data.ipynb` - Collect new data
- `pelatihan_model.ipynb` - Train models

### **4. Inference Example**
```python
# Load model and predict
sample_review = "game ini bikin nagih banget, seru parah!"
prediction = predict_sentiment_gru(sample_review)
print(f"Sentiment: {prediction}")  # Output: Positif
```

---

## 📈 **Sample Predictions**

| Review                                           | Prediction |
|--------------------------------------------------|-----------|
| "game ini bikin nagih banget, seru parah!"      | ✅ Positif |
| "fitur-fiturnya terlalu ribet dan berat"        | ✅ Positif |
| "grafik dan animasinya oke, tapi loading lama"  | 🔴 Negatif |
| "game nya jelek banget, uninstall langsung"     | 🔴 Negatif |
| "sangat membantu untuk mengisi waktu luang"     | ✅ Positif |

---

## 🎓 **Skills Demonstrated**

| Category                | Skills                                      |
|-------------------------|---------------------------------------------|
| **Data Engineering**    | Web scraping, Data cleaning, Class balancing |
| **NLP**                 | Indonesian text preprocessing, TF-IDF, Word embeddings |
| **Machine Learning**    | Gradient Boosting, Hyperparameter tuning    |
| **Deep Learning**       | RNN, GRU, Bidirectional layers, Regularization |
| **Tools & Libraries**   | TensorFlow, Scikit-learn, Pandas, NLTK      |
| **Best Practices**      | Model checkpointing, Early stopping, Learning rate scheduling |

---

## 📊 **Dataset Information**

- **Source:** Google Play Store (Honkai: Star Rail reviews)
- **Size:** 5,000 reviews
- **Language:** Indonesian
- **Date Range:** April 2023 - April 2025
- **Columns:** `reviewId`, `userName`, `content`, `score`, `at`, `thumbsUpCount`, etc.
- **File:** `ulasan_aplikasi.csv` (2.08 MB)

---

## 🔮 **Future Improvements**

- [ ] Add BERT-based model for Indonesian (IndoBERT)
- [ ] Create web app with Streamlit/Flask
- [ ] Implement aspect-based sentiment analysis
- [ ] Add real-time review monitoring
- [ ] Expand to other gaming platforms (iOS, Steam)

---

## 📚 **References**

- [Honkai: Star Rail on Google Play](https://play.google.com/store/apps/details?id=com.HoYoverse.hkrpgoversea)
- [Sastrawi - Indonesian Stemmer](https://github.com/sastrawi/sastrawi)
- [TensorFlow GRU Documentation](https://www.tensorflow.org/api_docs/python/tf/keras/layers/GRU)

---

## 👤 **Author**

**Alfanah Muhson**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/alfanah-muhson)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/Alfan345)

---

## 📝 **License**

This project is open source and available under the [MIT License](LICENSE).

---

## 🙏 **Acknowledgments**

- HoYoverse for creating Honkai: Star Rail
- Google Play Store reviewers for providing data
- Sastrawi team for Indonesian NLP tools
- TensorFlow community for documentation

---

**⭐ If you find this project useful, please give it a star!**
