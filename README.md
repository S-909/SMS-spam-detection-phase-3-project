# 📩 SMS Spam Detection

## 🚀 Overview

This project builds and evaluates machine learning models to classify SMS messages as either **spam** or **ham** (not spam). The goal is to develop an accurate and efficient spam detection system using natural language processing (NLP) and traditional machine learning techniques.

---

## 🎯 Business Understanding

Spam messages are more than just a nuisance—they can lead to financial scams, exposure to malicious links, and privacy threats. With the rising volume of SMS-based communication, an automated spam detection system is essential for:

- Enhancing user experience
- Protecting users from fraud
- Reducing SMS filtering costs for telecom providers

This project uses machine learning to build a system capable of reliably identifying spam messages in real-time.

---

## 📊 Data and Methodology

### 📁 Dataset

- **Source:** SMS Spam Collection dataset (publicly available)
- **Structure:**
  - `label`: Indicates whether a message is "ham" or "spam"
  - `message`: The SMS text content

### 🧹 Preprocessing

- Removed duplicates
- Transformed labels into binary format: `0 = ham`, `1 = spam`
- Cleaned text by:
  - Converting to lowercase
  - Removing punctuation, digits, and extra whitespace
  - (Optionally) Removing stop words

### 🔧 Feature Engineering

- **Text Vectorization Techniques:**
  - `CountVectorizer`: Bag-of-words representation
  - `TF-IDF Vectorizer`: Weighted importance of words

### 🤖 Models Used

| Model                     | Description                                             |
|--------------------------|---------------------------------------------------------|
| Multinomial Naive Bayes  | Fast, interpretable model based on word probabilities   |
| Logistic Regression      | Linear model using sigmoid function for classification  |

- **Training and Testing Split**
- **Evaluation Metrics:**
  - Accuracy
  - Precision
  - Recall
  - F1-Score

### 🛠️ Hyperparameter Tuning

- Used `GridSearchCV` with cross-validation to optimize:
  - `alpha` for Naive Bayes (smoothing)
  - `C` for Logistic Regression (regularization strength)

---

## 📈 Exploratory Data Analysis (EDA)

- **Class Distribution:** 
  - Dataset is imbalanced (more ham than spam)
- **Common Words in Messages:**
  - Spam: "free", "txt", "win", "urgent"
  - Ham: "ok", "home", "love", "see"
- **Message Length:**
  - Spam messages are generally longer than ham messages

Visualizations were generated using bar plots and histograms to understand the data distribution and feature importance.

---

## 🧪 Model Evaluation

| Model                   | Accuracy | Precision | Recall | F1-Score |
|------------------------|----------|-----------|--------|----------|
| Naive Bayes (tuned)    | ~97%     | High      | High   | High     |
| Logistic Regression    | ~96-98%  | High      | High   | High     |

- Both models performed well, but trade-offs were considered:
  - **Naive Bayes:** Faster, more interpretable
  - **Logistic Regression:** Slightly better at recall in some settings

---

## 🔍 Feature Importance

### 📢 Top Words Indicative of Spam:
- **"call"**, **"free"**, **"txt"**, **"claim"**, **"prize"**, **"win"**, **"stop"**, **"urgent"**

### 🧘 Top Words Indicative of Ham:
- **"ll"**, **"know"**, **"get"**, **"home"**, **"like"**, **"come"**

These features helped provide insight into what drives model decisions and could support future rule-based systems or explainability tools.

---

## 📝 Key Takeaways

- ⚖️ **Imbalanced Dataset**: Addressed through evaluation metrics like precision and recall
- 🧼 **Preprocessing**: Crucial for improving signal-to-noise ratio in text classification
- 🔠 **Vectorization**: CountVectorizer chosen for final model due to simplicity and good performance
- 🔍 **Hyperparameter Tuning**: Improved results and generalization
- 📚 **Explainability**: Feature importance gave clear insights into the model’s logic

---

## 🔍 Example Prediction

To demonstrate how the trained model works, here’s a sample prediction using two email messages:

```python
# Sample emails
emails = [
    "Congratulations! You've won a $1,000 gift card. Click here to claim your prize now!",  # Likely spam
    "Hello John, I hope you're doing well. Let's schedule a meeting for next week."          # Likely not spam
]

# Vectorize the input using the trained CountVectorizer
emails_count = V.transform(emails)

# Make predictions using the trained model
model.predict(emails_count)
```

The output of this prediction is:

```python
array([1, 0])
```

This means:

- The first message, `"Congratulations! You've won a $1,000 gift card. Click here to claim your prize now!"`, is classified as `1`, which indicates **spam**.
- The second message, `"Hello John, I hope you're doing well. Let's schedule a meeting for next week."`, is classified as `0`, which indicates **not spam** (ham).

### ✅ Interpretation

- `1` represents **spam**
- `0` represents **ham** (not spam)

| Message                                                                                          | Prediction | Interpretation |
|--------------------------------------------------------------------------------------------------|------------|----------------|
| "Congratulations! You've won a $1,000 gift card. Click here to claim your prize now!"            | 1          | Spam           |
| "Hello John, I hope you're doing well. Let's schedule a meeting for next week."                  | 0          | Not Spam       |

This example illustrates that the model correctly identifies promotional, urgent language as spam, while normal conversational text is recognized as ham.
##  Recommendations.
- 1.Implement the naive Bayes model in sms filtering system this will achieve >98% accuracy in spam detection.
- 2.Create user feedback mechanism for misclassified sms.Continuously improve model with new data
- 3.Set up performance monitoring dashboards.Track accuracy, false positive/negative rates, processing time# SMS-spam-detection-phase-3-project
