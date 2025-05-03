# Smishing-Detector
python version used 3.11.9

you should have version below 3.12 to install tensorflow

first
```cd smishing_detector```

then
```pip install requirements.txt```

to run the server
```python manage.py runserver```


in requirments there is a library called Twilio that I didn't implement it yet, it is used like a whatssap api so we can track irl messages


📄 Project Report: Spam Detection Using TensorFlow and Deep Learning
🔖 Title:
SMS Spam Detection using Deep Learning (LSTM) in TensorFlow

🧠 1. Introduction
The goal of this project is to classify SMS messages as "ham" (non-spam) or "spam". This is a binary classification problem that can be tackled using a variety of machine learning techniques. In this project, we opted for a deep learning-based approach using TensorFlow and Keras. The dataset used contains over 5900 messages with labels.

📦 2. Dataset Summary
Source: Public SMS spam dataset.

Format: Two columns – label (ham/spam) and text.

Imbalance:

Ham: 4825 samples

Spam: 1139 samples

Handling Imbalance: Class weighting was used during training to address this.

🔍 3. Data Preprocessing
3.1 Label Encoding
python code:
------
df["label"] = df["label"].map({"ham": 0, "spam": 1})
------
Converted text labels to integers (0 and 1).

3.2 Tokenization and Padding
Used Keras' Tokenizer to vectorize the text data. Each message was tokenized into integers, padded to a max length (50 tokens) to maintain uniform input size for the model.

3.3 Train-Test Split
python code:
------
X_train, X_test, y_train, y_test = train_test_split(padded, labels, test_size=0.2)
------
🏗️ 4. Model Architecture
We designed a simple yet effective deep learning architecture:

python code:
------
model = Sequential([
    Embedding(input_dim=10000, output_dim=16, input_length=50),
    LSTM(64, return_sequences=False),
    Dense(1, activation='sigmoid')
])
------
Embedding Layer: Converts word indices into dense vectors.

LSTM Layer: Captures sequential patterns and contextual meaning.

Dense Output Layer: Uses sigmoid for binary classification.

🧠 5. Why TensorFlow?
We chose TensorFlow over traditional models and other frameworks for several reasons:

✅ Benefits of TensorFlow:
Feature	Reason
🔁 Sequence Learning	TensorFlow's LSTM layer is ideal for text classification.
🧩 End-to-End Pipeline	TensorFlow provides everything: preprocessing, training, saving, and deploying.
📈 Fine Control	Access to hyperparameter tuning and GPU acceleration.
🌎 Deployment-Ready	Integration with TensorFlow Lite and TensorFlow.js for production.
🤖 Community & Docs	Massive ecosystem and support compared to PyTorch for small text projects.

🚫 6. Why Not Traditional Models?
Models like Naive Bayes, SVM, or Logistic Regression:
Traditional Model	Why Not Chosen
Naive Bayes	Too simplistic — assumes feature independence which isn’t true for language.
Logistic Regression	Good baseline but fails to capture word order/context.
SVM	Needs careful kernel tuning, lacks interpretability for sequences.

Though these are lighter and faster, they don't leverage the full power of word order, context, and sequential data. Our deep learning model better understands message semantics, especially for harder-to-catch spam.

📊 7. Evaluation
Loss and Accuracy tracked using .fit() history.

Prediction Probabilities analyzed during testing.

Example:
Prediction probability: 0.91 → SPAM
Prediction probability: 0.08 → HAM
⚖️ 8. Imbalance Handling
To combat the skewed distribution:

python code:
------
from sklearn.utils import class_weight

class_weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
-----
This technique ensures the model treats minority classes (spam) more seriously.

🧪 9. Reflection
Working with TensorFlow for this NLP task has shown the value of deep contextual learning. The LSTM-based model can understand subtle linguistic patterns and outperform rule-based or statistical models, especially in evolving spam strategies.

Even with imbalanced data, we achieved stable training and meaningful prediction probabilities.

🚀 10. Future Work
Replace LSTM with Bidirectional LSTM or GRU.

Integrate attention mechanisms.

Use pretrained embeddings like GloVe or BERT.

Expand to multi-class detection (phishing, promotions, etc.).

📁 11. Deliverables
notebook.ipynb: Full code implementation.

smishing_model2.h5: Saved model for inference.

tokenizer2.pkl: Saved tokenizer for production use.

Web interface for testing (Django + TensorFlow backend).

