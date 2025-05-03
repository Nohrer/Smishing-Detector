import joblib
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = tf.keras.models.load_model("smishing_model2.h5")
tokenizer = joblib.load("tokenizer2.pkl")

def predict_smishing(text):
    text=clean_text(text)
    seq = tokenizer.texts_to_sequences([text])    
    padded = pad_sequences(seq, maxlen=50)     
    pred = model.predict(padded)
    return "smishing" if pred[0][0] > 0.5 else "ham"

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text
