from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

word_index = imdb.get_word_index()
model = load_model("Models and Datasets\\imdb_sentiment_model.h5")

def encode_text(text, word_idx, max_len=200):
    # Lowercase and remove non-alphabetic characters (IMDB preprocessing style)
    text = re.sub(r"[^\w\s]", "", text.lower())
    words = text.split()

    # Convert words to indices (use 2 for unknown words)
    encoded = [word_idx.get(i, 2) for i in words]

    # Pad the sequence
    return pad_sequences([encoded], maxlen=max_len)

x = input("Enter the prompt to be analyzed: ")
x_encoded = encode_text(x, word_index)
result = model.predict(x_encoded)[0][0]

print(f"Sentiment Score: {result:.4f}")
print("Positive" if result > 0.5 else "Negative")