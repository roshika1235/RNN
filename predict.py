import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# -----------------------------
# Load trained model
# -----------------------------

model = tf.keras.models.load_model(
    "next_word_rnn.keras"
)

# -----------------------------
# Load tokenizer
# -----------------------------

with open("tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

# -----------------------------
# Load sequence length
# -----------------------------

with open("sequence_length.pkl", "rb") as file:
    max_sequence_length = pickle.load(file)


# -----------------------------
# Prediction function
# -----------------------------

def predict_next_word(text):

    text = text.lower()

    # Convert words to numbers
    token_list = tokenizer.texts_to_sequences([text])[0]

    # Padding
    token_list = pad_sequences(
        [token_list],
        maxlen=max_sequence_length - 1,
        padding="pre"
    )

    # Predict
    prediction = model.predict(
        token_list,
        verbose=0
    )

    # Get highest probability word
    predicted_index = np.argmax(prediction)

    # Convert number back to word
    for word, index in tokenizer.word_index.items():

        if index == predicted_index:
            return word

    return "Unknown"


# -----------------------------
# Take input from user
# -----------------------------

while True:

    text = input("\nEnter text (or type 'exit'): ")

    if text.lower() == "exit":
        print("Program ended.")
        break

    next_word = predict_next_word(text)

    print("Predicted next word:", next_word)
