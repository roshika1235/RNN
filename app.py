import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# Load Model
# -----------------------------
model = tf.keras.models.load_model("next_word_rnn.keras")

# -----------------------------
# Load Tokenizer
# -----------------------------
with open("tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

# -----------------------------
# Load Sequence Length
# -----------------------------
with open("sequence_length.pkl", "rb") as file:
    max_sequence_length = pickle.load(file)

# -----------------------------
# Next Word Prediction Function
# -----------------------------
def predict_next_word(seed_text):

    # Convert text to tokens
    token_list = tokenizer.texts_to_sequences([seed_text])[0]

    # Add padding
    token_list = pad_sequences(
        [token_list],
        maxlen=max_sequence_length - 1,
        padding="pre"
    )

    # Predict probabilities
    predicted = model.predict(token_list, verbose=0)

    # Get highest probability word index
    predicted_index = np.argmax(predicted)

    # Convert index back to word
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word

    return "Word not found"


# -----------------------------
# Interactive Loop
# -----------------------------
print("Next Word Prediction RNN")
print("Type 'exit' to quit\n")

while True:

    text = input("Enter text: ")

    if text.lower() == "exit":
        break

    next_word = predict_next_word(text)

    print(f"Predicted next word: {next_word}\n")