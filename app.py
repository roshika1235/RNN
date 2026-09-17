import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="RNN Next Word Prediction",
    page_icon="🧠",
    layout="wide"
)

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_assets():

    model = tf.keras.models.load_model(
        "next_word_rnn.keras"
    )

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("sequence_length.pkl", "rb") as f:
        sequence_length = pickle.load(f)

    return model, tokenizer, sequence_length


model, tokenizer, sequence_length = load_assets()

# =====================================================
# PREDICTION FUNCTION
# =====================================================

def predict_next_word(text):

    token_list = tokenizer.texts_to_sequences([text])[0]

    token_list = pad_sequences(
        [token_list],
        maxlen=sequence_length - 1,
        padding="pre"
    )

    prediction = model.predict(
        token_list,
        verbose=0
    )

    predicted_index = np.argmax(prediction)

    predicted_word = None

    for word, index in tokenizer.word_index.items():

        if index == predicted_index:
            predicted_word = word
            break

    return predicted_word


# =====================================================
# SIDEBAR
# =====================================================

page = st.sidebar.radio(
    "Navigation",
    [
        "RNN Algorithm",
        "Next Word Prediction"
    ]
)

# =====================================================
# PAGE 1
# =====================================================

if page == "RNN Algorithm":

    st.title("🧠 Recurrent Neural Network (RNN)")

    st.header("What is RNN?")

    st.write("""
A Recurrent Neural Network (RNN) is a Deep Learning algorithm
used for sequential data such as text, speech and time-series data.

Unlike traditional neural networks, RNN remembers previous
inputs using hidden states, allowing it to understand context.
""")

    st.header("Applications")

    st.write("""
✅ Next Word Prediction

✅ Text Generation

✅ Sentiment Analysis

✅ Speech Recognition

✅ Machine Translation

✅ Chatbots
""")

    st.header("How RNN Works")

    st.write("""
1. Convert text into tokens.
2. Send tokens through Embedding Layer.
3. RNN processes tokens sequentially.
4. Hidden State remembers previous information.
5. Dense Layer predicts the next word.
""")

    st.subheader("Architecture")

    st.code("""
Input Sentence
       ↓
Tokenizer
       ↓
Embedding Layer
       ↓
RNN Layer
       ↓
Dense Layer
       ↓
Next Word Prediction
""")

    st.header("Advantages")

    st.success("""
✔ Handles sequential data

✔ Understands context

✔ Useful for NLP tasks

✔ Good for language modeling
""")

    st.header("Disadvantages")

    st.error("""
✘ Vanishing Gradient Problem

✘ Struggles with long sequences

✘ Training can be slower
""")

    st.header("Project Workflow")

    st.code("""
story.txt
      ↓
Tokenizer
      ↓
Create Sequences
      ↓
Padding
      ↓
Train RNN
      ↓
Save Model
      ↓
Predict Next Word
""")

    st.header("Model Statistics")

    st.metric(
        "Vocabulary Size",
        len(tokenizer.word_index)
    )

    st.metric(
        "Sequence Length",
        sequence_length
    )

# =====================================================
# PAGE 2
# =====================================================

else:

    st.title("🔮 Next Word Prediction")

    st.write(
        "Upload a text file and enter a sentence to predict the next word."
    )

    uploaded_file = st.file_uploader(
        "Upload Text File",
        type=["txt"]
    )

    uploaded_text = ""

    if uploaded_file is not None:

        uploaded_text = uploaded_file.read().decode(
            "utf-8",
            errors="ignore"
        )

        st.success("File uploaded successfully!")

        st.subheader("File Statistics")

        st.write(
            f"Characters : {len(uploaded_text):,}"
        )

        st.write(
            f"Words : {len(uploaded_text.split()):,}"
        )

        st.subheader("Preview")

        st.text_area(
            "File Content",
            uploaded_text[:5000],
            height=250
        )

    st.markdown("---")

    st.subheader("Enter Text")

    user_input = st.text_input(
        "Type your sentence",
        placeholder="Once upon a"
    )

    if st.button("Predict Next Word"):

        if user_input.strip() == "":

            st.warning(
                "Please enter a sentence."
            )

        else:

            next_word = predict_next_word(
                user_input
            )

            if next_word:

                st.success(
                    f"Predicted Next Word: {next_word}"
                )

            else:

                st.error(
                    "Prediction could not be generated."
                )

    st.markdown("---")

    st.subheader("Example")

    st.info("""
Try:

Once upon a

The king was

In the forest

There lived a
""")
