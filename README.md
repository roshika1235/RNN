# 🧠 RNN Next Word Prediction

A Deep Learning project that predicts the next word in a sentence using a Recurrent Neural Network (RNN).

This project demonstrates how RNNs can understand sequential text data and use previous words to predict the most likely next word.

---

# 📖 What is RNN?

Let's take a very simple example.

When we type on our phone:

```text
I am going to the
```

our phone might predict the next word as:

```text
market
```

How does it know what word might come next?

It looks at the words we have already typed and uses that previous information to make a prediction.

So, whenever the order of previous information matters, we need a model that can remember what came before.

And that is where RNN (Recurrent Neural Network) becomes useful.

---

# 🔄 What is a Recurrent Neural Network?

A Recurrent Neural Network (RNN) is a type of Neural Network specifically designed for sequence data.

Unlike traditional neural networks that process each input independently, RNNs can remember previous inputs through a hidden state.

This memory allows RNNs to understand context and make predictions based on previously seen information.

For example:

```text
Input:
I am going to the

Output:
market
```

The prediction is made using both the current input and previous words.

---

# ✅ Why RNN is Better

## 1. Designed for Sequential Data

RNNs are specifically built to process sequence-based data such as:

- Text
- Speech
- Time Series
- Sensor Data

---

## 2. Maintains Memory of Previous Inputs

Unlike traditional neural networks, RNNs can retain information from earlier inputs.

This allows the model to understand context and relationships between words.

---

## 3. Captures Temporal Relationships

RNNs can identify patterns and trends occurring across different time steps.

This makes them valuable for:

- Forecasting
- Language Modeling
- Sequence Prediction

---

## 4. Flexible Input and Output Lengths

RNNs can work with sequences of varying lengths.

Applications include:

- Machine Translation
- Speech Recognition
- Text Generation
- Chatbots

---

## 5. Widely Applicable Across Domains

RNNs are used in:

- Next Word Prediction
- Sentiment Analysis
- NLP
- Speech Recognition
- Language Translation
- Stock Price Prediction
- Time Series Forecasting

---

# ❌ Drawbacks of RNN

Although RNNs are effective for sequence prediction, they have several limitations.

---

## Poor Performance on Long Sequences

As sequence length increases, the model tends to forget important context.

---

## Vanishing Gradient Problem

During training, gradients become very small, making it difficult for the network to learn long-term dependencies.

---

## Exploding Gradient Problem

Gradients can become extremely large, causing unstable training.

---

## Short-Term Memory

RNNs struggle to remember information from the beginning of long sequences.

---

## Slow Processing

Since RNN processes data sequentially, training can be time-consuming.

---

# 🚀 Evolution of RNN

To overcome the limitations of RNN, more advanced architectures were developed.

---

## LSTM (Long Short-Term Memory)

LSTM introduces:

- Memory Cells
- Forget Gate
- Input Gate
- Output Gate

Benefits:

✅ Retains information longer

✅ Reduces vanishing gradient problem

✅ Better performance on long sequences

✅ Improved accuracy

---

## GRU (Gated Recurrent Unit)

GRU is a simplified version of LSTM.

Benefits:

✅ Faster training

✅ Less computational cost

✅ Good performance on sequence tasks

---

# 🏗️ Project Architecture

```text
                         story.txt
                              │
                              ▼
                     Text Preprocessing
                              │
                              ▼
                         Tokenization
                              │
                              ▼
                      Create Sequences
                              │
                              ▼
                           Padding
                              │
                              ▼
                     Input Features (X)
                              │
                              ▼
                    ┌─────────────────┐
                    │   Embedding     │
                    │      Layer      │
                    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │      RNN        │
                    │     Layer       │
                    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │     Dense       │
                    │     Layer       │
                    └─────────────────┘
                              │
                              ▼
                     Next Word Prediction
```

---

# 🧠 Model Workflow

### Step 1

Read text data from:

```text
story.txt
```

### Step 2

Convert text into tokens using Tokenizer.

Example:

```text
I love Python
```

becomes

```text
[15, 23, 91]
```

### Step 3

Generate training sequences.

Example:

```text
I love
I love Python
```

### Step 4

Pad sequences to a fixed length.

### Step 5

Train the RNN model.

### Step 6

Save:

```text
next_word_rnn.keras
tokenizer.pkl
sequence_length.pkl
```

### Step 7

Predict next word for user input.

---

# 📂 Project Structure

```text
RNN/
│
├── app.py
├── next_word.ipynb
├── predict.py
├── story.txt
├── tokenizer.pkl
├── sequence_length.pkl
├── next_word_rnn.keras
├── requirements.txt
└── README.md
```

---

# 💻 Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Streamlit
- Pickle

---

# 📦 Clone the Repository

```bash
git clone https://github.com/roshika1235/RNN.git
```

Move into project directory:

```bash
cd RNN
```

---

# ⚙️ Install Dependencies

Using requirements file:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install streamlit tensorflow numpy
```

---

# ▶️ Run the Application

```bash
python -m streamlit run app.py
```

After running, open:

```text
http://localhost:8501
```

in your browser.

---

# 🎯 Features

✅ RNN Theory Page

✅ Complete Explanation of RNN

✅ Advantages and Drawbacks

✅ Project Architecture Visualization

✅ Upload Text File

✅ Enter Custom Sentence

✅ Predict Next Word

✅ Interactive Streamlit UI

---

# 📝 Example

Input:

```text
Once upon a
```

Output:

```text
time
```

---

# 🔮 Future Enhancements

- LSTM Implementation
- GRU Implementation
- Top 5 Word Predictions
- Transformer-Based Models
- Text Generation
- Probability Scores
- Model Comparison Dashboard

---

# 👩‍💻 Authors

### Roshika Challa
### Priyanka Sandhila
### Baddaram padhmini
### Harshitha Chakkirala
### Venkat
### Rupsa
### Sudhruti
### Ahijit 

GitHub Repository:

```text
https://github.com/roshika1235/RNN
```

---

# ⭐ If you found this project useful, please consider giving it a star on GitHub.
