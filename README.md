# RNN

Use a predictive text example—it’s very easy to explain:

“Let’s take a very simple example.

When we type on our phone, suppose we write: ‘I am going to the…’

Our phone might predict the next word as ‘market’.

How does it know what word might come next? It looks at the words we have already typed and uses that previous information to make a prediction.

So, whenever the order of previous information matters, we need a model that can remember what came before.

And that is where RNN, or Recurrent Neural Network, is useful.

--------------Why RNN is Better :-------------
•	Designed for Sequential Data
RNNs are specifically built to process sequence-based data such as text, speech, time series, and sensor data.
•	Maintains Memory of Previous Inputs
Unlike traditional neural networks, RNNs can retain information from earlier inputs, allowing them to learn dependencies over time.
•	Captures Temporal Relationships
RNNs can identify patterns and trends that occur across different time steps, making them useful for forecasting and sequence analysis.
•	Flexible Input and Output Lengths
RNNs can work with sequences of varying lengths, making them suitable for applications like translation, speech recognition, and text generation.
•	Widely Applicable Across Domains
RNNs are used in natural language processing, speech recognition, sentiment analysis, machine translation, stock prediction, and many other real-world applications.

-------------Drawbacks of RNN-----------

"Although RNNs are effective for sequence prediction, they have several limitations:

Poor Performance on Long Sequences: As sequence length increases, the model tends to forget important context."

Vanishing Gradient Problem: During training, gradients become very small, making it difficult for the network to learn long-term dependencies.

Exploding Gradient Problem: Gradients can become excessively large, causing unstable training and poor performance.

Short-Term Memory: RNNs struggle to remember information from earlier parts of long sequences.

Slow Processing: Since RNN processes data sequentially, training can be time-consuming.


"To overcome the limitations of RNN, advanced architectures were developed:

LSTM (Long Short-Term Memory):

Uses memory cells and gates.
Retains information for a longer duration.
Solves the vanishing gradient problem effectively.

Therefore, while RNN laid the foundation for sequence learning, LSTM provide better accuracy, faster training, and improved handling of long-term dependencies.
