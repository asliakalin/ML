# Natural Language Processing Notebooks:

## 1. Sentiment Analysis for Movie Reviews:
    - Generative probabilistic model for sentiment analysis of movie reviews
    - Bag of Words implementation
    - Sentiment Dictionaries, punctuation and syntax analysis
    - Feature engineering focus to increase model accuracy
    
## 2. Convolutional Neural Networks (CNNs) for Text Classification
    - implements the forward pass and backpropagation for a convolutional neural network with sparse inputs for text classification.
    - Forward Function calculates the probability of the positive class for an input text.
    - Using the gradient update equations for V and U, the backward function implements the gradient updates.
    - Once implemented the forward and backward functions, training the model shows increasing accuracy levels.
    
## 3. Distributed Word Embeddings via Skipgram Model, Term-Context Matrix, PPMI and Analogous Tasks:
    - Approximate a Skip-gram word embeddings in a list of movie plot summaries.
    - via positive pointwise mutual information (PPMI) and truncated singular value decomposition (SVD).
    - We have ~42000 summaries containing ~13000000 words, we proceed by creating a vocabulary.
    - Build a term-context matrix for words in the vocabulary.
    - Build Positive Pointwise Mutual Information (PPMI) Matrix.
    - Then obtain a dense low-dimensional vectors via truncated (rank-k) SVD. You should use svds function from Sicpy that is already imported for you to obtain the SVD factorization
    - Using cosine similarity as a measure of distance, now find the closest words to a certain word. 
    - Evaluate Word Embeddings via Analogous Tasks

## 4. LSTM Neural Sequence Labeling for Parts of Speech (POS) tags:
    - implementing, training, and evaluating an LSTM for part-of-speech tagging using the PyTorch library
    - using GloVe pretrained word embeddings
    - batches the data and given tags, trains and evaluates the model using recall, precision and f1 as evaluation models
    - Creates and uses a dictionary of 400,000 words
    - Creates and uses 50d, 100d, 200d and 300d (dimensional) embedded word representations
    - Implements bidirectionality and hyperparameter tuning to the base LSTM model to improve accuracy up to 92.6%
    
    
