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
    
## 5. Transition-Based Dependency Parser and Disambiguation:
    - implements a dependency parser based on tree structures of input sentences and running predictions
    - Uses GloVe pre-trained word embeddings for optimal performance
    - A tree structure is said to be projective if there are no crossing dependency edges and/or projection lines. The model uses the definition of projectivity in the algorithm by determining right and left arcs in the sentence and process the sentence following a dependency tree.
    - The greedy algorithm Design a dumber but really fast algorithm and let the machine learning do the rest. • Eisner’s algorithm searches over many different dependency trees at the same time. 
    - A transition-based dependency parser only builds one tree, in one left-to-right sweep over the input.Time complexity is linear, O(n), since we only have to treat each word once
    - This can be achieved since the algorithm is greedy, and only builds one tree, in contrast to Eisner’s algorithm, where all trees are explored 
    - There is no guarantee that we will even find the best tree given the model, with the arc-standard model 
    - There is a risk of error propagation • An advantage is that we can use very informative features, for the ML algorithm
    - 58-65% accuracy wrt to golden trees
    
## 6. Neural Coreference Resolution:
    - implementing parts of a Pytorch implementation for neural coreference resolution, inspired by Lee et al.(2017), “End-to-end Neural Coreference Resolution” (EMNLP).
    - implement the B3 coreference metric as discussed in class without importing external libraries. 

$`B^{_{precision}^{3}} = \frac{1}{n}\sum_{i}^{n} \frac{\left |Gold_{i} \cap  System_{i} \right |}{\left | System_{i} \right |}`$
$`B^{_{recall}^{3}} = \frac{1}{n}\sum_{i}^{n} \frac{\left |Gold_{i} \cap  System_{i} \right |}{\left | Gold_{i} \right |}`$

    - incorporate the word distance information by first initializing distance embedding in init() function, then concatenate the original embedding and the corresponding distance embedding in scorer() ~0.925 accuracy in trainingIn my fancy model implementation (more detailed explanation in writeup)
    - in my final model I used a 10K dictionary instead of 50K, I used 200-d representation of word embeddings instead of 50-d, included similarity between the tokens in word's sentence and in the mention's sentence (cosine similarity), used distance between two words in terms of absolute index, used parallelism between two words' positions in their respective sentences, using bins for the first 2 words, the first 5 words, the first 7 words, the first 10 words and further away. I also tried to included information about gender agreement (male, female, neutral) and number agreement (singular plural) but I couldn't find an already existing dictionary to download to use with the mentions.


## 7. Transformers in Question Answering & Response Generation:
    - Created a Transformer Model and a Decoder to analyze dialogues and allow model to respond to questions etc
    - Included the effect of emotions by training two different models by considering 2 equal-sized collections of emotions represented in dialogues:
```positive_emotions = ['anticipating', 'caring', 'confident', 'content', 'excited', 'faithful', 'grateful', 'hopeful', 'impressed', 'joyful', 'nostalgic', 'prepared', 'proud', 'sentimental','surprised','trusting']
negative_emotions = ['afraid', 'angry', 'annoyed', 'anxious', 'apprehensive', 'ashamed','devastated','disappointed','disgusted', 'embarrassed','furious','guilty','jealous','lonely','sad','terrified']```
    - Train 2 models one specifically trained on positive emotion-ed dialogues and other specifically trained on negative emotion-ed dialogues only.
        1) decode the positive model on some positive data to see the types of responses it produces
        2) decode the negative model on some negative data to see the types of responses it produces.
        3) decode both models on some positive data to see the types of responses it produces.
        4) decode both models on some negative data to see the types of responses it produces.
        5) see which model does better when evaluated on the other's development set

