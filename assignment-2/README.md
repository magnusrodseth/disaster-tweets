# IT3212 - Assignment 2

Written and developed by Haakon Tideman Kanter, Henrik Skog, Mattis Czternasty Hembre, Max Gunhamn, Sebastian Sole, and Magnus Rødseth.

## 1. Implement the preprocessing

We started by removing features, according to what was outlined in assignment 1. Additionally, we removed the `location` feature to preserve the privacy of the users and because it is not particularly relevant to determing whether a tweet is related to a disaster. Next, we removed rows with a confidence threshold below `1.0`.

Next, we preprocessed the textual data. We cleaned up the `keyword` column, ensuring that all keywords were lower case, and that all keywords were separated by a single space. For comparison, some keywords were on the format `airplane%20accident`, which should be `airplane accident`.

Furthermore, we cleaned the `text` column. The first part of this included removing links, line breaks, extra spaces, special characters and punctuation. Next, we removed English stopwords. Finally, we lemmatized the text.

When handling categorical data, we removed rows with a `choose_one` value of `Can't decide`, according to what was outlined in assignment 1. Next, we mapped the `choose_one` values `Relevant` and `Not Relevant` to `1` and `0`, respectively. This was stored in a new feature, called `target`.

Moreover, we removed duplicated rows with regards to the `text` column, as outlined in assignment 1.

## 2. Extract features

### 2.1. Counting hashtags

Hashtag counts were extracted, as we suspected tweets related to a disaster might contain more hashtags than a tweet which does not. This also had to be done before preprocessing, as we remove hashtags in the preprocessing step.

```py
# Extract the number of hashtags
df['hashtag_count'] = df['text'].apply(lambda x: len([c for c in str(x) if c == '#']))
```

### 2.2. Counting mentions

The `mention_count` is a feature where we see how many mentions there are in a tweet. This means how many other twitter users are mentioned in the tweet.
This was extracted because we thought there might be a connection betweeen how many users were tagged and if the tweets are disaster-related. The group thought that disaster-related tweets would have more mentions than tweets that were not disaster related.

```py
# Extract the number of mentions
df['mention_count'] = df['text'].apply(lambda x: len([c for c in str(x) if c == '@']))
```

### 2.3. Checking if the tweet contains a url

The `has_url` feature was extracted because tweets related to a disaster might point to an online resource where one can find more information about the situation. The column is `1` if the tweet contains a url and `0` if not. The `has_url` feature had to be done before preprocessing, as we remove links in the preprocessing step.

```py
# Extract the `has_url` feature
df['has_url'] = df['text'].apply(lambda x: 1 if 'http' in str(x) else 0)
```

### 2.4. N-grams

N-grams was a feature we wanted to look at, as we suspected there could be a correlation between bi- or trigrams and whether the tweet was related to a disaster or not. This is because bigrams and trigrams carry more context than than single words. We found the most used bigrams and trigrams, both for disaster-related tweets and non-disaster related tweets.

```py
from nltk.util import ngrams
from nltk.tokenize import word_tokenize


def create_ngrams(text, n):
    tokens = word_tokenize(text)
    n_grams = list(ngrams(tokens, n))
    return n_grams


df['bigrams'] = df['cleaned_text'].apply(lambda x: create_ngrams(x, 2))
df['trigrams'] = df['cleaned_text'].apply(lambda x: create_ngrams(x, 3))
```

We also inspected the most common bigrams and trigrams in disaster-related tweets and non-disaster related tweets:

```py
from collections import Counter


disaster_bigrams = df[df['target'] == 1]['bigrams']
non_disaster_bigrams = df[df['target'] == 0]['bigrams']

disaster_bigram_counts = Counter([gram for ngram_list in disaster_bigrams for gram in ngram_list])
non_disaster_bigram_counts = Counter([gram for ngram_list in non_disaster_bigrams for gram in ngram_list])


print("Most common n-grams in disaster-related tweets:")
print(disaster_bigram_counts.most_common(10))


print("\nMost common n-grams in non-disaster tweets:")
print(non_disaster_bigram_counts.most_common(10))
```

Result for bigrams:

```py
Most common n-grams in disaster-related tweets:
[(('suicide', 'bomber'), 78), (('northern', 'california'), 53), (('california', 'wildfire'), 44), (('home', 'razed'), 37), (('suicide', 'bombing'), 36), (('oil', 'spill'), 36), (('latest', 'home'), 36), (('razed', 'northern'), 36), (('severe', 'thunderstorm'), 35), (('70', 'year'), 34)]

Most common n-grams in non-disaster tweets:
[(('i', 'am'), 173), (('do', 'not'), 89), (('can', 'not'), 51), (('you', 'are'), 50), (('youtube', 'video'), 33), (('liked', 'youtube'), 32), (('i', 'have'), 26), (('cross', 'body'), 26), (('body', 'bag'), 26), (('going', 'to'), 25)]
```

Did the same for trigrams

```py

disaster_trigrams = df[df['target'] == 1]['trigrams']
non_disaster_trigrams = df[df['target'] == 0]['trigrams']


disaster_trigram_counts = Counter([gram for ngram_list in disaster_trigrams for gram in ngram_list])
non_disaster_trigram_counts = Counter([gram for ngram_list in non_disaster_trigrams for gram in ngram_list])


print("Most common trigrams in disaster-related tweets:")
print(disaster_trigram_counts.most_common(10))


print("\nMost common trigrams in non-disaster tweets:")
print(non_disaster_trigram_counts.most_common(10))
```

Result for trigrams:

```py
Most common trigrams in disaster-related tweets:
[(('northern', 'california', 'wildfire'), 38), (('latest', 'home', 'razed'), 36), (('home', 'razed', 'northern'), 36), (('razed', 'northern', 'california'), 35), (('watch', 'airport', 'get'), 34), (('airport', 'get', 'swallowed'), 34), (('get', 'swallowed', 'sandstorm'), 34), (('swallowed', 'sandstorm', 'minute'), 34), (('suicide', 'bomber', 'detonated'), 34), (('pkk', 'suicide', 'bomber'), 32)]

Most common trigrams in non-disaster tweets:
[(('liked', 'youtube', 'video'), 32), (('i', 'am', 'going'), 18), (('pick', 'fan', 'army'), 18), (('cross', 'body', 'bag'), 17), (('reddits', 'new', 'content'), 11), (('new', 'content', 'policy'), 11), (('low', 'selfimage', 'take'), 10), (('selfimage', 'take', 'quiz'), 10), (('deluged', 'invoice', 'make'), 10), (('likely', 'rise', 'top'), 10)]
```

### 2.5. Sentiment analysis

We implemented sentiment analysis, to investigate if the sentiment of the tweet would be relevant to classifying the tweet as disaster-related or not. We did this using the `SentimentIntensityAnalyzer` from `nltk`:

```py
from nltk.sentiment.vader import SentimentIntensityAnalyzer
def analyze_sentiment_vader(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(text)
    if sentiment_scores['compound'] >= 0.05:
        return "Positive"
    elif sentiment_scores['compound'] <= -0.05:
        return "Negative"
    else:
        return "Neutral"

df['sentiment'] = df['cleaned_text'].apply(analyze_sentiment_vader)

```

Based on this code, it seemed like it was a significant difference in the sentiment between the disaster-related and not disaster-related tweets. We then looked at the counts for each sentiment, and saw there was a distribution imbalance where the majority of disaster-related tweets had a neutral or negative sentiment, whilst there was way more positive sentiment amongst the "non-disaster-related" tweets. 

```py

from scipy import stats


df['sentiment_numeric'] = df['sentiment'].map({'Negative': 0, 'Neutral': 1, 'Positive': 2})

disaster_group = df[df['target'] == 1]['sentiment_numeric']
not_disaster_group = df[df['target'] == 0]['sentiment_numeric']

t_statistic, p_value = stats.ttest_ind(disaster_group, not_disaster_group, equal_var=False)

print(p_value)
if p_value < 0.05:
    print("There is a significant difference in sentiment between disaster and not disaster tweets.")
else:
    print("There is no significant difference in sentiment between disaster and not disaster tweets.")
```

### 2.6. TF-IDF vectorization

This is discussed further in the [modelling section](#4-implement-basic-modelling-methods).

## 3. Select features

...

## 4. Implement basic modelling methods

Before implementing any particular basic modelling method, we vectorized the features. In particular, we used `TfidfVectorizer`, which is equivalent to using `CountVectorizer`, followed by `TfidfTransformer`. Using `TfidfVectorizer` converts a collection of features to a matrix of TF-IDF features.

We used TF-IDF vectorization for the following reasons:

- **Dimensionality Reduction**. Raw text data is highly dimensional, as each unique word or token can be a dimension. By using TF-IDF, the importance of each word is quantified, allowing for potential dimensionality reduction by removing words with very low TF-IDF scores, as they are likely to be less informative.

- **Normalization**. The TF-IDF score balances out the term frequency (TF) with its inverse document frequency (IDF). This means that words that appear frequently in a single document but not in many documents throughout the corpus will get a high score. On the other hand, common words that appear frequently in many documents (like "and", "the", etc.) will get a low score. This normalization process ensures that less emphasis is placed on common words that do not carry much meaningful information.

- **Clean code**. Using `TfidfVectorizer` is equivalent to using `CountVectorizer` followed by `TfidfTransformer`. By using the `TfidfVectorizer`, you combine these two steps, which can be more efficient and require less code.

Next, the data was split into a train and test set, with a test size of `0.3`.

### 4.1. Based on the selected features

#### 4.1.1. Logistic regression

The first basic modelling method used was logistic regression. Logistic regression was chosen due to the following reasons:

- **Binary Outcome**. The target variable (`target`) is binary, with values either `0` (Not Relevant) or `1` (Relevant). Logistic regression is designed to model the probability of a binary outcome, making it a suitable choice for this dataset.

- **Linear Decision Boundary**. Logistic regression assumes a linear relationship between the log-odds of the outcome and the predictor variables. If the relationship between the vectorized text features and the target variable is approximately linear in the log-odds space, logistic regression will perform well.

##### Results of logistic regression

The logistic regression model achieved an accuracy of approximately 91.03% on the test dataset. Analyzing the classification report, we observe the following details for each class:

For class `0` (Not Relevant):

- **Precision**: 90% of the instances predicted as class `0` were correctly identified by the model.
- **Recall**: Out of all the actual instances of class `0`, the model successfully recognized 95% of them.
- **F1-Score**: The F1-score for class `0` is 92%.

For class `1` (Relevant):

- **Precision**: 93% of the instances predicted as class `1` were correctly identified by the model.
- **Recall**: Out of all the actual instances of class `1`, the model successfully recognized 86% of them.
- **F1-Score**: The F1-score for class `1` stands at 89%.

On a broader scale, the macro average indicates an average precision, recall, and F1-score all of approximately 91%. The weighted average, which considers the number of instances in each class, also suggests an average precision, recall, and F1-score all around 91%.

Inspecting the confusion matrix:

- 806 instances were correctly classified as class `0`, while 41 instances were incorrectly predicted as class `0` when they were actually class `1`.
- Conversely, 554 instances were correctly labeled as class `1`, but 93 instances were wrongly classified as class `1` when they were in fact class `0`.

#### 4.1.2. Support Vector Machine (SVM)

The second basic modelling method used was a support vector machine (SVM). SVM were chosen due to the following reasons:

- **High Dimensionality**. Text data, when transformed into a numerical format like TF-IDF or Count Vectorization, often results in a high-dimensional feature space, as each unique word or token becomes a dimension. SVMs are designed to handle high-dimensional data effectively.

- **Effective in Non-linear Problems**. While the basic SVM is a linear classifier, by applying the kernel trick, SVMs can solve non-linear classification problems. This means that if the boundary between 'Relevant' and 'Not Relevant' is not linear in the feature space, SVM can still find an optimal boundary.

- **Clear margin of separation**. SVMs aim to find the hyperplane that has the maximum margin between two classes. This often results in better generalization to unseen data, reducing the risk of overfitting.

##### Results of support vector machine

The Support Vector Machine model achieved an accuracy of approximately 91.43% on the test dataset. Delving deeper into the classification report, we can discern the following details for each class:

For class `0` (Not Relevant):

- **Precision**: Of all the instances the model predicted as class `0`, 91% were correctly identified.
- **Recall**: Out of all the true instances of class `0` in the test set, the model successfully classified 94% of them.
- **F1-Score**: The F1-score for this class is recorded at 93%.

For class `1` (Relevant):

- **Precision**: The model was accurate in predicting 92% of the instances as class `1`.
- **Recall**: It managed to detect 88% of the actual instances of class `1`.
- **F1-Score**: The F1-score for this class is recorded at 90%.

Zooming out to a more generalized view, the macro average indicates an average precision, recall, and F1-score of approximately 92%, 91%, and 91% respectively. The weighted average indicates an average precision, recall, and F1-score all rounding to about 91%.

Examining the confusion matrix:

- The model correctly classified 799 instances as class `0`, while mistakenly predicting 48 instances as class `0` which were actually class `1`.
- On the other hand, 567 instances were accurately labeled as class `1`. However, 80 instances were incorrectly classified as class `1` when they truly belonged to class `0`.

### 4.2. With hyperparameter tuning

For hyperparameter tuning, we used `GridSearchCV` to find the optimal hyperparameters for each model.

#### 4.2.1. Logistic regression

For logistic regression, we performed a grid search to identify the optimal combination of hyperparameters. We considered varying levels of regularization strength by adjusting the `C` parameter. The type of penalization was alternated between `l1` and `l2` using the `penalty` parameter. Additionally, different optimization algorithms were tested using the `solver` parameter, including `liblinear` and `saga`. The grid search cross-validated the model's performance for each combination.

The accuracy obtained after hyperparameter tuning (approximately 91.36%) is **slightly higher** than the accuracy from the initial logistic regression model, which was around 91.03%. In particular, it was the change of `C` and `lbfgs` parameter: from `C=1.0` to `C=2.0` and from `lbfgs` to `liblinear`. The hyperparameter tuning process for logistic regression managed to find a set of parameters that improved the model's accuracy slightly. This highlights the importance of such tuning processes.

#### 4.2.2. Support Vector Machine (SVM)

For the SVM, the regularization parameter `C` was adjusted to analyze the decision boundary. We explored different types of kernels using the `kernel` parameter, including `linear` and `rbf`. The coefficient for the kernel function, `gamma`, was also tuned, toggling between `scale` and `auto`. The grid search cross-validated the model's performance for each combination.

The accuracy obtained after hyperparameter tuning (approximately 91.63%) is **slightly higher** than the accuracy from the initial SVM model, which was around 91.43%. In particular, it was the change of `kernel` parameter: from `linear` to `rbf`. The hyperparameter tuning process for SVM managed to find a set of parameters that improved the model's accuracy slightly. This highlights the importance of such tuning processes.

### 4.3. Explaining the reasoning behind the hyperparameter tuning

#### 4.3.1. Logistic regression

- **Regularization (`C`)**. The hyperparameter `C` is the inverse of regularization strength. Smaller values of `C` indicate stronger regularization, which can prevent overfitting, but might also lead to underfitting if too strong. On the other hand, larger values of `C` mean weaker regularization, which might fit the training data more closely but risk overfitting. By tuning `C`, we aim to strike the right balance between underfitting and overfitting.

- **Penalty (`penalty`)**. The `penalty` parameter specifies the type of regularization to be applied. The most common types are L1 and L2 regularization. L1 regularization can lead to some coefficients becoming exactly zero, effectively selecting a simpler model with fewer features. On the other hands, L2 regularization tends to shrink coefficients but not set them to zero. The choice between L1 and L2 can impact the model's performance and interpretability.

- **Solver (`solver`)**. The `solver` parameter dictates the optimization algorithm to be used. Different solvers might converge at different rates and can have varying levels of accuracy, depending on the nature of the data and the problem. Some solvers work better with certain penalty types, making it important to consider the combination of `solver` and `penalty` when tuning the model.

#### 4.3.2. Support Vector Machine (SVM)

- **Regularization (`C`)**. Like in Logistic Regression, the `C` parameter in SVM controls the trade-off between obtaining a wider margin and classifying the training points correctly. A smaller `C` creates a wider margin, which might misclassify more points, while a larger `C` results in a narrower margin, making the model fit more closely to the training data. Adjusting `C` helps balance between overfitting (high variance) and underfitting (high bias).

- **Kernel (`kernel`)**. The kernel trick allows SVM to create non-linear decision boundaries. The choice of kernel can significantly impact the model's performance. The `rbf` (Radial Basis Function) kernel, on the other hand, can create more complex, non-linear boundaries.

- **Kernel Coefficient (`gamma`)**. Adjusting `gamma` can help control the shape and complexity of the decision boundary, impacting the model's generalization capability.

In summary, hyperparameter tuning for both Logistic Regression and SVM involves adjusting key parameters that control model complexity, regularization, and the nature of the decision boundary. The goal is to find the optimal combination of hyperparameters that results in the best performance on unseen data, ensuring that the model is both accurate and generalizable.

## 5. Comparing modelling methods

> TODO: Write a paragraph about the comparison of the modelling methods.

### 5.1. Selecting the modelling methods

...

### 5.2. Reasoning

...

## 6. Designing a pipeline

> Apart from that, design (not develop) your pipeline based on one Advanced modelling techniques.  Justify your choices (10%).

The advanced modelling technique we have chosen is Word2Vec embeddings coupled with a Random Forest classifier.

### Pipeline Design
Importing Data:

Similar to the previous pipeline, we start by importing the necessary data.

**Preprocessing the Data:**

We clean the tweets like the current pipeline.

**Word Embeddings with Word2Vec:**

Instead of using tf-idf vectorization, we use Word2Vec to generate word embeddings for each tweet. This is a more sophisticated approach that capture the semantic meaning of words.

We will first pre-trained a Word2Vec model on a large document corpus using the `gensim`-library. Then, we will use the pre-trained model to generate embeddings for each tweet in our dataset. 

**Training a Random Forest Classifier:**

Train a Random Forest classifier on the Word2Vec embeddings from the training set. Fine-tune the hyperparameters of the classifier to achieve the best possible performance.

**Evaluation:**

Use the trained Random Forest model to classify tweets in the test set. Evaluate the model's performance using F1-score.

### Justification

Word2Vec captures the semantic meaning of a word by representing it based on the context the word appears. The assumption is that words with similar meanings appear in similar contexts. This means words with similar meanings will have embeddings that are close in the vector space.

**Simplicity:**

Both Word2Vec and Random Forest are relatively simple models that are easy to implement and interpret. This makes the pipeline possible for us to actually understand. 

We considered these alternatives to Word2Vec:
- Glove
- FastText
- Universal Sentence Encoder by Google

FastText generates embeddings for subwords, so even if a word hasn't been seen during training, it can create a representation for it based on its subwords. This feature can be particularly useful for tweets, which often have misspellings, abbreviations, and neologisms.

However, we found Word2Vec to be the simplest and most intuitive approach. If we find that Word2Vec doesn't perform well, we can consider using FastText instead.

For the classifier, we considered using XGBoost or a neural network with x layers. However, we want to start with a simple model like Random Forest and then move on to more complex models if necessary.

In the end, we considered using BERT for the whole pipeline. This would probably result in very good scores using BERTs transfer learning structure. However, we found the transformer architecture and attention mechanism very complex to understand. Altough it is easy to implement a solution applying a small BERT model in code, we would not be able to understand how it works. 

Random Forest inherently has mechanisms (like bootstrapping and feature randomness) that can reduce the risk of overfitting, especially when dealing with high-dimensional data like Word2Vec embeddings.

Potential Improvements / Alternatives
BERT and its Variants:

As previously mentioned, BERT, along with its variants like DistilBERT and RoBERTa, can offer state-of-the-art performance in text classification tasks. By fine-tuning a pre-trained BERT model on our dataset, we might achieve better accuracy than Word2Vec + Random Forest.
Combining Multiple Embeddings:

Instead of just relying on Word2Vec, combining embeddings from multiple sources (e.g., FastText, GloVe) might enhance the feature set for the classifier.
Deep Learning Models:

Neural network architectures, especially recurrent (like LSTMs) or transformer-based models, can handle sequential data like tweets more effectively. Coupling Word2Vec with a neural network might further improve performance.
Ensemble Methods:

Instead of just using Random Forest, an ensemble of multiple classifiers might yield better results. Methods like stacking can be explored to combine predictions from various models.

### Advanced Modelling Technique

The advanced modelling technique we have chosen is transfer learning using the BERT model. BERT, or Bidirectional Encoder Representations from Transformers, is a family of deep learning language models developed by Google. BERT is a pre-trained model that can be fine-tuned for specific tasks, i.e. transfer learning. In our case, the knowledge gained from pre-training BERT on vast amounts of unlabeled text for a different source task can be transferred to the target task of classifying disaster-related tweets. 

### Pipeline Design

1. Importing data

We will import the data like in our current pipeline. 

2. Preprocessing the data

BERT does some preprocessing of the data itself. However, we will still do some preprocessing spesific to the domain of tweets. We will remove links, line breaks, extra spaces, special characters and punctuation. 

When we have the data preprocessed, we will use the BERT model to extract features from the text data. First we will load a pre-trained BERT classifier, configured to classify text into two classes. 

The next step of the pipeline is fine tuning the model on our dataset. 

In the final step of the pipeline, we will use the fine-tuned BERT model to classify the tweets in the test set.

### Justification
1. Bidirectional Contextual Understanding:
Traditional models like LSTM and GRU read text sequentially, either from left to right or right to left. In contrast, BERT comprehensively analyzes text bidirectionally, ensuring it captures context from both directions.

Tweets, due to their brevity, often contain condensed information where the context of a single word can be influenced by words both before and after it. BERT's bidirectional approach ensures nuanced understanding, making it especially suitable for tweet classification.

2. Proven Performance Across Diverse NLP Tasks:
Since its introduction, BERT has consistently set benchmark performances on various NLP challenges, ranging from question answering to sentiment analysis.

The wide-ranging success of BERT on different tasks suggests its adaptability and robustness. By selecting BERT, we're leveraging a model with a proven track record, increasing our chances of achieving high accuracy on our specific task of classifying disaster-related tweets.

3. Transfer Learning and Pre-training Advantages:
BERT is pre-trained on vast amounts of text data, learning a wealth of language patterns, structures, and nuances. This pre-trained knowledge can be fine-tuned on a smaller, task-specific dataset, making the model training faster and potentially more accurate.

BERT's capability to transfer knowledge from its extensive pre-training to our specific task can lead to strong performance even with only our limited training data.

4. BERT's Versatility:
Description: BERT isn't just a one-size-fits-all model. It has various versions tailored to different needs, from smaller, faster models like DistilBERT to more extensive, potentially more accurate versions like RoBERTa.

The flexibility in choosing a BERT variant allows us to strike a balance between computational efficiency and model performance. For instance, if we need faster predictions (e.g., in a real-time monitoring system), we might opt for DistilBERT. If we seek higher accuracy and have the computational resources, RoBERTa could be the choice.

### Potential improvements / alternatives
BERT isn't just a one-size-fits-all model. It has various versions tailored to different needs, from smaller, faster models like DistilBERT to more extensive, potentially more accurate versions like RoBERTa.

The flexibility in choosing a BERT variant allows us to strike a balance between computational efficiency and model performance. For instance, if we need faster predictions (e.g., in a real-time monitoring system), we might opt for DistilBERT. If we seek higher accuracy and have the computational resources, RoBERTa could be the choice.




## 7. Individual contributions

...

## 8. Personalized feedback form

1. Feature extraction: (YES/NO)
2. Feature selection: (YES/NO)
3. Choice of basic modelling methods: (YES/NO)
4. Choice of performance metrics: (YES/NO)
5. Comparison of modelling methods: (YES/NO)
6. Advanced Pipeline Design: (YES/NO)

### 8.1. Other questions

> TODO: Add other quesions if they arise

## 9
