# IT3212 - Assignment 2

Written and developed by Haakon Tideman Kanter, Henrik Skog, Mattis Czternasty Hembre, Max Gunhamn, Sebastian Sole, and Magnus Rødseth. 

## 1. Implement the preprocessing

We started by removing features, according to what was outlined in assignment 1. Additionally, we removed the `location` feature to preserve the privacy of the users and because it is not particularly relevant to determing whether a tweet is related to a disaster. Next, we removed rows with a confidence threshold below `1.0`.

Next, we preprocessed the textual data. We cleaned up the `keyword` column, ensuring that all keywords were lower case, and that all keywords were separated by a single space. For comparison, some keywords were on the format `airplane%20accident`, which should be `airplane accident`.

Furthermore, we cleaned the `text` column. The first part of this included removing links, line breaks, extra spaces, special characters and punctuation. Next, we removed English stopwords. Finally, we lemmatized the text.

When handling categorical data, we removeds rows with a `choose_one` value of `Can't decide`, according to what was outlined in assignment 1. Next, we mapped the `choose_one` values `Relevant` and `Not Relevant` to `1` and `0`, respectively. This was stored in a new feature, called `target`.

Moreover, we removed duplicated rows with regards to the `text` column, as outlined in assignment 1.

## 2. Extract features

### 2.1 Text feature
Machine learning models are not able to understand raw text, so the text must be converted into a numerical representation. In this delivery we have chosen to do this this by using tf-idf vectorization. 

TD-IDF, or term frequency–inverse document frequency, is an extenstion of the bag-of-words model. The bag-of-words algorithm represents a document by the occurrence of words within a it. You first build a vocabulary by looking at the set of all words used in the corpus. The amount of words in the vocabulary maps directly to the number of features it produces for a given document. The result of embedding a document with bag-of-words is simply a one hot encoding of the ocurrences of the words in the vocabulary in the document.

TF-IDF extends the bag-of-words model by also including how important a word is in the context of the entire corpus. The importance of a word increases proportionally to the number of times a word appears in the document, but is offset by the frequency of the word in the corpus. This means that words that appear frequently in a single document but not in many documents throughout the corpus will get a high score. On the other hand, common words that appear frequently in many documents (like "and", "the", etc.) will get a low score. This normalization process ensures that less emphasis is placed on common words that do not carry much meaningful information.

The implementation is done using the `TfidfVectorizer` from `sklearn.feature_extraction.text`. The `TfidfVectorizer` converts a collection of raw documents to a matrix of TF-IDF features. The `TfidfVectorizer` is equivalent to using `CountVectorizer` followed by `TfidfTransformer`. Using the `TfidfVectorizer` is more efficient and requires less code.

However, by converting the text into a numerical representation, we lose a lot of information. For example, the order of the words is lost. TF-IDF is also not able to capture the semantic meaning of the words.  To combat this, we try to extract some additional features from the text. 

### 2.1. Text length

### 2.1. Counting hashtags

Hashtag counts were extracted, as we suspected tweets related to a disaster might contain more hashtags than a tweet which does not. 
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

We implemented sentiment analysis to investigate if the sentiment of the tweet would be relevant to classifying the tweet as disaster-related or not. We did this using the `SentimentIntensityAnalyzer` from `nltk`:

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

Based on this code, it seemed like it was a significant difference in the sentiment between the disaster-related and not disaster-related tweets. We then looked at the counts for each sentiment, and saw there was a distribution imbalance where the majority of disaster-related tweets had a negative sentiment, whilst there was way more positive sentiment amongst the "non-disaster-related" tweets. 

We also tested with "text" instead of "cleaned_text" and the results were still the same (significant difference).


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

## 3. Select features

These are the features that we have to choose from:

| Feature        | Type | Description                                |
|----------------|------|--------------------------------------------|
| keyword        | ?    | Specific keyword associated with the text. |
| text_length    | ?    | The length of the text in characters.      |
| hashtag_count  | ?    | Number of hashtags used in the text.       |
| mention_count  | ?    | Number of mentions (@) in the text.        |
| has_url        | ?    | Boolean indicating if the text has a URL.  |
| sentiment      | ?    | Sentiment value of the text.               |
| bigrams        | ?    | Two consecutive words in the text.         |
| trigrams       | ?    | Three consecutive words in the text.       |


![Alt text](image-4.png)
We see that the distribution of the text-length is fairly similar for both disaster and non-disaster tweets. The difference may be significant, but it is not very large. We decided to keep the feature, as it might still be useful.

![Alt text](image-3.png)
The hashtag count also has a similar distribution for both disaster and non-disaster tweets. The difference is not very large, but we decided to keep the feature.

![Alt text](image-2.png)
The mention count has a similar distribution for both disaster and non-disaster tweets. The difference is not very large, but we decided to keep the feature.

![Alt text](image-1.png)
The url count has quite a large difference between disaster and non-disaster tweets. We can see that the majority of disaster-related tweets contain a url, whilst the majority of non-disaster tweets do not contain a url. This is of course not enough to conclude that the tweet is disaster-related, but it might be a useful feature.

## 4. Modelling
## 4.1 Logistic regression
The first basic modelling method used was logistic regression. Logistic regression was chosen due to the following reasons:

- **Binary Outcome**. The target variable (`target`) is binary, with values either `0` (Not Relevant) or `1` (Relevant). Logistic regression is designed to model the probability of a binary outcome, making it a suitable choice for this dataset.

- **Linear Decision Boundary**. Logistic regression assumes a linear relationship between the log-odds of the outcome and the predictor variables. If the relationship between the vectorized text features and the target variable is approximately linear in the log-odds space, logistic regression will perform well.

### Results
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

#### Hyperparameter tuning
For hyperparameter tuning, we used `GridSearchCV` to find the optimal hyperparameters for each model.

For logistic regression, we performed a grid search to identify the optimal combination of hyperparameters. We considered varying levels of regularization strength by adjusting the `C` parameter. The type of penalization was alternated between `l1` and `l2` using the `penalty` parameter. Additionally, different optimization algorithms were tested using the `solver` parameter, including `liblinear` and `saga`. The grid search cross-validated the model's performance for each combination.

The accuracy obtained after hyperparameter tuning (approximately 91.36%) is **slightly higher** than the accuracy from the initial logistic regression model, which was around 91.03%. In particular, it was the change of `C` and `lbfgs` parameter: from `C=1.0` to `C=2.0` and from `lbfgs` to `liblinear`. The hyperparameter tuning process for logistic regression managed to find a set of parameters that improved the model's accuracy slightly. This highlights the importance of such tuning processes.

## 4.2. Support Vector Machine (SVM)
The second basic modelling method used was a support vector machine (SVM). SVM were chosen due to the following reasons:

- **High Dimensionality**. Text data, when transformed into a numerical format like TF-IDF or Count Vectorization, often results in a high-dimensional feature space, as each unique word or token becomes a dimension. SVMs are designed to handle high-dimensional data effectively.

- **Effective in Non-linear Problems**. While the basic SVM is a linear classifier, by applying the kernel trick, SVMs can solve non-linear classification problems. This means that if the boundary between 'Relevant' and 'Not Relevant' is not linear in the feature space, SVM can still find an optimal boundary.

- **Clear margin of separation**. SVMs aim to find the hyperplane that has the maximum margin between two classes. This often results in better generalization to unseen data, reducing the risk of overfitting.

#### Hyperparameter tuning

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


### 4.3.1. Logistic regression
### 4.3.2. Support Vector Machine (SVM)


## 5. Comparing modelling methods

> TODO: Write a paragraph about the comparison of the modelling methods.

### 5.1. Selecting the modelling methods

...

### 5.2. Reasoning

...

## 6. Designing a pipeline

> Apart from that, design (not develop) your pipeline based on one Advanced modelling techniques. Justify your choices (10%).

The advanced modelling technique we have chosen are Word2Vec embeddings coupled with a Random Forest classifier. Both Word2Vec and Random Forest are relatively simple models that are easy to implement and interpret. This makes them suitable for this project.

### Word2Vec
Word2vec is a technique for natural language processing from 2013. The word2vec algorithm uses a neural network model to learn word associations from a large corpus of text. Once trained, it can detect synonymous words or suggest additional words for a partial sentence. In the same way as tf-idf (our current approach), word2vec represents each document with a vector. The vectors are chosen such that they capture the semantic and syntactic qualities of the words.

We will use the library `gensim` to train a Word2Vec model on a large corpus of text. Then, we will use the pre-trained model to generate embeddings for each tweet in our dataset. The embeddings will be used as features in our Random Forest classifier.

There exists many similar models to Word2Vec, such as Glove, FastText, Universal Sentence Encoder and BERT by Google. We chose Word2Vec because it is simple compared to the alternatives, and it is still a big step up from tf-idf.

### Random Forest Classifier
The Random Forest classifier is an ensemble learning method. It is an ensemble of decision trees, where each tree is trained on a random subset of the training data. The final prediction is made by aggregating the predictions of all the trees. Random Forests are robust to overfitting and can handle high-dimensional data effectively.

These are two important techniques the algorithm uses:

Bagging (Bootstrap Aggregating): For each tree, a random sample of the data is drawn with replacement, creating diverse sets of training data. This process introduces variability among the trees, ensuring individual trees are trained on slightly different versions of the data.
Feature Randomness: During tree construction, instead of considering all features for a split, a random subset of features is chosen. This adds another layer of randomness and reduces the correlation between individual trees.

Similarly to the reason we chose Word2Vec, we chose Random Forest because it is a simple and intuitive approach compared to other ensemble methods like XGBoost or neural networks.

### Side note about transformer models
We heavily considered using BERT for the whole pipeline. This would probably result in very good scores using BERTs transfer learning structure. However, we found the transformer architecture and attention mechanism very complex to understand. Altough it is easy to implement a solution applying a small BERT model in code, we would not be able to understand how it works. 

Another approach is to use a LLM like gpt-3. This would result in even better scores, but the same problem as with BERT applies. We would not be able to understand how it works. Using a LLM would also be very expensive, as we would have to pay for the API.

### Pipeline Design

**Importing Data:**

Similar to the previous pipeline, we start by importing the necessary data.

**Preprocessing the Data:**

We clean the tweets like the current pipeline.

**Feature extraction using Word2Vec:**

Instead of using tf-idf vectorization, we use Word2Vec to generate word embeddings for each tweet.  We will first pre-trained a Word2Vec model on a large document corpus using the `gensim`-library. Then, we will use the pre-trained model to generate embeddings for each tweet in our dataset.

**Modelling using a Random Forest Classifier:**

Train a Random Forest classifier on the Word2Vec embeddings from the training set. 

**Evaluation:**
Use the trained Random Forest model to classify tweets in the test set. Evaluate the model's performance using F1-score. Fine-tune the hyperparameters of the classifier using grid search to achieve the best possible performance.


## 7. Individual contributions
Henrik Skog: Plotting, preprocessing, feature extraction, modelling, pipeline design, report writing

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
