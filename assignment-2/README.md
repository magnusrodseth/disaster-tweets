# IT3212 - Assignment 2

Written and developed by Haakon Tideman Kanter, Henrik Skog, Mattis Czternasty Hembre, Max Gunhamn, Sebastian Sole, and Magnus Rødseth.

## 1. Implement the preprocessing

We started by removing features, according to what was outlined in assignment 1. Next, we removed rows with a confidence threshold below `1.0`.

Next, we preprocessed the textual data. We cleaned up the `keyword` column, ensuring that all keywords were lower case, and that all keywords were separated by a single space. For comparison, some keywords were on the format `airplane%20accident`, which should be `airplane accident`.

Furthermore, we cleaned the `text` column. The first part of this included removing links, line breaks, extra spaces, special characters and punctuation. Next, we removed English stopwords. Finally, we lemmatized the text.

When handling categorical data, we removed rows with a `choose_one` value of `Can't decide`, according to what was outlined in assignment 1. Next, we mapped the `choose_one` values `Relevant` and `Not Relevant` to `1` and `0`, respectively. This was stored in a new feature, called `target`.

Moreover, we removed duplicated rows with regards to the `text` column, as outlined in assignment 1.

## 2. Extract features


- `hashtag_count`

Hashtag counts were extracted, as we suspected tweets related to a disaster might contain more hashtags than a tweet which does not. This also had to be done before preprocessing, as we remove hashtags in the preprocessing step.

```py
# Extract the number of hashtags
df['hashtag_count'] = df['text'].apply(lambda x: len([c for c in str(x) if c == '#']))
```

- `mention_count`
  
Mention count is a feature where we see how many mentions there are in a tweet. This means how many other twitter users are mentioned in the tweet. 
This was extracted because we thought there might be a connection betweeen how many users were tagged and if the tweets are disaster-related. The group thought that disaster-related tweets would have more mentions than tweets that were not disaster related.

```py
# Extract the number of mentions
df['mention_count'] = df['text'].apply(lambda x: len([c for c in str(x) if c == '@']))
```

- `has_url`

Has_url was extracted because tweets related to a disaster might point to an online resource where one can find more info about the situation. The column is 1 if the tweet contains a url and 0 if not. 
The has_url feature had to be done before preprocessing, as we remove links in the preprocessing step.  

```py
# Extract the `has_url` feature
df['has_url'] = df['text'].apply(lambda x: 1 if 'http' in str(x) else 0)
```

- `n-grams`

N-grams was a feature we wanted to look at, as we suspected there could be a correlation between bigrams/ trigrams and if the tweet was disaster-related or not, as bigrams and trigrams have more context than just single words. We found the most used bigrams and trigrams, both for disaster-related tweets and non-disaster related tweets. 

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

To check which bigrams were most common, both in disaster-related tweets and non-disaster related tweets (according to choose_one column)
```py
from collections import Counter
df['is_disaster'] = df['choose_one'].map({"Relevant": 1, "Not Relevant": 0})

disaster_bigrams = df[df['is_disaster'] == 1]['bigrams']
non_disaster_bigrams = df[df['is_disaster'] == 0]['bigrams']

disaster_bigram_counts = Counter([gram for ngram_list in disaster_bigrams for gram in ngram_list])
non_disaster_bigram_counts = Counter([gram for ngram_list in non_disaster_bigrams for gram in ngram_list])

# Example: Print the most common n-grams in disaster-related tweets
print("Most common n-grams in disaster-related tweets:")
print(disaster_bigram_counts.most_common(10))

# Example: Print the most common n-grams in non-disaster tweets
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
df['is_disaster'] = df['choose_one'].map({"Relevant": 1, "Not Relevant": 0})

# Create separate lists of trigrams for disaster and non-disaster tweets
disaster_trigrams = df[df['is_disaster'] == 1]['trigrams']
non_disaster_trigrams = df[df['is_disaster'] == 0]['trigrams']

# Count the frequency of trigrams in both categories
disaster_trigram_counts = Counter([gram for ngram_list in disaster_trigrams for gram in ngram_list])
non_disaster_trigram_counts = Counter([gram for ngram_list in non_disaster_trigrams for gram in ngram_list])

# Example: Print the most common trigrams in disaster-related tweets
print("Most common trigrams in disaster-related tweets:")
print(disaster_trigram_counts.most_common(10))

# Example: Print the most common trigrams in non-disaster tweets
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




- `sentiment analysis`

We implemented sentiment analysis, to investigate if the sentiment of the tweet would be relevant to classifying the tweet as disaster-related or not. We did this through nltks "vader":

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

When we checked, it was under a 5% connection between sentiment of the tweet and disaster relation. Therefore, we chose to drop this column. (IKKE GJORT ENDA)  We tested this both before and after preprocessing of the text, but the results were the same. 

```py
from scipy import stats
disaster_group = df[df['choose_one'] == "Disaster"]
not_disaster_group = df[df['choose_one'] == "Not Disaster"]

t_statistic, p_value = stats.ttest_ind(disaster_group['sentiment'], not_disaster_group['sentiment'], equal_var=False)

if p_value < 0.05:
    print("There is a significant difference in sentiment between disaster and not disaster tweets.")
else:
    print("There is no significant difference in sentiment between disaster and not disaster tweets.")
```

- `TF_IDF vectorizing`

Will be discussed in [implement basic modelling methods](#4-implement-basic-modelling-methods). 

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

> Based on one advanced modelling technique. Justify choices.

- Ensamble learning
  - Max voting (generally used for classification problems)
  - Average (Can be used for making predictions in regression problems, or while calculating probabilities for classification problems)
- Stacking
- Blending
- Bagging
  - Random forest telles her som en advanced modelling technique, så kanskje vi ikke skal ha den med på basic.
- Boosting

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
