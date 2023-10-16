# IT3212 - Assignment 2

Written and developed by Haakon Tideman Kanter, Henrik Skog, Mattis Hembre, Max Gunhamn, Sebastian Sole, and Magnus Rødseth.

## 1. Implement the preprocessing

We started by removing features, according to what was outlined in assignment 1. Next, we removed rows with a confidence threshold below `1.0`.

Next, we preprocessed the textual data. We cleaned up the `keyword` column, ensuring that all keywords were lower case, and that all keywords were separated by a single space. For comparison, some keywords were on the format `airplane%20accident`, which should be `airplane accident`.

Furthermore, we cleaned the `text` column. The first part of this included removing links, line breaks, extra spaces, special characters and punctuation. Next, we removed English stopwords. Finally, we lemmatized the text.

When handling categorical data, we removed rows with a `choose_one` value of `Can't decide`, according to what was outlined in assignment 1. Next, we mapped the `choose_one` values `Relevant` and `Not Relevant` to `1` and `0`, respectively. This was stored in a new feature, called `target`.

Moreover, we removed duplicated rows with regards to the `text` column, as outlined in assignment 1.

## 2. Extract features

- `hashtag_count`
- `mention_count`
- `has_url`
- n-grams

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

- **Binary Outcome**. The target variable (`target`) is binary, with values either `0` (Not Relevant) or `1` (Relevant). Logistic regression is specifically designed to model the probability of a binary outcome, making it a suitable choice for this dataset.

- **Linear Decision Boundary**. Logistic regression assumes a linear relationship between the log-odds of the outcome and the predictor variables. If the relationship between the vectorized text features and the target variable is approximately linear in the log-odds space, logistic regression will perform well.

##### Results of logistic regression

The logistic regression model achieved an accuracy of approximately 91.03% on the test dataset. Analyzing the classification report, we observe the following details for each class:

For class `0` ("Not Relevant"):

- **Precision**: 90% of the instances predicted as class `0` were correctly identified by the model.
- **Recall**: Out of all the actual instances of class `0`, the model successfully recognized 95% of them.
- **F1-Score**: The F1-score for class `0` is 92%.

For class `1` ("Relevant"):

- **Precision**: 93% of the instances predicted as class `1` were correctly identified by the model.
- **Recall**: Out of all the actual instances of class `1`, the model successfully recognized 86% of them.
- **F1-Score**: The F1-score for class `1` stands at 89%.

On a broader scale, the macro average—which gives equal weight to each class—indicates an average precision, recall, and F1-score all of approximately 91%. The weighted average, which considers the number of instances in each class, also suggests an average precision, recall, and F1-score all around 91%.

Inspecting the confusion matrix:

- 806 instances were correctly classified as class `0`, while 41 instances were incorrectly predicted as class `0` when they were actually class `1`.
- Conversely, 554 instances were correctly labeled as class `1`, but 93 instances were wrongly classified as class `1` when they were in fact class `0`.

#### 4.1.2 TODO: The rest of the basic modeling

### 4.2. With hyperparameter tuning

...

### 4.3. Explaining the reasoning behind the hyperparameter tuning

...

## 5. Comparing modelling methods

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
