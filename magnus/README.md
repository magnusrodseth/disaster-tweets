# IT3212 - Assignment 2

Written and developed by Haakon Tideman Kanter, Henrik Skog, Mattis Hembre, Max Gunhamn, Sebastian Sole, and Magnus Rødseth.

## 1. Implement the preprocessing

We started by removing features, according to what was outlined in assignment 1. Next, we removed rows with a confidence threshold below `1.0`.

Next, we preprocessed the textual data. We cleaned up the `keyword` column, ensuring that all keywords were lower case, and that all keywords were separated by a single space. For comparison, some keywords were on the format `airplane%20accident`, which should be `airplane accident`.

Furthermore, we cleaned the `text` column. The first part of this included removing links, line breaks, extra spaces, special characters and punctuation. Next, we removed English stopwords. Finally, we lemmatized the text.

When handling categorical data, we removed rows with a `choose_one` value of `Can't decide`, according to what was outlined in assignment 1. Next, we mapped the `choose_one` values `Relevant` and `Not Relevant` to `1` and `0`, respectively. This was stored in a new feature, called `target`.

Moreover, we removed duplicated rows with regards to the `text` column, as outlined in assignment 1.

## 2. Extract features

...

## 3. Select features

...

## 4. Implement basic modelling methods

### 4.1 Based on the selected features

...

- Linear regression
- Polynomial regression
- Jeg tror logistic regression passer oss godt her, etter vi har fjernet outliers.
- Passer det med random forest her, som basic modelling method nummer 2?
- Andre relevante basic modelling methods inkluderer Naive Bayes og Support Vector Machines.

### 4.2 With hyperparameter tuning

...

### 4.3 Explaining the reasoning behind the hyperparameter tuning

...

## 5. Comparing modelling methods

### 5.1 Selecting the modelling methods

...

### 5.2 Reasoning

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

### 8.1 Other questions

> TODO: Add other quesions if they arise
