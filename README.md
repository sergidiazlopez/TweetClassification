## Project Objectives

The main objectives of this practice are:

- Applying Bayesian learning in the classification of tweets based on their sentiment.
- Using validation methods explained in the theory.
- Applying the theory to real-world problems.
- Enhancing the ability to present results effectively.

## Required Materials

- A file with the tweet database for conducting the practice. **Not uploaded**.

## Project Description

In this project, a file named "FinalStemmedSentimentAnalysisDataset.csv" contains a dataset of tweets that have already been processed, with over a million examples. The goal is to create a filter to determine whether the tweets are positive or negative using Bayesian networks. To achieve this, the database will be divided into training and testing sets randomly.

It is important to balance the training and testing sets so that they contain the same percentage of cases from different data classes. This helps avoid overfitting issues in a single class.

The tweets have been preprocessed using the Python Lancaster Stemmer, which reduces words to their roots. If you believe you can apply a different processing method, you can do so on the raw database provided, as long as you document the changes adequately. Punctuation marks have also been removed to reduce noise in the data.

### Exercise 1

In this exercise, you are required to implement a Bayesian network to determine if the test tweets are positive or negative. For this part, the entire tweet database and the dictionary obtained during training will be used. The report should explain how the problem was solved and analyze the results, including dictionary generation, justification of the validation method, and the metric used.

### Exercise 2

This exercise involves evaluating the network using training sets and dictionaries of different sizes. You should consider:

1. Increasing the training set and observing its impact on the dictionary.
2. Keeping the training set constant while varying the dictionary size.
3. Using the same dictionary size but modifying the training set.

The report should analyze how each of these changes affects performance.

### Exercise 3

In this exercise, you need to evaluate the network using training sets and dictionaries of different sizes, but this time, implementing 'Laplace smoothing.' The report should analyze how 'Laplace smoothing' affects the results.
