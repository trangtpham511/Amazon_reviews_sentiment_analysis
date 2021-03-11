# Amazon_reviews_sentiment_analysis

## About the dataset

This dataset is imported from https://nijianmo.github.io/amazon/index.html with 147194 customers' reviews on the gift cards. It contains review with the overall rating from from 1.0 to 5.0, with 1.0 is highly negative and 5.0 is highly positive. The dataset is hugely imblance with the positive reviews account for 90.6% of total reviews.

## Objective 

My objective is to create a model to predict whether a review is positive, neutral or negative, based on the 'reviewText' column only

## Libraries

The libraries used for this project is : Pandas, Numpy, Sklearn, NLTK, Keras, Matplotlib, Seaborn, Imblearn

## Techniques

I use NLP ( Natural Language Processing ) for the sentiment analysis, with the following steps:

- Text processing includes tokenization, removing stopwords, lemmatization
- Count, weigh and convert words into vector (CounTVectorizer/TFIDF /TfidfVectorizer)

 The algorithms used are Logistic Regression , Naive Bayes, RandomForest with N-grams(Unigrams and Bigrams), and also Convolutional Neural Network with Keras.
 
 Manual class balancing and oversampling with SMOTE are used to deal with the class balancing. My focus is on getting a model with high Kappa score, high accuracy and high f-1 score. 
 
 A Grid Search is performed in order to get the best parameters for the Random Forest model.
 
 ## Result
 
 - With the text-processed imbalanced dataset, the Logistic Regression with Unigrams and the Convolutional Neural Network yield very similar results in terms of evaluation metrics and Kappa score. 
 - After performing class balancing method, I have the best model using Random Forest with Bigrams, with the Kappa Score of 0.79, and evaluation metrics for all classes are above 0.84. My model is very good for imbalanced dataset.
 - The class balancing does not improve the performance of the Convolutional Neural Network, but it is due to the fact that I have not used hyperparameter for the algorithm.
 
 
