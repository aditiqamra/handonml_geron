# Machine learning end to end example

## Problem statement

Build a model of housing prices in CA state using CA census data
Metrics would include population, median income , median housing price for each district in CA
Model should be able to predict median housing price in any district given all other metrics

## Objective
Feed district prices into another ML model to determine whether one should invest in that area or not

## Current state of art / solution
currently prices are determined by manual estimation by experts. these have been found to be off by 20%

## What type of ML task is it ?
It is a supervised because we have labeled training examples
it is a regression task because we are predicting a value
Plain batch learning will suit this problem

## Select a performance measure
typically Root mean square error (RMSE) is used 
![](https://www.includehelp.com/ml-ai/Images/rmse-1.jpg)

Incase there are outliers in the training data, mean absolute error [(MAE)]
might be better also called [average absolute deviation]

Both the RMSE and the MAE are ways to measure the distance between two vectors: the vector of predictions and the vector of target values

MAE is manhattan norm
RMSE is euclidean norm

## Check assumptions
make sure output needed will not be converted to categories - in that case exact price is not the end goal but rather this becomes a classification task

## Now start modeling

### Load data
### Check assumptions
### Create test data set !!
pick randomly 20% of your dataset and set it aside
In some case random sampling will not be suitable in that case make sure we
so stratified  sampling specially if you know which attributes are more important than others. you don;t want your test dataset to be biased e.g. only low median income group districts



