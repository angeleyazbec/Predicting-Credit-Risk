# Predicting-Credit-Risk
## Background

LendingClub is a peer-to-peer lending services company that allows individual investors to partially fund personal loans as well as buy and sell notes backing the loans on a secondary market. LendingClub offers their previous data through an API.

I used this data to create machine learning models to classify the risk level of given loans. Specifically, I compared the Logistic Regression model and Random Forest Classifier.

### Retrieving the data

In the `Generator` folder in `Resources`, there is a [GenerateData.ipynb](/Resources/Generator/GenerateData.ipynb) notebook that will download data from LendingClub and output two CSVs: 

* `2019loans.csv`
* `2020Q1loans.csv`

I used an entire year's worth of data (2019) to predict the credit risk of loans from the first quarter of the next year (2020).


## Preprocessing: Convert categorical data to numeric

Created a training set from the 2019 loans using `pd.get_dummies()` to convert the categorical data to numeric columns. Similarly, create a testing set from the 2020 loans, also using `pd.get_dummies()`.  

## Consider the models

Created and compared two models on this data: a logistic regression and a random forests classifier. 

## Fit a LogisticRegression model and RandomForestClassifier model

Created a LogisticRegression model, fit it to the data, and printed the model's score. Conducted the same for the RandomForestClassifier. 

## Revisit the Preprocessing: Scale the data

Used `StandardScaler` to scale the training and testing sets. 

Fit and scored the LogisticRegression and RandomForestClassifier models on the scaled data. 
