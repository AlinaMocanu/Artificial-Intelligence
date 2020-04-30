# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 14:53:08 2020

@author: Alina
"""
# Import libraries

from sklearn import metrics, svm, linear_model
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection  import GridSearchCV
import pandas as pd

 ##########     DATA PREPARATION ############

# read training data

train_samples = pd.read_fwf('data/train_samples.txt', header = None, delimiter = '\t')
train_labels = pd.read_fwf('data/train_labels.txt', header = None, delimiter = '\t')


# read validation data

validation_samples = pd.read_fwf('data/validation_samples.txt', header = None, delimiter = '\t')
validation_labels =  pd.read_fwf('data/validation_labels.txt', header = None, delimiter = '\t')


# read test data

test_samples = pd.read_fwf('data/test_samples.txt', header = None, delimiter = '\t')


texts = []
labels = []

# build dataframe for train data

for i  in range(len(train_samples)):
        texts.append(train_samples[1][i])
        labels.append(train_labels[1][i])
        
trainDF = pd.DataFrame()
trainDF['text'] = texts
trainDF['label'] = labels
trainDF['text'] = [entry.lower() for entry in trainDF['text']]


texts.clear()
labels.clear()

# build dataframe for validation data

for i  in range(len(validation_samples)):
        texts.append(validation_samples[1][i])
        labels.append(validation_labels[1][i])
        
validationDF = pd.DataFrame()
validationDF['text'] = texts
validationDF['label'] = labels
validationDF['text'] = [entry.lower() for entry in validationDF['text']]


frames = [trainDF, validationDF]
Corpus = pd.concat(frames)

############### FEATURE EXTRACTION #########################

# characters level tf-idf
 
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer = 'char', ngram_range = (4, 5), max_df = 0.25, min_df = 2, max_features = 6000, sublinear_tf = 1)

tfidf_vect_ngram_chars.fit(Corpus['text'])

xtrain_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(trainDF['text']) 

xvalid_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(validationDF['text'])

xcorpus_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(Corpus['text'])

xtest_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(test_samples[1]) 


################# MODEL TRAINING AND EVALUATION###############

# Utility function for training the model and computing its accuracy,
# confussion matrix and F1-score

def train_model(classifier, feature_vector_train, train_label, valid_label, feature_vector_valid, test_or_valid):
    
    # fit the training dataset on the classifier
    
    classifier.fit(feature_vector_train, train_label)
    

    # Plot confusion matrix
    titles_options = [("Confusion matrix, without normalization", None),
                      ("Normalized confusion matrix", 'true')]
    
    for title, normalize in titles_options:
        disp = metrics.plot_confusion_matrix(classifier, feature_vector_valid, valid_label,
                                 display_labels = [0, 1],
                                 cmap = plt.cm.Blues,
                                 normalize = normalize)
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)

    plt.show()
    
    # predict the labels on validation dataset
    
    predictions = classifier.predict(feature_vector_valid)
    
    # write predictions in .csv file
    
    if test_or_valid == 'test':
        prediction = pd.DataFrame(predictions, test_samples[0]).to_csv('prediction.csv', index_label = 'id', header = ['label'])
        
    if test_or_valid == 'validation':
        # compute F1-score

        print("F1-score: ", metrics.f1_score(valid_label, predictions, average='macro'))
        
        # compute accuracy
        
        print("accuracy: ", metrics.accuracy_score(predictions, valid_label))
        
        
# Perform Grid Search

# svm
clf_ = svm.SVC(kernel='rbf')
hyperparameters = {
                        'C' : [1, 10, 100, 1000],
                        'gamma' : [ 1e-4, 1.0]
                  }
clf = GridSearchCV(clf_, hyperparameters, cv = 3)

clf.fit(xtrain_tfidf_ngram_chars, trainDF['label'])

print("best parameters for svm: ", clf.best_params_)

# baggingClasiifier
clf_ = BaggingClassifier(random_state = 0, bootstrap_features = True)
hyperparameters = {
                        'n_estimators' : [ 10, 20, 30]
                  }
clf = GridSearchCV(clf_, hyperparameters, cv = 3)

clf.fit(xtrain_tfidf_ngram_chars, trainDF['label'])

print("best parameters for BaggingClassifiers: ", clf.best_params_)

    
# SVM
train_model(BaggingClassifier(base_estimator = svm.SVC(C = 10, gamma = 1.0), n_estimators = 30, random_state = 0, bootstrap_features = True), xtrain_tfidf_ngram_chars, trainDF['label'], validationDF['label'], xvalid_tfidf_ngram_chars, 'validation')

# Logistic Regression
train_model(linear_model.LogisticRegression(C = 10), xtrain_tfidf_ngram_chars, trainDF['label'], validationDF['label'], xvalid_tfidf_ngram_chars, 'validation')

# print number of tweets in each class
print(validationDF['label'].value_counts())
