#!/usr/bin/env python
# coding: utf-8

# In[29]:


from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
samples = 3619
num_features = 4096 # dimension of latent vector
train_percent = 0.8 # used when splitting data into train and test sets

def get_model(model_name = 'random_forest'):
    if model_name == 'random_forest':
        return RandomForestClassifier(n_estimators=100, random_state=0)
    elif model_name == 'ada_boost':
        return AdaBoostClassifier(n_estimators=100)
    elif model_name == 'gradient_boost':
        return GradientBoostingClassifier(n_estimators=100)
    elif model_name == 'kmeans':
        return KMeans(n_clusters=2)
    elif model_name == 'knearest':
        return KNeighborsClassifier(n_neighbors=3)
    
    
def feature_classifier(data_filename = './data.csv', labels_filename = './labels.csv', train_percent = 0.8, model_name = 'random_forest', model_filename = './my_model.clf', do_pca = 1, components = 2):
    # Read in data and labels
    my_features = np.genfromtxt(data_filename, delimiter=',')
    my_labels = np.genfromtxt(labels_filename, delimiter=',')
    my_features = np.nan_to_num(my_features)
#     scaler = pickle.load(open('siamese_scaler.pkl', 'rb'))
#     my_features = scaler.transform(my_features)
#     scaler.fit(my_features)
    # Get model
    model = get_model(model_name)
    Xtrain, Xtest, ytrain, ytest = train_test_split(my_features, my_labels, train_size=train_percent)
    if do_pca == 1:
        pca = PCA(components)
        model = make_pipeline(pca, model)
    model.fit(Xtrain, ytrain)
    # Save model
#     pickle.dump(scaler, open('siamese_scaler.pkl', 'wb'))
    pickle.dump(model, open(model_filename, 'wb'))
    # Load model
    model = pickle.load(open(model_filename, 'rb'))
    y = model.predict(Xtest)
    print('Accuracy = ', accuracy_score(ytest, y))
    print('Confusion matrix: ', confusion_matrix(ytest, y))
    
def clf_predict(data_filename = './data.csv', model_filename = './my_model.clf'):
    # Returns an array of predicted labels
    # Read in data
    X = np.genfromtxt(data_filename, delimiter=',')
    # Load model
    model = pickle.load(open(model_filename, 'rb'))
    return model.predict(X)


# In[30]:


def clf_predict_matrix(data_in, model_fn):
    model = pickle.load(open(model_fn, 'rb'))
    return 1 - model.predict_proba(data_in)[:, 0]


# In[ ]:




