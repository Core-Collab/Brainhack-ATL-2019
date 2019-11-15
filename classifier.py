from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
samples = 3500
num_features = 512 # dimension of latent vector
train_percent = 0.8 # used when splitting data into train and test sets
good_percent = 0.8 # used to determine how much data is 'good'

# Generate test data
my_means = 50 * np.random.rand(num_features)
my_stds = 20 * np.random.rand(num_features)
features = np.random.normal(my_means, my_stds, (samples, num_features))
labels = np.random.rand(samples)

# Set 'good' data as class 0, otherwise class 1
labels[labels < good_percent] = 0
labels[labels >= good_percent] = 1

# Perturb samples that have 'bad' labels
features[labels == 1, :] += 4 * (2 * np.random.rand(num_features) - 1)

# Save test data and labels to file
np.savetxt('./data.csv', features, delimiter=',')
np.savetxt('./labels.csv', labels, delimiter='.')

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
    # Get model
    model = get_model(model_name)
    Xtrain, Xtest, ytrain, ytest = train_test_split(my_features, my_labels, train_size=train_percent)
    if do_pca == 1:
        pca = PCA(components)
        model = make_pipeline(pca, model)
    model.fit(Xtrain, ytrain)
    # Save model
    pickle.dump(model, open(model_filename, 'wb'))
    # Load model
    model = pickle.load(open(model_filename, 'rb'))
    y = model.predict(Xtest)
    print('Accuracy =', accuracy_score(ytest, y))
    print('Confusion matrix: \n', confusion_matrix(ytest, y))

def clf_predict(data_filename = './data.csv', model_filename = './my_model.clf'):
    # Returns an array of predicted labels
    # Read in data
    X = np.genfromtxt(data_filename, delimiter=',')
    # Load model
    model = pickle.load(open(model_filename, 'rb'))


feature_classifier()