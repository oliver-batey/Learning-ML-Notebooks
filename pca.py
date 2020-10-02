import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA 
from sklearn.linear_model import LogisticRegression


'''Download MNIST data set'''
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784',version=1)
print(mnist.keys())

X,y = mnist['data'],mnist['target']
X_train,y_train = X[:50000],y[:50000]
X_test,y_test = X[50000:],y[50000:]




'''Train and test Random Forest Classifier on data'''
classifier_1 = RandomForestClassifier(n_estimators=10)

t1 = time.time()#initiate clock before training 
classifier_1.fit(X_train,y_train)
t2 = time.time()#get time agter training 

predictions_1 = classifier_1.predict(X_test)
print('RFC accuracy with original data:', accuracy_score(y_test,predictions_1))

runtime1 = t2-t1
print('Training time of RFC on original data:', runtime1,'seconds')




'''Now use PCA to retain 95% variance and train new classifer'''
pca = PCA(n_components=0.95)
X_train_r = pca.fit_transform(X_train)
X_test_r = pca.transform(X_test)# use transform not fit_transform since pca was already fit to training set

classifier_2 = RandomForestClassifier(n_estimators=10)

t1= time.time()#initiate clock before training 
classifier_2.fit(X_train_r,y_train)
t2 = time.time()# get time after training 

predictions_2 = classifier_2.predict(X_test_r)
print('RFC accuracy with compressed data:', accuracy_score(y_test,predictions_2))

runtime2 = t2-t1
print('Training time of RFC on compressed data:', runtime2,'seconds')

'''The classification is less accurate and training takes longer when using PCA.
This illustrates the fact that PCA is not always benefitial, for example, reducing
the decision function may be much more complex in a lower dimension than in a higher
one. For logistic regression using the softmax function, PCA reduces training time
and improves accuracy.'''




logistic_1 = LogisticRegression(multi_class='multinomial',solver='lbfgs')
t1 = time.time()#initiate clock before training 
logistic_1.fit(X_train,y_train)
t2 = time.time()#get time agter training 

logistic_predictions_1 = logistic_1.predict(X_test)
print('Logistic regression accuracy with original data:', accuracy_score(y_test,logistic_predictions_1))

runtime3 = t2-t1
print('Training time of Logistic regression on original data:', runtime3,'seconds')


'''Run new Logistic regression algorithm on compressed data'''
logistic_2 = LogisticRegression(multi_class='multinomial',solver='lbfgs')
t1 = time.time()#initiate clock before training 
logistic_2.fit(X_train_r,y_train)
t2 = time.time()#get time agter training 

logistic_predictions_2 = logistic_2.predict(X_test_r)
print('Logistic regression accuracy with compressed data:', accuracy_score(y_test,logistic_predictions_2))

runtime4 = t2-t1
print('Training time of Logistic regression on compressed data:', runtime4,'seconds')


