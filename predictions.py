'''
Created on December 28, 2016
@author: Anusha Bilakanti
'''

import pandas

from sklearn import preprocessing 
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

temp_data = pandas.DataFrame({
	'Outlook':['Sunny','Sunny','Overcast', 'Rain','Rain','Rain','Overcast','Sunny','Sunny', 'Rain', 'Sunny', 'Overcast','Overcast', 'Rain'], 
	'Temp':['Hot', 'Hot','Hot','Mild', 'Cool','Cool','Cool','Mild', 'Cool','Mild','Mild','Mild', 'Hot','Mild'],
	'Humidity':['High','High','High','High', 'Normal','Normal','Normal', 'High', 'Normal','Normal','Normal', 'High','Normal', 'High'],
	'Wind':['Weak', 'Strong','Weak','Weak','Weak','Strong','Strong','Weak','Weak','Weak','Strong','Strong','Weak', 'Strong']})


target = pandas.DataFrame({'Play':['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']})



temp_data=temp_data.apply(preprocessing.LabelEncoder().fit_transform)
target=target.apply(preprocessing.LabelEncoder().fit_transform)


validation_size = 0.20
seed = 7
X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(temp_data, target, test_size=validation_size, random_state=seed)






# Make predictions on test dataset

print "***************LogisticRegression***************"
lr = LogisticRegression()
Y_train=Y_train.values.ravel()
lr.fit(X_train, Y_train)
predictions = lr.predict(X_test)
print(accuracy_score(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))


print "*************LinearDiscriminantAnalysis**********"
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, Y_train)
predictions = lda.predict(X_test)
print(accuracy_score(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))


print "*****************KNeighborsClassifier************"
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_test)
print(accuracy_score(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))


print "*****************DecisionTreeClassifier*********"
dtc = DecisionTreeClassifier()
dtc.fit(X_train, Y_train)
predictions = dtc.predict(X_test)
print(accuracy_score(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))

print "******************GaussianNB******************"
gnb = GaussianNB()
gnb.fit(X_train, Y_train)
predictions = gnb.predict(X_test)
print(accuracy_score(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))

print "**********************SVC********************"
svc = SVC()
svc.fit(X_train, Y_train)
predictions = svc.predict(X_test)
print(accuracy_score(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))






