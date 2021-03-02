import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import time

import warnings
warnings.filterwarnings(action="ignore")

pd.set_option('display.width', 1000)
pd.set_option('display.max_column', 35)

data = pd.read_csv('BrainTumorData.csv', index_col=False)
print("\n\n\nSample BrainTumor dataset head(5) :- \n\n", data.head(5) )



print("\n\n\nShape of the BrainTumor dataset  data.shape = ", end="")
print( data.shape)


print("\n\n\nBrainTumor data decription : \n")
print( data.describe() )


print( "\n\n\ndata.diagnosis.unique() : " , data.diagnosis.unique() )





data['diagnosis'] = data['diagnosis'].map({'M':1,'B':0})

print("\n\n\nAfter updation of  diagnosis feature: \n", data.head() )




plt.hist(data['diagnosis'])
plt.title('Diagnosis (M=1 , B=0)')
plt.show()



data = data.set_index('id')
print("\n\n\nAfter id feature is set as row index: \n", data)


del data['Unnamed: 32']
print("\n\nAfter Deletion of 'Unnamed: 32' column\n", data)

print("\n\n\ndata.groupby('diagnosis').size()\n")
print(data.groupby('diagnosis').size())

data.plot(kind='density', subplots=True, layout=(5,7), sharex=False, legend=False, fontsize=1)
plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(111)
cax = ax1.imshow(data.corr() )
ax1.grid(True)
plt.title('Cancer Attributes Correlation')

fig.colorbar(cax, ticks=[.75,.8,.85,.90,.95,1])
plt.show()


Y = data['diagnosis'].values
X = data.drop('diagnosis', axis=1).values

X_train, X_test, Y_train, Y_test = train_test_split (X, Y, test_size = 0.33, random_state=21)




models_list = []
models_list.append(('CART', DecisionTreeClassifier()))
models_list.append(('SVM', SVC()))
models_list.append(('NB', GaussianNB()))
models_list.append(('KNN', KNeighborsClassifier()))


num_folds = 10

results = []
names = []

for name, model in models_list:
    kfold = KFold(n_splits=num_folds, random_state=123)
    startTime = time.time()
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    endTime = time.time()
    results.append(cv_results)
    names.append(name)
    print( "%s: %f (%f) (run time: %f)" % (name, cv_results.mean(), cv_results.std(), endTime-startTime))




fig = plt.figure()
fig.suptitle('Performance Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

pipelines = []

pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeClassifier())])))
pipelines.append(('ScaledSVM', Pipeline([('Scaler', StandardScaler()),('SVM', SVC( ))])))
pipelines.append(('ScaledNB', Pipeline([('Scaler', StandardScaler()),('NB', GaussianNB())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsClassifier())])))


results = []
names = []



print("\n\n\nAccuracies of algorithm after scaled dataset\n")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    kfold = KFold(n_splits=num_folds, random_state=123)
    for name, model in pipelines:
        start = time.time()
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
        end = time.time()
        results.append(cv_results)
        names.append(name)
        print( "%s: %f (%f) (run time: %f)" % (name, cv_results.mean(), cv_results.std(), end-start))

fig = plt.figure()
fig.suptitle('Performance Comparison after Scaled Data')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


scaler = StandardScaler().fit(X_train)

X_train_scaled = scaler.transform(X_train)
model = SVC()
start = time.time()
model.fit(X_train_scaled, Y_train)   #Training of algorithm
end = time.time()
print( "\n\nSVM Training Completed. It's Run Time: %f" % (end-start))

X_test_scaled = scaler.transform(X_test)
predictions = model.predict(X_test_scaled)
print("All predictions done successfully by SVM Machine Learning Algorithms")
print("\n\nAccuracy score %f" % accuracy_score(Y_test, predictions))

print("\n\n")
print("confusion_matrix = \n")
print( confusion_matrix(Y_test, predictions))




from sklearn.externals import joblib
filename =  "finalized_BrainTumor_model.sav"
joblib.dump(model, filename)
print( "Best Performing Model dumped successfully into a file by Joblib")


