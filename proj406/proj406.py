#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#data from https://www.kaggle.com/ronitf/heart-disease-uci
df = pd.read_csv("C:/Users/Tim/Desktop/proj406/projdata/heartdisease/heart.csv")
df.describe()

#%%
#rename for easier interpretation
df.columns=['age','sex','chestpain','restingbp','cholesterol','fastingbloodsugar','restecg','maxheartrate',
       'exerciseangina','stdepression','stslope','majorvessels','thalassemia','target']

df.dtypes

#%%
#check for null
df.isnull().sum().max()

total=df.isnull().sum().sort_values(ascending=False)
percent=(df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
missing=pd.concat([total,percent],axis=1,keys=['Total','Percent'])
missing.head(20)
print(missing)

#%%
#exploratory
e1 = sns.countplot(x='chestpain',data=df,hue='sex')
e2 = sns.countplot(x='chestpain',data=df,hue='target')

plt.scatter(x=df.age[df.target==1], y=df.thalassemia[(df.target==1)], c="red")
plt.scatter(x=df.age[df.target==0], y=df.thalassemia[(df.target==0)])
plt.legend(["Disease", "No Disease"])
plt.xlabel("Age")
plt.ylabel("Thalassemia")
plt.show()

plt.scatter(x=df.age[df.target==1], y=df.maxheartrate[(df.target==1)], c="red")
plt.scatter(x=df.age[df.target==0], y=df.maxheartrate[(df.target==0)])
plt.legend(["Disease", "No Disease"])
plt.xlabel("Age")
plt.ylabel("Max Heart Rate")
plt.show()

#chestpaint = []
#for chestpain in df['chestpain']:
#    if chestpain in [0]:
#        chestpaint.append('typical')
#    elif chestpain in [1]:
#        chestpaint.append('atypical')
#    elif chestpain in [2]:
#        chestpaint.append('nonang')
#    elif chestpain in [3]:
#        chestpaint.append('asymp')

rfeatures = df[['age', 'sex', 'restingbp', 'cholesterol', 'maxheartrate']]
#rfeatures['chestpain'] = chestpaint
sns.set(style="ticks", color_codes=True)
sns.pairplot(
    rfeatures, 
#    hue = 'chestpaint', 
    diag_kind = 'kde', 
    palette = 'rocket', 
    plot_kws=dict(alpha = 0.7),
    diag_kws=dict(shade=True)
)

corrmat=df.corr()
sns.heatmap(
    corrmat,
    square=True
)
plt.show()

#%%
#recode
df['sex']=df['sex'].replace(
    [0,1],['female','male']
)
df['chestpain']=df['chestpain'].replace(
    [0,1,2,3],['typical','atypical','nonang','asymp']
)
df['fastingbloodsugar']=df['fastingbloodsugar'].replace(
    [0,1],['<120','>120']
)
df['restecg']=df['restecg'].replace(
    [0,1,2],['normal','waveabn','venthyper']
)
df['exerciseangina']=df['exerciseangina'].replace(
    [0,1],['no','yes']
)
df['stslope']=df['stslope'].replace(
    [0,1,2],['up','flat','down']
)
df['thalassemia']=df['thalassemia'].replace(
    [1,2,3],['normal','fixed','reversable']
)

#%%
df=pd.get_dummies(df,drop_first=True)
df.head()

#%%
y = df['target'].values
x = df.drop(['target'],axis=1)
#normalize
from sklearn import preprocessing
xorig = x.values 
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(xorig)
x2 = pd.DataFrame(x_scaled)

#%%
#split for cv
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.25, random_state=123)

#%%
accuracies = {}

#%%
#model 1: random forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators= 1000, random_state=123)
rf.fit(X_train,y_train)

ypred = rf.predict(X_test)
errors = abs(ypred-y_test)
from sklearn.metrics import accuracy_score
print(f"Random Forest Accuracy: {accuracy_score(y_test, ypred)}")
print('Average absolute error:', round(np.mean(errors), 2))

acc = rf.score(X_test,y_test)*100
accuracies['Random Forest']=acc

#confusion matrix
from sklearn.metrics import confusion_matrix
cmr = confusion_matrix(y_test,ypred)
sns.heatmap(
    cmr,
    annot=True,
    fmt=".3f", 
    linewidths=.5, 
    square = True, 
    cmap = 'Blues_r'
)
plt.title("Random Forest Confusion Matrix")

#%%
#model 2: naive bayes
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train,y_train)

ypred = nb.predict(X_test)
print(f"Naive Bayes Accuracy: {accuracy_score(y_test, ypred)}")
acc = nb.score(X_test,y_test)*100
accuracies['Naive Bayes']=acc

#cm
cmn = confusion_matrix(y_test,ypred)
sns.heatmap(
    cmn,
    annot=True,
    fmt=".3f", 
    linewidths=.5, 
    square = True, 
    cmap = 'Blues_r'
)
plt.title("Naive Bayes Confusion Matrix")

#%%
#model 3: SVM
from sklearn.svm import SVC
svm = SVC(random_state=123)
svm.fit(X_train,y_train)

ypred = svm.predict(X_test)
print(f"SVM Accuracy: {accuracy_score(y_test, ypred)}")
acc = svm.score(X_test,y_test)*100
accuracies['SVM']=acc

#cm
cms = confusion_matrix(y_test,ypred)
sns.heatmap(
    cms,
    annot=True,
    fmt=".3f", 
    linewidths=.5, 
    square = True, 
    cmap = 'Blues_r'
)
plt.title("SVM Confusion Matrix")

#%%
#model 4: KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train,y_train)

ypred = knn.predict(X_test)
print(f"2NN Accuracy: {accuracy_score(y_test, ypred)}")

#best #neighbors?
scoreL=[]
for i in range(1,10):
    kn = KNeighborsClassifier(n_neighbors=i)
    kn.fit(X_train,y_train)
    scoreL.append(kn.score(X_test, y_test))

scoreL

accc = max(scoreL)
acc = max(scoreL)*100
accuracies['KNN'] = acc

print(f"Max KNN Accuracy: {accc}")

#cm
knn3 = KNeighborsClassifier(n_neighbors=3)
knn3.fit(X_train,y_train)

ypred = knn3.predict(X_test)

cmk = confusion_matrix(y_test,ypred)
sns.heatmap(
    cmk,
    annot=True,
    fmt=".3f", 
    linewidths=.5, 
    square = True, 
    cmap = 'Blues_r'
)
plt.title("KNN Confusion Matrix")

#%%
#comparing models
sns.barplot(
    x=list(accuracies.keys()),
    y=list(accuracies.values()),
    palette="rocket"
)
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.yticks(np.arange(0,100,10))

#%%
#best models are nb and random forest
#go with random forest
#plot tree
rf = RandomForestClassifier(n_estimators= 1000, random_state=123)
rf.fit(X_train,y_train)

ypred = rf.predict(X_test)

m = rf.estimators_[1]
fnames = [i for i in X_train.columns]

cnames = y_train.astype('str')
cnames[cnames=='0'] = 'No Disease'
cnames[cnames=='1'] = 'Disease'
#cnames=cnames.values

from sklearn.tree import export_graphviz
export_graphviz(
    m, out_file='tree.dot', 
    feature_names = fnames,
    class_names = cnames,
    rounded = True, proportion = True, 
    label='root',
    precision = 2, filled = True
)

#from subprocess import call
#call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

#from IPython.display import Image
#Image(filename = 'tree.png')
#source: https://towardsdatascience.com/how-to-visualize-a-decision-tree-from-a-random-forest-in-python-using-scikit-learn-38ad2d75f21c

#code not working so converted using online tool

#%%
#evaluating using AUC
from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(y_test, ypred)
metrics.auc(fpr,tpr)

#feature importance
importances = list(rf.feature_importances_)
xlist = list(x.columns)
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(xlist, importances)]
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

importance = rf.feature_importances_
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
plt.bar([x for x in range(len(importance))], importance)
plt.show()

#alt feature importance
#ELI5 is a Python package which helps to debug machine learning classifiers and explain their predictions.
#https://eli5.readthedocs.io/en/latest/overview.html 
import eli5
from eli5.sklearn import PermutationImportance
p = PermutationImportance(rf, random_state=123).fit(X_test, y_test)
eli5.show_weights(p, feature_names = X_test.columns.tolist())

#%%
#hypertuning - RSCV
#https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
#n_estimators = number of trees in the foreset
#max_features = max number of features considered for splitting a node
#max_depth = max number of levels in each decision tree
#min_samples_split = min number of data points placed in a node before the node is split
#min_samples_leaf = min number of data points allowed in a leaf node
#bootstrap = method for sampling data points (with or without replacement)
from sklearn.model_selection import RandomizedSearchCV
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(
    estimator = rf, 
    param_distributions = random_grid, 
    n_iter = 100, 
    cv = 5, 
    verbose=2, 
    random_state=123, 
    n_jobs = -1
)
rf_random.fit(X_train, y_train)

rf_random.best_params_

def evaluate(model, X_test, y_test):
    predictions = model.predict(X_test)
    errors = abs(predictions - y_test)
    #mape = 100 * np.mean(errors / y_test)
    #accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    #print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    #return accuracy

baserf = RandomForestClassifier(n_estimators = 1000, random_state = 123)
baserf.fit(X_train, y_train)
#base_accuracy = evaluate(baserf, X_test, y_test)
baserf.score(X_test,y_test)

bestrf = rf_random.best_estimator_
#random_accuracy = evaluate(bestrf, X_test, y_test)
bestrf.score(X_test,y_test)

#print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))

#%%
#hypertuning GSCV ###CHANGE PARAMGRID###
from sklearn.model_selection import GridSearchCV
param_grid = {
    'bootstrap': [True],
    'max_depth': [70, 80, 90, 100],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [3, 5, 7],
    'n_estimators': [1000, 1200, 1400, 2000]
}

rf = RandomForestClassifier()
gs = GridSearchCV(
    estimator = rf, 
    param_grid = param_grid, 
    cv = 5, 
    n_jobs = -1, 
    verbose = 2
)

gs.fit(X_train, y_train)
gs.best_params_

bestgrid = gs.best_estimator_
#grid_accuracy = evaluate(bestgrid, X_test, y_test)
bestgrid.score(X_test,y_test)

#print('Improvement of {:0.2f}%.'.format( 100 * (grid_accuracy - base_accuracy) / base_accuracy))

#source: https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74

#%%
#shap feature importance post-hypertuning
#https://www.kaggle.com/dansbecker/shap-values
#https://www.kaggle.com/dansbecker/advanced-uses-of-shap-values
#https://github.com/slundberg/shap

import shap
#model with gs best params
model = RandomForestClassifier(
    n_estimators=2000,
    max_depth=70,
    max_features=2,
    min_samples_leaf=4,
    min_samples_split=3,
    bootstrap=True
)

model.fit(X_train,y_train)

#shap
srf = shap.TreeExplainer(model)
sv = srf.shap_values(X_test)

shap.summary_plot(sv[1],X_test,plot_type='bar')
shap.summary_plot(sv[1],X_test)
#%%
#sample predictions with shap
def predict(model,patient):
    srf = shap.TreeExplainer(model)
    sv = srf.shap_values(patient)
    shap.initjs()
    return shap.force_plot(srf.expected_value[1], sv[1], patient, matplotlib=True)

#%% 
#predicting patients
patient = X_test.iloc[42,:].astype(float)
predict(model, patient)

patient = X_test.iloc[15,:].astype(float)
predict(model, patient)

patient = X_test.iloc[1,:].astype(float)
predict(model, patient)

shap.initjs()
sv = srf.shap_values(X_train.iloc[:50])
shap.force_plot(srf.expected_value[1],sv[1],X_test.iloc[:50])
