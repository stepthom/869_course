import numpy as np

import pandas as pd
import altair as alt
import sys
import streamlit as st

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.metrics import silhouette_score, silhouette_samples, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder
from sklearn.tree.export import export_text
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

import urllib

import itertools
import scipy

if sys.version_info[0] < 3:
    reload(sys)
    sys.setdefaultencoding("utf-8")
	
	
@st.cache
def get_and_prepare_data(dataset='Marketing'):
	base = "C:\\Users\\st50\\Documents\\869_course\\data\\"

	if dataset == 'Portuguese Bank':
		target_name = 'y'
		file = 'bank.csv'
	elif dataset == 'Adult':
                target_name = 'high_salary'
                file = 'adult.csv'
	else:
		target_name = 'Class'
		file = 'GermanCredit.csv'
		
		
	df = pd.read_csv(base + file, sep=r'\s*,\s*')

	
	# Need to do some encoding?
	categorical_feature_mask = df.dtypes == object
	categorical_cols = df.columns[categorical_feature_mask].tolist()
	if categorical_cols:
		le = LabelEncoder()
		df[categorical_cols] = df[categorical_cols].apply(lambda col: le.fit_transform(col))
	
	
	return target_name, df
	

# Interactive Streamlit elements, like these sliders, return their value.
# This gives you an extremely simple interaction model.
dataset = st.sidebar.selectbox("Dataset", ('German Credit', 'Portuguese Bank', 'Adult'))
classifier = st.sidebar.selectbox("Classifier", ('Logistic Regression', 'Decision Tree'))
random_state_str = st.sidebar.text_input("Random State", "99")
#st.sidebar.markdown("**Decision Tree Controls**")
#depth = st.sidebar.slider("Max Depth", 1, 20, 4, 1)
#impurity = st.sidebar.selectbox("Impurity Metric", ('entropy', 'gini'))
#show_tree = st.sidebar.checkbox('Show Tree')
st.sidebar.markdown('**Imbalanced Data Controls**')
class_weight = st.sidebar.selectbox("Class Weights", ('None', 'balanced'))
sampling = st.sidebar.selectbox("Sampling", ('None', 'Up', 'Down', 'SMOTE'))

depth=4
impurity="entropy"
show_tree=False


random_state = int(random_state_str)
if class_weight == "None":
    class_weight = None
if sampling == "None":
    sampling = None

try:
	target_name, df = get_and_prepare_data(dataset)
except urllib.error.URLError:
    st.error("Connection Error. This demo requires internet access")

	
st.write("# Uncle Steve's Imbalanced Data Playground")
st.write("Decide how to handle class imbalance in the left panel, and the results will be displayed here.")

vc = df[target_name].value_counts(normalize=True)

st.write("## Data Summary")
st.write("Name: " + dataset)
st.write("Number of rows: " + str(df.shape[0]))
st.write("Target name: " + target_name)
st.write("Target levels and counts: " + vc.to_string())
st.write("#### First 30 rows")

st.write(df.head(30))




st.write("## " + classifier + " on Original Data")

st.write("#### Training Data Stats")

X_train, X_test, y_train, y_test = train_test_split(df.drop([target_name], axis=1), df[target_name], test_size=0.2, random_state=random_state)
st.write("Train rows: " + str(X_train.shape[0]))
unique, counts = np.unique(y_train, return_counts=True)
vcs = dict(zip(unique, counts))
st.write("Train target levels and counts: " + str(vcs))


clf = LogisticRegression(random_state=random_state, class_weight=None)
if classifier == "Decision Tree":
    clf = DecisionTreeClassifier(random_state=random_state, criterion=impurity, class_weight = None, max_depth=depth)

clf.fit(X_train, y_train)
feature_names = list(X_train.columns)
class_names = [str(x) for x in clf.classes_]


if show_tree:
    st.code(export_text(clf, feature_names=feature_names, show_weights=True))



#st.write("#### Performance on Training Data")
#y_pred_training_dt = clf.predict(X_train)
#st.code(classification_report(y_train, y_pred_training_dt, target_names=class_names))

st.write("#### Performance on Testing Data")

y_pred_dt = clf.predict(X_test)
y_pred_proba_dt = clf.predict_proba(X_test)[:,1]

#st.code(classification_report(y_test, y_pred_dt, target_names=class_names))
st.code(confusion_matrix(y_test, y_pred_dt))
st.write("Accuracy: {0:.3f}".format(accuracy_score(y_test, y_pred_dt)))
st.write("Precision: {0:.3f}".format(precision_score(y_test, y_pred_dt)))
st.write("Recall: {0:.3f}".format(recall_score(y_test, y_pred_dt)))
st.write("F1-Score: {0:.3f}".format(f1_score(y_test, y_pred_dt)))
st.write("ROC AUC: {0:.3f}".format(roc_auc_score(y_test, y_pred_dt)))




st.write("## " + classifier + " with Tricks :)")

st.write("#### Training Data Stats")

ros = RandomOverSampler(random_state=random_state)
rus = RandomUnderSampler(random_state=random_state)
smote = SMOTE(random_state = random_state)

#from sklearn.utils import class_weight
#class_weights = class_weight.compute_class_weight('balanced',
#                                                 np.unique(y_train),
#                                                 y_train)

X_train_better = X_train
y_train_better = y_train

if sampling == 'Up':
    X_train_better, y_train_better = ros.fit_resample(X_train, y_train)
elif sampling == 'Down':
    X_train_better, y_train_better = rus.fit_resample(X_train, y_train)
elif sampling == 'SMOTE':
    X_train_better, y_train_better = smote.fit_resample(X_train, y_train)
    

st.write("Train rows: " + str(X_train_better.shape[0]))
unique, counts = np.unique(y_train_better, return_counts=True)
vcs_better = dict(zip(unique, counts))
st.write("Train target levels and counts: " + str(vcs_better))


clf_better = LogisticRegression(random_state=random_state, class_weight=class_weight)
if classifier == "Decision Tree":
    clf_better = DecisionTreeClassifier(random_state=random_state, criterion=impurity, class_weight = class_weight, max_depth=depth)
clf_better.fit(X_train_better, y_train_better)

if show_tree:
    st.code(export_text(clf_better, feature_names=feature_names, show_weights=True))



st.write("#### Performance on Testing Data")

y_pred_dt_better = clf_better.predict(X_test)
y_pred_proba_dt_better = clf_better.predict_proba(X_test)[:,1]

#st.code(classification_report(y_test, y_pred_dt_better, target_names=class_names))
st.code(confusion_matrix(y_test, y_pred_dt_better))
st.write("Accuracy: {0:.3f}".format(accuracy_score(y_test, y_pred_dt_better)))
st.write("Precision: {0:.3f}".format(precision_score(y_test, y_pred_dt_better)))
st.write("Recall: {0:.3f}".format(recall_score(y_test, y_pred_dt_better)))
st.write("F1-Score: {0:.3f}".format(f1_score(y_test, y_pred_dt_better)))
st.write("ROC AUC: {0:.3f}".format(roc_auc_score(y_test, y_pred_dt_better)))


st.button("Re-run")
