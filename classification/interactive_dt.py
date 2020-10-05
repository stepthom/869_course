import numpy as np

import pandas as pd
import altair as alt
import sys
import streamlit as st

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.metrics import silhouette_score, silhouette_samples
import sklearn.metrics
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder

import urllib

import itertools
import scipy

if sys.version_info[0] < 3:
    reload(sys)
    sys.setdefaultencoding("utf-8")
	
	
@st.cache
def get_and_prepare_data(dataset='Marketing'):
	base = "C:\\Users\\st50\\Documents\\869_course\\data\\"

	if dataset is 'Marketing':
		target_name = 'Bought'
		file = 'marketing.csv'
	elif dataset is 'KCU':
		target_name = 'paid_back'
		file = 'KCU.csv'
	else:
		target_name = 'Class'
		file = 'GermanCredit.csv'
		
		
	df = pd.read_csv(base + file)
	
	# Need to do some encoding?
	categorical_feature_mask = df.dtypes == object
	categorical_cols = df.columns[categorical_feature_mask].tolist()
	if categorical_cols:
		le = LabelEncoder()
		df[categorical_cols] = df[categorical_cols].apply(lambda col: le.fit_transform(col))
	
	
	return target_name, df
	

# Interactive Streamlit elements, like these sliders, return their value.
# This gives you an extremely simple interaction model.
dataset = st.sidebar.selectbox("Dataset", ('Marketing', 'KCU', 'German Credit'))
random_state_str = st.sidebar.text_input("Random State", "42")
depth = st.sidebar.slider("Max Depth", 1, 50, 10, 1)
impurity = st.sidebar.selectbox("Impurity Metric", ('entropy', 'gini'))
class_weight = st.sidebar.selectbox("Class Weights", ('balanced', 'None'))

random_state = int(random_state_str)
if class_weight is "None":
	class_weight = None

try:
	target_name, df = get_and_prepare_data(dataset)
except urllib.error.URLError:
    st.error("Connection Error. This demo requires internet access")

	
st.write("# Uncle Steve's Decision Tree Playground")
st.write("Design your decision tree in the left panel, and the results will be displayed here.")





st.write("## Data Sample")

st.write(df)



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df.drop([target_name], axis=1), df[target_name], test_size=0.2, random_state=random_state)

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state=random_state, criterion=impurity, class_weight = class_weight, max_depth=depth)
clf.fit(X_train, y_train)
feature_names = list(X_train.columns)
class_names = [str(x) for x in clf.classes_]



st.write("## Tree")

# This is buggy, for some reason, and crashes

#from sklearn.tree import plot_tree
#plt.figure();
#plot_tree(clf, filled=True, feature_names=feature_names, class_names=class_names, proportion=False, fontsize=6);
#st.pyplot()

from sklearn.tree.export import export_text


st.code(export_text(clf, feature_names=feature_names, show_weights=True))


st.write("## Performance on Training Data")

from sklearn.metrics import classification_report



y_pred_training_dt = clf.predict(X_train)
st.code(classification_report(y_train, y_pred_training_dt, target_names=class_names))

st.write("## Performance on Held-out Testing Data")

y_pred_dt = clf.predict(X_test)
y_pred_proba_dt = clf.predict_proba(X_test)[:,1]

st.code(classification_report(y_test, y_pred_dt, target_names=class_names))

st.write("## Predictions and Errors on Testing Data")

answers = X_test.copy()
answers[target_name] = y_test
answers['PredictedProba'] = y_pred_proba_dt
answers['Predicted'] = y_pred_dt
answers['Error'] = abs(y_test - y_pred_dt)
answers['ErrorProba'] = abs(y_test - y_pred_proba_dt)

st.write(answers.sort_values(['ErrorProba'], ascending=False))


# Streamlit widgets automatically run the script from top to bottom. Since
# this button is not connected to any other logic, it just causes a plain
# rerun.
st.button("Re-run")
