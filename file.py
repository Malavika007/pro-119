from io import StringIO
import nntplib
import pandas as pd
from scipy.fftpack import ss_diff
df = pd.read_csv('titanic.csv')
col_names = ['PassengerId','Pclass','Sex','Age','SibSp', 'Parch', 'Survived']
print(df.head())

features = ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Survived']

X = df[features]
y = df.Survived

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

clf = DecisionTreeClassifier()

clf = clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))

from sklearn.tree import export_graphviz
from sklearn.externals.six StringIO
from IPython.display import Image
import pydotplus

dot_data = StringIO()

export_graphviz(clf, out_file=dot_data, filled = True, rounded =True, special_characters=True, feature_names=True, class_names=['0','1'])
print(dot_data.getvalue())

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('Titanic.png')
Image(graph.create_png())