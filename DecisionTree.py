import numpy as np
import pandas
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn import tree

# read in our weather data
rawData = pandas.read_csv('data/dataWeather.txt',delimiter='\t')

# the labels are in the last column
label = rawData['Play']
# the actual "data" is in all other columns
tmp = rawData.iloc[:,0:-1]

# now convert the data to one-hot encoding!
data = pandas.get_dummies(tmp)

# construct a decision tree model using the ID information gain
dt = tree.DecisionTreeClassifier(criterion='entropy')

# fit it to our data
dt.fit(data,label)

# test with a new day

# first do the one-hot encoding for the new day
tmp = pandas.get_dummies(pandas.DataFrame({'Outlook':['Sunny'],'Temperature':['Cool'],'Humidity':['High'],'Windy':[True]}))

# now re-index this with the original encoding, making sure to
# add "0" to every column that does NOT appear in our tmp variable!
newDay=tmp.reindex(columns = data.columns, fill_value=0)

# finally, we can predict the decision:
print('on the tested day, the decision to Play is:',dt.predict(newDay)[0],'with',np.max(dt.predict_proba(newDay)),'probability')

import pydotplus
from IPython.display import Image
# export the tree using the correct feature names
# colored by purity
dot_data = tree.export_graphviz(dt,out_file=None,
                                feature_names=data.columns,
                                rounded=True,filled=True)
# convert this to a picture structure
graph = pydotplus.graph_from_dot_data(dot_data)
# show this picture as a PNG in the browser
Image(graph.create_png())