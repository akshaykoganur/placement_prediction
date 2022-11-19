
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing

df = pd.read_csv('collegePlace.csv')


x = df.drop('PlacedOrNot',axis='columns')
x = x.drop('Age',axis='columns')
x = x.drop('Hostel',axis='columns')
y = df['PlacedOrNot']
le = preprocessing.LabelEncoder()
x['Gender'] = le.fit_transform(x['Gender'])
x['Stream'] = le.fit_transform(x['Stream'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 100)

classify = DecisionTreeClassifier()
classify=classify.fit(x_train,y_train)

pickle.dump(classify, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
print(model.predict([[1,1,1,0,0]]))
