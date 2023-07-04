import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import preprocessing

data= pd.read_csv("car.data",sep=',')
#print(data.head())
le=preprocessing.LabelEncoder()

#convert features into numerical variables
buying=le.fit_transform(list(data["buying"]))
maint=le.fit_transform(list(data["maint"]))
doors=le.fit_transform(list(data["doors"]))
persons=le.fit_transform(list(data["persons"]))
lug_boot=le.fit_transform(list(data["lug_boot"]))
safety=le.fit_transform(list(data["safety"]))
cls=le.fit_transform(list(data["class"]))

x=list(zip(buying,maint,doors,persons,lug_boot,safety))
y=list(cls)


x_train, x_test, y_trian , y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.1)
#print(x_train)

model=KNeighborsClassifier(n_neighbors=5)
model.fit(x_train,y_trian)
acc= model.score(x_test,y_test)
print(acc)
predictions= model.predict(x_test)

classes=["unacc", "acc", "good", "vgood"]
for i in range(len(predictions)):
    print("predict:",classes[predictions[i]],"real:" ,classes[y_test[i]] )