# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.
2. Calculate the null values present in the dataset and apply label encoder.
3. Determine test and training data set and apply decison tree classification in dataset.
4. calculate Accuracy,data prediction.


## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: NITHIYANANDAN N 
RegisterNumber: 212222230099
*/
import pandas as pd
data=pd.read_csv("/content/Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()   
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
### data.head()
![image](https://github.com/NITHIYANANDAN278/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121784636/18ee9daf-d1dd-45bb-b10c-ff70f4d20714)
### data.info()
![image](https://github.com/NITHIYANANDAN278/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121784636/b78155c6-c3db-4a7a-9c77-b792fbf29df8)
### isnull() and sum()
![image](https://github.com/NITHIYANANDAN278/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121784636/b0bc6e9a-a519-49e4-b004-2aebc64abe9a)
### data value counts()
![image](https://github.com/NITHIYANANDAN278/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121784636/1408e9e5-498b-47a9-b7eb-9f4ae090da97)
### data.head() for salary
![image](https://github.com/NITHIYANANDAN278/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121784636/ba1870ba-7ede-4064-b49e-226233dd96c6)
### x.head()
![image](https://github.com/NITHIYANANDAN278/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121784636/3755ab0b-675d-41b7-9309-067d0d9b9112)
### accuracy value
![image](https://github.com/NITHIYANANDAN278/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121784636/89b6db1e-cba1-48d3-9501-ee9befa18865)
### data prediction
![image](https://github.com/NITHIYANANDAN278/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121784636/28cbe6f6-7b17-4541-88cb-39577c7f04db)









## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
