# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import required libraries
2. Load the dataset
3. View the dataset
4. Select the features (X) and target (y)
5. Split the dataset
6. Create the Logistic Regression model
7. Train the model
8. Predict the results
9. Evaluate the model
10. 10. Predict for a new student
11. Display the output
 

## Program:
```
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull()

print(data1.duplicated().sum())

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x
y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
print("Prediction Array : \n", y_pred)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy : ",accuracy)

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
print("Confusion Matrix : \n", confusion)

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print("Classification Report : \n\n",classification_report1)


from sklearn import metrics
cm_display=metrics.ConfusionMatrixDisplay(confusion_matrix=confusion,display_labels=[True,False])
cm_display.plot()

```

Developed by: LOKESH S

RegisterNumber:  212224240079



## Output:


![Screenshot 2025-04-28 161742](https://github.com/user-attachments/assets/abac9958-d043-4a27-bc74-bf950479c0b9)

![Screenshot 2025-04-28 161748](https://github.com/user-attachments/assets/2404a734-1312-4625-8ef9-24e68d663471)

![Screenshot 2025-04-28 161754](https://github.com/user-attachments/assets/76c4c53a-db93-426d-bbb9-36b2af9b6d9f)

![Screenshot 2025-04-28 161800](https://github.com/user-attachments/assets/214c2a57-641f-4a72-a8b9-8c4f26723219)

![Screenshot 2025-04-28 161806](https://github.com/user-attachments/assets/8730b74c-18e9-45d7-9bd3-19f83164bfeb)

![Screenshot 2025-04-28 161817](https://github.com/user-attachments/assets/1680422d-673a-41a5-a411-c100a32f78a4)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
