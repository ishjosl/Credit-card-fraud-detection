#Import all the necessary libraries 
 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

#Load the data

df=pd.read_csv('/Users/ishimwejoslin/Downloads/creditcard.csv')

#Explore the data

df.head()
   Time        V1        V2        V3        V4  ...       V26       V27       V28  Amount  Class
0   0.0 -1.359807 -0.072781  2.536347  1.378155  ... -0.189115  0.133558 -0.021053  149.62      0
1   0.0  1.191857  0.266151  0.166480  0.448154  ...  0.125895 -0.008983  0.014724    2.69      0
2   1.0 -1.358354 -1.340163  1.773209  0.379780  ... -0.139097 -0.055353 -0.059752  378.66      0
3   1.0 -0.966272 -0.185226  1.792993 -0.863291  ... -0.221929  0.062723  0.061458  123.50      0
4   2.0 -1.158233  0.877737  1.548718  0.403034  ...  0.502292  0.219422  0.215153   69.99      0

[5 rows x 31 columns]

df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 284807 entries, 0 to 284806
Data columns (total 31 columns):
 #   Column  Non-Null Count   Dtype  
---  ------  --------------   -----  
 0   Time    284807 non-null  float64
 1   V1      284807 non-null  float64
 2   V2      284807 non-null  float64
 3   V3      284807 non-null  float64
 4   V4      284807 non-null  float64
 5   V5      284807 non-null  float64
 6   V6      284807 non-null  float64
 7   V7      284807 non-null  float64
 8   V8      284807 non-null  float64
 9   V9      284807 non-null  float64
 10  V10     284807 non-null  float64
 11  V11     284807 non-null  float64
 12  V12     284807 non-null  float64
 13  V13     284807 non-null  float64
 14  V14     284807 non-null  float64
 15  V15     284807 non-null  float64
 16  V16     284807 non-null  float64
 17  V17     284807 non-null  float64
 18  V18     284807 non-null  float64
 19  V19     284807 non-null  float64
 20  V20     284807 non-null  float64
 21  V21     284807 non-null  float64
 22  V22     284807 non-null  float64
 23  V23     284807 non-null  float64
 24  V24     284807 non-null  float64
 25  V25     284807 non-null  float64
 26  V26     284807 non-null  float64
 27  V27     284807 non-null  float64
 28  V28     284807 non-null  float64
 29  Amount  284807 non-null  float64
 30  Class   284807 non-null  int64  
dtypes: float64(30), int64(1)
memory usage: 67.4 MB
df.describe()
                Time            V1            V2  ...           V28         Amount          Class
count  284807.000000  2.848070e+05  2.848070e+05  ...  2.848070e+05  284807.000000  284807.000000
mean    94813.859575  1.168375e-15  3.416908e-16  ... -1.227390e-16      88.349619       0.001727
std     47488.145955  1.958696e+00  1.651309e+00  ...  3.300833e-01     250.120109       0.041527
min         0.000000 -5.640751e+01 -7.271573e+01  ... -1.543008e+01       0.000000       0.000000
25%     54201.500000 -9.203734e-01 -5.985499e-01  ... -5.295979e-02       5.600000       0.000000
50%     84692.000000  1.810880e-02  6.548556e-02  ...  1.124383e-02      22.000000       0.000000
75%    139320.500000  1.315642e+00  8.037239e-01  ...  7.827995e-02      77.165000       0.000000
max    172792.000000  2.454930e+00  2.205773e+01  ...  3.384781e+01   25691.160000       1.000000

[8 rows x 31 columns]

# Visualize data using plots

## How many are fraud and how many are not frauds
class_names={0:'Not Fraud', 1: 'Fraud'}

print(df.Class.value_counts().rename(index=class_names))
Class
Not Fraud    284315
Fraud           492
Name: count, dtype: int64

## Plotting the variables using subplots
fig=plt.figure(figsize=(15,12))
fig = plt.figure(figsize = (15, 12))


plt.subplot(5, 6, 1) ; plt.plot(df.V1) ; plt.subplot(5, 6, 15) ; plt.plot(df.V15)
plt.subplot(5, 6, 2) ; plt.plot(df.V2) ; plt.subplot(5, 6, 16) ; plt.plot(df.V16)
plt.subplot(5, 6, 3) ; plt.plot(df.V3) ; plt.subplot(5, 6, 17) ; plt.plot(df.V17)
plt.subplot(5, 6, 4) ; plt.plot(df.V4) ; plt.subplot(5, 6, 18) ; plt.plot(df.V18)
plt.subplot(5, 6, 5) ; plt.plot(df.V5) ; plt.subplot(5, 6, 19) ; plt.plot(df.V19)
plt.subplot(5, 6, 6) ; plt.plot(df.V6) ; plt.subplot(5, 6, 20) ; plt.plot(df.V20)
plt.subplot(5, 6, 7) ; plt.plot(df.V7) ; plt.subplot(5, 6, 21) ; plt.plot(df.V21)
plt.subplot(5, 6, 8) ; plt.plot(df.V8) ; plt.subplot(5, 6, 22) ; plt.plot(df.V22)
plt.subplot(5, 6, 9) ; plt.plot(df.V9) ; plt.subplot(5, 6, 23) ; plt.plot(df.V23)
plt.subplot(5, 6, 10) ; plt.plot(df.V10) ; plt.subplot(5, 6, 24) ; plt.plot(df.V24)
plt.subplot(5, 6, 11) ; plt.plot(df.V11) ; plt.subplot(5, 6, 25) ; plt.plot(df.V25)
plt.subplot(5, 6, 12) ; plt.plot(df.V12) ; plt.subplot(5, 6, 26) ; plt.plot(df.V26)
plt.subplot(5, 6, 13) ; plt.plot(df.V13) ; plt.subplot(5, 6, 27) ; plt.plot(df.V27)
plt.subplot(5, 6, 14<Axes: >
[<matplotlib.lines.Line2D object at 0x115fdeb40>]
<Axes: >
[<matplotlib.lines.Line2D object at 0x116046f60>]

plt.subplot(5, 6, 2) ; plt.plot(df.V2) ; plt.subplot(5, 6, 16) ; plt.plot(df.V16)
<Axes: >
plt.subplot(5, 6, 3) ; plt.plot(df.V3) ; plt.subplot(5, 6, 17) ; plt.plot(df.V17)
plt.subplot(5, 6, 4) ; plt.plot(df.V4) ; plt.subplot(5, 6, 18) ; plt.plot(df.V18)
plt.subplot(5, 6, 5) ; plt.plot(df.V5) ; plt.subplot(5, 6, 19) ; plt.plot(df.V19)
plt.subplot(5, 6, 6) ; plt.plot(df.V6) ; plt.subplot(5, 6, 20) ; plt.plot(df.V20)
plt.subplot(5, 6, 7) ; plt.plot(df.V7) ; plt.subplot(5, 6, 21) ; plt.plot(df.V21)
plt.subplot(5, 6, 8) ; plt.plot(df.V8) ; plt.subplot(5, 6, 22) ; plt.plot(df.V22)
plt.subplot(5, 6, 9) ; plt.plot(df.V9) ; plt.subplot(5, 6, 23) ; plt.plot(df.V23)
plt.subplot(5, 6, 10) ; plt.plot(df.V10) ; plt.subplot(5, 6, 24) ; plt.plot(df.V24)
plt.subplot(5, 6, 11) ; plt.plot(df.V11) ; plt.subplot(5, 6, 25) ; plt.plot(df.V25)
plt.subplot(5, 6, 12) ; plt.plot(df.V12) ; plt.subplot(5, 6, 26) ; plt.plot(df.V26)
plt.subplot(5, 6, 13) ; plt.plot(df.V13) ; plt.subplot(5, 6, 27) ; plt.plot(df.V27)
plt.subplot(5, 6, 14) ; plt.plot(df.V14) ; plt.subplot(5, 6, 28) ; plt.plot(df.V28)
plt.subplot(5, 6, [<matplotlib.lines.Line2D object at 0x11607be60>]
<Axes: >
[<matplotlib.lines.Line2D object at 0x119ed09e0>]
 
plt.subplot(5, 6, 3) ; plt.plot(df.V3) ; plt.subplot(5, 6, 17) ; plt.plot(df.V17)
<Axes: >
plt.subplot(5, 6, 4) ; plt.plot(df.V4) ; plt.subplot(5, 6, 18) ; plt.plot(df.V18)
plt.subplot(5, 6, 5) ; plt.plot(df.V5) ; plt.subplot(5, 6, 19) ; plt.plot(df.V19)
plt.subplot(5, 6, 6) ; plt.plot(df.V6) ; plt.subplot(5, 6, 20) ; plt.plot(df.V20)
plt.subplot(5, 6, 7) ; plt.plot(df.V7) ; plt.subplot(5, 6, 21) ; plt.plot(df.V21)
plt.subplot(5, 6, 8) ; plt.plot(df.V8) ; plt.subplot(5, 6, 22) ; plt.plot(df.V22)
plt.subplot(5, 6, 9) ; plt.plot(df.V9) ; plt.subplot(5, 6, 23) ; plt.plot(df.V23)
plt.subplot(5, 6, 10) ; plt.plot(df.V10) ; plt.subplot(5, 6, 24) ; plt.plot(df.V24)
plt.subplot(5, 6, 11) ; plt.plot(df.V11) ; plt.subplot(5, 6, 25) ; plt.plot(df.V25)
plt.subplot(5, 6, 12) ; plt.plot(df.V12) ; plt.subplot(5, 6, 26) ; plt.plot(df.V26)
plt.subplot(5, 6, 13) ; plt.plot(df.V13) ; plt.subplot(5, 6, 27) ; plt.plot(df.V27)
plt.subplot(5, 6, 14) ; plt.plot(df.V14) ; plt.subplot(5, 6, 28) ; plt.plot(df.V28)
plt.subplot(5, 6, 29) ; plt.plot(df.Amount)
plt.show()[<matplotlib.lines.Line2D object at 0x119efd580>]
<Axes: >
[<matplotlib.lines.Line2D object at 0x119f2d8b0>]
 
plt.subplot(5, 6, 4) ; plt.plot(df.V4) ; plt.subplot(5, 6, 18) ; plt.plot(df.V18)
<Axes: >
[<matplotlib.lines.Line2D object at 0x119f5e5a0>]
<Axes: >
[<matplotlib.lines.Line2D object at 0x119f93260>]
 
plt.subplot(5, 6, 5) ; plt.plot(df.V5) ; plt.subplot(5, 6, 19) ; plt.plot(df.V19)
<Axes: >
[<matplotlib.lines.Line2D object at 0x119fc3d40>]
<Axes: >
[<matplotlib.lines.Line2D object at 0x11a0201d0>]
 
plt.subplot(5, 6, 6) ; plt.plot(df.V6) ; plt.subplot(5, 6, 20) ; plt.plot(df.V20)
<Axes: >
[<matplotlib.lines.Line2D object at 0x11a023b30>]
<Axes: >
[<matplotlib.lines.Line2D object at 0x11a06c710>]
 
plt.subplot(5, 6, 7) ; plt.plot(df.V7) ; plt.subplot(5, 6, 21) ; plt.plot(df.V21)
<Axes: >
[<matplotlib.lines.Line2D object at 0x11a0983b0>]
<Axes: >
[<matplotlib.lines.Line2D object at 0x11a0c5dc0>]
>>> plt.subplot(5, 6, 8) ; plt.plot(df.V8) ; plt.subplot(5, 6, 22) ; plt.plot(df.V22)
<Axes: >
[<matplotlib.lines.Line2D object at 0x11a0f67e0>]
<Axes: >
[<matplotlib.lines.Line2D object at 0x11a126f60>]
 
plt.subplot(5, 6, 9) ; plt.plot(df.V9) ; plt.subplot(5, 6, 23) ; plt.plot(df.V23)
<Axes: >
[<matplotlib.lines.Line2D object at 0x11a153bf0>]
<Axes: >
[<matplotlib.lines.Line2D object at 0x11a1bc560>]

plt.subplot(5, 6, 10) ; plt.plot(df.V10) ; plt.subplot(5, 6, 24) ; plt.plot(df.V24) 
<Axes: >
[<matplotlib.lines.Line2D object at 0x11a1e4ec0>]
<Axes: >
[<matplotlib.lines.Line2D object at 0x11a215700>]

plt.subplot(5, 6, 11) ; plt.plot(df.V11) ; plt.subplot(5, 6, 25) ; plt.plot(df.V25) 
<Axes: >
[<matplotlib.lines.Line2D object at 0x11a242090>]
<Axes: >
[<matplotlib.lines.Line2D object at 0x11a26ea20>]
>>> plt.subplot(5, 6, 12) ; plt.plot(df.V12) ; plt.subplot(5, 6, 26) ; plt.plot(df.V26) 
<Axes: >
[<matplotlib.lines.Line2D object at 0x11a2a7410>]
<Axes: >
[<matplotlib.lines.Line2D object at 0x11a2d3d70>]

plt.subplot(5, 6, 13) ; plt.plot(df.V13) ; plt.subplot(5, 6, 27) ; plt.plot(df.V27) 
<Axes: >
[<matplotlib.lines.Line2D object at 0x11a334620>]
<Axes: >
[<matplotlib.lines.Line2D object at 0x11a368f80>]

plt.subplot(5, 6, 14) ; plt.plot(df.V14) ; plt.subplot(5, 6, 28) ; plt.plot(df.V28) 
<Axes: >
[<matplotlib.lines.Line2D object at 0x11a3919d0>]
<Axes: >
[<matplotlib.lines.Line2D object at 0x11a3ca3f0>] 
plt.subplot(5, 6, 29) ; plt.plot(df.Amount)
<Axes: >
[<matplotlib.lines.Line2D object at 0x11a402cf0>] 
plt.show()

from sklearn.cross_validation import train_test_split 
plt.show()
 
# Split the data into features and target
# Split the data into training and testing sets

from sklearn.model_selection import train_test_split
feature_names = df.iloc[:, 1:30].columns 
target = df.iloc[:1, 30: ].columns
print(feature_names)
Index(['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11',
       'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21',
       'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'],
      dtype='object')
 
print(target)
Index(['Class'], dtype='object') 
data_features = df[feature_names]
data_target = df[target]
 
X_train, X_test, y_train, y_test = train_test_split(data_features, data_target, train_size=0.70, test_size=0.30, random_state=1) 
print("Length of X_train is: {X_train}".format(X_train = len(X_train)))
Length of X_train is: 199364
 
print("Length of X_test is: {X_test}".format(X_test = len(X_test)))
Length of X_test is: 85443 
print("Length of y_train is: {y_train}".format(y_train = len(y_train)))
Length of y_train is: 199364
>>> print("Length of y_test is: {y_test}".format(y_test = len(y_test)))
Length of y_test is: 85443

# Build and Train a logistic regression model
 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import confusion_matrix 
model = LogisticRegression() 
model.fit(X_train, y_train.values.ravel()) 
model = LogisticRegression(max_iter=1000)  # Increase from default (100) 
model.fit(X_train, y_train.values.ravel())
LogisticRegression(max_iter=1000)
 
# Evaluate the model using confusion matrix, F1 Score, and Recall score 
 
pred = model.predict(X_test) 
class_names = ['not_fraud', 'fraud'] 
matrix = confusion_matrix(y_test, pred)

## Create pandas dataframe
dataframe = pd.DataFrame(matrix, index=class_names, columns=class_names)
 
## Create heatmap 
sns.heatmap(dataframe, annot=True, cbar=None, cmap="Blues", fmt = 'g')
<Axes: > 
plt.title("Confusion Matrix"), plt.tight_layout()
(Text(0.5, 1.0, 'Confusion Matrix'), None) 
plt.ylabel("True Class"), plt.xlabel("Predicted Class")
(Text(47.109375, 0.5, 'True Class'), Text(0.5, 19.909374999999997, 'Predicted Class'))
 
plt.show()
 
## F1 score and Recall score  
from sklearn.metrics import f1_score, recall_score 
f1_score = round(f1_score(y_test, pred), 2)
 
recall_score = round(recall_score(y_test, pred), 2) 
print("Sensitivity/Recall for Logistic Regression Model 1 : {recall_score}".format(recall_score = recall_score))
Sensitivity/Recall for Logistic Regression Model 1 : 0.56 
print("F1 Score for Logistic Regression Model 1 : {f1_score}".format(f1_score = f1_score))

F1 Score for Logistic Regression Model 1 : 0.66
>>> 
