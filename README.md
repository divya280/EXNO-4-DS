# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.

STEP 2:Clean the Data Set using Data Cleaning Process.

STEP 3:Apply Feature Scaling for the feature in the data set.

STEP 4:Apply Feature Selection for the feature in the data set.

STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
### DEVELOPED BY: V DIVYASHREE
### REGISTER NO: 212223230051

 ```
      import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/bmi.csv")
df.head()
```
![image](https://github.com/user-attachments/assets/1c991935-303a-4e05-ba01-2d1efb35408f)
```
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/user-attachments/assets/d268de5c-e441-475a-9db0-339af12eb25f)
```
max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals
```
![image](https://github.com/user-attachments/assets/812743f8-07b3-4772-a386-2acd47ebb916)
```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/user-attachments/assets/f1648b9d-181e-44bd-92b4-1dd405ebc75f)

```
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```
![image](https://github.com/user-attachments/assets/330aacba-8c59-4116-a2bd-857445158261)
```
from sklearn.preprocessing import MaxAbsScaler
sc=MaxAbsScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df
```
![image](https://github.com/user-attachments/assets/6f9d1141-7dc2-4a25-99ee-105881cfc68d)

```
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head()
```
![image](https://github.com/user-attachments/assets/81145a5f-6bd8-44d5-bf2f-3f18b82c53f2)
```
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
data=pd.read_csv('/content/income(1) (1) (1).csv',na_values=[" ?"])
data
```
![image](https://github.com/user-attachments/assets/340dde6d-9990-414a-a9dd-e7c68dc349e5)

```
data.isnull().sum()
```
![image](https://github.com/user-attachments/assets/753fa52d-44e7-4de1-b5e6-86f37b8040af)
```
missing=data[data.isnull().any(axis=1)]
missing
```
![image](https://github.com/user-attachments/assets/4a5d2da3-1157-417d-a0fa-2c9fb0fd488d)
```
data2=data.dropna(axis=0)
data2
```
![image](https://github.com/user-attachments/assets/b5f52c7e-228f-4aa0-949e-85300b282afa)
```
sal=data['SalStat']
data2['SalStat']=data['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![image](https://github.com/user-attachments/assets/0620f444-8535-41bd-8fd1-8e2826118d06)
```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```

![image](https://github.com/user-attachments/assets/a8c65b30-59bb-49e5-9144-9890808d5165)
```
data2
```
![image](https://github.com/user-attachments/assets/b4158029-e3ff-4b67-9488-38dadbbef3fb)

```
new_data=pd.get_dummies(data2,drop_first=True)
new_data
```
![image](https://github.com/user-attachments/assets/6ac8e5d8-15f5-45c7-a71c-e5e4d593eae0)
```
columns_list=list(new_data.columns)
print(columns_list)
```
![image](https://github.com/user-attachments/assets/35dedbc6-948f-4b5e-a73c-1dff3a87ece8)
```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![image](https://github.com/user-attachments/assets/df80eadc-b772-4ce1-a02c-b39453eeb9be)

```
y=new_data['SalStat'].values
print(y)
```
![image](https://github.com/user-attachments/assets/98752cb0-6951-45e4-8a7d-6d0e11cacc2e)
```
x=new_data[features].values
print(x)
```

![image](https://github.com/user-attachments/assets/fc534417-aa1f-4b4d-ba1e-5d4e74c47773)

## ALGORITHM IMPLEMENTATION
```
train_x, test_x, train_y, test_y=train_test_split(x,y,test_size=0.3, random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors=5)
KNN_classifier.fit(train_x, train_y)
```

![image](https://github.com/user-attachments/assets/9c08455a-c1dd-4a12-81e4-743b7d93aa9c)
```
prediction=KNN_classifier.predict(test_x)
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```

![image](https://github.com/user-attachments/assets/eb8f6bda-d4b2-4659-9f87-f4590c24658d)
```
accuracy_score=accuracy_score(test_y, prediction)
print(accuracy_score)
```

![image](https://github.com/user-attachments/assets/3aaf9ff8-de60-48bd-b3de-6986a220b281)

```
print('Misclassified samples: %d' % (test_y !=prediction).sum())
```

![image](https://github.com/user-attachments/assets/8ab9aff8-d6bc-4def-ad10-5590ca951788)
```
data.shape
```

![image](https://github.com/user-attachments/assets/341f20ae-ac82-44a6-b002-ac204d1a2e24)
```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```

![image](https://github.com/user-attachments/assets/4dd3ce1a-a0ba-48df-944b-9a3532f354d3)

```
tips.time.unique()
```

![image](https://github.com/user-attachments/assets/18084dae-86f0-44f6-a47f-f74eceab0dfc)
```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```

![image](https://github.com/user-attachments/assets/d5965090-4eb9-411b-8ce0-839adf43e3fd)
```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```

![image](https://github.com/user-attachments/assets/52e10856-7816-4239-920f-4513e1295ca0)
```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1': [1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3': [0,1,1,0,1],
    'Target'  : [0,1,1,0,1]
}
df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]
selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```

![image](https://github.com/user-attachments/assets/c11336c7-fbaf-45ac-9fa2-0325a97b0f55)

# RESULT:
       Thus  Feature Scaling and Feature Selection process and save the data to a file.
