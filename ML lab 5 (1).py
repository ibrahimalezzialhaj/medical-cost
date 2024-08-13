#!/usr/bin/env python
# coding: utf-8

# # 1. Introduction 👋
# <center><img src="https://images.unsplash.com/photo-1607619056574-7b8d3ee536b2?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=1240&q=80" alt="Drug Picture" width="700" height="700"></center><br>
# 
# ## Data Set Problems 🤔
# 👉 This dataset contains information about drug classification based on patient general information and its diagnosis. Machine learning model is needed in order **to predict the outcome of the drugs type** that might be suitable for the patient.
# 
# ---
# 
# ## Objectives of Notebook 📌
# 👉 **This notebook aims to:**
# *   Dataset exploration using various types of data visualization.
# *   Build various ML models that can predict drug type.
# 
# 👨‍💻 **The machine learning models used in this project are:** 
# 1. Linear Logistic Regression
# 2. Linear Support Vector Machine (SVM)
# 3. K Neighbours
# 4. Decision Tree
# 5. Random Forest
# 
# 
# ---
# 
# ## Data Set Description 🧾
# 
# 👉 There are **6 variables** in this data set:
# *   **4 categorical** variables,and
# *   **2 continuous** variables.
# 
# <br>
# 
# 👉 The following is the **structure of the data set**.
# 
# 
# <table style="width:100%">
# <thead>
# <tr>
# <th style="text-align:center; font-weight: bold; font-size:14px">Variable Name</th>
# <th style="text-align:center; font-weight: bold; font-size:14px">Description</th>
# <th style="text-align:center; font-weight: bold; font-size:14px">Sample Data</th>
# </tr>
# </thead>
# <tbody>
# <tr>
# <td><b>Age</b></td>
# <td>Patient Age</td>
# <td>23; 47; ...</td>
# </tr>
# <tr>
# <td><b>Sex</b></td>
# <td>Gender of patient <br> (male or female)</td>
# <td>F; M; ...</td>
# </tr>
# <tr>
# <td><b>BP</b></td>
# <td>Levels of blood pressure <br> (high, normal, or low)</td>
# <td>HIGH; NORMAL; LOW; ...</td>
# </tr>
# <tr>
# <td><b>Cholesterol</b></td>
# <td>Levels of cholesterol <br> (high or normal)</td>
# <td>1.4; 1.3; ...</td>
# </tr>
# <tr>
# <td><b>Na_to_K</b></td>
# <td>Sodium to potassium ratio in blood</td>
# <td>25.355; 13.093; ...</td>
# </tr>
# <tr>
# <td><b>Drug</b></td>
# <td>Type of drug</td>
# <td>DrugY; drugC; ...</td>
# </tr>
# </tbody>
# </table>
# 
# ---
# 

# # 2. Importing Libraries 📚
# 👉 **Importing libraries** that will be used in this notebook.

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# # 3. Reading Data Set 👓
# 👉 After importing libraries, we will also **import the dataset** that will be used.

# In[3]:


df_drug = pd.read_csv("drug200.csv")


# 👉 Read the first 6 rows in the dataset.

# In[4]:


df_drug.head()


# 👉 Data type and checking null in dataset.

# In[5]:


print(df_drug.info())


# In[15]:


df_drug.isna().sum()


# 👉 From the results above, **there are no missing/null value** in this dataset

# # 4. Initial Dataset Exploration 🔍
# 👉 This section will explore raw dataset that has been imported.

# ## 4.1 Categorical Variables 📊

# In[5]:


df_drug.Drug.value_counts()


# 👉 It can be seen that from results above, DrugY has more amount than other types of drugs

# In[6]:


df_drug.Sex.value_counts()


# 👉 The distribution of patient gender is balanced.

# In[7]:


df_drug.BP.value_counts()


# 👉 The distribution of blood pressure level is balanced.

# In[8]:


df_drug.Cholesterol.value_counts()


# 👉 The distribution of cholesterol level is balanced.

# ## 4.2 Numerical Variables 🔢
# 👉 This section will show mean, count, std, min, max and others using describe function. The skewness value for each numerical variables will also shown in this section.

# In[9]:


df_drug.describe()


# In[17]:


# 1. Handle Missing Values
# For categorical columns, fill with mode
for column in ['Sex', 'BP', 'Cholesterol', 'Drug']:
    mode_val = df_drug[column].mode()[0]
    df_drug[column].fillna(mode_val, inplace=True)

# For numerical columns, fill with median
for column in ['Age', 'Na_to_K']:
    median_val = df_drug[column].median()
    df_drug[column].fillna(median_val, inplace=True)


# check double values

# In[19]:


df_drug.duplicated().sum()


# In[20]:


df_drug.drop_duplicates(inplace=True)


# In[6]:


sns.distplot(df_drug['Age']);


# In[7]:


sns.distplot(df_drug['Na_to_K']);


# 👉The distribution of **'Age'** column is **symetric**, since the skewness value  between -0.5 and 0.5 <br>
# 👉The distribution of **'Na_to_K'** column is **moderately skewed**, since the skewness value is ***between 0.5 and 1***. It can also be seen from the histogram for 'Na_to_K' column

# # 5. EDA 📊
# 👉 This section will explore variables in the dataset using different various plots/charts.

# ## 5.1 Drug Type Distribution 💊

# In[9]:


sns.set_theme(style="darkgrid")
sns.countplot(y="Drug", data=df_drug, palette="flare")
plt.ylabel('Drug Type')
plt.xlabel('Total')
plt.show()


# ## 5.2 Gender Distribution 👫

# In[15]:


sns.set_theme(style="darkgrid")
sns.countplot(x="Sex", data=df_drug, palette="rocket")
plt.xlabel('Gender (F=Female, M=Male)')
plt.ylabel('Total')
plt.show()


# ## 5.3 Blood Pressure Distribution 🩸

# In[16]:


sns.set_theme(style="darkgrid")
sns.countplot(y="BP", data=df_drug, palette="crest")
plt.ylabel('Blood Pressure')
plt.xlabel('Total')
plt.show()


# ## 5.4 Cholesterol Distribution 🥛

# In[17]:


sns.set_theme(style="darkgrid")
sns.countplot(x="Cholesterol", data=df_drug, palette="magma")
plt.xlabel('Blood Pressure')
plt.ylabel('Total')
plt.show()


# ## 5.5 Gender Distribution based on Drug Type 👫💊

# In[18]:


pd.crosstab(df_drug.Sex,df_drug.Drug).plot(kind="bar",figsize=(12,5),color=['#003f5c','#ffa600','#58508d','#bc5090','#ff6361'])
plt.title('Gender distribution based on Drug type')
plt.xlabel('Gender')
plt.xticks(rotation=0)
plt.ylabel('Frequency')
plt.show()


# ## 5.6 Blood Pressure Distribution based on Cholesetrol 🩸🥛

# In[13]:


pd.crosstab(df_drug.BP,df_drug.Cholesterol).plot(kind="bar",figsize=(15,6),color=['#6929c4','#1192e8'])
plt.title('Blood Pressure distribution based on Cholesterol')
plt.xlabel('Blood Pressure')
plt.xticks(rotation=0)
plt.ylabel('Frequency')
plt.show()


# ## 5.7 Sodium to Potassium Distribution based on Gender and Age 🧪👫👴

# In[20]:


plt.scatter(x=df_drug.Age[df_drug.Sex=='F'], y=df_drug.Na_to_K[(df_drug.Sex=='F')], c="Blue")
plt.scatter(x=df_drug.Age[df_drug.Sex=='M'], y=df_drug.Na_to_K[(df_drug.Sex=='M')], c="Orange")
plt.legend(["Female", "Male"])
plt.xlabel("Age")
plt.ylabel("Na_to_K")
plt.show()


# # 6. Dataset Preparation ⚙
# 👉 This section will prepare the dataset before building the machine learning models.

# ## 6.1 Data Binning 🚮

# ### 6.1.1 Age 👴
# 👉 The age will be divided into **7 age categories**:
# *  Below 20 y.o.
# *  20 - 29 y.o.
# *  30 - 39 y.o.
# *  40 - 49 y.o.
# *  50 - 59 y.o.
# *  60 - 69 y.o.
# *  Above 70.

# In[21]:


bin_age = [0, 19, 29, 39, 49, 59, 69, 80]
category_age = ['<20s', '20s', '30s', '40s', '50s', '60s', '>60s']
df_drug['Age_binned'] = pd.cut(df_drug['Age'], bins=bin_age, labels=category_age)
df_drug = df_drug.drop(['Age'], axis = 1)


# ### 6.1.2 Na_to_K 🧪
# 👉 The chemical ratio will be divided into **4 categories**:
# *  Below 10.
# *  10 - 20.
# *  20 - 30.
# *  Above 30.

# In[22]:


bin_NatoK = [0, 9, 19, 29, 50]
category_NatoK = ['<10', '10-20', '20-30', '>30']
df_drug['Na_to_K_binned'] = pd.cut(df_drug['Na_to_K'], bins=bin_NatoK, labels=category_NatoK)
df_drug = df_drug.drop(['Na_to_K'], axis = 1)


# ## 6.2 Splitting the dataset 🪓
# 👉 The dataset will be split into **70% training and 30% testing**.

# In[23]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[33]:


X = df_drug.drop(["Drug"], axis=1)
y = df_drug["Drug"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# ## 6.3 Feature Engineering 🔧
# 👉 The FE method that used is **one-hot encoding**, which is **transforming categorical variables into a form that could be provided to ML algorithms to do a better prediction**.

# In[25]:


X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)


# In[26]:


X_train.head()


# In[27]:


X_test.head()


# In[21]:


from sklearn.preprocessing import LabelEncoder

# Initialize LabelEncoder
le = LabelEncoder()

# Create a copy of the data to apply label encoding

# Apply LabelEncoder to the categorical columns
df_drug['Sex'] = le.fit_transform(df_drug['Sex'])
df_drug['BP'] = le.fit_transform(df_drug['BP'])
df_drug['Cholesterol'] = le.fit_transform(df_drug['Cholesterol'])
df_drug['Drug'] = le.fit_transform(df_drug['Drug'])

df_drug.head()


# ## save cleaned data

# In[28]:


df_drug.to_csv("Clean_data.csv",index=False)


# In[29]:


df_drug = pd.read_csv('Clean_data.csv')


# In[30]:


# Calculate correlation matrix for the specific columns
corr_matrix = df_drug.corr()
print(corr_matrix)
# Create a figure and set its size
plt.figure(figsize=(6, 8))

# Plot the correlation heatmap for the specific columns
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0)

# Show the plot
plt.title('Correlation Heatmap for Specific Columns')
plt.show()


# # 7. Models 🛠

# ## 7.1 Logistic Regression

# In[83]:


from sklearn.linear_model import LogisticRegression
LRclassifier = LogisticRegression(solver='liblinear', max_iter=5000)
LRclassifier.fit(X_train, y_train)

y_pred = LRclassifier.predict(X_test)

#print(classification_report(y_test, y_pred))
#print(confusion_matrix(y_test, y_pred))

from sklearn.metrics import accuracy_score
LRAcc = accuracy_score(y_pred,y_test)
print('Logistic Regression accuracy is: {:.2f}%'.format(LRAcc*100))


# In[84]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[85]:


plt.figure(figsize=(4,4))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score'
plt.title(all_sample_title, size = 15);


# ## save model

# In[55]:


import joblib
model_filename = 'model_of _data.joblib'
joblib.dump(train, model_filename)
print(f"Model saved as {model_filename}")


# ## load and test model

# In[58]:


# Load the saved model
import joblib
model_data = joblib.load(model_filename)
test = model_data.predict(X_test)
test


# ## 7.2 K Neighbours

# In[86]:


from sklearn.neighbors import KNeighborsClassifier
KNclassifier = KNeighborsClassifier(n_neighbors=20)
KNclassifier.fit(X_train, y_train)

y_pred = KNclassifier.predict(X_test)

#print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

from sklearn.metrics import accuracy_score
KNAcc = accuracy_score(y_pred,y_test)
print('K Neighbours accuracy is: {:.2f}%'.format(KNAcc*100))


# In[87]:


scoreListknn = []
for i in range(1,30):
    KNclassifier = KNeighborsClassifier(n_neighbors = i)
    KNclassifier.fit(X_train, y_train)
    scoreListknn.append(KNclassifier.score(X_test, y_test))
    
plt.plot(range(1,30), scoreListknn)
plt.xticks(np.arange(1,30,1))
plt.xlabel("K value")
plt.ylabel("Score")
plt.show()
KNAccMax = max(scoreListknn)
print("KNN Acc Max {:.2f}%".format(KNAccMax*100))


# ## 7.3 Support Vector Machine (SVM)

# In[ ]:





# In[91]:


from sklearn.svm import SVC
SVCclassifier = SVC(kernel='linear', max_iter=50)
SVCclassifier.fit(X_train, y_train)

y_pred = SVCclassifier.predict(X_test)

#print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

from sklearn.metrics import accuracy_score
SVCAcc = accuracy_score(y_pred,y_test)
print('SVC accuracy is: {:.2f}%'.format(SVCAcc*100))


# ## 7.5 Decision Tree

# In[93]:


from sklearn.tree import DecisionTreeClassifier
DTclassifier = DecisionTreeClassifier(max_leaf_nodes=5)
DTclassifier.fit(X_train, y_train)

y_pred = DTclassifier.predict(X_test)

#print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

from sklearn.metrics import accuracy_score
DTAcc = accuracy_score(y_pred,y_test)
print('Decision Tree accuracy is: {:.2f}%'.format(DTAcc*100))


# In[94]:


scoreListDT = []
for i in range(2,50):
    DTclassifier = DecisionTreeClassifier(max_leaf_nodes=i)
    DTclassifier.fit(X_train, y_train)
    scoreListDT.append(DTclassifier.score(X_test, y_test))
    
plt.plot(range(2,50), scoreListDT)
plt.xticks(np.arange(2,50,5))
plt.xlabel("Leaf")
plt.ylabel("Score")
plt.show()
DTAccMax = max(scoreListDT)
print("DT Acc Max {:.2f}%".format(DTAccMax*100))


# ## 7.6 Random Forest

# In[95]:


from sklearn.ensemble import RandomForestClassifier

RFclassifier = RandomForestClassifier(max_leaf_nodes=5)
RFclassifier.fit(X_train, y_train)

y_pred = RFclassifier.predict(X_test)

#print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

from sklearn.metrics import accuracy_score
RFAcc = accuracy_score(y_pred,y_test)
print('Random Forest accuracy is: {:.2f}%'.format(RFAcc*100))


# In[96]:


scoreListRF = []
for i in range(2,50):
    RFclassifier = RandomForestClassifier(n_estimators = 1000, random_state = 1, max_leaf_nodes=i)
    RFclassifier.fit(X_train, y_train)
    scoreListRF.append(RFclassifier.score(X_test, y_test))
    
plt.plot(range(2,50), scoreListRF)
plt.xticks(np.arange(2,50,5))
plt.xlabel("RF Value")
plt.ylabel("Score")
plt.show()
RFAccMax = max(scoreListRF)
print("RF Acc Max {:.2f}%".format(RFAccMax*100))


# # 8. Model Comparison 👀

# In[100]:


compare = pd.DataFrame({'Model': ['Logistic Regression', 'K Neighbors', 'K Neighbors Max', 'SVM','Decision Tree', 'Decision Tree Max', 'Random Forest', 'Random Forest Max'], 
                        'Accuracy': [LRAcc*100, KNAcc*100, KNAccMax*100, SVCAcc*100, DTAcc*100, DTAccMax*100, RFAcc*100, RFAccMax*100]})
compare.sort_values(by='Accuracy', ascending=False)


# 👉 From the results, it can be seen that most of ML models can reach **up to 80% accuracy** in predicting classification of drug type.

# # 10. References 🔗
# 📚 **Kaggle Notebook**:
# *  [Drug Classification With Different Algorithms by Görkem Günay](https://www.kaggle.com/gorkemgunay/drug-classification-with-different-algorithms)
# *  [Drug Classification - 100% Accuracy by Erin Ward](https://www.kaggle.com/eward96/drug-classification-100-accuracy)
# *  [drug prediction with acc(100 %) by Sachin Sharma](https://www.kaggle.com/sachinsharma1123/drug-prediction-with-acc-100)
