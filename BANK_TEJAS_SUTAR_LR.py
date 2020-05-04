# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 17:35:43 2020

@author: tejas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("D:\\TEJAS FORMAT\\EXCELR ASSIGMENTS\\COMPLETED\\LOGISTICS REGRESSION\\BANK FULL\\bank-full.csv")
df=pd.read_csv("D:\\TEJAS FORMAT\\EXCELR ASSIGMENTS\\COMPLETED\\LOGISTICS REGRESSION\\BANK FULL\\bank-full.csv", sep=";")
df.isnull().sum()
sns.heatmap(df.isnull(),yticklabels= False ,cbar=False ,cmap='viridis')

##############Checking for histograms####################
df.age.hist()
plt.xlabel("age")
plt.ylabel("frequency of purchases by age")
plt.title("Frequency of purchases by age")
#In age histogram data is not normally distributed.Also,Client whose age from 30-40 are most buyers.

df.job.hist()
plt.xlabel("job")
plt.ylabel("frequency of purchases by job")
plt.title("Fruency of purchases by JOB") 
#In job histogram data is not normally distributed.Also,Client whose job from management are most buyers.

df.marital.hist()
plt.xlabel("marital")
plt.ylabel("frequency of purchases by marital")
plt.title("Frequency of purchases by marital")
#In marital histogram data is not normally distributed.Also,Client whore married are most buyers.

df.education.hist()
plt.xlabel("education")
plt.ylabel("Frquency of purchases by education")
plt.title("Frquency of purchases by education")
#In education histogram data is not normally distributed.Also,Client with secondary education are most buyers.

df.month.hist()
plt.xlabel("month")
plt.ylabel("frequency of purchases by month")

#In months histogram data is not normally distributed.Also, most of buyers are from may-june.

bank=pd.get_dummies(df,drop_first=True)
columns=bank.head()


##########Splitting the data into training and testing#################
from sklearn.model_selection import train_test_split
train_data,test_data=train_test_split(bank,test_size=0.2)
train_data1=train_data.reset_index()
test_data1=test_data.reset_index()
train_data1=train_data1.drop("index",axis=1)
test_data1=test_data1.drop("index",axis=1)
train_data1
X1_train=train_data1.iloc[:,0:42]

########Buliding the logistics model############
import statsmodels.formula.api as sm
m1=sm.logit("y_yes~X1_train",data=train_data1).fit()
m1.summary()
m1.summary2()
##AIC=17326.9176#######
###Train pred####
train_pred=m1.predict(train_data1)
train_data1

#####Setting the probability value#############
train_data1["model_pred"]=np.zeros(36168)
train_data1.loc[train_pred>=0.50,"model_pred"]=1

#####Checking for model accuracy############
from sklearn.metrics import classification_report
train_classification=classification_report(train_data1["y_yes"],train_data1["model_pred"])


########Confusion matrix##############
confusion_matrix_train=pd.crosstab(train_data1["y_yes"],train_data1["model_pred"])

##########Accuracy#########
train_accuracy=(31154+780)/36168
#model ccuracy=88%

##########ROC Curve##########
from sklearn import metrics
fpr,tpr,threshold=metrics.roc_curve(train_data1["y_yes"],train_data1["model_pred"])
plt.plot(fpr,tpr);plt.xlabel("FALSE POSITIVE RATE");plt.ylabel("TRUE POSITIVE RATE")
roc_auc_train=metrics.auc(fpr,tpr)


X2_test=test_data1.iloc[:,0:42]
########Buliding the logistics model############
import statsmodels.formula.api as sm
m2=sm.logit("y_yes~X2_test",data=test_data1).fit()
m2.summary()
m2.summary2()
##AIC=4366#######

test_pred=m2.predict(test_data1)
test_data1

#####Setting the probability value#############
test_data1["model_pred0"]=np.zeros(9043)
test_data1.loc[test_pred>=0.50,"model_pred0"]=1

#####Checking for model accuracy############
from sklearn.metrics import classification_report
test_classification=classification_report(test_data1["y_yes"],test_data1["model_pred0"])


########Confusion matrix##############
confusion_matrix_test=pd.crosstab(test_data1["y_yes"],test_data1["model_pred0"])

##########Accuracy#########
test_accuracy=(7796+386)/9043
#model accuracy=90%

##########ROC Curve##########
from sklearn import metrics
fpr,tpr,threshold=metrics.roc_curve(test_data1["y_yes"],test_data1["model_pred0"])
plt.plot(fpr,tpr);plt.xlabel("FALSE POSITIVE RATE");plt.ylabel("TRUE POSITIVE RATE")
roc_auc_test=metrics.auc(fpr,tpr)

