#importing basic packages for eda and cleaning
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#importing the raw data 
train = pd.read_csv(r"D:\Kaggle\Titanic Dataset\all\Logistic Regression\train.csv")
test = pd.read_csv(r"D:\Kaggle\Titanic Dataset\all\Logistic Regression\test.csv")

#combining the training and test
#important as this matches by column names
master = pd.concat([train, test], ignore_index=True)

#for better view
pd.set_option("display.max_rows",500)
pd.set_option("display.max_columns",100)
pd.set_option("display.max_colwidth",20)


## EDA:

#data overview 
master.sample(10)
master.describe()
master.info()

#heatmap for missing values
sns.heatmap(master.isnull(),yticklabels = False, cbar = False)

## As we have a lot of missing Age values, imputing the Age using linear regression model

#Removing Survived as that is the final goal and can't be used to impute and Cabin as it has too much missing data
age_data = master.drop("Survived", axis = 1)
sns.heatmap(age_data.isnull(),yticklabels = False, cbar = False)
age_data = age_data.drop("Cabin", axis = 1)
sns.heatmap(age_data.isnull(),yticklabels = False, cbar = False)

#Splitting the age data into training and test
age_data_train =  age_data[age_data.Age.notnull()]
age_data_test =   age_data[age_data.Age.isnull()]  

sns.heatmap(age_data_train.isnull(),yticklabels = False, cbar = False)

#imputing the value of Fare based on Pclass
sns.boxplot(x='Pclass',y='Fare',data=age_data_train) 
age_data_train["Fare"] = age_data_train.groupby("Pclass").transform(lambda x: x.fillna(x.median()))['Fare']

sns.heatmap(age_data_train.isnull(),yticklabels = False, cbar = False)

#imputing the value of Embarked based on Pclass (More challenging):
sns.countplot(x = "Embarked",hue = "Pclass", data = age_data_train)
age_data_train["Embarked"] = age_data_train.groupby("Pclass").transform(lambda x: x.fillna(max(x.value_counts())))['Embarked'] #.mode doesnt work in group by

sns.heatmap(age_data_train.isnull(),yticklabels = False, cbar = False)

sns.heatmap(age_data_test.isnull(),yticklabels = False, cbar = False)

# data types (Encoding of categorical data)


# Spliting Features and Lebel Column
X = age_data_train.drop("Age",axis = 1)
y = age_data_train["Age"]
