import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv(r"C:\Users\adity\Desktop\Python\Training\ML\Logistic Regression\titanic_train.csv")
test = pd.read_csv(r"C:\Users\adity\Desktop\Python\Training\ML\Logistic Regression\titanic_test.csv")

pd.set_option("display.max_rows",500)
pd.set_option("display.max_columns",100)
pd.set_option("display.max_colwidth",20)

train.head()
train.info()
train.describe()

#check where we have the most missing data using seaborn's heatmap
train.isnull()
sns.heatmap(train.isnull(),yticklabels = False, cbar = False) #looking at this we need to impute age and maybe drop cabin

#EDA
train.columns
sns.set_style("whitegrid")

sns.countplot(x = "Survived",data = train)

sns.countplot(x = "Survived",hue = "Sex", data = train)

sns.countplot(x = "Survived",hue = "Pclass", data = train)

sns.countplot(x = "Survived",hue = "Age", data = train)

sns.distplot(train["Age"].dropna(),kde=False,bins=30)

sns.countplot(x = "SibSp", data = train)

sns.distplot(train["Fare"].dropna(),kde=False,bins=40)

#if you want interactive plots, 
#import cufflinks as cf
#cf.go_offline()
#train['Fare'].iplot(kind='hist',bins=30,color='green')

#Cleaning the data
#imputing age by using pclass average age (Hypothesis: Higher the class higher the age)
sns.boxplot(x='Pclass',y='Age',data=train) 

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age

train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)

sns.heatmap(train.isnull(),yticklabels = False, cbar = False)
#too much missing values in cabin so dropping
train.drop("Cabin",axis=1,inplace = True)
sns.heatmap(train.isnull(),yticklabels = False, cbar = False)
train.dropna(inplace=True)

train.head()
train.info()
#categorical variables: dummy variables
sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)
pclass = pd.get_dummies(train["Pclass"], drop_first = True)
train = pd.concat([train, sex,embark],axis = 1)
train.head()
train.info()

#dropping the columns we don't need, we won't be featue engineering in this one. Embarked can be engineered using NLP
train.drop(["Name","Sex","Ticket","Embarked"], axis =1, inplace = True)
train.drop(["PassengerId"], axis = 1, inplace = True)    #forgot to drop this

train = pd.concat([train, pclass],axis = 1) #forgot to add pclass

train.drop(["Pclass"], axis = 1, inplace = True)

train.head()
train.info()


#creating the actual logistic regression model:
X = train.drop("Survived",axis = 1)
y = train["Survived"]

#Assuming we just want to work with training data 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,random_state=101)

from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

predictions = logmodel.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,predictions))
