#Aditya Punnai
#Linear Regression - Housing Price Prediction 

import pandas as pd
df_main = pd.read_csv(r"C:\Users\adity\Desktop\DS - Bootcamp\DS-ML-Python-Bootcamp\Python-Data-Science-and-Machine-Learning-Bootcamp\Machine Learning Sections\Linear-Regression\Ecommerce Customers")

len(df_main)
len(df_main.columns)


pd.set_option("display.max_rows",500)
pd.set_option("display.max_columns",100)
pd.set_option("display.max_colwidth",20)

df_main.head()

df_main.info()

df_main.describe()

#EDA
#simple plot if data isn't very large
import seaborn as sns
sns.pairplot(df_main)
sns.distplot(df_main["Yearly Amount Spent"])

df_main.corr()
sns.heatmap(df_main.corr(), annot = True)

#we ignore the address column for now as we don't want to bring NLP to deal with it
df_main.columns


#Feature 
X = df_main[['Avg. Session Length', 'Time on App',
       'Time on Website', 'Length of Membership']] 
#Label
y = df_main['Yearly Amount Spent']

## ** everything from below is standard control enter, no changes needed
#Train-Test Split
from sklearn.model_selection import train_test_split
#?train_test_split and copy the line which says X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from sklearn.linear_model import LinearRegression
lm = LinearRegression()

lm.fit(X_train,y_train) #we are training the model so use only training x and y

print(lm.intercept_)

lm.coef_
X_train.columns
#make a data frame
import pandas as pd

pd.DataFrame(lm.coef_,X_train.columns,columns = ["Coeff"])

# =============================================================================
# #we will use this on Boston housing dataset which is a real dataset (although old)
# #from sklearn.datasets import load_boston
# #boston = load_boston()
# #boston.keys()
# #boston['feature_names']
# =============================================================================

#How to make predictions on our test sets and grab results
predictions = lm.predict(X_test) #give features it can use to predict

#to see how good the predictions did compared to the actual y_test we can do a simple scatter plot
import matplotlib.pyplot as plt
plt.scatter(y_test,predictions)

#to check for residuals
sns.distplot((y_test - predictions)) #because its normally distributed, we know linear regression was a good choice 

#evaulation metrics: MAE, MSE, RMSE
from sklearn import metrics
metrics.mean_absolute_error(y_test,predictions)
metrics.mean_squared_error(y_test,predictions)
# room mean sqr error 
metrics.mean_squared_error(y_test,predictions)**0.5




