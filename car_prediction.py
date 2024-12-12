import warnings
warnings.filterwarnings('ignore')
import pandas as pd
data=pd.read_csv('car data.xls')
data.head() #Displays top 5 rows
 data.tail() #Displays bottom 5 rows
data.shape
print("Number of rows:",data.shape[0])
print("Number of columns:",data.shape[1])
#Get information like total no. of rows,cols,datatypes of each column and memory requirement
data.info()
#To check null values in the dataset
data.isnull()
data.isnull().sum() #Total no. of cols with null values..
#Get overall statistics of the dataset
data.describe()
#Data Preprocessing
data.head(1)
import datetime
date_time=datetime.datetime.now()
date_time
data['Age']=date_time.year - data['Year'] #adding a new column Age to determine the age of the car
data.head()
data.drop('Year',axis=1,inplace=True) #discarding the Year column of the dataset #axis='index' 
data.head()
import seaborn as sns 
sns.boxplot(x=data['Selling_Price'])
sorted(data['Selling_Price'],reverse=True) # Here 33 and 35 are the outliers
data=data[~(data['Selling_Price']>=33.0) & (data['Selling_Price']<=35.0)]
data.shape
#Encoding the categorical columns Ex: Gender-->male,female [columns with different categories]
data.head(1)
data['Fuel_Type'].unique()
data['Fuel_Type'] =data['Fuel_Type'].map({'Petrol':0,'Diesel':1,'CNG':2})
data['Fuel_Type'].unique()
data['Seller_Type'].unique()
data['Seller_Type'] = data['Seller_Type'].map({'Dealer':0,'Individual':1}) 
data['Seller_Type'].unique()
data['Transmission'].unique()
data['Transmission']=data['Transmission'].map({'Manual':0,'Automatic':1})
data['Transmission'].unique()
#After encoding ccategorical columns we get
data.head()
#Storing Feature Matrix in X and Response(Target) in Vector Y
X=data.drop(['Car_Name','Selling_Price'],axis=1)
Y=data['Selling_Price']
X
Y
#splitting the Dataset into training and testing set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=42)
! pip install Xgboost 
! pip install catboost
#Importing the Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
#Now we need to train the model i,e Model training
lr=LinearRegression()
lr.fit(X_train,Y_train)

rf=RandomForestRegressor()
rf.fit(X_train,Y_train)
xgb=GradientBoostingRegressor()
xgb.fit(X_train,Y_train)

xg=XGBRegressor()
xg.fit(X_train,Y_train)

#Prediction on tested data
Y_pred1=lr.predict(X_test)
Y_pred2=rf.predict(X_test)
Y_pred3=xgb.predict(X_test)
Y_pred4=xg.predict(X_test)
#Evaluation of the algorithm
from sklearn import metrics
score1=metrics.r2_score(Y_test,Y_pred1)
score2=metrics.r2_score(Y_test,Y_pred2)
score3=metrics.r2_score(Y_test,Y_pred3)
score4=metrics.r2_score(Y_test,Y_pred4)
print(score1,score2,score3,score4)
final_data=pd.DataFrame({'Models':['LR','RF','GBR','XG'],
                         "R2_SCORE":[score1,score2,score3,score4]})
final_data
sns.barplot(x=final_data['Models'],y=final_data['R2_SCORE'])
xg=XGBRegressor()
xg_final=xg.fit(X,Y)
import joblib
joblib.dump(xg_final,'car_price_predictor')
model=joblib.load('car_price_predictor')
#Prediction on new data
import pandas as pd
new_data=pd.DataFrame({
    'Present_Price':5.59,
    'Kms_Driven':27000,
    'Fuel_Type':0,
    'Seller_Type':0,
    'Transmission':0,
    'Owner':0,
    'Age':8
},index=[0])
model.predict(new_data)