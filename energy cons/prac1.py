import pandas as pd
import pandas_datareader as data
import seaborn as sns
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import sklearn.metrics 
from sklearn.metrics import mean_squared_error
plt.style.use('fivethirtyeight')

df=pd.read_csv('C:\\Users\\kirta\\Documents\\ML\\energy cons\\PJME_hourly.csv')
df=df.set_index('Datetime')
df.index=pd.to_datetime(df.index)
df.plot(style='.',figsize=(15,5),color='blue',title='pjme energy used in mw')
#plt.show()
train=df.loc[df.index < '01-01-2015']
test= df.loc[df.index >= '01-01-2015']

fig, ax = plt.subplots(figsize=(15,5))
train.plot(ax=ax, label='training set', title='Date train/test split')
test.plot(ax=ax, label='test set')
ax.axvline('01-01-2015',color='black',ls='--')
ax.legend(['Training set', 'Test set'])
df.loc[(df.index > '01-01-2010') & (df.index < '01-08-2010')].plot()
#plt.show()

def features(df):
    df = df.copy()
    df['hour']=df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df

df=features(df)

fig, ax = plt.subplots(figsize=(10, 8))
sns.boxplot(data=df, x='hour', y='PJME_MW')
ax.set_title('MW by Hour')
#plt.show()

fig, ax = plt.subplots(figsize=(10, 8))
sns.boxplot(data=df, x='month', y='PJME_MW', palette='Blues')
ax.set_title('MW by Month')
#plt.show()

train = features(train)
test = features(test)

FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year']
TARGET = 'PJME_MW'

X_train = train[FEATURES]
y_train = train[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]

reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',    
                       n_estimators=1000,
                       early_stopping_rounds=50,
                       objective='reg:linear',
                       max_depth=3,
                       learning_rate=0.01)
reg.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=100)

fi = pd.DataFrame(data=reg.feature_importances_,
             index=reg.feature_names_in_,
             columns=['importance'])
fi.sort_values('importance').plot(kind='barh', title='Feature Importance')
#plt.show()

test['prediction']=reg.predict(X_test)
df=df.merge(test[['prediction']],how='left',left_index=True,right_index=True)
df['prediction'].plot(ax=ax,style='.')
plt.legend(['Truth data','predictions'])
ax.set_title('raw data and predictions')
#plt.show()

ax = df.loc[(df.index > '04-01-2018') & (df.index < '04-08-2018')]['PJME_MW'] \
    .plot(figsize=(15, 5), title='Week Of Data')
df.loc[(df.index > '04-01-2018') & (df.index < '04-08-2018')]['prediction'] \
    .plot(style='.')
plt.legend(['Truth Data','Prediction'])
plt.show()

score = np.sqrt(mean_squared_error(test['PJME_MW'], test['prediction']))
print(f'RMSE Score on Test set: {score:0.2f}')
test['error'] = np.abs(test[TARGET] - test['prediction'])
test['date'] = test.index.date
test.groupby(['date'])['error'].mean().sort_values(ascending=False).head(10)




#print(df)

