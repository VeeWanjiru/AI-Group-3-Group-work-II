# data analysis libraries
import pandas as pd
import numpy as np
import random as rnd

# data visualization libraries

import seaborn as sns
import matplotlib.pyplot as plt


# scaling and train test split libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# cModel design
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam

# evaluation metrics
from sklearn.metrics \
    import mean_squared_error,mean_absolute_error,explained_variance_score
from sklearn.metrics import classification_report,confusion_matrix

#loading the dataset

dataset= pd.read_csv("C:\ Users\ virgi\PycharmProjects\ANNgroupwork2\heart_failure_clinical_records_dataset")

#analysing the features

print(dataset.columns.values)

#checking for null values
dataset.isnull().sum()

#checking the datatype of all the columns(features)
dataset.info()

#dropping an ID and Zip code columns

df = dataset.drop('id',axis=1)
df=dataset.drop('zipcode',axis=1)

#Feature engineering
df['date'] = pd.to_datetime(df['date'])
df['month']= df['date'].apply(lambda date:date.month)
df['year'] = df['date'].apply(lambda date:date.year)
df = df.drop('date', axis=1)

#We can now view the new columns

print(df.columns.values)

#specifying independent and dependent variables
x = df.drop('price',axis=1)
y=df['price']

#splitting the dataset into train and validation

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=101)

#Data normalization/scaling

scalar = MinMaxScalar()

X_train = scalar.fit_transform(X_train)
X_test=scalar.transform(X_test)

print('Max:' , X_train.max())
print('Min',X_train.min())

model = Sequential()
model.add(Dense(19,activation='relu'))

#hidden layers

model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))

model.add(Dense(1))
model.compile(optimizer:='adam',loss='mse')

print(model)

model.fit(x=X_train,y=y_train.values),
validation_data=(X_test,y_test.values),


losses=pd.DataFrame(model.history.history)
plt.figure(figsize=(15,5))
sns.lineplot(data=losses,lw=3)
plt.xlabel('Epochs')
ply.ylabel('')
plt.title('Training Loss per Epoch')
sns.despine()

predictions = model.predict(X_test)
print('MAE :'),mean_absolute_error(y_test,predictions)
print('MSE: ',mean_squared_error(y_test,predictions))
print('RMSE: ',np.sqrt(mean_squared_error(y_test,predictions)))
print('Variance Regression Score:',explained_variance_score(y_test,predictions))
print('\n\nDescriptive Statistics:\n', df['price'].describe())

f, axes =plt.subplots(1,2,figsize=(15,5))
plt.scatter(y_test,predictions)
plt.plot(y_test,y_test,'r')
errors = y_test.values.reshape(6484,1)-prediction
sns.distplot(errors,ax=axes[0])
sns.despine(left=True,bottom=True)
axes[0].set(xlabel='Error',ylabel='',title="Error Histogram")
axes[0].set(xlabel='Test True Y',ylabel='Model Predictions',title="Model Predictions vs perfect fit ")

heart_failure_rate=df('rate',axis=1).iloc[0]
print('heart_failure_rate:\n{heart_failure_rate}')
heart_failure_rate=scalar.transform(heart_failure_rate.values.reshape(-1,20))

print('\n Prediction',model.predict(heart_failure_rate)[0,1])
print('\n Original ',df.iloc[0]['heart_failure rate'])


