
import numpy as np
import pandas as pd

!pip install keras --upgrade

import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
print(keras.__version__)

!pip install --upgrade --no-cache-dir gdown
!gdown 1jIl8uXsrdRz3ZBxljjQNo0ssDT8LbJBx


!unzip will_not_travel_again_data.zip -d .

train = pd.read_csv('./data/train.csv',engine="pyarrow")
test = pd.read_csv('./data/test.csv',engine="pyarrow")
# train.head()


# Drop column ['user'] for both train and test sets
train.drop(columns=["user"],inplace = True)
test.drop(columns=['',"user"],inplace = True)

def reduce_mem_usage(df):
    a=df.memory_usage().sum() / 1024**2
    df=df.copy()
    numerics_int = ['int16', 'int32', 'int64']
    numerics_float = ['float16', 'float32', 'float64']

    for col in df.columns:
        col_types = df[col].dtypes
        if col_types in numerics_int:
            c_min = df[col].min()
            c_max = df[col].max()
            for types in numerics_int:
                if c_min > np.iinfo(types).min and c_max < np.iinfo(types).max:
                    df[col]=df[col].astype(types)
                    break


        elif col_types in numerics_float:
            c_min = df[col].min()
            c_max = df[col].max()
            for types in numerics_float:
                if c_min > np.finfo(types).min and c_max < np.finfo(types).max:
                    df[col]= df[col].astype(types).apply(lambda x: round(x, 2))
                    break
    b = df.memory_usage().sum() / 1024**2
    print('before = ',int(a), 'after = ', int(b))
    return df

train=reduce_mem_usage(train)

# check is_booking where checkIn_date or checkOut_date is null
train['is_booking'].unique()

train['is_booking'][train['checkOut_date'].isnull()].unique()

train['is_booking'][train['checkIn_date'].isnull()].unique()

# handle missing checkOut_date where checkIn_date values
train.dropna(subset=['checkIn_date','checkOut_date'],inplace = True)

train.isna().sum()['destination_distance']

train['destination_distance'] = train['destination_distance'].fillna(train.groupby(
    ['user_location_city','destination'])['destination_distance'].transform('mean'))

train['destination_distance'].fillna(0,inplace = True)
train['destination_distance'] = train['destination_distance'].astype('int64')
train['destination_distance'].isnull().sum()

test['destination_distance'] = test['destination_distance'].fillna(train.groupby(
    ['user_location_city','destination'])['destination_distance'].transform('mean'))

test['destination_distance'].fillna(0,inplace=True)
test['destination_distance'] = test['destination_distance'].astype('int64')
test['destination_distance'].isnull().sum()

train.isna().sum()


time_columns = ['search_date', 'checkIn_date', 'checkOut_date']
for column in time_columns:
    train[column] = pd.to_datetime(train[column]) 
    test[column] = pd.to_datetime(test[column]) 



# Add Days of Stay

duration = train['checkOut_date'] - train['checkIn_date'] 
train['duration'] = duration.dt.days.astype(int) 
duration = test['checkOut_date'] - test['checkIn_date'] 
test['duration'] = duration.dt.days.astype(int) 

# Add Days between search_date and checkIn_date

days_between = train['checkIn_date'] - train['search_date'] 
train['days_between'] = days_between.dt.days.astype(int) 
days_between = test['checkIn_date'] - test['search_date'] 
test['days_between'] = days_between.dt.days.astype(int) 


train['search_date_hour'] = train['search_date'].dt.hour 
train['search_date_dayofweek'] = train['search_date'].dt.dayofweek 
train['checkIn_date_dayofweek'] = train['checkIn_date'].dt.dayofweek 
train['search_date_month'] = train['search_date'].dt.month 
train['checkIn_date_month'] = train['checkIn_date'].dt.month 

test['search_date_hour'] = test['search_date'].dt.hour 
test['search_date_dayofweek'] = test['search_date'].dt.dayofweek 
test['checkIn_date_dayofweek'] = test['checkIn_date'].dt.dayofweek 
test['search_date_month'] = test['search_date'].dt.month 
test['checkIn_date_month'] = test['checkIn_date'].dt.month 


is_booked = train[train['is_booking'] == 1]
not_booked = train[train['is_booking'] == 0]


import plotly.graph_objects as go

trace_not_booked = go.Bar(y = not_booked['search_date_hour'].value_counts().sort_index()/len(not_booked) , name='Not Booked') 
trace_is_booked = go.Bar(y = is_booked['search_date_hour'].value_counts().sort_index()/len(is_booked) , name='Booked') 

data = [trace_is_booked, trace_not_booked]

layout = go.Layout(
    xaxis=dict(title='Search Hour', tickangle=45, automargin=True),
    yaxis=dict(title='Frequency')
)

fig = go.Figure(data=data, layout=layout)
fig.show()
fig.write_json('./search_hour.json')

fig.write_html("./file1.html")

trace_not_booked = go.Bar(y = not_booked['checkIn_date_dayofweek'].value_counts().sort_index()/len(not_booked) ,
                          name='Not Booked') 
trace_is_booked = go.Bar(y = is_booked['checkIn_date_dayofweek'].value_counts().sort_index()/len(is_booked) ,
                         name='Booked') 

ticktext = ['دوشنبه', 'سه‌شنبه', 'چهارشنبه', 'پنج‌شنبه', 'جمعه', 'شنبه', 'یکشنبه'] 


data = [trace_is_booked, trace_not_booked]

layout = go.Layout(
    xaxis=dict(title='Day of Week', tickangle=45, automargin=True,
               tickvals = [0,1,2,3,4,5,6], ticktext= ticktext
 ),
    yaxis=dict(title='Frequency'),
)

fig = go.Figure(data=data, layout=layout)
fig.show()

fig.write_html("./file2.html")



trace_not_booked = go.Bar(y = not_booked['checkIn_date_month'].value_counts().sort_index()/len(not_booked) , name='Not Booked') 
trace_is_booked = go.Bar(y = is_booked['checkIn_date_month'].value_counts().sort_index()/len(is_booked) , name='Booked') 


data = [trace_is_booked, trace_not_booked]

ticktext = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

layout = go.Layout(
    xaxis=dict(title='Month', tickangle=45, automargin=True,
             ticktext = ticktext ,tickvals = np.arange(0,12)),
    yaxis=dict(title='Frequency')
)

fig = go.Figure(data=data, layout=layout)
fig.show()


fig.write_html("./file3.html")


trace_not_booked = go.Scatter(y = not_booked['days_between'].value_counts().sort_index()/len(not_booked) , name='Not Booked', opacity= 0.5) 
trace_is_booked = go.Scatter(y = is_booked['days_between'].value_counts().sort_index()/len(is_booked) , name='Booked') 


data = [trace_is_booked, trace_not_booked]

layout = go.Layout(
    xaxis=dict(title='Days between search and checking time', tickangle=45, automargin=True),
    yaxis=dict(title='Frequency')
)

fig = go.Figure(data=data, layout=layout)
fig.show()


fig.write_html("./file4.html")


trace_not_booked = go.Scatter(y = not_booked['duration'].value_counts().sort_index()/len(not_booked) , name='Not Booked', opacity= 0.5) 
trace_is_booked = go.Scatter(y = is_booked['duration'].value_counts().sort_index()/len(is_booked) , name='Booked') 

data = [trace_is_booked, trace_not_booked]

layout = go.Layout(
    xaxis=dict(title='Length of Stay', tickangle=45, automargin=True),
    yaxis=dict(title='Frequency')
)

fig = go.Figure(data=data, layout=layout)
fig.show()
fig.write_json('./los.json')

fig.write_html("./file5.html")

del trace_not_booked, trace_is_booked, is_booked, not_booked, data

# Add is_abroad column

train['is_abroad'] = train['hotel_country'] == train['user_location_country']
test['is_abroad'] = test['hotel_country'] == test['user_location_country']

train['is_abroad'] = train['is_abroad'].apply(lambda x : 1 if x == True else 0)
test['is_abroad'] = test['is_abroad'].apply(lambda x : 1 if x == True else 0)


# Preprocessing

train['checkOut_date_dayofweek'] = train['checkOut_date'].dt.day_name()
train['checkOut_date_month'] = train['checkOut_date'].dt.month_name()
test['checkOut_date_dayofweek'] = test['checkOut_date'].dt.day_name()
test['checkOut_date_month'] = test['checkOut_date'].dt.month_name()

#Preprocessing (Drop Unnecessary Columns)
train.drop(['search_date','checkIn_date','checkOut_date','destination_distance',
            'search_date_month'],
           axis = 1, inplace= True)


test.drop(['search_date','checkIn_date','checkOut_date','destination_distance',
            'search_date_month'],
           axis = 1, inplace= True)

train=reduce_mem_usage(train)
test=reduce_mem_usage(test)

y_train = train['is_booking']
train = train.loc[:, train.columns != 'is_booking']
train['is_booking'] = y_train
del y_train

# Preprocessing (SS)
columns = ['user_location_country','user_location_region','user_location_city',
           'destination','hotel_country','hotel_market']

for column in columns:
  mean = train[column].mean() # (Mean of train_data[column])
  std = train[column].std() # (Standard Deviation of train_data[column])
  train[column]=(train[column]- mean)/std
  test[column]=(test[column] - mean)/std

# Preprocessing (Make One-hotted Columns)

dummy_columns = ['channel']

train = pd.get_dummies(train, columns = dummy_columns,dtype='int8')
test = pd.get_dummies(test, columns = dummy_columns,dtype='int8')

test['channel_10'] = 0

# Preprocessing (Encode Categorical Columns)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

columns_to_encode = ['search_date_dayofweek','checkIn_date_dayofweek',
                     'checkIn_date_month','checkOut_date_dayofweek',
                     'checkOut_date_month']

for column in columns_to_encode:
    train.loc[:, column] = le.fit_transform(train[column])
    test.loc[:, column] = le.transform(test[column])

train_columns = train.columns
test = test[train_columns.drop('is_booking')]

train = train.drop(index = train[train['is_booking'] == False].sample(frac =.905).index)
train = train.astype(np.float32)
test = test.astype(np.float32)

#  Make Validation Set
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(train.drop(columns = ['is_booking'])

#  Design Model

from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
from tensorflow.keras import Input


input = Input(shape=(X_train.shape[1],))
x = layers.Dense(128, activation='relu', name ='x')(input)
x = layers.Dense(64, activation='relu', name ='x1')(x)
x = layers.Dense(32, activation='relu', name ='x2')(x)
x = layers.Dense(16, activation='relu', name ='x3')(x)
out= layers.Dense(1, activation='sigmoid', name='out')(x)

model = Model(inputs=input, outputs=out)
model.summary()

#Complie the Model
from tensorflow.keras import optimizers, losses, metrics

model.compile(optimizer= 'adam',
              loss='binary_crossentropy',metrics=["auc"])

# Train your Model
checkpoint_cb = keras.callbacks.ModelCheckpoint("my_keras_model.keras", save_best_only=True)
history = model.fit(X_train, y_train,
                    epochs=30,
                    batch_size=512,
                    validation_data=(X_valid, y_valid),
                    callbacks=[checkpoint_cb]
                      )



from sklearn.metrics import roc_auc_score
roc_auc_score(y_valid, model.predict(X_valid))
# Evaluate Model

submission = pd.DataFrame(model.predict(test), columns = ['prediction'])
