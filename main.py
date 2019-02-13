import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


df_train = pd.read_csv('train.csv')

df_test = pd.read_csv('test.csv')

labelencoder = LabelEncoder()

labels = labelencoder.fit_transform(df_train['species'])

trainX = df_train.drop(['species', 'id'], axis=1)
testX = df_test.drop(['id'], axis=1)

scaler = StandardScaler()
trainX = scaler.fit_transform(trainX)
testX = scaler.fit(trainX)



# Split the data into train and test. The stratify parm will ensure train and test will have the same proportions of class labels as the input dataset
#x_train, x_test, y_train, y_test = train_test_split(train, df_train['species'], test_size=0.2, stratify=np.array(df_train['breed']), random_state=100)


print(trainX)