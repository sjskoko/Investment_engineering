import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout
from tensorflow.keras.constraints import max_norm
import matplotlib.pyplot as plt
import numpy as np

# 변수 재로드
X = pd.read_csv('./content/loan_train_preprocessed.csv')
X = X.drop(['id', 'loan_status', 'out_prncp', 'out_prncp_inv', \
'funded_amnt', 'loan_amnt', 'funded_amnt_inv', 'total_rev_hi_lim'], axis=1)
X = X.drop(['home_ownershipRENT', 'home_ownershipMORTGAGE', 'home_ownershipOTHER',
       'home_ownershipOWN', 'home_ownershipNONE', 'purposedebt_consolidation',
       'purposecredit_card', 'purposehome_improvement',
       'purposesmall_business', 'purposeother', 'purposemajor_purchase',
       'purposewedding', 'purposecar', 'purposehouse', 'purposemoving',
       'purposemedical', 'purposerenewable_energy', 'purposevacation',
       'purposeeducational', 'earliest_cr_line2010', 'earliest_cr_line1990',
       'earliest_cr_line2000', 'earliest_cr_line1970', 'earliest_cr_line1980',
       'earliest_cr_line1960', 'earliest_cr_line1950'], axis=1)

y = pd.read_csv('./content/loan_train_label.csv')
y = y.drop(['id'], axis=1)


# Dividing the data into train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.20 ,random_state = 2)
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size = 0.20 ,random_state = 2)

x_train.shape,y_train.shape,x_test.shape,y_test.shape




from keras import backend as K

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# model building
model = Sequential()


# input layer
model.add(Dense(4096,  activation='relu'))
model.add(Dropout(0.2))
# input layer
model.add(Dense(2048,  activation='relu'))
model.add(Dropout(0.2))
# input layer
model.add(Dense(1024,  activation='relu'))
model.add(Dropout(0.2))
# input layer
model.add(Dense(512,  activation='relu'))
model.add(Dropout(0.2))

# hidden layer
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))

# hidden layer
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))

# hidden layer
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))

# output layer
model.add(Dense(units=1,activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy', f1])


# model training
model.fit(x=x_train, 
          y=y_train, 
          epochs=20,
          batch_size=128,
          validation_data=(x_val, y_val), 
          )

# result
# losses = pd.DataFrame(model.history.history)
# losses[['loss','val_loss']].plot()
# plt.show()


from sklearn.metrics import classification_report,confusion_matrix
predictions = model.predict(x_test)
predictions=np.argmax(predictions,axis=1)
print(classification_report(y_test,predictions))

confusion_matrix(y_test,predictions)

len(y_test)
print(predictions)