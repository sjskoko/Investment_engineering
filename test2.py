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
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.20 )
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size = 0.20 )

x_train.shape,y_train.shape,x_val.shape,y_val.shape,x_test.shape,y_test.shape




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

import warnings
warnings.filterwarnings("ignore")

from keras import optimizers
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.callbacks import EarlyStopping
from matplotlib import pyplot 

model = Sequential()
model.add(Dense(2048, input_shape = (len(x_train.columns),), activation='ELU'))
#model.add(Dropout(0.3))
# model.add(Dense(4096, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(2048, activation='relu'))
# model.add(Dropout(0.3))
model.add(Dense(1024, activation='ELU'))
model.add(Dropout(0.3))
model.add(Dense(512, activation='ELU'))
model.add(Dropout(0.3))
model.add(Dense(256, activation='ELU'))
model.add(Dropout(0.3))
model.add(Dense(256, activation='ELU'))
model.add(Dropout(0.3))
model.add(Dense(256, activation='ELU'))
model.add(Dropout(0.3))
model.add(Dense(256, activation='ELU'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='ELU'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='ELU'))
model.add(Dense(1, activation = 'sigmoid'))


model.compile(optimizer = 'adam',
             loss = 'binary_crossentropy',
             metrics = ['accuracy', f1])

# Adjust the weights of the classes since your dataset is HIGHLY IMBALANCED!
class_weight = {0: 1.,
                1: 1.}

model.fit(x_train.values,
         y_train.values,
         epochs = 300,
         batch_size = 2048,
         validation_data = (x_val.values, y_val.values), class_weight=class_weight,
         callbacks=[EarlyStopping(monitor='val_f1', mode='max', patience=100, restore_best_weights=True)])

from sklearn.metrics import accuracy_score
y_prediction = model.predict(x_test.values)
y_prediction= [1 if i>=0.5 else 0 for i in y_prediction]
print("The Test Accuracy of the model is: {} %".format(accuracy_score(y_test.values, y_prediction) * 100.)) 
print()

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test.values, y_prediction))
print()

from sklearn.metrics import classification_report
target_names = ['Fully Paid', 'Charged Off']
print(classification_report(y_test, y_prediction, target_names=target_names))
