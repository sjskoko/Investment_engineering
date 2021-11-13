import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score

# 변수 재로드
X = pd.read_csv('./content/loan_train_preprocessed.csv')
# X = X[['term', 'initial_list_status', 'int_rate', 
# 'emp_length', 'annual_inc', 'dti', 'delinq_2yrs', 
# 'inq_last_6mths', 'revol_util', 'recoveries', 
# 'collection_recovery_fee', 'tot_cur_bal', 
# 'home_ownershipRENT', 'purposesmall_business', 
# 'purposewedding', 'earliest_cr_line2000']]

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

acc = []
# Dividing the data into train and test
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.20 , stratify =y)

x_train.shape,y_train.shape,x_test.shape,y_test.shape




# 사이킷런 래퍼 XGBoost 클래스인 XGBClassifier 임포트

evals = [(x_test, y_test)]

xgb_wrapper = XGBClassifier(n_estimators=400, learning_rate=0.1, max_depth=3)
xgb_wrapper.fit(x_train , y_train,  early_stopping_rounds=100,eval_set=evals, eval_metric="logloss",  verbose=True)

y_preds = xgb_wrapper.predict(x_test)
y_pred_proba = xgb_wrapper.predict_proba(x_test)[:, 1]

# 정확도 확인

# 수정된 get_clf_eval() 함수 
def get_clf_eval(y_test, pred=None, pred_proba=None):
    confusion = confusion_matrix( y_test, pred)
    accuracy = accuracy_score(y_test , pred)
    precision = precision_score(y_test , pred)
    recall = recall_score(y_test , pred)
    f1 = f1_score(y_test,pred)
    # ROC-AUC 추가 
    roc_auc = roc_auc_score(y_test, pred_proba)
    print('오차 행렬')
    print(confusion)
    # ROC-AUC print 추가
    print('정확도: {0:.4f}, 정밀도: {1:.4f}, 재현율: {2:.4f},\
    F1: {3:.4f}, AUC:{4:.4f}'.format(accuracy, precision, recall, f1, roc_auc))
    acc.append(accuracy)

# Dividing the data into train and test
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.20 , stratify =y)

x_train.shape,y_train.shape,x_test.shape,y_test.shape

# 사이킷런 래퍼 XGBoost 클래스인 XGBClassifier 임포트

evals = [(x_test, y_test)]

xgb_wrapper = XGBClassifier(n_estimators=400, learning_rate=0.1, max_depth=3)
xgb_wrapper.fit(x_train , y_train,  early_stopping_rounds=100,eval_set=evals, eval_metric="logloss",  verbose=True)

y_preds = xgb_wrapper.predict(x_test)
y_pred_proba = xgb_wrapper.predict_proba(x_test)[:, 1]


get_clf_eval(y_test, y_preds, y_pred_proba)


a=list(y_preds)
b=list(y_test['loan_status'])
o=0
x=0
for i in range(len(list(y_preds))):
    if a[i] == b[i]:
        o += 1
    else:
        x += 1
