import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import ElasticNet
from sklearn.utils import validation
from sklearn.model_selection import train_test_split

X = pd.read_csv('./content/loan_train_preprocessed.csv')
X = X.drop(['id', 'loan_status', 'funded_amnt', 'installment', 'funded_amnt_inv', 'collection_recovery_fee'], axis=1)

y = pd.read_csv('./content/loan_train_label.csv')
y = y.drop(['id'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.2,
                                                    random_state=1)








# elastic net 사용
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0, random_state=42)
elastic_net.fit(X_train, y_train)
print(elastic_net.score(X_test, y_test))



X[['revol_bal', 'total_rev_hi_lim']]
X.columns
X['term']
X.count().sort_values()
import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor




# for b0, 상수항 추가
X = pd.read_csv('./content/loan_train_preprocessed.csv')
X = X.drop(['id', 'loan_status', 'out_prncp', 'out_prncp_inv', \
'funded_amnt', 'loan_amnt', 'funded_amnt_inv', 'total_rev_hi_lim'], axis=1)


x_data1 = sm.add_constant(X, has_constant = "add")

# OLS 검정
##https://todayisbetterthanyesterday.tistory.com/8
multi_model = sm.OLS(y, x_data1)
fitted_multi_model = multi_model.fit()
fitted_multi_model.summary()


# 제거 칼럼
'''
initial_list_status : 중요하지 않음
out_prncp : 의미 없음
out_prncp_inv : 의미 없음
total_rev_hi_lim : 의미 없음
funded_amnt, loan_amnt, funded_amnt_inv : 공선성 높음

'''
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
a = pd.DataFrame({'vif': vif[:]}, index=X.columns)
a



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

X.columns
['term', 'initial_list_status', 'int_rate', 
'emp_length', 'annual_inc', 'dti', 'delinq_2yrs', 
'inq_last_6mths', 'revol_util', 'recoveries', 
'collection_recovery_fee', 'tot_cur_bal', 
'home_ownershipRENT', 'purposesmall_business', 
'purposewedding', 'earliest_cr_line2000']

['int_rate', 'recoveries', 'annual_inc', 'emp_length', 
'dti', 'initial_list_status', 'collection_recovery_fee', 
'home_ownershipRENT', 'term', 'inq_last_6mths', 'tot_cur_bal', 
'delinq_2yrs', 'revol_util', 'earliest_cr_line2000', 'purposewedding', 
'purposesmall_business']

y = pd.read_csv('./content/loan_train_label.csv')
y = y.drop(['id'], axis=1)


# Dividing the data into train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.20 , stratify =y)

x_train.shape,y_train.shape,x_test.shape,y_test.shape





##https://towardsdatascience.com/end-to-end-case-study-classification-lending-club-data-489f8a1b100a
# Trying few more ML algos to find the best fit model
# Importing libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn import model_selection
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(rc={'figure.figsize':(3,3)})
sns.set_style('whitegrid')

import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)


DT=DecisionTreeClassifier(criterion='gini',random_state = 100,max_depth=5,class_weight='balanced', min_samples_leaf=5)
RF=RandomForestClassifier(criterion='gini',random_state = 100,max_depth=5,class_weight='balanced', min_samples_leaf=5,n_estimators=20)
Bagged=BaggingClassifier(n_estimators=50)
AdaBoost=AdaBoostClassifier(n_estimators=50)
GBoost=GradientBoostingClassifier(n_estimators=50)

models = []

models.append(('DT',DT))
models.append(('RandomForest',RF))
models.append(('Bagged',Bagged))
models.append(('AdaBoost',AdaBoost))
#models.append(('AdaBoostRF',AB_RF))
models.append(('GradientBoost',GBoost))

# evaluating each model in turn using KFold CV with 3 splits
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(shuffle=True,n_splits=3,random_state=0)
	cv_results = model_selection.cross_val_score(model, x_train, y_train,cv=kfold, scoring='f1_micro')
	results.append(cv_results)
	names.append(name)
	print("%s: %f (%f)" % (name, 1-np.mean(cv_results),np.std(cv_results,ddof=1)))
   # boxplot algorithm comparison
fig = plt.figure()
sns.set_style('whitegrid')
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
sns.set(rc={'figure.figsize':(10,10)})
plt.xticks(rotation='horizontal')
plt.ylabel('Weighted F1 Score')
ax.set_xticklabels(names)
plt.show()



# XGboost
## https://libertegrace.tistory.com/entry/Classification-4-%EC%95%99%EC%83%81%EB%B8%94-%ED%95%99%EC%8A%B5Ensemble-Learning-Boosting3-XGBoost


# Dividing the data into train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.20 , stratify =y)

x_train.shape,y_train.shape,x_test.shape,y_test.shape




# 사이킷런 래퍼 XGBoost 클래스인 XGBClassifier 임포트
from xgboost import XGBClassifier

evals = [(x_test, y_test)]

xgb_wrapper = XGBClassifier(n_estimators=400, learning_rate=0.1, max_depth=3)
xgb_wrapper.fit(x_train , y_train,  early_stopping_rounds=100,eval_set=evals, eval_metric="logloss",  verbose=True)

y_preds = xgb_wrapper.predict(x_test)
y_pred_proba = xgb_wrapper.predict_proba(x_test)[:, 1]

# 정확도 확인
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score

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

get_clf_eval(y_test, y_preds, y_pred_proba)

#feature 중요도도 그려볼 수 있다. 
from xgboost import plot_importance
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 12))
# 사이킷런 래퍼 클래스를 입력해도 무방. 
plot_importance(xgb_wrapper, ax=ax)
plt.show()