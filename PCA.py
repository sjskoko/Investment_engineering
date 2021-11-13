from sklearn.decomposition import PCA
import pandas as pd

# 데이터 로드
X = pd.read_csv('./content/loan_train_preprocessed.csv')
X = X.drop(['id', 'loan_status', 'out_prncp', 'out_prncp_inv', 'total_rev_hi_lim'], axis=1) # 무의미/연관 없는 변수 제거
X = X.drop(['funded_amnt', 'loan_amnt', 'funded_amnt_inv'], axis=1) # 공선성 높은 변수 제거
X = X.drop(['home_ownershipRENT', 'home_ownershipMORTGAGE', 'home_ownershipOTHER',
       'home_ownershipOWN', 'home_ownershipNONE'], axis=1) # categorical 변수 제거 (home_ownership)
X = X.drop(['purposedebt_consolidation',
       'purposecredit_card', 'purposehome_improvement',
       'purposesmall_business', 'purposeother', 'purposemajor_purchase',
       'purposewedding', 'purposecar', 'purposehouse', 'purposemoving',
       'purposemedical', 'purposerenewable_energy', 'purposevacation',
       'purposeeducational'], axis=1) # categorical 변수 제거 (purpose)
X = X.drop(['earliest_cr_line2010', 'earliest_cr_line1990',
       'earliest_cr_line2000', 'earliest_cr_line1970', 'earliest_cr_line1980',
       'earliest_cr_line1960', 'earliest_cr_line1950'], axis=1) # categorical 변수 제거 (purpose)



# X = X.drop(['id', 'loan_status', 'out_prncp', 'out_prncp_inv', \
# 'funded_amnt', 'loan_amnt', 'funded_amnt_inv', 'total_rev_hi_lim'], axis=1)
# X = X.drop(['home_ownershipRENT', 'home_ownershipMORTGAGE', 'home_ownershipOTHER',
#        'home_ownershipOWN', 'home_ownershipNONE', 'purposedebt_consolidation',
#        'purposecredit_card', 'purposehome_improvement',
#        'purposesmall_business', 'purposeother', 'purposemajor_purchase',
#        'purposewedding', 'purposecar', 'purposehouse', 'purposemoving',
#        'purposemedical', 'purposerenewable_energy', 'purposevacation',
#        'purposeeducational', 'earliest_cr_line2010', 'earliest_cr_line1990',
#        'earliest_cr_line2000', 'earliest_cr_line1970', 'earliest_cr_line1980',
#        'earliest_cr_line1960', 'earliest_cr_line1950'], axis=1)

y = pd.read_csv('./content/loan_train_label.csv')
y = y.drop(['id'], axis=1)





# 주성분 분석
pca = PCA(n_components=37) # 주성분을 몇개로 할지 결정
printcipalComponents = pca.fit_transform(X)

# 주성분으로 이루어진 데이터 프레임 구성
principalDf = pd.DataFrame(data=printcipalComponents, 
                        columns = [f'principal component{i}' for i in range(37)]) 

# 변수 설명력 확인
pca.explained_variance_ratio_
sum(pca.explained_variance_ratio_)


# 변수 설명력 확인 그래프
import matplotlib.pyplot as plt

plt.plot([i for i in range(len(pca.explained_variance_ratio_))], pca.explained_variance_ratio_)
plt.show()

# 누적 변수 설명력 확인
for i in range(len(pca.explained_variance_ratio_)):
    print(i, sum(pca.explained_variance_ratio_[:i]))

# 누적 변수 설명력 확인 그래프
plt.plot([i for i in range(len(pca.explained_variance_ratio_))], [sum(pca.explained_variance_ratio_[:i]) for i in range(len(pca.explained_variance_ratio_))])
plt.show()

# n=19 주성분 분석 진행
pca = PCA(n_components=19) # 주성분을 몇개로 할지 결정
printcipalComponents = pca.fit_transform(X)

# 주성분으로 이루어진 데이터 프레임 구성
principalDf = pd.DataFrame(data=printcipalComponents, 
                        columns = [f'principal component{i}' for i in range(19)]) 


#################################

# Dividing the data into train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(principalDf,y,test_size = 0.20 , stratify =y)

x_train.shape,y_train.shape,x_test.shape,y_test.shape





# xgboost 실행

# XGboost
## https://libertegrace.tistory.com/entry/Classification-4-%EC%95%99%EC%83%81%EB%B8%94-%ED%95%99%EC%8A%B5Ensemble-Learning-Boosting3-XGBoost
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

# fig, ax = plt.subplots(figsize=(10, 12))
# # 사이킷런 래퍼 클래스를 입력해도 무방. 
# plot_importance(xgb_wrapper, ax=ax)
# plt.show()

'''
정확도는 공선성 변수까지 제거했을 때
약 0.66~0.67 수준에서 형성
차원축소보다 공선성에 따른 변수 선택이 더 효율적
'''