import pandas as pd
import statsmodels.api as sm
import time
import itertools 

# 데이터 로드
X = pd.read_csv('./content/loan_train_preprocessed.csv')
X = X.drop(['id', 'loan_status', 'out_prncp', 'out_prncp_inv', 'total_rev_hi_lim'], axis=1) # 무의미/연관 없는 변수 제거
y = pd.read_csv('./content/loan_train_label.csv')
y = y.drop(['id'], axis=1)

## 후진 소거법
variables = X.columns[:].tolist() ## 설명 변수 리스트
 
selected_variables = variables ## 초기에는 모든 변수가 선택된 상태
sl_remove = 0.05
 
sv_per_step = [] ## 각 스텝별로 선택된 변수들
adjusted_r_squared = [] ## 각 스텝별 수정된 결정계수
steps = [] ## 스텝
step = 0
while len(selected_variables) > 0:
    temp_X = sm.add_constant(X[selected_variables])
    p_vals = sm.OLS(y,temp_X).fit().pvalues[1:] ## 절편항의 p-value는 뺀다
    max_pval = p_vals.max() ## 최대 p-value
    if max_pval >= sl_remove: ## 최대 p-value값이 기준값보다 크거나 같으면 제외
        remove_variable = p_vals.idxmax()
        selected_variables.remove(remove_variable)
 
        step += 1
        steps.append(step)
        adj_r_squared = sm.OLS(y,sm.add_constant(X[selected_variables])).fit().rsquared_adj
        adjusted_r_squared.append(adj_r_squared)
        sv_per_step.append(selected_variables.copy())
    else:
        break



'''
['term', 'initial_list_status', 'int_rate', 
'emp_length', 'annual_inc', 'dti', 'delinq_2yrs', 
'inq_last_6mths', 'revol_util', 'recoveries', 
'collection_recovery_fee', 'tot_cur_bal', 
'home_ownershipRENT', 'purposesmall_business', 
'purposewedding', 'earliest_cr_line2000']
'''