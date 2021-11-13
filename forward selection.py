import pandas as pd
import statsmodels.api as sm
import time
import itertools 

# 데이터 로드
X = pd.read_csv('./content/loan_train_preprocessed.csv')
X = X.drop(['id', 'loan_status', 'out_prncp', 'out_prncp_inv', 'total_rev_hi_lim'], axis=1) # 무의미/연관 없는 변수 제거
y = pd.read_csv('./content/loan_train_label.csv')
y = y.drop(['id'], axis=1)

## 전진 선택법
variables = list(X.columns[:])## 설명 변수 리스트
X.columns

selected_variables = [] ## 선택된 변수들
sl_enter = 0.05
 
sv_per_step = [] ## 각 스텝별로 선택된 변수들
adjusted_r_squared = [] ## 각 스텝별 수정된 결정계수
steps = [] ## 스텝
step = 0
while len(variables) > 0:
    print(len(X.columns))
    remainder = list(set(variables) - set(selected_variables))
    pval = pd.Series(index=remainder) ## 변수의 p-value
    ## 기존에 포함된 변수와 새로운 변수 하나씩 돌아가면서 
    ## 선형 모형을 적합한다.
    print(len(X.columns))

    for col in remainder: 
        temp_X = X[selected_variables+[col]]
        temp_X = sm.add_constant(temp_X)
        model = sm.OLS(y,temp_X).fit()
        pval[col] = model.pvalues[col]
    print(len(X.columns))
 
    min_pval = pval.min()
    if min_pval < sl_enter: ## 최소 p-value 값이 기준 값보다 작으면 포함
        selected_variables.append(pval.idxmin())
        
        step += 1
        steps.append(step)
        adj_r_squared = sm.OLS(y,sm.add_constant(X[selected_variables])).fit().rsquared_adj
        adjusted_r_squared.append(adj_r_squared)
        sv_per_step.append(selected_variables.copy())
        print(len(X.columns))
    else:
        break
        print("")



'''
['int_rate', 'recoveries', 'annual_inc', 'emp_length', 
'dti', 'initial_list_status', 'collection_recovery_fee', 
'home_ownershipRENT', 'term', 'inq_last_6mths', 'tot_cur_bal', 
'delinq_2yrs', 'revol_util', 'earliest_cr_line2000', 'purposewedding', 
'purposesmall_business']
'''