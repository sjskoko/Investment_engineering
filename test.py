import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.datasets import load_boston

boston = load_boston()

# 데이터 세트 DF 로 변환 (데이터 입력, 컬럼 입력)
boston_df = pd.DataFrame(boston.data, columns = boston.feature_names)
print(boston.feature_names)

boston_df['PRICE'] = boston.target
print('Boston 데이터 세트 크기:', boston_df.shape)
boston_df.head()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# y 에 타겟데이터도 PRICE 넣음
# X 에 PRICE 컬럼(axis=1) 제외하고 넣음
y_target = boston_df['PRICE']
X_data = boston_df.drop(['PRICE'], axis=1, inplace=False)

# train_test_split(X값,Y값,테스트 데이터 비중)
X_train, X_test, y_train, y_test = train_test_split(X_data, y_target, test_size = 0.3,
                                                   random_state = 156)

from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import ElasticNet

# alpha = 10 으로 설정해 릿지회귀 수행
ridge = Ridge(alpha = 1)
ridge.fit(X_data, y_target)
print(ridge.score(X_test, y_test))

# elastic net 사용
elastic_net = ElasticNet(alpha=.01, l1_ratio=0.5, random_state=42)
elastic_net.fit(X_data, y_target)
print(elastic_net.score(X_test, y_test))
