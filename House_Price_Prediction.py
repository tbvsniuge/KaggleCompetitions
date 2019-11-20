import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

train = pd.read_csv('DataSource/train.csv')
test = pd.read_csv('DataSource/test.csv')

train_x = train.drop('SalePrice', axis=1)
train_y = train['SalePrice']
train_test_raw = train_x.append(test)

train_test_raw_c2n = train_test_raw.copy()

train_test_raw_c2n = train_test_raw_c2n.drop(['Street', 'Utilities', 'Condition2', 'RoofMatl', 'Heating'], axis=1)

lot_frontage_by_neighborhood_all = train_test_raw_c2n["LotFrontage"].groupby(train_test_raw_c2n["Neighborhood"])
for key, group in lot_frontage_by_neighborhood_all:
    lot_f_nulls_nei = train_test_raw_c2n['LotFrontage'].isnull() & (train_test_raw_c2n['Neighborhood'] == key)
    train_test_raw_c2n.loc[lot_f_nulls_nei, 'LotFrontage'] = group.median()

dic_num = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}

train_test_raw_c2n['MSSubClass'] = train_test_raw_c2n['MSSubClass'].astype('category')

cat_hasnull = [col for col in train_test_raw_c2n.select_dtypes(['object']) if train_test_raw_c2n[col].isnull().any()]
cat_hasnull.remove('Electrical')  # 实际意义上，Electrical不允许空
null_idx_el = train_test_raw_c2n['Electrical'].isnull()
train_test_raw_c2n.loc[null_idx_el, 'Electrical'] = 'SBrkr'
for col in cat_hasnull:
    null_idx = train_test_raw_c2n[col].isnull()
    train_test_raw_c2n.loc[null_idx, col] = 'None'

has_rank = [col for col in train_test_raw_c2n if 'TA' in list(train_test_raw_c2n[col])]

train_test_raw_c2n = pd.get_dummies(train_test_raw_c2n)
dtype_df = train_test_raw_c2n.dtypes.reset_index()
dtype_df.columns = ["Count", "Column Type"]
train_test_cols = train_test_raw_c2n.select_dtypes(include=['number']).columns
train_test_raw_c2n = train_test_raw_c2n[train_test_cols]

train_test_raw_c2n = train_test_raw_c2n.fillna(train_test_raw_c2n.median())
train_med = train_test_raw_c2n[:1460]
test_med = train_test_raw_c2n[1460:]
# skew
from scipy import stats

cols = [col for col in train_med if '_2num' in col or '_' not in col]
skew = [abs(stats.skew(train_med[col])) for col in train_med if '_2num' in col or '_' not in col]

skews = pd.DataFrame()
skews['Columns'] = cols
skews['Skew_Magnintudes'] = skew

cols_unskew = skews[skews.Skew_Magnintudes > 1].Columns

train_unskew2 = train_med.copy()
test_unskew2 = test_med.copy()

for col in cols_unskew:
    train_unskew2[col] = np.log1p(train_med[col])

for col in cols_unskew:
    test_unskew2[col] = np.log1p(test_med[col])

bonf_outlier = [88, 462, 523, 588, 632, 968, 1298, 1324]

train_unskew3 = train_unskew2.drop(bonf_outlier)
y_train = train_y.drop(bonf_outlier)
y_train = np.log1p(y_train)
drop_cols = ["MSSubClass_160", "MSZoning_C (all)"]

train_unskew3 = train_unskew3.drop(drop_cols, axis=1)
test_unskew2 = test_unskew2.drop(drop_cols, axis=1)

X_train = train_unskew3.drop(['Id'], axis=1)

X_test = test_unskew2.drop(['Id'], axis=1)

model = XGBRegressor(learning_rate=0.05, n_estimators=560, max_depth=5, min_child_weight=3, gamma=0, subsample=0.4,
                     colsample_bytree=0.4, objective='reg:squarederror', nthread=4, scale_pos_weight=1, seed=0,
                     reg_alpha=0, reg_lambda=1)

from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
LCV = LassoCV()
scale_LCV = Pipeline([('scaler', scaler), ('LCV', LCV)])

# cvscore = cross_val_score(cv=5, estimator=scale_LCV, X=X_train, y=y_train, n_jobs=1)
# print('CV Score is: ' + str(np.mean(cvscore)))

model.fit(X_train,y_train)
scale_LCV.fit(X_train, y_train)

preds_XGB = np.expm1(model.predict(X_test));
preds_LCV = np.expm1(scale_LCV.predict(X_test));

preds_mix = (preds_XGB + preds_LCV) / 2


out_preds_XGB = pd.DataFrame()
out_preds_XGB['Id'] = test['Id']
out_preds_XGB['SalePrice'] = preds_XGB
out_preds_XGB.to_csv('preds_XGB.csv', index=False)

out_preds_LCV = pd.DataFrame()
out_preds_LCV['Id'] = test['Id']
out_preds_LCV['SalePrice'] = preds_LCV
out_preds_LCV.to_csv('preds_LCV.csv', index=False)

out_preds_MIX = pd.DataFrame()
out_preds_MIX['Id'] = test['Id']
out_preds_MIX['SalePrice'] = preds_mix
out_preds_MIX.to_csv('preds_mix.csv', index=False)
