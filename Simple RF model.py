# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 16:57:04 2017

@author: gyujin
"""

##----------------- feature engineering -------------------##

# registration_init time , exiration date 연, 월, 일로 분할
# train--------------------
df_train_merged.info()
df_train_merged.registration_init_time = pd.to_datetime(df_train_merged.registration_init_time, format='%Y%m%d')
df_train_merged['registration_init_time_year'] = df_train_merged['registration_init_time'].dt.year
df_train_merged['registration_init_time_month'] = df_train_merged['registration_init_time'].dt.month
df_train_merged['registration_init_time_day'] = df_train_merged['registration_init_time'].dt.day

df_train_merged.expiration_date = pd.to_datetime(df_train_merged.expiration_date,  format='%Y%m%d')
df_train_merged['expiration_date_year'] = df_train_merged['expiration_date'].dt.year
df_train_merged['expiration_date_month'] = df_train_merged['expiration_date'].dt.month
df_train_merged['expiration_date_day'] = df_train_merged['expiration_date'].dt.day

# test------------------------

df_test_merged.registration_init_time = pd.to_datetime(df_test_merged.registration_init_time, format='%Y%m%d')
df_test_merged['registration_init_time_year'] = df_test_merged['registration_init_time'].dt.year
df_test_merged['registration_init_time_month'] = df_test_merged['registration_init_time'].dt.month
df_test_merged['registration_init_time_day'] = df_test_merged['registration_init_time'].dt.day

df_test_merged.expiration_date = pd.to_datetime(df_test_merged.expiration_date,  format='%Y%m%d')
df_test_merged['expiration_date_year'] = df_test_merged['expiration_date'].dt.year
df_test_merged['expiration_date_month'] = df_test_merged['expiration_date'].dt.month
df_test_merged['expiration_date_day'] = df_test_merged['expiration_date'].dt.day



# Object data to category
df_train_merged['registration_init_time'] = df_train_merged['registration_init_time'].astype('category')
df_train_merged['expiration_date'] = df_train_merged['expiration_date'].astype('category')
df_test_merged['registration_init_time'] = df_test_merged['registration_init_time'].astype('category')
df_test_merged['expiration_date'] = df_test_merged['expiration_date'].astype('category')
for col in df_train_merged.select_dtypes(include=['object']).columns:
    df_train_merged[col] = df_train_merged[col].astype('category')
for col in df_test_merged.select_dtypes(include=['object']).columns:
    df_test_merged[col] = df_test_merged[col].astype('category')    
# Encoding categorical features
for col in df_train_merged.select_dtypes(include=['category']).columns:
    df_train_merged[col] = df_train_merged[col].cat.codes

for col in df_test_merged.select_dtypes(include=['category']).columns:
    df_test_merged[col] = df_test_merged[col].cat.codes



# RF Model (변수그냥 싹다 때려박음)
from sklearn import cross_validation, grid_search, metrics, ensemble

model = ensemble.RandomForestClassifier(n_estimators=250, max_depth=25)
model.fit(df_train_merged[df_train_merged.columns[df_train_merged.columns != 'is_train']], df_train_target)


df_train_merged_plot = pd.DataFrame({'features': df_train_merged.columns[df_train_merged.columns != 'is_train'],
                        'importances': model.feature_importances_})
df_train_merged_plot = df_train_merged_plot.sort_values('importances', ascending=False)

# importance 0.05 보다 낮은거 다 drop
df_train_test = df_train_merged.drop(df_train_merged_plot.features[df_train_merged_plot.importances < 0.05].tolist(), 1)


# train data split
train_data, test_data, train_labels, test_labels = cross_validation.train_test_split(df_train_merged, df_train_target, test_size = 0.3)
model.fit(train_data, train_labels)

# predict
pred_1 = model.predict(test_data)

# accuracy print!

print(metrics.classification_report(test_labels, pred_1))

-----------------------------------------------------------
             precision    recall  f1-score   support

          0       0.67      0.63      0.65     55065
          1       0.65      0.69      0.67     55597

avg / total       0.66      0.66      0.66    110662

-----------------------------------------------------------
