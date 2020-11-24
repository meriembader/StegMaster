import lightgbm as lgb

lgb_params = {
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'n_jobs': -1,
    'learning_rate': 0.007,
    'num_leaves': 2**8,
    'max_depth': -1,
    'tree_learner': 'serial',
    'colsample_bytree': 0.7,
    'subsample_freq': 1,
    'subsample': 0.7,
    'n_estimators': 10,
    'max_bin': 255,
    'verbose': -1,
    'seed': 0
}

model = lgb.LGBMClassifier(**lgb_params)
