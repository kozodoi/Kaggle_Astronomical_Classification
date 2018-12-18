import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import logging
import itertools

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

#modify to work with kfold
#def smoteAdataset(Xig, yig, test_size=0.2, random_state=0):
#def smoteAdataset(Xig_train, yig_train, Xig_test, yig_test):
#    sm=SMOTE(random_state=2)
#    Xig_train_res, yig_train_res = sm.fit_sample(Xig_train, yig_train.ravel())
#    return Xig_train_res, pd.Series(yig_train_res), Xig_test, pd.Series(yig_test)

def create_logger():
    logger_ = logging.getLogger('main')
    logger_.setLevel(logging.DEBUG)
    fh = logging.FileHandler('simple_lightgbm.log')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(levelname)s]%(asctime)s:%(name)s:%(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger_.addHandler(fh)
    logger_.addHandler(ch)

def get_logger():
    return logging.getLogger('main')

def lgb_multi_weighted_logloss(y_true, y_preds):
    classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
    class_weight = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1}
    if len(np.unique(y_true)) > 14:
        classes.append(99)
        class_weight[99] = 2
    y_p = y_preds.reshape(y_true.shape[0], len(classes), order='F')
    y_ohe = pd.get_dummies(y_true)
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1 - 1e-15)
    y_p_log = np.log(y_p)
    y_log_ones = np.sum(y_ohe.values * y_p_log, axis=0)
    nb_pos = y_ohe.sum(axis=0).values.astype(float)
    class_arr = np.array([class_weight[k] for k in sorted(class_weight.keys())])
    y_w = y_log_ones * class_arr / nb_pos
    loss = - np.sum(y_w) / np.sum(class_arr)
    return 'wloss', loss, False


def multi_weighted_logloss(y_true, y_preds):
    classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
    class_weight = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1}
    if len(np.unique(y_true)) > 14:
        classes.append(99)
        class_weight[99] = 2
    y_p = y_preds
    y_ohe = pd.get_dummies(y_true)
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1 - 1e-15)
    y_p_log = np.log(y_p)
    y_log_ones = np.sum(y_ohe.values * y_p_log, axis=0)
    nb_pos = y_ohe.sum(axis=0).values.astype(float)
    class_arr = np.array([class_weight[k] for k in sorted(class_weight.keys())])
    y_w = y_log_ones * class_arr / nb_pos
    loss = - np.sum(y_w) / np.sum(class_arr)
    return loss

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def predict_chunk(df_, clfs_, meta_, features, train_mean):
    df_, aux_df_ = preprocess_ts_df(df_)
    
    auxs = make_features(df_, aux_df_)
    
    aggs = get_aggregations()
    aggs = get_aggregations()
    new_columns = get_new_columns(aggs)
    agg_ = df_.groupby('object_id').agg(aggs)
    agg_.columns = new_columns
    agg_ = add_features_to_agg(df=agg_)
    
    full_test = agg_.reset_index().merge(
        right=meta_,
        how='left',
        on='object_id'
    )
    
    for aux in auxs:
        full_test = pd.merge(full_test, aux, on='object_id', how='left')

    full_test = postprocess_df(full_test)

    #full_test = full_test.fillna(train_mean)
    preds_ = None
    for clf in clfs_:
        if preds_ is None:
            preds_ = clf.predict_proba(full_test[features]) / len(clfs_)
        else:
            preds_ += clf.predict_proba(full_test[features]) / len(clfs_)
    preds_99 = np.ones(preds_.shape[0])
    for i in range(preds_.shape[1]):
        preds_99 *= (1 - preds_[:, i])
    preds_df_ = pd.DataFrame(preds_, columns=['class_' + str(s) for s in clfs_[0].classes_])
    preds_df_['object_id'] = full_test['object_id']
    preds_df_['class_99'] = 0.14 * preds_99 / np.mean(preds_99) 
    print(preds_df_['class_99'].mean())
    del agg_, full_test, preds_
    gc.collect()
    return preds_df_


def save_importances(importances_):
    mean_gain = importances_[['gain', 'feature']].groupby('feature').mean()
    importances_['mean_gain'] = importances_['feature'].map(mean_gain['gain'])
    plt.figure(figsize=(8, 12))
    sns.barplot(x='gain', y='feature', data=importances_.sort_values('mean_gain', ascending=False))
    plt.tight_layout()
    plt.savefig('importances.png')


def train_classifiers(full_train=None, y=None):

    folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)
    clfs = []
    importances = pd.DataFrame()
    
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_class': 14,
        'metric': 'multi_logloss',
        'learning_rate': 0.03,
        'subsample': .9,
        'colsample_bytree': .6,
        'reg_alpha': .01,
        'reg_lambda': .01,
        'min_split_gain': 0.02,
        'min_child_weight': 5,
        'n_estimators': 10000,
        'silent': -1,
        'verbose': -1,
        'max_depth': 3,
        'seed': 159
    }
    oof_preds = np.zeros((len(full_train), np.unique(y).shape[0]))
    full_ids = np.zeros(len(full_train))
    w = y.value_counts()
    ori_weights  = {i : np.sum(w) / w[i] for i in w.index}
    weights = {i : np.sum(w) / w[i] for i in w.index}
    classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
    class_weight = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1}
    for value in classes:
        weights[value] = weights[value] * class_weight[value]
    for fold_, (trn_, val_) in enumerate(folds.split(y, y)):
        lgb_params['seed'] += fold_ 
        trn_x, trn_y = full_train.iloc[trn_], y.iloc[trn_]
        val_x, val_y = full_train.iloc[val_], y.iloc[val_]
        full_ids[val_] = val_x['object_id']
        del val_x['object_id'], trn_x['object_id']
        
#        trn_xa, trn_y, val_xa, val_y=smoteAdataset(trn_x.values, trn_y.values, val_x.values, val_y.values)
#        trn_x=pd.DataFrame(data=trn_xa, columns=trn_x.columns)
#        val_x=pd.DataFrame(data=val_xa, columns=val_x.columns)
        
        clf = lgb.LGBMClassifier(**lgb_params)
        clf.fit(
            trn_x, trn_y,
            eval_set=[(trn_x, trn_y), (val_x, val_y)],
            eval_metric=lgb_multi_weighted_logloss,
            verbose=100,
            early_stopping_rounds=50,
            sample_weight=trn_y.map(weights)
        )
        oof_preds[val_, :] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)
        get_logger().info(multi_weighted_logloss(val_y, clf.predict_proba(val_x, num_iteration=clf.best_iteration_)))

        imp_df = pd.DataFrame()
        imp_df['feature'] = trn_x.columns
        imp_df['gain'] = clf.feature_importances_
        imp_df['fold'] = fold_ + 1
        importances = pd.concat([importances, imp_df], axis=0, sort=False)

        clfs.append(clf)
    get_logger().info('MULTI WEIGHTED LOG LOSS : %.5f ' % multi_weighted_logloss(y_true=y, y_preds=oof_preds))
    preds_df_ = pd.DataFrame(oof_preds, columns=['class_' + str(s) for s in clfs[0].classes_])
    preds_df_['object_id'] = full_ids
    print(preds_df_.head())
    preds_df_.to_csv("oof_predictions.csv", index=False)
    unique_y = np.unique(y)
    class_map = dict()
    for i,val in enumerate(unique_y):
        class_map[val] = i
            
    y_map = np.zeros((y.shape[0],))
    y_map = np.array([class_map[val] for val in y])
    
    # Compute confusion matrix
    from sklearn.metrics import confusion_matrix
    cnf_matrix = confusion_matrix(y_map, np.argmax(oof_preds,axis=-1))
    np.set_printoptions(precision=2) 
    
    sample_sub = pd.read_csv('../input/sample_submission.csv')
    class_names = list(sample_sub.columns[1:-1])
    del sample_sub;gc.collect()
    
    # Plot non-normalized confusion matrix
    plt.figure(figsize=(12,12))
    foo = plot_confusion_matrix(cnf_matrix, classes=class_names,normalize=True,
                          title='Confusion matrix')
    return clfs, importances

def get_aggregations():
    return {
        'flux': ['min', 'max', 'mean', 'median', 'std', 'skew'],
        'flux_err': ['min', 'max', 'mean', 'median', 'std', 'skew'],
        'detected': ['sum'],
        'flux_ratio_sq': ['sum','skew'],
        'flux_by_flux_ratio_sq': ['sum','skew'],
    }

def get_new_columns(aggs):
    return [k + '_' + agg for k in aggs.keys() for agg in aggs[k]]

def add_features_to_agg(df):
    df['flux_diff'] = df['flux_max'] - df['flux_min']
    df['flux_dif2'] = (df['flux_max'] - df['flux_min']) / df['flux_mean']
    df['flux_w_mean'] = df['flux_by_flux_ratio_sq_sum'] / df['flux_ratio_sq_sum']
    df['flux_dif3'] = (df['flux_max'] - df['flux_min']) / df['flux_w_mean']
    return df

def agg_per_obj_passband(df, col, agg):
    aux = df[['object_id','passband']+[col]]
    aggs = {col: [agg]}
    aux = df.groupby(['object_id','passband']).agg(aggs).reset_index()
    new_df = pd.DataFrame()
    new_df['object_id'] = aux['object_id'].unique()
    for x in range(0,6):
        new_aux = aux[aux['passband'] == x]
        del new_aux['passband']
        new_aux.columns = ['object_id',col+'_'+agg+'_passband_'+str(x)]
        new_df = pd.merge(new_df, new_aux, on='object_id', how='left')
        new_df = new_df.fillna(0)
    return new_df
    
def mjd_diff_detected(df, col):
    mjd_max = df.groupby('object_id')[col].max().reset_index()
    mjd_min = df.groupby('object_id')[col].min().reset_index()
    mjd_max.columns = ['object_id',col+'_max']
    mjd_min.columns = ['object_id',col+'_min']
    df = pd.merge(df, mjd_max, on='object_id', how='left')
    df = pd.merge(df, mjd_min, on='object_id', how='left')
    df[col+'_diff_detected'] = df[col+'_max'] - df[col+'_min']
    aux_df = df.groupby('object_id')[col+'_diff_detected'].max().reset_index()
    return aux_df
    
def mjd_diff2_detected(df, col):
    mjd_max = df.groupby('object_id')[col].max().reset_index()
    mjd_min = df.groupby('object_id')[col].min().reset_index()
    mjd_mean = df.groupby('object_id')[col].mean().reset_index()
    mjd_max.columns = ['object_id',col+'_max']
    mjd_min.columns = ['object_id',col+'_min']
    mjd_mean.columns = ['object_id',col+'_mean']
    df = pd.merge(df, mjd_max, on='object_id', how='left')
    df = pd.merge(df, mjd_min, on='object_id', how='left')
    df = pd.merge(df, mjd_mean, on='object_id', how='left')
    df[col+'_diff2_detected'] = (df[col+'_max'] - df[col+'_min']) / df[col+'_mean']
    aux_df = df.groupby('object_id')[col+'_diff2_detected'].max().reset_index()
    return aux_df

def mjd_diff_detected_passband(df, col):
    mjd_max = df.groupby(['object_id','passband'])[col].max().reset_index()
    mjd_min = df.groupby(['object_id','passband'])[col].min().reset_index()
    mjd_max.columns = ['object_id','passband',col+'_max']
    mjd_min.columns = ['object_id','passband',col+'_min']
    df = pd.merge(df, mjd_max, on=['object_id','passband'], how='left')
    df = pd.merge(df, mjd_min, on=['object_id','passband'], how='left')
    df[col+'_diff'] = df[col+'_max'] - df[col+'_min']
    aux = df.groupby(['object_id','passband'])[col+'_diff'].max().reset_index()
    new_df = pd.DataFrame()
    new_df['object_id'] = aux['object_id'].unique()
    for x in range(0,6):
        new_aux = aux[aux['passband'] == x]
        del new_aux['passband']
        new_aux.columns = ['object_id',col+'_detected_passband_'+str(x)]
        new_df = pd.merge(new_df, new_aux, on='object_id', how='left')
        new_df = new_df.fillna(0)
    return new_df
    
def flux_around_max_passband(df, tam_window):
	max_flux = df.groupby('object_id')['flux'].max().reset_index()
	max_flux.columns = ['object_id','max_flux_obj']
	df = pd.merge(df, max_flux, on='object_id', how='left')
	df['RWMF'] = 0
	df['RWMF'][df['flux'] == df['max_flux_obj']] = 1
	df = df.sort_values(['object_id','mjd'])
	max_mjd = df[df['RWMF'] == 1]
	max_mjd = max_mjd[['object_id','mjd']]
	max_mjd.columns = ['object_id','mjd_where_flux_max']
	df = pd.merge(df, max_mjd, on='object_id', how='left')
	df['time_walk'] = df['mjd_where_flux_max'] + tam_window 
	df['mjd_where_flux_max'] = df['mjd_where_flux_max'] - tam_window
	aux_df = df[(df['mjd'] > df['mjd_where_flux_max'])&(df['mjd'] < df['time_walk'])]
	aux_df = aux_df[['object_id','passband','flux']]
	rtn_df = aux_df.groupby(['object_id','passband'])['flux'].mean().reset_index()
	q_df = pd.DataFrame()
	q_df['object_id'] = rtn_df['object_id'].unique()
	for x in range(0,6):
		p_df = rtn_df[rtn_df['passband'] == x]
		p_df = p_df[['object_id','flux']]
		p_df.columns = ['object_id','flux_around_max_passband_'+str(x)]
		q_df = pd.merge(q_df, p_df, on='object_id', how='left')
	return q_df

def flux_around_max_geral(df, tam_window):
	max_flux = df.groupby('object_id')['flux'].max().reset_index()
	max_flux.columns = ['object_id','max_flux_obj']
	df = pd.merge(df, max_flux, on='object_id', how='left')
	df['RWMF'] = 0
	df['RWMF'][df['flux'] == df['max_flux_obj']] = 1
	df = df.sort_values(['object_id','mjd'])
	max_mjd = df[df['RWMF'] == 1]
	max_mjd = max_mjd[['object_id','mjd']]
	max_mjd.columns = ['object_id','mjd_where_flux_max']
	df = pd.merge(df, max_mjd, on='object_id', how='left')
	df['time_walk'] = df['mjd_where_flux_max'] + tam_window 
	aux_df_1 = df[(df['mjd'] > df['mjd_where_flux_max'])&(df['mjd'] < df['time_walk'])]
	aux_df_1 = aux_df_1[['object_id','flux']]
	df['time_walk'] = df['mjd_where_flux_max'] - tam_window 
	aux_df_2 = df[(df['mjd'] > df['time_walk'])&(df['mjd'] < df['mjd_where_flux_max'])]
	aux_df_2 = aux_df_2[['object_id','flux']]
	df['time_walk'] = df['mjd_where_flux_max'] + tam_window
	df['mjd_where_flux_max'] = df['mjd_where_flux_max'] - tam_window
	aux_df_3 = df[(df['mjd'] > df['mjd_where_flux_max'])&(df['mjd'] < df['time_walk'])]
	aux_df_3 = aux_df_3[['object_id','flux']]
	auxs = [aux_df_1, aux_df_2, aux_df_3]
	q_df = pd.DataFrame()
	q_df['object_id'] = aux_df_1['object_id'].unique()
	for df_ in auxs:
		df_  = df_.groupby('object_id')['flux'].mean().reset_index()
		q_df = pd.merge(q_df, df_, on='object_id', how='left')
	q_df.columns = ['object_id', 'g_flux_after_max', 'g_flux_before_max', 'g_flux_around_max']
	return q_df

def increasead_flux_seq(df, col):
    df = df.sort_values(["object_id", "passband", "mjd"])
    df['prev_'+col] = df.groupby(['object_id','passband'])[col].shift().fillna(0)
    df['diff_last_'+col] = df[col] - df['prev_'+col]
    del df['prev_'+col]
    df['is_positive'] = 0
    df['is_positive'][df['diff_last_'+col] < 0] = 1
    df['block'] = (df['is_positive'] != df['is_positive'].shift(1)).cumsum()
    df['max_number_of_incresead_flux_in_seq'] = df.groupby(['object_id', 'passband', 'block']).cumcount()+1
    aux_df = df[df['is_positive'] == 1]
    aux_df = aux_df[['object_id','max_number_of_incresead_flux_in_seq']].drop_duplicates()
    aux = aux_df.groupby('object_id')['max_number_of_incresead_flux_in_seq'].count().reset_index()
    return aux

def find_noise(df):
    df = df.sort_values(["object_id", "passband", "mjd"])
    df['prev_flux'] = df.groupby(['object_id','passband'])['flux'].shift()
    df['diff_last_flux'] = df['flux'] - df['prev_flux']
    df = df.groupby(['object_id','passband'])['diff_last_flux']

def full_width_at_half_maximum(df, agg):
    df = df.sort_values(["object_id", "mjd"])
    df['block'] = ((df['mjd'] - df.groupby('object_id')['mjd'].shift(-1)) > 100).cumsum()
    df['flux_max'] = df.groupby(['object_id','block'])['flux'].transform('max')
    df['half_flux_max'] = df['flux_max'] / 2
    df = df[df['flux'] > df['half_flux_max']]
    df['mjd_min'] = df.groupby(['object_id','block'])['mjd'].transform('min')
    df['mjd_max'] = df.groupby(['object_id','block'])['mjd'].transform('max')
    df['FWHM'] = df['mjd_max'] - df['mjd_min']
    aggs = {'FWHM': [agg]}
    aux = df.groupby('object_id').agg(aggs).reset_index()
    aux.columns = ['object_id','FWHM_'+agg]
    return aux

def score_bands(df, agg):
    aux = agg_per_obj_passband(df, 'flux', agg)
    cols = ['flux_'+agg+'_passband_0', 'flux_'+agg+'_passband_1', 'flux_'+agg+'_passband_2', 'flux_'+agg+'_passband_3', 'flux_'+agg+'_passband_4', 'flux_'+agg+'_passband_5']
    aux['score_band_std_'+agg] = np.std(aux[cols], axis=1)
    return aux[['object_id','score_band_std_'+agg]]

def what_passband(df):
    df['flux_min'] = df.groupby('object_id')['flux'].transform('min')
    df['flux_max'] = df.groupby('object_id')['flux'].transform('max')
    df_1 = df[df['flux'] == df['flux_max']]
    df_1 = df_1[['object_id','passband']]
    df_2 = df[df['flux'] == df['flux_min']]
    df_2 = df_2[['object_id','passband']]
    df_1 = pd.merge(df_1, df_2, on='object_id', how='left')
    df_1.columns = ['object_id','band_of_max','band_of_min']
    return df_1

def select_objects(df):
    aux = df.copy()
    aux['flux_max'] = aux.groupby('object_id')['flux'].transform('max')
    aux['mjd_min'] = aux.groupby('object_id')['mjd'].transform('min')
    df_aux = aux[aux['flux_max'] == aux['flux']]
    df_aux = df_aux[['object_id','mjd']]
    df_aux.columns = ['object_id','mjd_where_max']
    aux = pd.merge(aux, df_aux, on='object_id', how='left')
    aux = aux[(aux['mjd_where_max'] - aux['mjd_min']) > 10]
    aux['std_flux'] = aux.groupby('object_id')['flux'].transform('std')
    aux = aux[aux['std_flux'] >= 10]
    aux.drop(['flux_max','mjd_min','std_flux'], axis=1, inplace=True)
    return aux['object_id'].unique()
    
def preprocess_ts_df(df):
    df['flux_ratio_sq'] = np.power(df['flux'] / df['flux_err'], 2.0)
    df['flux_by_flux_ratio_sq'] = df['flux'] * df['flux_ratio_sq']
    aux_df = df[df['detected'] == 1]
    return df, aux_df

def postprocess_df(full_df):
    full_df['magnitude'] = -2.5*np.log10(full_df['flux_max'])
    full_df['abs_magnitude'] = full_df['magnitude'] - full_df['distmod']
    
    #full_df['luminosity_mean'] = 4*3.1415*(full_df['distmod']**2)*full_df['flux_mean']
    
    cols = ['flux_around_max_passband_0', 'flux_around_max_passband_1',
            'flux_around_max_passband_2', 'flux_around_max_passband_3',
            'flux_around_max_passband_4', 'flux_around_max_passband_5']
    for col in cols:
        full_df['flux_max_/_'+col] = full_df['flux_max'] / full_df[col]
    
    full_df['diff_flux_in_max_1'] = full_df['g_flux_before_max'] - full_df['g_flux_after_max']
    full_df['diff_flux_in_max_2'] = full_df['g_flux_after_max']	- full_df['g_flux_before_max']
    
    cols = ['flux_std_passband_0', 'flux_std_passband_1', 'flux_std_passband_2',
    		'flux_std_passband_3', 'flux_std_passband_4', 'flux_std_passband_5']
    for col_1 in cols:
        for col_2 in cols:
            if (col_1 != col_2):
                full_df[col_1+'_/_'+col_2] = full_df[col_1] / full_df[col_2]
    
    return full_df

def make_features(df, aux_df):
    auxs = []
    auxs.append(mjd_diff_detected_passband(aux_df, 'mjd'))
    auxs.append(mjd_diff_detected(aux_df, 'mjd'))
    auxs.append(mjd_diff_detected(aux_df, 'flux'))
    auxs.append(mjd_diff2_detected(aux_df, 'flux'))
    auxs.append(agg_per_obj_passband(df, 'flux', 'mean'))
    auxs.append(agg_per_obj_passband(df, 'flux', 'std'))
    auxs.append(agg_per_obj_passband(df, 'flux', 'skew'))
    auxs.append(agg_per_obj_passband(df, 'flux_ratio_sq', 'mean'))
    auxs.append(flux_around_max_passband(df, 50))
    auxs.append(flux_around_max_geral(df, 100))
    auxs.append(increasead_flux_seq(df, 'flux'))
    auxs.append(full_width_at_half_maximum(df, 'mean'))
    auxs.append(full_width_at_half_maximum(aux_df,'mean'))
    auxs.append(score_bands(df, 'max'))
    auxs.append(score_bands(df, 'min'))
    auxs.append(score_bands(aux_df, 'max'))
    auxs.append(score_bands(aux_df, 'min'))
    auxs.append(what_passband(df))
    auxs.append(what_passband(aux_df))
    return auxs

# 05-FOLD - MULTI WEIGHTED LOG LOSS : 0.51775
# 10-FOLD - MULTI WEIGHTED LOG LOSS : 0.50680
def main():
    train = pd.read_csv('../input/training_set.csv')

    unique_ids = train['object_id'].unique()
    print('Number of objects after: ', len(unique_ids))
    new_obj_id = select_objects(train)
    print('Number of objects before: ', len(new_obj_id))
    train = train[train['object_id'].isin(new_obj_id)]

    train, aux_train = preprocess_ts_df(train)
    
    ## My feature engineering part ##
    auxs = make_features(train, aux_train)
    
    aggs = get_aggregations()
    agg_train = train.groupby('object_id').agg(aggs)
    new_columns = get_new_columns(aggs)
    agg_train.columns = new_columns
    agg_train = add_features_to_agg(df=agg_train)
    agg_train.head()

    del train
    gc.collect()

    meta_train = pd.read_csv('../input/training_set_metadata.csv')
    meta_train = meta_train[meta_train['object_id'].isin(new_obj_id)]
    meta_train.head()

    full_train = agg_train.reset_index().merge(
        right=meta_train,
        how='outer',
        on='object_id'
    )
    
    for aux in auxs:
        full_train = pd.merge(full_train, aux, on='object_id', how='left')

    full_train = postprocess_df(full_train)

    y = full_train['target']
    del full_train['target']
    del full_train['ddf'], full_train['hostgal_specz'], full_train['decl'], full_train['ra'], full_train['gal_l'], full_train['gal_b'], full_train['mwebv']

    train_mean = full_train.median(axis=0)
    #full_train.fillna(train_mean, inplace=True)
    get_logger().info(full_train.columns)
    clfs, importances = train_classifiers(full_train, y)
    del full_train['object_id']
    save_importances(importances_=importances)

    is_valid = False
    
    if (is_valid == False):

        meta_test = pd.read_csv('../input/test_set_metadata.csv')

        meta_test.loc[~meta_test['hostgal_specz'].isnull(), 'hostgal_photoz'] = meta_test.loc[~meta_test['hostgal_specz'].isnull(), 'hostgal_specz']
        meta_test.loc[~meta_test['hostgal_specz'].isnull(), 'hostgal_photoz_err'] = 0
        
        galatic_ids  	 = (meta_test['object_id'][meta_test['hostgal_photoz'] == 0]).unique()
        extragalatic_ids = (meta_test['object_id'][meta_test['hostgal_photoz'] != 0]).unique()
        
        import time

        start = time.time()
        chunks = 5000000
        remain_df = None

        for i_c, df in enumerate(pd.read_csv('../input/test_set.csv', chunksize=chunks, iterator=True)):
            unique_ids = np.unique(df['object_id'])
            new_remain_df = df.loc[df['object_id'] == unique_ids[-1]].copy()

            if remain_df is None:
                df = df.loc[df['object_id'].isin(unique_ids[:-1])].copy()
            else:
                df = pd.concat([remain_df, df.loc[df['object_id'].isin(unique_ids[:-1])]], axis=0)

            remain_df = new_remain_df

            preds_df = predict_chunk(df_=df,
                                     clfs_=clfs,
                                     meta_=meta_test,
                                     features=full_train.columns,
                                     train_mean=train_mean)

            if i_c == 0:
                preds_df.to_csv('predictions_v3.csv', header=True, index=False, float_format='%.6f')
            else:
                preds_df.to_csv('predictions_v3.csv', header=False, mode='a', index=False, float_format='%.6f')

            del preds_df
            gc.collect()

            if (i_c + 1) % 10 == 0:
                get_logger().info('%15d done in %5.1f' % (chunks * (i_c + 1), (time.time() - start) / 60))
                print('%15d done in %5.1f' % (chunks * (i_c + 1), (time.time() - start) / 60))

        preds_df = predict_chunk(df_=remain_df,
                                 clfs_=clfs,
                                 meta_=meta_test,
                                 features=full_train.columns,
                                 train_mean=train_mean)

        preds_df.to_csv('predictions_v3.csv', header=False, mode='a', index=False, float_format='%.6f')

        z = pd.read_csv('predictions_v3.csv')
        z = z.groupby('object_id').mean().reset_index()
        
        z.to_csv('single_predictions.csv', index=False, float_format='%.6f')

        sub_ext = z[z['object_id'].isin(extragalatic_ids)]
        sub_gal = z[z['object_id'].isin(galatic_ids)]

        galatic_y = [6, 16, 53, 65, 92]
        extraga_y = [15, 42, 52, 62, 64, 67, 88, 90, 95]
        
        for y in extraga_y:
            sub_gal['class_'+str(y)] = 0
        
        for y in galatic_y:
            sub_ext['class_'+str(y)] = 0
        
        sub = pd.concat([sub_gal, sub_ext])

        sub.to_csv('postprocess_predictions_x.csv', index=False, float_format='%.6f')

if __name__ == '__main__':
    gc.enable()
    create_logger()
    try:
        main()
    except Exception:
        get_logger().exception('Unexpected Exception Occured')
        raise