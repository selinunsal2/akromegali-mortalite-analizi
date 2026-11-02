"""Akromegali mortalite analizi: tekrarlanabilir, ön çalışma.
Usage: python run_project.py [--no-mlflow] [--output results]
MLflow server is expected at http://127.0.0.1:5000 unless --no-mlflow is used.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURES = ['Age_in_Diagnosis', 'Gender', 'Preop_GH', 'Preop_IGF_1', 'Preop_Tm_Size_MAX_mm']
NUMERIC = ['Age_in_Diagnosis', 'Preop_GH', 'Preop_IGF_1', 'Preop_Tm_Size_MAX_mm']
CATEGORICAL = ['Gender']
SCORING = {'accuracy': 'accuracy', 'roc_auc': 'roc_auc', 'pr_auc': 'average_precision',
           'precision_death': 'precision', 'recall_death': 'recall', 'f1_death': 'f1'}

def models():
    num_lr = Pipeline([('imputer', SimpleImputer(strategy='median', add_indicator=True)),
                       ('scaler', StandardScaler())])
    cat = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OneHotEncoder(handle_unknown='ignore'))])
    lr_prep = ColumnTransformer([('num', num_lr, NUMERIC), ('cat', cat, CATEGORICAL)])
    rf_prep = ColumnTransformer([('num', SimpleImputer(strategy='median', add_indicator=True), NUMERIC),
                                 ('cat', cat, CATEGORICAL)])
    return {
        'DummyClassifier': DummyClassifier(strategy='most_frequent'),
        'LogisticRegression': Pipeline([('preprocessor', lr_prep), ('classifier', LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=42))]),
        'RandomForest': Pipeline([('preprocessor', rf_prep), ('classifier', RandomForestClassifier(
            n_estimators=200, max_depth=5, class_weight='balanced', random_state=42, n_jobs=-1))])
    }

def json_safe(x):
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating,)): return float(x)
    if isinstance(x, (np.bool_,)): return bool(x)
    if isinstance(x, Path): return str(x)
    return str(x)

def evaluate(y, pred, proba):
    return dict(accuracy=accuracy_score(y,pred), precision_death=precision_score(y,pred,zero_division=0),
                recall_death=recall_score(y,pred,zero_division=0), f1_death=f1_score(y,pred,zero_division=0),
                roc_auc=roc_auc_score(y,proba),pr_auc=average_precision_score(y,proba))

def data_contract(df, out):
    rule_alive = df['Alive'].isin([0,1]) & df['Alive'].notna()
    ages = df[['Age_in_Diagnosis','Age_at_Operation','Age']]
    valid_age = ages.notna().all(axis=1)
    age_rule = (ages['Age_in_Diagnosis']<=ages['Age_at_Operation']) & (ages['Age_at_Operation']<=ages['Age'])
    pair = df[['TSS_VAR_YOK','Remission_3_Month']]
    valid_pair = pair.notna().all(axis=1)
    match = pair['TSS_VAR_YOK'].eq('YOK') == pair['Remission_3_Month'].eq('NOT_OPERATED')
    summary = pd.DataFrame([
        {'rule':'Alive in {0,1}','checked':len(df),'violations':int((~rule_alive).sum()),'not_checkable':0},
        {'rule':'Diagnosis <= operation <= Age','checked':int(valid_age.sum()),'violations':int((valid_age & ~age_rule).sum()),'not_checkable':int((~valid_age).sum())},
        {'rule':'TSS YOK iff NOT_OPERATED','checked':int(valid_pair.sum()),'violations':int((valid_pair & ~match).sum()),'not_checkable':int((~valid_pair).sum())}])
    summary.to_csv(out/'data_contract.csv', index=False)
    # Deliberate corruption on a copy, without touching the source data
    altered=df.copy(deep=True)
    idx=valid_age[valid_age].index[0]
    altered.loc[idx,'Age_at_Operation']=altered.loc[idx,'Age']+10
    detected=not (altered.loc[idx,'Age_in_Diagnosis'] <= altered.loc[idx,'Age_at_Operation'] <= altered.loc[idx,'Age'])
    (out/'corruption_test.json').write_text(json.dumps({'row_index':int(idx),'detected':bool(detected)},indent=2), encoding='utf-8')
    return summary

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',default=str(Path(__file__).with_name('akromegali_alistirma.csv')))
    ap.add_argument('--output',default=str(Path(__file__).with_name('results')))
    ap.add_argument('--no-mlflow',action='store_true')
    ap.add_argument('--tracking-uri', default='http://127.0.0.1:5000')
    args=ap.parse_args()
    datafile=Path(args.data); out=Path(args.output);out.mkdir(parents=True, exist_ok=True)
    df=pd.read_csv(datafile)
    required=set(FEATURES+['Alive','Age_at_Operation','Age','TSS_VAR_YOK','Remission_3_Month'])
    absent=required-set(df.columns)
    if absent: raise ValueError(f'Missing columns: {sorted(absent)}')
    if df['Alive'].isna().any() or not df['Alive'].isin([0,1]).all():raise ValueError('Invalid target values')
    digest=hashlib.md5(datafile.read_bytes()).hexdigest()
    data_contract(df,out)
    missing=(pd.DataFrame({'count':df.isna().sum(),'rate':df.isna().mean()}).sort_values('rate',ascending=False))
    missing.to_csv(out/'missingness.csv')
    df['Alive'].value_counts().sort_index().rename_axis('Alive').to_csv(out/'target_distribution.csv')
    fig,ax=plt.subplots(figsize=(7,4));df['Alive'].value_counts().sort_index().plot.bar(ax=ax)
    ax.set_xticklabels(['Death (Alive=0)','Alive (Alive=1)'],rotation=0);ax.set_ylabel('Patients');ax.set_title('Outcome distribution')
    fig.tight_layout();fig.savefig(out/'target_distribution.png',dpi=180);plt.close(fig)
    X=df[FEATURES].copy();y=(df['Alive']==0).astype(int)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.20,random_state=42,stratify=y)
    models_dict=models();test_results=[];cv_results=[];fitted={}
    cv=RepeatedStratifiedKFold(n_splits=5,n_repeats=5,random_state=42)
    mlflow=None
    if not args.no_mlflow:
        try:
            import mlflow as ml
            import mlflow.sklearn
            ml.set_tracking_uri(args.tracking_uri)
            ml.set_experiment('Akromegali_Mortalite_Tahmini')
            mlflow=ml
        except Exception as e:
            raise RuntimeError('MLflow connection failed. Start the server, or use --no-mlflow') from e
    for name,model in models_dict.items():
        print(f'Training {name}... ',flush=True)
        # Repeated CV is confined to the training portion, and re-fits pipelines in each fold.
        scores=cross_validate(model,X_train,y_train,cv=cv,scoring=SCORING,n_jobs=1,error_score='raise')
        cvrow={'Model':name}
        for metric in SCORING:
            vals=scores['test_'+metric];cvrow[metric+'_mean']=float(vals.mean());cvrow[metric+'_std']=float(vals.std())
        cv_results.append(cvrow)
        model.fit(X_train,y_train);fitted[name]=model
        predictions=model.predict(X_test);prob=model.predict_proba(X_test)[:,1]
        testrow={'Model':name,**evaluate(y_test,predictions,prob)}
        test_results.append(testrow)
        pd.DataFrame(confusion_matrix(y_test,predictions,labels=[0,1]),index=['true_alive','true_death'],columns=['pred_alive','pred_death']).to_csv(out/(name+'_confusion.csv'))
        if mlflow is not None:
            # New run name prevents silently changing existing runs.
            with mlflow.start_run(run_name=f'{name}_reproducible'):
                mlflow.log_params({'seed':42,'test_size':.20,'cv_folds':5,'cv_repeats':5,'model':name})
                mlflow.log_metrics(testrow | {f'cv_{k}':v for k,v in cvrow.items() if k!='Model'})
                mlflow.log_dict({'features':FEATURES},'feature_list.json')
                mlflow.set_tag('dataset_md5',digest)
                mlflow.set_tag('preliminary_model','timing_of_preoperative_features_unverified')
                # Explicit skops serialization, reviewed types only, as in the user's installed MLflow 3.
                trust=['numpy.dtype']
                if name=='RandomForest': trust.append('sklearn.tree._tree.Tree')
                mlflow.sklearn.log_model(model,name='model',serialization_format='skops',skops_trusted_types=trust)
    cvdf=pd.DataFrame(cv_results);cvdf.to_csv(out/'cross_validation_results.csv',index=False)
    testdf=pd.DataFrame(test_results);testdf.to_csv(out/'test_results.csv',index=False)
    fig,ax=plt.subplots(figsize=(9,5));metrics=['roc_auc','pr_auc','recall_death','f1_death'];xx=np.arange(len(metrics))
    for i,row in cvdf.iterrows():ax.bar(xx+(i-1)*.25,[row[x+'_mean'] for x in metrics],width=.25,label=row['Model'])
    ax.set_xticks(xx, ['ROC-AUC','Average precision','Recall (death)','F1 (death)']);ax.set_ylabel('Repeated CV mean');ax.legend();fig.tight_layout()
    fig.savefig(out/'model_comparison_cv.png',dpi=200);plt.close(fig)
    lr=fitted['LogisticRegression'];pred=lr.predict(X_test);prob=lr.predict_proba(X_test)[:,1]
    fp=X_test.loc[(y_test==0)&(pred==1)].copy()
    fp['death_score']=prob[(y_test==0)&(pred==1)]
    # Patient-level details are written locally; consider appropriate clinical data restrictions before sharing artifacts.
    if not fp.empty:
        top=fp.sort_values('death_score',ascending=False).iloc[[0]]
        prep=lr.named_steps['preprocessor'];clf=lr.named_steps['classifier']
        v=prep.transform(top[FEATURES]);v=v.toarray() if hasattr(v,'toarray') else np.asarray(v)
        vals=v[0];coefs=clf.coef_[0];contrib=vals*coefs
        explanation=pd.DataFrame({'feature':prep.get_feature_names_out(), 'transformed_value':vals,
                                  'coefficient':coefs, 'log_odds_contribution':contrib})
        explanation.to_csv(out/'local_contributions.csv',index=False)
        check=float(expit(clf.intercept_[0]+contrib.sum()))
        details={'row_index':int(top.index[0]),'predicted_score':float(top['death_score'].iloc[0]),'reconstructed_score':check,
                 'match':bool(np.isclose(check,top['death_score'].iloc[0]))}
        (out/'selected_false_positive.json').write_text(json.dumps(details,indent=2),encoding='utf-8')
        coefdf=pd.DataFrame({'feature':prep.get_feature_names_out(),'coefficient':coefs})
        coefdf.to_csv(out/'global_coefficients.csv',index=False)
        coefdf=coefdf.sort_values('coefficient');fig,ax=plt.subplots(figsize=(10,6))
        ax.barh(coefdf['feature'],coefdf['coefficient']);ax.axvline(0,color='black',lw=1)
        ax.set_title('Global Logistic Regression coefficients');fig.tight_layout()
        fig.savefig(out/'global_coefficients.png',dpi=200);plt.close(fig)
    metadata={'source_file':datafile.name,'md5':digest,'rows':len(df),'columns':len(df.columns),
              'features':FEATURES,'target': '1=death (transformed from Alive==0)',
              'split_seed':42,'test_size':.20,'cv':'RepeatedStratifiedKFold(5 x 5, seed=42)',
              'caution':'Preoperative measurement timing unverified; no external validation; probabilities uncalibrated; test set explored previously.'}
    (out/'run_metadata.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding='utf-8')
    print('\nRepeated CV results:\n',cvdf.round(4).to_string(index=False))
    print('\nTest results (descriptive; test set was previously inspected):\n',testdf.round(4).to_string(index=False))
    print(f'\nOutputs: {out.resolve()}')

if __name__=='__main__':main()
