from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix, classification_report

ROOT=Path(__file__).resolve().parents[1]
(ROOT/"models").mkdir(exist_ok=True)
(ROOT/"reports").mkdir(exist_ok=True)
DATA_PATH = ROOT / "data" / "online_shoppers_intention.csv"
if DATA_PATH.exists():
    df = pd.read_csv(DATA_PATH)
else:
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=468)
    df = ds.data.features.copy()
    df["Revenue"] = ds.data.targets.iloc[:, 0].values
y=df["Revenue"].astype(int); X=df.drop(columns=["Revenue","PageValues"])
cat=X.select_dtypes(include=["object","bool","category"]).columns.tolist()
num=X.columns.difference(cat).tolist()
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
prep=ColumnTransformer([
 ("num",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num),
 ("cat",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),cat)])
cv=StratifiedKFold(5,shuffle=True,random_state=42)
baseline=Pipeline([("prep",prep),("model",DummyClassifier(strategy="most_frequent"))]).fit(Xtr,ytr)
basepred=baseline.predict(Xte)
models={
 "Logistic Regression":(LogisticRegression(max_iter=2000,class_weight="balanced"),{"model__C":[.1,1,5]}),
 "Random Forest":(RandomForestClassifier(random_state=42,class_weight="balanced",n_jobs=-1),{"model__n_estimators":[200,400],"model__max_depth":[None,12,20],"model__min_samples_leaf":[1,3,5]}),
 "Extra Trees":(ExtraTreesClassifier(random_state=42,class_weight="balanced",n_jobs=-1),{"model__n_estimators":[200,400],"model__max_depth":[None,12,20],"model__min_samples_leaf":[1,3,5]})}
searches={}
for name,(est,params) in models.items():
 p=Pipeline([("prep",prep),("model",est)])
 s=RandomizedSearchCV(p,params,n_iter=min(8,int(np.prod([len(v) for v in params.values()]))),scoring="f1",cv=cv,n_jobs=-1,random_state=42)
 s.fit(Xtr,ytr); searches[name]=s; print(name,s.best_score_,s.best_params_)
name,best=max(searches.items(),key=lambda kv:kv[1].best_score_)
prob=best.predict_proba(Xte)[:,1]; pred=(prob>=.5).astype(int)
m={"selected_model":name,"best_cv_f1":float(best.best_score_),
"test_precision":float(precision_score(yte,pred,zero_division=0)),
"test_recall":float(recall_score(yte,pred,zero_division=0)),
"test_f1":float(f1_score(yte,pred,zero_division=0)),
"test_roc_auc":float(roc_auc_score(yte,prob)),
"test_pr_auc":float(average_precision_score(yte,prob)),
"confusion_matrix":confusion_matrix(yte,pred).tolist(),
"classification_report":classification_report(yte,pred,output_dict=True,zero_division=0),
"threshold":.5,"best_params":best.best_params_,
"baseline":{"precision":float(precision_score(yte,basepred,zero_division=0)),"recall":float(recall_score(yte,basepred,zero_division=0)),"f1":float(f1_score(yte,basepred,zero_division=0)),"roc_auc":None,"pr_auc":None,"confusion_matrix":confusion_matrix(yte,basepred).tolist()},
"dataset":{"rows":int(len(df)),"columns_including_target":int(df.shape[1]),"missing_values":int(df.isna().sum().sum()),"duplicate_rows":int(df.duplicated().sum()),"class_counts":{str(k):int(v) for k,v in y.value_counts().sort_index().items()}},
"split":{"test_size":0.2,"random_state":42,"stratified":True}}
joblib.dump(best.best_estimator_,ROOT/"models"/"purchase_intent_pipeline.joblib")
(ROOT/"reports"/"test_metrics.json").write_text(json.dumps(m,indent=2))
comparison=[{"model":"Dummy Classifier","cv_best_f1":None,"test_precision":m["baseline"]["precision"],"test_recall":m["baseline"]["recall"],"test_f1":m["baseline"]["f1"],"test_roc_auc":None,"test_pr_auc":None,"best_params":"majority class"}]
for model_name,search in searches.items():
    model_prob=search.best_estimator_.predict_proba(Xte)[:,1]
    model_pred=(model_prob>=0.5).astype(int)
    comparison.append({"model":model_name,"cv_best_f1":float(search.best_score_),"test_precision":float(precision_score(yte,model_pred,zero_division=0)),"test_recall":float(recall_score(yte,model_pred,zero_division=0)),"test_f1":float(f1_score(yte,model_pred,zero_division=0)),"test_roc_auc":float(roc_auc_score(yte,model_prob)),"test_pr_auc":float(average_precision_score(yte,model_prob)),"best_params":str(search.best_params_)})
pd.DataFrame(comparison).to_csv(ROOT/"reports"/"model_comparison.csv",index=False)
print(json.dumps(m,indent=2))
