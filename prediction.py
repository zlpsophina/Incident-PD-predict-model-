import os
import nibabel
import numpy as np
import pandas as pd
# import ants
# import cv2
import os
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from importlib import reload
import pickle
import random
# from predict_wlb.helper_funcs import *
# import utilites_useless
# reload(utilites)
# from utilites_useless import *
import scipy
import xgboost as xgb
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from scipy.stats import ttest_ind,ttest_1samp,pearsonr,ttest_rel,spearmanr,zscore,chi2_contingency
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split,GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, auc,accuracy_score,classification_report, make_scorer,confusion_matrix
import seaborn as sns
import shap
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
# plt.rc('font',family='Times New Roman')
from matplotlib.font_manager import FontProperties
from matplotlib import rcParams
from matplotlib import font_manager
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from lightgbm import LGBMClassifier
# import joypy
from collections import Counter
# import scanpy as sc
# from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.cluster import hierarchy
from collections import defaultdict
from sklearn.metrics import r2_score
from sklearn.utils import resample
# from imblearn.over_sampling import SMOTE





def get_data():


    abbrevation=pd.read_csv("./dataset/UKB_blood_category.csv")
    abbrevation=abbrevation[['Description','Abbreviation']]
    pd_patients=pd.read_csv("./dataset/UKB_PD_patients.csv",low_memory=False)
    # print("pd_patients",pd_patients.columns.to_list())
    blood=pd.read_csv("./dataset/ukb_bl_blood_data.csv",low_memory=False)
    print(blood.columns.to_list())
    print(blood.shape)
    print( "Townsend" in blood.columns.to_list() ) #True
    # blood.drop(columns=['Oestradiol','SHBG','Testosterone'],inplace=True)
    # blood['Description']=blood['Description'].replace('Systolic_blood_pressure__automated_reading','Systolic_blood_pressure')
    # blood['Description'] = blood['Description'].replace('Diastolic_blood_pressure__automated_reading',
    #                                                     'Diastolic_blood_pressure')
    # print("blood colunms ori", blood.columns.to_list())
    blood.rename(columns={'Systolic blood pressure, automated reading':'Systolic blood pressure',"Diastolic blood pressure, automated reading":'Diastolic blood pressure'},inplace=True)
    # print("blood colunms rename",blood.columns.to_list())
    # print(len(blood.columns.to_list())) #66
    rename_blood=[]
    for i in blood.columns.to_list():
        if i in abbrevation['Description'].values.tolist():
            rename_blood.append(abbrevation[abbrevation['Description']==i]['Abbreviation'].values.tolist()[0])
            # print(i,abbrevation[abbrevation['Description']==i]['Abbreviation'].values.tolist()[0])
        else:
            rename_blood.append(i)

    blood.columns=rename_blood


    conv=pd.read_csv("./dataset/ukb_cov_info_withnan.csv",low_memory=False)

    conv.rename(columns={'age':'Age','sex':'Sex','education':'Education'},inplace=True)
    conv.rename(columns={'education':'Education'},inplace=True)

    control=pd_patients[(pd_patients["incident_pd"]!=1) & (pd_patients["dignosised_pd"]!=1)]
    incident_pd_all=pd_patients[(pd_patients["incident_pd"]==1) & (pd_patients["dignosised_pd"]!=1)]
    dignosised_pd_all=pd_patients[(pd_patients["incident_pd"]!=1) & (pd_patients["dignosised_pd"]==1)]

    print("incident_pd_all",incident_pd_all.shape)#(3591, 20)
    print("dignosised_pd_all",dignosised_pd_all.shape)#(861, 20)
    print("control",control.shape) #(497783, 20)


    incident_pd_control=pd.concat([control,incident_pd_all],axis=0)
    control_diagnosed=control.sample(n=dignosised_pd_all.shape[0],random_state=2024)
    dignosised_pd_control=pd.concat([control,dignosised_pd_all],axis=0)
    # dignosised_pd_control=dignosised_pd_all

    incident_pd_control_blood=pd.merge(incident_pd_control[['eid',"incident_pd"]],blood,on='eid',how='inner')
    incident_pd_control_blood.rename(columns={"incident_pd": "label"}, inplace=True)
    incident_pd_control_blood_conv=pd.merge(incident_pd_control_blood,conv,on='eid',how='inner')
    dignosised_pd_control_blood=pd.merge(dignosised_pd_control[['eid','dignosised_pd']],blood,on='eid',how='inner')
    dignosised_pd_control_blood.rename(columns={"dignosised_pd":"label"},inplace=True)
    dignosised_pd_control_blood_conv=pd.merge(dignosised_pd_control_blood,conv,on='eid',how='inner')


    incideng_pd_control_eid_label=incident_pd_control_blood_conv[['eid', 'label']]
    dignosised_pd_control_eid_label = dignosised_pd_control_blood_conv[['eid', 'label']]

    incident_pd_control_blood_conv.drop(columns=['eid'],inplace=True)
    dignosised_pd_control_blood_conv.drop(columns=['eid'], inplace=True)

    print(incident_pd_control_blood_conv.shape) #(501374, 74)
    print(dignosised_pd_control_blood_conv.shape) #(498644, 74)
    incident_pd_control_blood_conv= incident_pd_control_blood_conv.iloc[:, ~incident_pd_control_blood_conv.columns.duplicated()]
    dignosised_pd_control_blood_conv = dignosised_pd_control_blood_conv.iloc[:,
                                     ~dignosised_pd_control_blood_conv.columns.duplicated()]

    ori_replace = pd.DataFrame(np.array(incident_pd_control_blood_conv.columns.to_list()), columns=['ori_name'])


    incident_pd_control_blood_conv.columns = incident_pd_control_blood_conv.columns.str.replace('[^a-zA-Z0-9_]', '_', regex=True)
    dignosised_pd_control_blood_conv.columns = dignosised_pd_control_blood_conv.columns.str.replace('[^a-zA-Z0-9_]', '_', regex=True)

    ori_replace['Features'] = pd.DataFrame(np.array(incident_pd_control_blood_conv.columns.to_list()))
    ori_replace.to_csv("./dataset/ori_replace.csv",index=False)

    print(incident_pd_control_blood_conv.shape) #(501374, 70)


    con_list=['Age', 'Sex','Education', 'BMI', 'Smoking', 'Alcohol', 'Townsend',  'label']# 'TBI','Age', 'Sex',
    incident_pd_control_conv=incident_pd_control_blood_conv[con_list]
    dignosised_pd_control_conv=dignosised_pd_control_blood_conv[con_list]

    blood_list=list(set(incident_pd_control_blood_conv.columns.to_list())-set(con_list))

    blood_list.append('label')
    incident_pd_control_blood=incident_pd_control_blood_conv[blood_list]
    dignosised_pd_control_blood=dignosised_pd_control_blood_conv[blood_list]
    print("finnal")


    return incident_pd_control_blood,dignosised_pd_control_blood,incident_pd_control_blood_conv,dignosised_pd_control_blood_conv,incident_pd_control_conv,dignosised_pd_control_conv,ori_replace,incideng_pd_control_eid_label,dignosised_pd_control_eid_label

def get_data_diff_year():

    abbrevation=pd.read_csv("./dataset/UKB_blood_category.csv")
    abbrevation=abbrevation[['Description','Abbreviation']]
    pd_patients=pd.read_csv("./dataset/UKB_PD_patients.csv",low_memory=False)
    # print("pd_patients",pd_patients.columns.to_list())
    blood=pd.read_csv("./dataset/ukb_bl_blood_data.csv",low_memory=False)

    blood.rename(columns={'Systolic blood pressure, automated reading':'Systolic blood pressure',"Diastolic blood pressure, automated reading":'Diastolic blood pressure'},inplace=True)
    # print("blood colunms rename",blood.columns.to_list())
    # print(len(blood.columns.to_list())) #66
    rename_blood=[]
    for i in blood.columns.to_list():
        if i in abbrevation['Description'].values.tolist():
            rename_blood.append(abbrevation[abbrevation['Description']==i]['Abbreviation'].values.tolist()[0])
            # print(i,abbrevation[abbrevation['Description']==i]['Abbreviation'].values.tolist()[0])
        else:
            rename_blood.append(i)

    blood.columns=rename_blood
    print("blood.shap",blood.shape)
    missing_rates = blood.isna().mean()
    print("每列的缺失值比例 proteins:")
    missing_rates_sorted = missing_rates.sort_values(ascending=False)
    print(missing_rates_sorted[-1])

    conv=pd.read_csv("./dataset/ukb_cov_info_withnan.csv",low_memory=False)
    # print("conv",conv.columns.to_list())
    # conv=conv.drop(columns=['Ethnic', 'centre','age','sex'])#
    conv.rename(columns={'age':'Age','sex':'Sex','education':'Education'},inplace=True)
    conv.rename(columns={'education':'Education'},inplace=True)

    control=pd_patients[(pd_patients["incident_pd"]!=1) & (pd_patients["dignosised_pd"]!=1)]
    incident_pd_all=pd_patients[(pd_patients["incident_pd"]==1) & (pd_patients["dignosised_pd"]!=1)]
    incident_pd_less5y=pd_patients[pd_patients["incident_pd_within5_years"]==1]
    incident_pd_5_10y = pd_patients[pd_patients["incident_pd_within10_years"] == 1]
    incident_pd_more10y = pd_patients[pd_patients["incident_pd_morethan10_years"] == 1]

    dignosised_pd_all=pd_patients[(pd_patients["incident_pd"]!=1) & (pd_patients["dignosised_pd"]==1)]

    print("incident_pd_all",incident_pd_all.shape)#(3591, 20)
    print("incident_pd_less5y",incident_pd_less5y.shape)
    print("incident_pd_5_10y",incident_pd_5_10y.shape)
    print("incident_pd_more10y",incident_pd_more10y.shape)
    print("dignosised_pd_all",dignosised_pd_all.shape)#(861, 20)
    print("control",control.shape) #(497783, 20)

    incident_pd_control=pd.concat([control,incident_pd_all],axis=0)
    incident_pd_control_less5y = incident_pd_less5y
    incident_pd_control_5_10y =  incident_pd_5_10y
    incident_pd_control_more10y = incident_pd_more10y

    dignosised_pd_control=pd.concat([control,dignosised_pd_all],axis=0)

    incident_pd_control_blood=pd.merge(incident_pd_control[['eid',"incident_pd"]],blood,on='eid',how='inner')
    incident_pd_control_blood.rename(columns={"incident_pd": "label"}, inplace=True)
    incident_pd_control_blood_conv=pd.merge(incident_pd_control_blood,conv,on='eid',how='inner')

    incident_pd_control_less5y_blood = pd.merge(incident_pd_control_less5y[['eid', "incident_pd_within5_years"]], blood, on='eid', how='inner')
    incident_pd_control_less5y_blood.rename(columns={"incident_pd_within5_years": "label"}, inplace=True)
    incident_pd_control_less5y_blood_conv = pd.merge(incident_pd_control_less5y_blood, conv, on='eid', how='inner')
    incident_pd_control_5_10y_blood = pd.merge(incident_pd_control_5_10y[['eid', "incident_pd_within10_years"]], blood, on='eid', how='inner')
    incident_pd_control_5_10y_blood.rename(columns={"incident_pd_within10_years": "label"}, inplace=True)
    incident_pd_control_5_10y_blood_conv = pd.merge(incident_pd_control_5_10y_blood, conv, on='eid', how='inner')
    incident_pd_control_more10y_blood = pd.merge(incident_pd_control_more10y[['eid', "incident_pd_morethan10_years"]], blood, on='eid', how='inner')
    incident_pd_control_more10y_blood.rename(columns={"incident_pd_morethan10_years": "label"}, inplace=True)
    incident_pd_control_more10y_blood_conv = pd.merge(incident_pd_control_more10y_blood, conv, on='eid', how='inner')


    dignosised_pd_control_blood=pd.merge(dignosised_pd_control[['eid','dignosised_pd']],blood,on='eid',how='inner')
    dignosised_pd_control_blood.rename(columns={"dignosised_pd":"label"},inplace=True)
    dignosised_pd_control_blood_conv=pd.merge(dignosised_pd_control_blood,conv,on='eid',how='inner')


    tdi=pd.read_csv("./dataset/UKB_TBI_500K.csv") #缺失率达到0。2
    tdi.rename(columns={'RF_TBI':'TBI'},inplace=True)

    #
    incident_pd_control_blood_conv=pd.merge(incident_pd_control_blood_conv,tdi,how='left',on='eid')
    incident_pd_control_less5y_blood_conv = pd.merge(incident_pd_control_less5y_blood_conv, tdi, how='left', on='eid')
    incident_pd_control_5_10y_blood_conv = pd.merge(incident_pd_control_5_10y_blood_conv, tdi, how='left', on='eid')
    incident_pd_control_more10y_blood_conv = pd.merge(incident_pd_control_more10y_blood_conv, tdi, how='left', on='eid')

    dignosised_pd_control_blood_conv = pd.merge( dignosised_pd_control_blood_conv, tdi, how='left', on='eid')
    incident_pd_control_blood_conv.drop(columns=['eid'],inplace=True)
    incident_pd_control_less5y_blood_conv.drop(columns=['eid'], inplace=True)
    incident_pd_control_5_10y_blood_conv.drop(columns=['eid'], inplace=True)
    incident_pd_control_more10y_blood_conv.drop(columns=['eid'], inplace=True)

    dignosised_pd_control_blood_conv.drop(columns=['eid'], inplace=True)
    print(incident_pd_control_blood_conv.shape) #(501374, 74)
    print(dignosised_pd_control_blood_conv.shape) #(498644, 74)
    incident_pd_control_blood_conv= incident_pd_control_blood_conv.iloc[:, ~incident_pd_control_blood_conv.columns.duplicated()]
    incident_pd_control_less5y_blood_conv = incident_pd_control_less5y_blood_conv.iloc[:,~incident_pd_control_less5y_blood_conv.columns.duplicated()]
    incident_pd_control_5_10y_blood_conv = incident_pd_control_5_10y_blood_conv.iloc[:,~incident_pd_control_5_10y_blood_conv.columns.duplicated()]
    incident_pd_control_more10y_blood_conv = incident_pd_control_more10y_blood_conv.iloc[:,~incident_pd_control_more10y_blood_conv.columns.duplicated()]

    dignosised_pd_control_blood_conv = dignosised_pd_control_blood_conv.iloc[:,
                                     ~dignosised_pd_control_blood_conv.columns.duplicated()]

    ori_replace = pd.DataFrame(np.array(incident_pd_control_blood_conv.columns.to_list()), columns=['ori_name'])


    incident_pd_control_blood_conv.columns = incident_pd_control_blood_conv.columns.str.replace('[^a-zA-Z0-9_]', '_', regex=True)
    incident_pd_control_less5y_blood_conv.columns = incident_pd_control_less5y_blood_conv.columns.str.replace('[^a-zA-Z0-9_]', '_', regex=True)
    incident_pd_control_5_10y_blood_conv.columns = incident_pd_control_5_10y_blood_conv.columns.str.replace('[^a-zA-Z0-9_]', '_', regex=True)
    incident_pd_control_more10y_blood_conv.columns = incident_pd_control_more10y_blood_conv.columns.str.replace('[^a-zA-Z0-9_]', '_', regex=True)

    dignosised_pd_control_blood_conv.columns = dignosised_pd_control_blood_conv.columns.str.replace('[^a-zA-Z0-9_]', '_', regex=True)

    ori_replace['Features'] = pd.DataFrame(np.array(incident_pd_control_blood_conv.columns.to_list()))
    ori_replace.to_csv("/home1/HWGroup/daiyx/zhenglp/dataset/wlb/ori_replace.csv",index=False)

    print(incident_pd_control_blood_conv.shape) #(501374, 70)

    print(dignosised_pd_control_blood_conv.shape) #(498644, 70)


    con_list=['TBI','Age', 'Sex','Education', 'BMI', 'Smoking', 'Alcohol', 'Townsend',  'label']# 'TBI','Age', 'Sex',
    incident_pd_control_conv=incident_pd_control_blood_conv[con_list]
    dignosised_pd_control_conv=dignosised_pd_control_blood_conv[con_list]

    blood_list=list(set(incident_pd_control_blood_conv.columns.to_list())-set(con_list))
    blood_list.append('label')
    incident_pd_control_blood=incident_pd_control_blood_conv[blood_list]
    dignosised_pd_control_blood=dignosised_pd_control_blood_conv[blood_list]

    incident_pd_control_less5y_blood=incident_pd_control_less5y_blood_conv[blood_list]
    incident_pd_control_5_10y_blood=incident_pd_control_5_10y_blood_conv[blood_list]
    incident_pd_control_more10y_blood=incident_pd_control_more10y_blood_conv[blood_list]
    print("finnal")
    print(incident_pd_control_less5y_blood.columns.to_list())


    return incident_pd_control_blood_conv,dignosised_pd_control_blood_conv,ori_replace,incident_pd_control_less5y_blood,incident_pd_control_5_10y_blood,incident_pd_control_more10y_blood

def replace_outline_data_with_mean(df):
    for col in df.columns:
        if df[col].dtype != 'object':  # Exclude non-numeric columns
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Identify outliers
            lower_bound_outliers = df[col] < lower_bound
            upper_bound_outliers = df[col] > upper_bound

            # Replace outliers with the column mean
            col_mean = df[col].mean()
            df[col][lower_bound_outliers | upper_bound_outliers] = col_mean
    return df

def check_data_outline(df):
    outlier_counts = {}
    for col in df.columns:
        if df[col].dtype != 'object':  # Exclude non-numeric columns
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            lower_bound_outliers = df[df[col] < lower_bound]
            upper_bound_outliers = df[df[col] > upper_bound]
            total_outliers = len(lower_bound_outliers) + len(upper_bound_outliers)
            outlier_counts[col] = total_outliers
    return outlier_counts

# 可视化并保存前50个特征的 SHAP 值图
def save_shap_summary_plot(shap_values, X, feature_names, title, file_name,save_path):
    # 提取前50个特征
    shap_abs = np.abs(shap_values).mean(axis=0)
    top50_features = np.argsort(shap_abs)[:,:-20]
    print(top50_features.shape) #(20, 2920)
    # print(top50_features[:10]) #[968 973 990 970 971 985 972 969 977  52]

    # top50_feature_names = [feature_names[i] for i in top50_features]
    # print(type(X))
    # print(X.head())
    # 绘制 SHAP summary plot
    plt.figure()
    # shap.summary_plot(shap_values[:, top50_features], X.iloc[:, top50_features], feature_names=top50_feature_names,show=False)
    shap.summary_plot(shap_values, X, feature_names=feature_names,show=False)
    plt.title(title)
    plt.savefig(os.path.join(save_path,file_name),dpi=300)
    plt.show()
    plt.close()



def setup_seed(seed):
   torch.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   np.random.seed(seed)
   random.seed(seed)
   torch.backends.cudnn.deterministic = True

def train_lightgbm_10fold_classific_feature_important_each_protiens(data_all_proteins,label_all,save_path,type1):
    proteins_name_list=data_all_proteins.columns.to_list()
    my_params = {'n_estimators': 500,
                 'max_depth': 15,
                 'num_leaves': 10,
                 'subsample': 0.7,
                 'learning_rate': 0.01,
                 'colsample_bytree': 0.7}


    def normal_imp(mydict):
        mysum = sum(mydict.values())
        mykeys = mydict.keys()
        for key in mykeys:
            mydict[key] = mydict[key] / mysum
        return mydict

    tg_imp_cv = Counter()
    tc_imp_cv = Counter()
    shap_imp_cv = np.zeros(len(proteins_name_list))
    mykf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
    AUC_cv=[]
    y_pred_full = np.zeros(shape=[1, 1])
    all_people_shap=np.zeros_like(data_all_proteins)
    cat_f=[]
    for f in proteins_name_list:
         if f in ['Sex', 'Smoking', 'Alcohol', 'TBI']:
            cat_f.append(f)
    for train_idx, test_idx in mykf.split(data_all_proteins, label_all):
        X_train, X_test = data_all_proteins.iloc[train_idx, :], data_all_proteins.iloc[test_idx, :]
        y_train, y_test = label_all.iloc[train_idx], label_all.iloc[test_idx]

        # smote = SMOTE(random_state=2024)
        # X_train, y_train = smote.fit_resample(X_train, y_train)

        my_lgb = LGBMClassifier(objective='binary', metric='auc', is_unbalance=True, verbosity=1,class_weight='balanced', seed=42)
        my_lgb.set_params(**my_params)
        if "conv" in type1:
            print("has category")#,'RF_TBI'
            my_lgb.fit(X_train, y_train,categorical_feature=cat_f)#['sex','Smoking', 'Alcohol']
        else:
            my_lgb.fit(X_train, y_train)

        y_pred_prob = my_lgb.predict_proba(X_test)[:, 1]
        AUC_cv.append(np.round(roc_auc_score(y_test, y_pred_prob), 3))
        y_pred_full = np.concatenate([y_pred_full, np.expand_dims(y_pred_prob, -1)])
        totalgain_imp = my_lgb.booster_.feature_importance(importance_type='gain')
        totalgain_imp = dict(zip(my_lgb.booster_.feature_name(), totalgain_imp.tolist()))
        totalcover_imp = my_lgb.booster_.feature_importance(importance_type='split')
        totalcover_imp = dict(zip(my_lgb.booster_.feature_name(), totalcover_imp.tolist()))
        tg_imp_cv += Counter(normal_imp(totalgain_imp))
        tc_imp_cv += Counter(normal_imp(totalcover_imp))
        explainer = shap.TreeExplainer(my_lgb)
        shap_values = explainer.shap_values(X_test)
        shap_values = np.abs(np.average(shap_values[0], axis=0))
        all_people_shap += shap_values
        shap_imp_cv += shap_values / np.sum(shap_values)
        print("shape values",  np.array(shap_values).shape) #(2920,)

    all_people_shap=all_people_shap/10
    all_people_shap=pd.DataFrame(all_people_shap,columns=proteins_name_list)
    print("all people shap")
    print(all_people_shap.head()) #
    # all_people_shap.to_csv(os.path.join(save_path,"all_people_each_proteins_shap_average_10fold_{}.csv".format(type1)),index=False)

    shap_imp_df = pd.DataFrame({'Features': proteins_name_list,
                                'ShapValues_cv': shap_imp_cv/10})
    shap_imp_df.sort_values(by='ShapValues_cv', ascending=False, inplace=True)
    print("ShapeValues ")
    print(shap_imp_df.head())
    tg_imp_cv = normal_imp(tg_imp_cv)
    tg_imp_df = pd.DataFrame({'Features': list(tg_imp_cv.keys()),
                              'TotalGain_cv': list(tg_imp_cv.values())})

    tc_imp_cv = normal_imp(tc_imp_cv)
    tc_imp_df = pd.DataFrame({'Features': list(tc_imp_cv.keys()),
                              'TotalCover_cv': list(tc_imp_cv.values())})

    my_imp_df = pd.merge(left=shap_imp_df, right=tg_imp_df, how='left', on=['Features'])
    my_imp_df = pd.merge(left=my_imp_df, right=tc_imp_df, how='left', on=['Features'])
    my_imp_df['Ensemble'] = (my_imp_df['ShapValues_cv'] + my_imp_df['TotalGain_cv'] + my_imp_df['TotalCover_cv']) / 3
    my_imp_df.sort_values(by='Ensemble', ascending=False, inplace=True)
    my_imp_df.to_csv(os.path.join(save_path, 'lightgbm_10fold_Importance_{}.csv'.format(type1)), index=False)


def show_shape_values_significant_important(save_path,feature_name,type1,type2):
    df=pd.read_csv(os.path.join(save_path,"all_people_each_proteins_shap_average_10fold_{}_{}.csv".format(tyep1,type2)))
    # 找到 'feature_of_interest' 特征在数据集中的索引
    shap_values=df
    feature_index = list(df.columns).index(feature_name)
    plt.figure()
    # 提取特定特征的 SHAP 值数据
    shap_values_selected = shap_values[:, feature_index]
    # 调用 save_shap_summary_plot() 函数来保存仅包含特定特征 SHAP 值的摘要图
    save_shap_summary_plot(shap_values_selected,show=False)
    plt.savefig(os.path.join(save_path, "significant_important_proteins_shap_values_{}_{}.jpg".format(type1,type2)), dpi=300)
    plt.show()
    plt.close()

def computed_auc_cumulated_by_add_each_feature(data_all_proteins,label_all,save_path,type1):
    # select_feature=['Basal_metabolic_rate', 'Cystatin_C', 'HDL_cholesterol', 'Monocyte_count', 'Lymphocyte_percentage',
    #                 'Eosinophill_percentage', 'Basophill_percentage', 'Immature_reticulocyte_fraction', 'Mean_reticulocyte_volume',
    #                 'Red_blood_cell__erythrocyte__distribution_width', 'LDL_direct', 'Albumin', 'Total_bilirubin',
    #                 'Platelet_distribution_width', 'Nucleated_red_blood_cell_count', 'Gamma_glutamyltransferase',
    #                 'Systolic_blood_pressure__automated_reading', 'Glycated_haemoglobin__HbA1c_', 'age', 'C_reactive_protein',
    #                 'RF_TBI', 'Townsend', 'Pulse_rate__automated_reading'] #(23, 4)
    featrue_imp=pd.read_csv(os.path.join(save_path,'lightgbm_10fold_Importance_{}.csv'.format(type1)))
    # featrue_imp=featrue_imp[featrue_imp['Features'].isin(select_proteins)]
    print("featrue_imp",featrue_imp.shape)
    featrue_imp.sort_values(by='TotalGain_cv', ascending=False, inplace=True)
    feature_ranking_list=featrue_imp['Features'].values.tolist()
    my_params = {'n_estimators': 500,
                 'max_depth': 15,
                 'num_leaves': 10,
                 'subsample': 0.7,
                 'learning_rate': 0.01,
                 'colsample_bytree': 0.7}

    # my_params = {'n_estimators': 400,
    #              'max_depth': 5,
    #              'num_leaves': 20,
    #              'subsample': 0.8,
    #              'learning_rate': 0.01,
    #              'colsample_bytree': 0.7}

    y_test_full = np.zeros(shape=[1, 1])
    mykf = StratifiedKFold(n_splits=10, random_state=2024, shuffle=True)
    for train_idx, test_idx in mykf.split(data_all_proteins, label_all):
        y_test_full = np.concatenate([y_test_full, np.expand_dims(label_all.iloc[test_idx], -1)])

    y_pred_full_prev = y_test_full
    tmp_f, AUC_cv_lst = [], []
    # feature_ranking_list=feature_ranking_list[:50]
    cate_f=[]
    feature_ranking_list=['Cystatin_C','Systolic_blood_pressure','HbA1c','Cholesterol',"PCT",'GGT','Townsend','HLR_P',"Lp_a_",'AST','Albumin','ALP','IRF','Urea','Total_bilirubin',
                          'Basal_metabolic_rate','RET_P','RDW','MSCV','Diastolic_blood_pressure','Direct_bilirubin','Triglycerides','RET_C','Phosphate','Creatinine','MRV','PDW',
                          'BASO_P',"ALT",'CRP','Apo_B','Urate','Pulse_rate__automated_reading','EOS_P',"BMI","Total_protein","LYMP_C","MPV",'Platelet_count','MONO_C','Vitamin_D','MONO_P',
                          'BASO_C','WBC','MCV','NEUT_P','EOS_C','Calcium','IGF_1','Glucose']
    for f in feature_ranking_list:
        tmp_f.append(f)
        my_X = data_all_proteins[tmp_f]
        AUC_cv = []
        y_pred_full = np.zeros(shape=[1, 1])
        if f in ['Sex', 'Smoking', 'Alcohol', 'TBI']:
            cate_f.append(f)
        for train_idx, test_idx in mykf.split(my_X, label_all):
            X_train, X_test = my_X.iloc[train_idx, :], my_X.iloc[test_idx, :]
            y_train, y_test = label_all.iloc[train_idx], label_all.iloc[test_idx]

            # smote = SMOTE(random_state=2024)
            # X_train, y_train = smote.fit_resample(X_train, y_train)

            my_lgb = LGBMClassifier(objective='binary', metric='auc', is_unbalance=True, n_jobs=4, verbosity=-1,
                                    seed=2024)
            my_lgb.set_params(**my_params)
            # if "conv" in type1 :
            #     print("has category")
            # if f in ['sex', 'Smoking', 'Alcohol', 'RF_TBI']:#['sex','Smoking', 'Alcohol','RF_TBI']
            #     cate_f.append(f)
            my_lgb.fit(X_train, y_train, categorical_feature=cate_f)
            # else:
            #     my_lgb.fit(X_train, y_train)
            y_pred_prob = my_lgb.predict_proba(X_test)[:, 1]
            AUC_cv.append(np.round(roc_auc_score(y_test, y_pred_prob), 3))
            y_pred_full = np.concatenate([y_pred_full, np.expand_dims(y_pred_prob, -1)])
        log10_p = delong_roc_test(y_test_full[:, 0], y_pred_full_prev[:, 0], y_pred_full[:, 0])
        y_pred_full_prev = y_pred_full
        tmp_out = np.array([np.round(np.mean(AUC_cv), 3), np.round(np.std(AUC_cv), 3), 10 ** log10_p[0][0]] + AUC_cv)
        AUC_cv_lst.append(tmp_out)
        print((f, np.mean(AUC_cv), 10 ** log10_p[0][0]))

    AUC_df = pd.DataFrame(AUC_cv_lst,
                          columns=['AUC_mean', 'AUC_std', 'p_delong'] + ['AUC_' + str(i) for i in range(10)])

    AUC_df = pd.concat((pd.DataFrame({'Features': tmp_f}), AUC_df), axis=1)
    # myout = pd.merge(AUC_df, pro_dict, how='left', on=['Pro_code'])
    print("AUC_df")
    print(AUC_df.head())
    AUC_df.to_csv(os.path.join(save_path,"AUC_camculated_lightgbm_each_proteins_{}_ori.csv".format(type1)), index=False)

def get_nb_f_3(mydf):
    mydf['p_delong']= mydf['p_delong'].astype(float)
    p_lst = mydf.p_delong.tolist()
    i = 0
    k=0
    # while((p_lst[i]<0.05)|(p_lst[i+1]<0.05)):
    #     i+=1
    for p_value in p_lst:
        print(p_value)
        if p_value > 0.05:
            i += 1
            k+=1
        else:
            i = 0  # 重置计数器
            k += 1

        # 检查是否达到了 3 次连续不显著
        if i >= 3:
            print("Stopping training: No significant improvement in 3 consecutive iterations.")
            break
    k=k-3
    return k
def get_nb_f(mydf):
    p_lst = mydf.p_delong.tolist()
    i = 0
    while((p_lst[i]<0.05)|(p_lst[i+1]<0.05)):
        i+=1
    return i
def get_nb_f_max(mydf,max_value):
    # mydf['p_delong']= mydf['p_delong'].astype(float)
    # p_lst = mydf.p_delong.tolist()
    i = 0
    for p_value in mydf['AUC_mean'].values.tolist():
        if p_value < max_value:
            i += 1
        else:
            break
    return i
def convert_to_float(complex_str):
    return complex(complex_str).real  # 将字符串转换为复数，并提取实部
def plot_cumulated_AUC_and_important_of_each_features(save_path,type1,ori_replace):
    auc_df=pd.read_csv(os.path.join(save_path,"AUC_camculated_lightgbm_each_proteins_{}_ori.csv".format(type1)))
    feature_imp=pd.read_csv(os.path.join(save_path,'lightgbm_10fold_Importance_{}.csv'.format(type1)))
    # feature_imp=feature_imp[feature_imp['Features'].isin(select_proteins)]
    mydf = pd.merge(auc_df, feature_imp, how='left', on='Features')
    mydf=pd.merge(mydf,ori_replace,how='left',on='Features')
    mydf.drop(columns=['Features'],inplace=True)
    mydf.rename(columns={'ori_name':'Features'},inplace=True)
    mydf['AUC_mean']=mydf['AUC_mean'].apply(convert_to_float)
    mydf['AUC_std'] = mydf['AUC_std'].apply(convert_to_float)
    mydf['p_delong'] = mydf['p_delong'].apply(convert_to_float)
      # print(mydf.columns.to_list())'Systolic_blood_pressure__automated_reading':'Systolic_blood_pressure',"Diastolic_blood_pressure__automated_reading":'Diastolic_blood_pressure'
    print(mydf['Features'])
    mydf.loc[mydf['Features'] == 'Systolic blood pressure, automated reading', 'Features'] = 'Systolic blood pressure'
    mydf.loc[mydf['Features'] == 'Diastolic blood pressure, automated reading', 'Features'] = 'Diastolic blood pressure'
    mydf.loc[mydf['Features'] == 'Cholesterol', 'Features'] = 'Total cholesterol'
    mydf = mydf.iloc[:50, :]
    result=mydf[['Features',"TotalGain_cv",'AUC_mean','AUC_std','p_delong']]
    result.to_csv(os.path.join(save_path,'Top50_predictor_important_accumulative_AUC_proteins.csv'),index=False)
    print(mydf.head())
    mydf['AUC_lower'] = mydf['AUC_mean'] - 1.96 * mydf['AUC_std']
    mydf['AUC_upper'] = mydf['AUC_mean'] + 1.96 * mydf['AUC_std']
    mydf['Features_idx'] = [i for i in range(1, len(mydf) + 1)]
    nb_f = 5#get_nb_f_3(mydf)
    print("nb_f",nb_f)
    fig, ax = plt.subplots(figsize=(18, 6.5))
    palette = sns.color_palette("Blues", n_colors=len(mydf))
    palette.reverse()
    sns.barplot(ax=ax, x="Features", y="TotalGain_cv", palette=palette, data=mydf.sort_values(by="TotalGain_cv", ascending=False))
    y_imp_up_lim = round(mydf['TotalGain_cv'].max() + 0.01, 2)
    ax.set_ylim([0, y_imp_up_lim])
    ax.tick_params(axis='y', labelsize=14)
    ax.set_xticklabels(mydf['Features'], rotation=45, fontsize=10, horizontalalignment='right')
    my_col = ['r'] * nb_f + ['k'] * (len(mydf) - nb_f)
    for ticklabel, tickcolor in zip(plt.gca().get_xticklabels(), my_col):
        ticklabel.set_color(tickcolor)

    ax.set_ylabel('Predictor Importance', weight='bold', fontsize=18)
    # ax.set_title(my_title, y=1.0, pad=-25, weight='bold', fontsize=24)
    ax.set_xlabel('')
    ax.grid(which='minor', alpha=0.2, linestyle=':')
    ax.grid(which='major', alpha=0.5, linestyle='--')
    ax.set_axisbelow(True)

    ax2 = ax.twinx()
    ax2.plot(np.arange(nb_f + 1), mydf['AUC_mean'][:nb_f + 1], 'red', alpha=0.8, marker='o')
    ax2.plot(np.arange(nb_f + 1, len(mydf)), mydf['AUC_mean'][nb_f + 1:], 'black', alpha=0.8, marker='o')
    ax2.plot([nb_f, nb_f + 1], mydf['AUC_mean'][nb_f:nb_f + 2], 'black', alpha=0.8, marker='o')
    plt.fill_between(mydf['Features_idx'] - 1, mydf['AUC_lower'], mydf['AUC_upper'], color='tomato', alpha=0.2)
    ax2.set_ylabel('Cumulative AUC', weight='bold', fontsize=18)
    ax2.tick_params(axis='y', labelsize=14)
    y_auc_up_lim = round(mydf['AUC_upper'].max() + 0.02, 2)
    y_auc_low_lim = round(mydf['AUC_lower'].min() - 0.02, 2)
    ax2.set_ylim([y_auc_low_lim, y_auc_up_lim])

    fig.tight_layout()
    plt.xlim([-.6, len(mydf) - .2])
    for spine in plt.gca().spines.values():
        spine.set_visible(True)
    plt.savefig(os.path.join(save_path,"cumulated_auc_and_feature_important_{}_ori.jpg".format(type1)),dpi=300)
    plt.show()
    plt.close()
def get_top_pros(mydf):
    p_lst = mydf.p_delong.tolist()
    i = 0
    while((p_lst[i]<0.05)|(p_lst[i+1]<0.05)):
        i+=1
    return i

def prepare_lightgbm_input(data,data_val,type):
    # data=data.d
    label_all = data['label']
    label_all_val=data_val['label']
    data_all = data.drop(columns=['label'])
    data_all_val=data_val.drop(columns=['label'])
    if "eid" in data_all.columns.to_list():
        data_all=data_all.drop(columns=['eid'])
    if "eid" in data_all_val.columns.to_list():
        data_all_val=data_all_val.drop(columns=['eid'])
    if "conv" in type: #'sex', 'age','education', 'Townsend', 'BMI', 'Smoking', 'Alcohol','RF_TBI',
        print("has category")
        cat_list=[]
        for i in ['Sex', 'Smoking', 'Alcohol', 'TBI']:
            if i in data_all.columns.to_list():
                cat_list.append(i)
        # data_all[['sex','Smoking', 'Alcohol','RF_TBI']]=data_all[['sex','Smoking', 'Alcohol','RF_TBI']].astype('category')
        # data_all_val[['sex', 'Smoking', 'Alcohol', 'RF_TBI']] = data_all_val[
        #     ['sex', 'Smoking', 'Alcohol', 'RF_TBI']].astype('category')
        data_all[cat_list]=data_all[cat_list].astype('category')
        data_all_val[cat_list]=data_all[cat_list].astype('category')

    # if "conv" in type:
    #     data_all_val[['sex','Smoking', 'Alcohol','RF_TBI']]=data_all_val[['sex','Smoking', 'Alcohol','RF_TBI']].astype('category')
    return data_all,label_all,data_all_val,label_all_val



def computed_auc_shapvalue_by_selected_Feature(data_proteins,label,data_all_val,label_all_val,save_path,type1,ori_replace,scale_pos_weight):

    # data_proteins=data_proteins[selected_feature]
    print("data_proteins",data_proteins.shape)
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=2024)
    auc_scores = []
    auc_scores_thr= []
    tprs = []
    auc_scores_val=[]
    auc_scores_val_thr=[]
    tprs_val=[]
    mean_fpr = np.linspace(0, 1, 100)
    mean_fpr_val=np.linspace(0, 1, 100)
    plt.figure(figsize=(10, 8))
    # 模型参数

    # my_params = {'n_estimators': 800,
    #              'max_depth': 6,
    #              'num_leaves': 30,
    #              'min_child_samples': 30,
    #              'subsample': 0.6,
    #              'learning_rate': 0.01,
    #              'reg_alpha': 0.1,
    #               'reg_lambda': 0.1,
    #              'colsample_bytree': 0.6,
    #              # 样本不平衡处理
    #              'is_unbalance': True,  # 保留（开启内置不平衡处理）
    #              # 'scale_pos_weight': scale_pos_weight,  # 新增（正类权重=负样本数/正样本数=3028/1082≈2.8）
    #              'class_weight': 'balanced'  # 新增（显式平衡类别权重，与is_unbalance互补）
    #              }
    #best params
    my_params = {'n_estimators': 500,
                 'max_depth': 10,
                 'num_leaves': 10,
                 'subsample': 0.7,
                 'learning_rate': 0.01, #best 0.01
                 'colsample_bytree': 0.7, "is_unbalance": True, }
    #only age gender  tbi
    # my_params = {'n_estimators': 200,
    #              'max_depth': 5,
    #              'num_leaves': 5 ,
    #              'subsample': 0.6,
    #              'learning_rate': 0.03, #best 0.01
    #              'colsample_bytree': 0.6,}
    # my_params = {'n_estimators': 200,
    #              'max_depth': 5,
    #              'num_leaves': 5 ,
    #              'subsample': 0.6,
    #              'learning_rate': 0.1 , #best 0.01
    #              'colsample_bytree': 0.6,}
    #sex1
    # my_params = {'n_estimators': 500,
    #              'max_depth': 8,
    #              'num_leaves': 15,
    #              'subsample': 0.7,
    #              'learning_rate': 0.01,
    #              'colsample_bytree': 0.7, }
    #sex0
    # my_params = {'n_estimators': 500,
    #              'max_depth': 8,
    #              'num_leaves': 8,
    #              'subsample': 0.7,
    #              'learning_rate': 0.001,
    #              'colsample_bytree': 0.7, }

    cat_list=[]
    # cat_list=data_proteins.columns.to_list( )
    for i in ['Sex','Smoking', 'Alcohol','TBI']:
        if i in data_proteins.columns.to_list():
            cat_list.append(i)
            data_all_val[i]=data_all_val[i].astype('category')
            data_proteins[i]=data_proteins[i].astype('category')
    cm_list=[]
    cm_list_val = []
    predictions_all=[]
    label_match_prediction=[]
    predictions_all_val = []
    best_thresholds_val_all=[]
    sensitivity_list=[]
    specificity_list=[]
    smote = SMOTE(random_state=42)
    predictions_all_val_proba=[]
    # data_proteins_resampled, label_resampled = smote.fit_resample(data_proteins, label)

    for fold, (train_index, val_index) in enumerate(kf.split(data_proteins, label)):
        X_train,y_train = data_proteins.iloc[train_index,:], label.iloc[train_index]
        X_val , y_val = data_proteins.iloc[val_index,:], label.iloc[val_index]

        # model = lgb.train(params, train_data, valid_sets=[val_data])
        my_lgb = LGBMClassifier(objective='binary', metric=['auc', 'binary_logloss', 'f1'], n_jobs=4, verbosity=-1,
                                seed=2024)
        my_lgb.set_params(**my_params)
        # my_lgb.fit(X_train, y_train)
        if "conv" in type1:
            my_lgb.fit(X_train, y_train, categorical_feature=cat_list) #['sex','Smoking', 'Alcohol','RF_TBI']
        else:
            my_lgb.fit(X_train, y_train)

        y_pred = my_lgb.predict_proba(X_val)[:, 1]
        # y_pred = my_lgb.predict(X_val, num_iteration=my_lgb.best_iteration)
        y_pred_val=my_lgb.predict_proba(data_all_val,num_iteration=my_lgb._best_iteration)[:,1] #,num_iteration=my_lgb._best_iteration
        predictions_all_val_proba.append(y_pred_val)
        # 绘制ROC曲线

        auc = roc_auc_score(y_val, y_pred)
        auc_scores.append(round(auc,5))
        print("flold {} auc {}".format(fold,round(auc,5)))
        fpr, tpr, thresholds = roc_curve(y_val, y_pred)
        tprs.append(np.interp(mean_fpr, fpr, tpr))

        youden_index = tpr - fpr
        best_index = np.argmax(youden_index)
        best_threshold = thresholds[best_index]
        # print(f"最佳阈值 (Fold {fold + 1}): {best_threshold}")
        y_pred_class=np.where(y_pred > best_threshold, 1, 0)
        predictions_all.extend(y_pred_class)
        label_match_prediction.extend(y_val)
        auc_best_thrd = accuracy_score(y_val, y_pred_class)
        # print("auc_最佳阈值", auc_best_thrd)
        auc_scores_thr.append(auc_best_thrd)

        cm_fold = confusion_matrix(y_val, y_pred_class)  # y_val是真实标签，y_pred_class是预测类别

        # 2. 提取TP、TN、FP、FN（混淆矩阵结构：[[TN, FP], [FN, TP]]）
        TN, FP, FN, TP = cm_fold.ravel()  # ravel()将矩阵展平为一维数组 [TN, FP, FN, TP]
        # 3. 计算sensitivity和specificity（处理除零情况）
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0  # 避免分母为0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        # 4. 存储结果并打印
        sensitivity_list.append(round(sensitivity,5))
        specificity_list.append(round(specificity,5))
        print(f"Fold {fold} - Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}")


        auc_val=roc_auc_score(label_all_val,y_pred_val)
        print("val flold {} auc {}".format(fold, auc_val))
        auc_scores_val.append(round(auc_val,5))
        # print("auc_val", auc_val)
        fpr_val, tpr_val, thresholds_val = roc_curve(label_all_val, y_pred_val)
        tprs_val.append(np.interp(mean_fpr_val, fpr_val, tpr_val))

        youden_index_val = tpr_val - fpr_val
        best_index_val = np.argmax(youden_index_val)
        best_threshold_val = thresholds_val[best_index_val]
        best_thresholds_val_all.append(best_threshold_val)
        # print(f"val 最佳阈值 (Fold {fold + 1}): {best_threshold_val}")
        y_test_pred_labels =np.where(y_pred_val > best_threshold_val,1,0)# (y_pred_val > thresholds_val).astype(int)
        auc_val_best_thrd = accuracy_score(label_all_val, y_test_pred_labels)
        auc_scores_val_thr.append(auc_val_best_thrd)
        # print("auc_val_最佳阈值",auc_val_best_thrd)
        predictions_all_val.append(y_test_pred_labels)

    cm = confusion_matrix(label_match_prediction, predictions_all)
    print("avg_cm")
    print(cm)
    mean_proba_val = np.mean(predictions_all_val_proba, axis=0)  # 平均所有折的概率
    best_threshold_avg = np.mean(best_thresholds_val_all)  # 平均最佳阈值
    y_pred_val_class = np.where(mean_proba_val > best_threshold_avg, 1, 0)  # 分类
    cm_val = confusion_matrix(label_all_val, y_pred_val_class)  # 计算测试集混淆矩阵
    print("avg_cm_val")
    print(cm_val)

    print("auc list")
    print(auc_scores)
    print("auc list val")
    print(auc_scores_val)
    print("sensitivity")
    print(sensitivity_list)
    print("specificity")
    print(specificity_list)
    # 平均ROC曲线
    # top_3_indices = sorted(range(len(auc_scores)), key=lambda i: auc_scores[i], reverse=True)[:3]
    # tprs=[tprs[i] for i in top_3_indices]
    # auc_scores=[auc_scores[i] for i in top_3_indices]
    result=pd.DataFrame(np.array(auc_scores),columns=['auc_scores'])
    result["tprs"]=tprs#pd.DataFrame(np.array(tprs))
    result.to_csv(os.path.join(save_path,"incident_auc_scores_tprs_{}_youden.csv").format(type1),index=False)

    result_val=pd.DataFrame({'auc_scores':auc_scores_val,
                             'tprs':tprs_val})
    result_val.to_csv(os.path.join(save_path,"diagnosed_auc_scores_tprs_{}_youden.csv").format(type1),index=False)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(auc_scores)
    # mean_auc = roc_auc_score(data_proteins, my_lgb.predict(label))
    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.4f)' % mean_auc, lw=2, alpha=0.8,)
    plt.fill_between(mean_fpr, np.minimum(mean_tpr + np.std(tprs, axis=0), 1), mean_tpr - np.std(tprs, axis=0),
                     color='grey', alpha=0.2)
    # 完善图像细节
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('10-Fold Cross-Validation ROC Curves')
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig(os.path.join(save_path,"incident_AUC_selected_feature_{}_youden.jpg").format(type1))
    plt.close()
    print(f'Average AUC: {np.mean(auc_scores):.4f}')
    print(f'Average ACC best threshold: {np.mean(auc_scores_thr):.4f}')
    print(f'Average AUC val: {np.mean(auc_scores_val):.4f}')
    print(f'Average ACC val_best_threshold: {np.mean(auc_scores_val_thr):.4f}')

    mean_tpr_val = np.mean(tprs_val, axis=0)
    mean_tpr_val[-1] = 1.0
    mean_auc_val = np.mean(auc_scores_val)
    plt.plot(mean_fpr_val, mean_tpr_val, color='b', label=r'Mean ROC (AUC = %0.4f)' % mean_auc_val, lw=2, alpha=0.8,)
    plt.fill_between(mean_fpr_val, np.minimum(mean_tpr_val + np.std(tprs_val, axis=0), 1), mean_tpr_val - np.std(tprs_val, axis=0),
                     color='grey', alpha=0.2)
    # 完善图像细节
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('10-Fold Cross-Validation ROC Curves')
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig(os.path.join(save_path,"diagnosed_AUC_selected_feature_{}_youden.jpg").format(type1))
    plt.close()

    explainer = shap.TreeExplainer(my_lgb)
    shap_values = explainer.shap_values(data_proteins)
    print(shap_values.shape) #(36142, 14)
    shap_values_df=pd.DataFrame(shap_values)
    # shap_values_df.to_csv(os.path.join(save_path,"incident_shap_values_{}.csv").format(type1),index=False)

    shap_values_val = explainer.shap_values(data_all_val)
    print(shap_values_val.shape)  # (36142, 14)
    shap_values_df_val = pd.DataFrame(shap_values_val)
    # shap_values_df_val.to_csv(os.path.join(save_path, "diagnosed_shap_values_{}.csv").format(type1), index=False)
    # 绘制SHAP Summary Plot

    rename_colunm=[]
    for i in data_proteins.columns.to_list():
        if i in ori_replace['Features'].values.tolist():
            rename_colunm.append(ori_replace[ori_replace['Features']==i]['ori_name'].values.tolist()[0])
        else:
            rename_colunm.append(i)

    data_proteins.columns=rename_colunm
    data_all_val.columns=rename_colunm

    data_proteins.rename(columns={'Cholesterol':'Total cholesterol'},inplace=True)
    data_all_val.rename(columns={'Cholesterol': 'Total cholesterol'}, inplace=True)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, data_proteins)
    plt.savefig(os.path.join(save_path, "incident_Shapvalue_selected_feature_{}.jpg").format(type1),dpi=300)
    plt.close()
    #
    # plt.figure(figsize=(10, 6))
    # shap.summary_plot(shap_values_val, data_all_val)
    # # plt.savefig(os.path.join(save_path, "diagnosed_Shapvalue_selected_feature_{}.jpg").format(type1),dpi=300)
    # plt.close()

def prepare_tprs(df):
    trps=np.array(df['tprs'].values)
    trps_arr=[]
    for i in range(10):
        trps_l=trps[i][1:-1].split(" ")
        trps_l=list(filter(lambda x: x.strip() != "", trps_l))
        trps_l=[float(item.strip()) for item in trps_l]
        trps_arr.append(trps_l)
    return trps_arr
def plot_auc(auc_proteins,auc_DGinfo,auc_proteins_DGinfo,type,save_path):
    #'steelblue', 'deepskyblue', 'yellowgreen'
    col1, col2, col3 ='#1f77b4', '#ff7f0e', '#d62728'
    mean_fpr_proteins = np.linspace(0, 1, 100)
    tprs_proteins=prepare_tprs(auc_proteins)
    auc_scores_prteins=auc_proteins['auc_scores'].values
    mean_tpr_proteins = np.mean(tprs_proteins, axis=0)
    mean_tpr_proteins[-1] = 1.0
    mean_auc_proteins = np.mean(auc_scores_prteins)

    mean_fpr_proteins_DGinfo = np.linspace(0, 1, 100)
    tprs_proteins_DGinfo=prepare_tprs(auc_proteins_DGinfo)
    auc_scores_prteins_DGinfo=auc_proteins_DGinfo['auc_scores'].values
    mean_tpr_proteins_DGinfo = np.mean(tprs_proteins_DGinfo, axis=0)
    mean_tpr_proteins_DGinfo[-1] = 1.0
    mean_auc_proteins_DGinfo = np.mean(auc_scores_prteins_DGinfo)

    mean_fpr_DGinfo = np.linspace(0, 1, 100)
    tprs_DGinfo=prepare_tprs(auc_DGinfo)
    auc_scores_DGinfo=auc_DGinfo['auc_scores'].values
    mean_tpr_DGinfo = np.mean(tprs_DGinfo, axis=0)
    mean_tpr_DGinfo[-1] = 1.0
    mean_auc_DGinfo = np.mean(auc_scores_DGinfo)
    plt.figure(figsize=(5, 4))
    plt.plot(mean_fpr_proteins, mean_tpr_proteins,col1, label=r'Blood (AUC = %0.2f)' % mean_auc_proteins, lw=2, alpha=0.8,)
    plt.plot(mean_fpr_DGinfo, mean_tpr_DGinfo, col2, label=r'Demographic (AUC = %0.2f)' % mean_auc_DGinfo, lw=2,
             alpha=0.8, )
    plt.plot(mean_fpr_proteins_DGinfo, mean_tpr_proteins_DGinfo, col3, label=r'Blood+Demographic (AUC = %0.3f)' % mean_auc_proteins_DGinfo, lw=2,
             alpha=0.8, )

    # plt.fill_between(mean_fpr, np.minimum(mean_tpr + np.std(tprs, axis=0), 1), mean_tpr - np.std(tprs, axis=0),
    #                  color='grey', alpha=0.2)
    # 完善图像细节
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('10-Fold Cross-Validation ROC Curves')
    plt.legend(loc="lower right",fontsize=8)
    # plt.show()
    plt.savefig(os.path.join(save_path,"AUC_incident_all_{}.jpg").format(type),dpi=300)
    plt.close()
def plot_auc_incident_diagnosed(auc_incident,auc_diagnosed,type,save_path):
    #'steelblue', 'deepskyblue', 'yellowgreen'
    col1, col2, col3 ='#1f77b4', '#ff7f0e', '#d62728'
    mean_fpr_proteins = np.linspace(0, 1, 100)
    tprs_proteins=prepare_tprs(auc_incident)
    auc_scores_prteins=auc_incident['auc_scores'].values
    mean_tpr_proteins = np.mean(tprs_proteins, axis=0)
    mean_tpr_proteins[-1] = 1.0
    mean_auc_proteins = np.mean(auc_scores_prteins)
    std_auc_proteins=np.std(auc_scores_prteins)

    mean_fpr_proteins_DGinfo = np.linspace(0, 1, 100)
    tprs_proteins_DGinfo=prepare_tprs(auc_diagnosed)
    auc_scores_prteins_DGinfo=auc_diagnosed['auc_scores'].values
    mean_tpr_proteins_DGinfo = np.mean(tprs_proteins_DGinfo, axis=0)
    mean_tpr_proteins_DGinfo[-1] = 1.0
    mean_auc_proteins_DGinfo = np.mean(auc_scores_prteins_DGinfo)
    std_auc_proteins_DGinfo=np.std(auc_scores_prteins_DGinfo)

    plt.figure(figsize=(5, 4))
    plt.plot(mean_fpr_proteins, mean_tpr_proteins,col1, label="Incident   PD (AUC = {} ± {})".format(np.round(mean_auc_proteins,2),np.round(std_auc_proteins,2)), lw=2, alpha=0.8,)
    plt.plot(mean_fpr_proteins_DGinfo, mean_tpr_proteins_DGinfo, col3, label="Diagnosed PD(AUC = {} ± {})".format(np.round(mean_auc_proteins_DGinfo,2),np.round(std_auc_proteins_DGinfo,3)), lw=2,
             alpha=0.8, )
    #


    # plt.fill_between(mean_fpr, np.minimum(mean_tpr + np.std(tprs, axis=0), 1), mean_tpr - np.std(tprs, axis=0),
    #                  color='grey', alpha=0.2)
    # 完善图像细节
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('10-Fold Cross-Validation ROC Curves')
    plt.legend(loc="lower right",fontsize=8)
    # plt.show()
    plt.savefig(os.path.join(save_path,"AUC_incident_diagnosed_{}.jpg".format(type)),dpi=300)
    plt.close()



def GridSesarch_find_best_param(X_train, y_train):
    cat_list=X_train.columns.to_list()
    for c in cat_list:
        X_train[c]= X_train[c].astype('category')  # 将分类特征转换为类别类型
    my_params = {'n_estimators': 500,
                 'max_depth': 15,
                 'num_leaves': 10,
                 'subsample': 0.7,
                 'learning_rate': 0.01,
                 'colsample_bytree': 0.7}
    model = lgb.LGBMClassifier()
    # 定义参数网格
    param_grid = {
        'num_leaves': [10, 20, 31, 40],
        'max_depth': [5,10, 15,20],
        'learning_rate': [0.1, 0.01, 0.005],
        'n_estimators': [200, 400,800],
        'lambda_l1': [0, 0.1, 1],
        'lambda_l2': [0, 0.1, 1],
        'subsample': [0.8, 1.0],
        'scale_pos_weight': [1, 2, 5]

    }
    grid_search = GridSearchCV(model, param_grid, scoring='roc_auc', cv=5)
    grid_search.fit(X_train, y_train, categorical_feature=cat_list)
    print("Best parameters found: ", grid_search.best_params_)
    #Best parameters found:  {'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 400, 'num_leaves': 20, 'subsample': 0.8}
    print("Best AUC: ", grid_search.best_score_) #Best AUC:  0.6747595682873203

def calculate_risk_metrics_with_ci(df, n_bootstrap=1000, ci=95):
    """
    计算 Top vs Bottom Decile 的 Risk Difference 和 RR，并带上 95% CI
    """

    def get_metrics(data):
        # 按照当前数据的分位数划组
        data = data.copy()
        # 使用 rank 处理相同预测值的边界问题
        data['decile'] = pd.qcut(data['pred_prob'].rank(method='first'), 10, labels=range(1, 11))

        risk_stats = data.groupby('decile')['true_label'].mean()
        r_bottom = risk_stats.iloc[0]
        r_top = risk_stats.iloc[-1]

        diff = r_top - r_bottom
        rr = r_top / r_bottom if r_bottom > 0 else np.nan
        return diff, rr

    # 1. 计算原始观测值
    obs_diff, obs_rr = get_metrics(df)

    # 2. Bootstrap 重采样
    boot_diffs = []
    boot_rrs = []

    print(f"Running {n_bootstrap} bootstrap iterations...")
    for i in range(n_bootstrap):
        boot_sample = resample(df, replace=True)
        try:
            d, r = get_metrics(boot_sample)
            boot_diffs.append(d)
            if not np.isnan(r):
                boot_rrs.append(r)
        except:
            continue  # 防止极端抽样导致分位数计算失败

    # 3. 计算置信区间 (百分位数法)
    lower_bound = (100 - ci) / 2
    upper_bound = 100 - lower_bound

    diff_ci = np.percentile(boot_diffs, [lower_bound, upper_bound])
    rr_ci = np.percentile(boot_rrs, [lower_bound, upper_bound])

    return {
        'diff': obs_diff, 'diff_ci': diff_ci,
        'rr': obs_rr, 'rr_ci': rr_ci
    }
def calculate_net_benefit(y_true, y_prob, thresholds):
    net_benefit = []
    n = len(y_true)
    for pt in thresholds:
        # 计算在当前阈值下的 TP 和 FP
        y_pred = (y_prob >= pt).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))

        # 净获益公式: (TP / n) - (FP / n) * (pt / (1 - pt))
        if pt == 1.0:  # 避免除以 0
            nb = 0
        else:
            nb = (tp / n) - (fp / n) * (pt / (1 - pt))
        net_benefit.append(nb)
    return net_benefit
def draw_multi_model_DCA_calibration_figure_v124(models_config, stage_name, save_path, type_name):
    """
    绘制多模型 DCA 和 Calibration 曲线（支持 CV 平均值 和 外部验证单次值）

    :param models_config: list of dict, e.g. [{'name': 'XGB', 'file': 'path.csv'}, ...]
    :param stage_name: str, e.g. 'Internal_Validation'
    :param save_path: str, 保存路径
    :param type_name: str, e.g. 'Comparison' (用于文件名)
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    colors_list = ['#c0392b', '#2980b9', '#27ae60', '#8e44ad', '#d35400', '#f1c40f', '#34495e']

    # --- 阈值范围 (DCA用) ---
    thresholds = np.linspace(0.001, 0.2, 100)

    print(f"Plotting for Stage: {stage_name}")

    # ==========================================
    # 1. 绘制基准线 (基于第一个模型的数据概况)


    # ==========================================
    try:
        base_df = pd.read_csv(models_config[0]['file'])
        # 处理列名空格
        base_df.columns = [c.strip() for c in base_df.columns]

        # 计算总体 Treat All (如果是CV，就基于所有Fold的总患病率)
        y_test_base = base_df['true_label'].values

        # 这里的 calculate_net_benefit_all 需要你之前定义的函数
        # 简单实现一下逻辑，防止报错
        prevalence = np.mean(y_test_base)

        nb_all = prevalence - (1 - prevalence) * thresholds / (1 - thresholds)
        nb_none = np.zeros(len(thresholds))

        # 1. Calibration 基准线
        ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated", alpha=0.6)

        # 2. DCA 基准线
        ax2.plot(thresholds, nb_all, color='gray', linestyle='--', label="Treat All", alpha=0.5)
        ax2.plot(thresholds, nb_none, color='black', linestyle=':', label="Treat None", alpha=0.5)

    except Exception as e:
        print(f"Warning: Failed to draw baselines. {e}")

    # ==========================================
    # 2. 循环处理每个模型
    # ==========================================
    for i, model_cfg in enumerate(models_config):
        model_name = model_cfg['name']
        file_path = model_cfg['file']
        color = colors_list[i % len(colors_list)]

        if not os.path.exists(file_path):
            print(f"Skipping {model_name}: File not found.")
            continue

        try:
            df = pd.read_csv(file_path)
            # 清洗列名
            df.columns = [c.strip() for c in df.columns]


            # 判断是 CV 还是 External
            # unique_folds = df['fold'].unique()
            # print("Processing Model: {}, Folds Detected: {}".format(model_name, unique_folds))
            # is_cv = len(unique_folds) > 1 #and 'None' not in unique_folds and 'External' not in unique_folds

            # 初始化变量
            final_frac_pos = []
            final_mean_pred = []
            final_nb = []
            final_brier = 0.0
            # for diagnosied, external validation, contain 10 fold cv, but only use the mean value of true label for calibration curve and DCA curve
            # report error  Only binary classification is supported. The type of the target is continuous.

            # if model_name=='Diagnosed':
            #     df['sample_idx'] = df.groupby('fold').cumcount()
            #
            #     # 3. 计算平均概率
            #     # 我们保留 true_label（取 mean 还是 original 都行，因为它们相等）
            #     df = df.groupby('sample_idx').agg({
            #         'true_label': 'first',  # 取每一组的第一个标签即可
            #         'pred_prob': 'mean'  # 核心：计算 10 折概率的平均值
            #     }).reset_index(drop=True)
            #     df['fold']=0
            is_cv = 'fold' in df.columns and len(df['fold'].unique()) > 1

            # ---------------------------------------------------
            # 分支 A: 交叉验证模式 (计算平均值 / Pooled)
            # ---------------------------------------------------
            if is_cv:
                unique_folds= df['fold'].unique()
                # 1. Calibration (使用 Pooled 方法：合并所有数据)
                # 这是最科学的方法，展示该建模策略在整个数据集上的整体校准度
                y_true_all = df['true_label'].values
                y_prob_all = df['pred_prob'].values  # 或者是 y_prob

                final_brier = brier_score_loss(y_true_all, y_prob_all)
                final_frac_pos, final_mean_pred = calibration_curve(y_true_all, y_prob_all, n_bins=20,
                                                                    strategy='quantile')#strategy='quantile' uniform

                # 2. DCA (计算每一折 Net Benefit 然后求平均)
                nb_list = []
                for fold in unique_folds:
                    fold_data = df[df['fold'] == fold]
                    nb_fold = calculate_net_benefit_model(fold_data['true_label'].values,
                                                          fold_data['pred_prob'].values,
                                                          thresholds)
                    nb_list.append(nb_fold)

                # 按列求平均
                final_nb = np.mean(nb_list, axis=0)

                label_suffix = " (Mean CV)"

            # ---------------------------------------------------
            # 分支 B: 单次/外部验证模式
            # ---------------------------------------------------
            else:
                y_true = df['true_label'].values
                y_prob = df['pred_prob'].values  # 或者是 y_prob

                final_brier = brier_score_loss(y_true, y_prob)
                final_frac_pos, final_mean_pred = calibration_curve(y_true, y_prob, n_bins=20, strategy='quantile')

                final_nb = calculate_net_benefit_model(y_true, y_prob, thresholds)
                label_suffix = ""

            # ---------------------------------------------------
            # 绘图
            # ---------------------------------------------------

            # Plot Calibration
            ax1.plot(final_mean_pred, final_frac_pos, "s-",
                     color=color,
                     label=f"{model_name} (Brier={final_brier:.3f})",
                     linewidth=1.5, markersize=5, alpha=0.8)

            # Plot DCA
            ax2.plot(thresholds, final_nb,
                     color=color,
                     linewidth=2,
                     label=f"{model_name}{label_suffix}", alpha=0.8)

            print(f"   Processed {model_name}: Brier={final_brier:.3f}")

        except Exception as e:
            print(f"Error processing {model_name}: {e}")
            import traceback
            traceback.print_exc()

    # ==========================================
    # 3. 美化图表
    # ==========================================
    limit_val = max(prevalence * 3, 0.05)  # 动态设置，通常看 5% 就够了
    ax1.set_xlim([0, limit_val])
    ax1.set_ylim([0, limit_val])
    ax1.set_xlabel("Predicted Probability", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Observed Prevalence", fontsize=12, fontweight='bold')
    ax1.set_title("Calibration Curve", fontsize=14, fontweight='bold')
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(True, which='both', linestyle=':', alpha=0.5)

    # --- 美化 DCA (右图) ---
    ax2.set_xlim([0, 0.1])  # 罕见病重点看 0-5% 阈值
    # Y轴范围设置：稍微给一点负值空间，上方留出患病率的1.5倍
    ax2.set_ylim([-0.1,0.1])
    ax2.set_xlabel("Threshold Probability", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Net Benefit", fontsize=12, fontweight='bold')
    ax2.set_title("Decision Curve Analysis(DCA))", fontsize=14, fontweight='bold')
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()

    if not os.path.exists(save_path): os.makedirs(save_path)
    save_filename = f'{stage_name}_{type_name}_Final_DCA_Calibration.png'
    plt.savefig(os.path.join(save_path, save_filename), dpi=600, bbox_inches='tight')
    plt.show()

def survival_analysis_and_plot(df, save_path):
    # df = pd.read_csv("all_folds_predictions.csv")
    # 2. 如果同一 eid 在不同 fold 出现（虽然 CV 不会，但为了保险），按 eid 取平均概率
    df_grouped = df.groupby('eid').agg({
        'true_label': 'first',
        'pred_prob': 'mean',
        'duration': 'first'
    }).reset_index()
    # 3. 根据预测风险概率 pred_prob 进行分位切分
    # 计算分位数阈值
    top_threshold = df_grouped['pred_prob'].quantile(0.9)  # 前 10% 的高风险阈值
    median_low = df_grouped['pred_prob'].quantile(0.45)  # 中间 10% 的下限
    median_high = df_grouped['pred_prob'].quantile(0.55)  # 中间 10% 的上限
    bottom_threshold = df_grouped['pred_prob'].quantile(0.1)  # 最后 10% 的低风险阈值

    # 4. 定义分类函数
    def segment_risk(prob):
        if prob >= top_threshold:
            return 'Top 10% (High Risk)'
        elif median_low <= prob <= median_high:
            return 'Median 10%'
        elif prob <= bottom_threshold:
            return 'Bottom 10% (Low Risk)'
        else:
            return 'Others'

    df_grouped['risk_group'] = df_grouped['pred_prob'].apply(segment_risk)

    plot_df = df_grouped[df_grouped['risk_group'] != 'Others'].copy()
    plot_df = plot_df[plot_df['duration'] > 0]  # 过滤掉持续时间为0的数据

    # 2. 计算 Cox HR 值 (以 Bottom 10% 为参照)
    group_order = ['Bottom 10% (Low Risk)', 'Median 10%', 'Top 10% (High Risk)']
    plot_df['risk_group'] = pd.Categorical(plot_df['risk_group'], categories=group_order, ordered=True)
    cox_model_data = pd.get_dummies(plot_df[['duration', 'true_label', 'risk_group']],
                                    columns=['risk_group'], drop_first=True)

    cph = CoxPHFitter()
    cph.fit(cox_model_data, duration_col='duration', event_col='true_label')
    hr_summary = cph.summary[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']]
    print(hr_summary)
    # 3. 准备表格文字
    table_data = [
        ["Group", "HR (95% CI)", "P-value"],
        ["Bottom 10%", "Ref.", "-"],
        ["Median 10%",
         f"{hr_summary.loc['risk_group_Median 10%', 'exp(coef)']:.2f} ({hr_summary.loc['risk_group_Median 10%', 'exp(coef) lower 95%']:.2f}-{hr_summary.loc['risk_group_Median 10%', 'exp(coef) upper 95%']:.2f})",
         f"{hr_summary.loc['risk_group_Median 10%', 'p']:.3f}"],
        ["Top 10%",
         f"{hr_summary.loc['risk_group_Top 10% (High Risk)', 'exp(coef)']:.2f} ({hr_summary.loc['risk_group_Top 10% (High Risk)', 'exp(coef) lower 95%']:.2f}-{hr_summary.loc['risk_group_Top 10% (High Risk)', 'exp(coef) upper 95%']:.2f})",
         f"{hr_summary.loc['risk_group_Top 10% (High Risk)', 'p']:.3f}"]
    ]

    # 4. 绘图
    plt.figure(figsize=(10, 8), dpi=300)
    kmf = KaplanMeierFitter()
    colors = {'Top 10% (High Risk)': '#d63031', 'Median 10%': '#fdcb6e', 'Bottom 10% (Low Risk)': '#00b894'}

    for group in group_order:
        mask = (plot_df['risk_group'] == group)
        kmf.fit(plot_df[mask]['duration'], event_observed=plot_df[mask]['true_label'], label=group)
        kmf.plot_survival_function(color=colors[group], lw=2.5, ci_show=True, alpha=0.8)

    # 5. 计算整体 Log-rank P 值
    lr_results = multivariate_logrank_test(plot_df['duration'], plot_df['risk_group'], plot_df['true_label'])
    plt.annotate(f'Log-rank p < 0.001' if lr_results.p_value < 0.001 else f'Log-rank p = {lr_results.p_value:.3f}',
                 xy=(0.05, 0.05), xycoords='axes fraction', fontsize=12, fontweight='bold')

    # 6. 插入 HR 统计表格
    the_table = plt.table(cellText=table_data,
                          colWidths=[0.2, 0.35, 0.15],
                          loc='lower left',
                          cellLoc='center',
                          edges='horizontal',
                          bbox=[0.05, 0.15, 0.45, 0.2])  # [left, bottom, width, height]
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(9)
    for key, cell in the_table.get_celld().items():
        cell.set_linewidth(0.6)  # 设置一个很细的线，或者设为 0 完全隐藏

    # 7. 细节美化
    plt.title('Survival Stratification by Model Predicted Risk', fontsize=16, pad=20, fontweight='bold')
    plt.xlabel('Follow-up Duration (Years)', fontsize=13)
    plt.xlim(0,16.5)
    plt.xticks(np.arange(0, 17.5, 2.5))
    plt.ylabel('Probability of PD-free Survival', fontsize=13)
    plt.ylim(0.96,1)  # 针对 PD 这种低发病率，Y轴通常截断在 0.8-1.0
    plt.yticks(np.arange(0.96,1.0,0.02))
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.legend(loc='lower right', frameon=True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path,"Survival_Curve_with_HR_Table.png"))
    plt.show()


if __name__ == '__main__':

    save_path = './prediction_result'
    if os.path.exists(save_path) is False:
        os.mkdir(save_path)
    #step1 load data  from .csv file
    incident_blood,dignosised_blood,incident_blood_conv_all,dignosised_blood_conv_all,incident_conv,dignosised_conv,ori_replace,incident_eid_label,diagnosied_eid_label=get_data()
    # preprare model input data
    data_all_proteins, label_all, data_all_val, label_all_val=prepare_lightgbm_input(incident_blood_conv_all,dignosised_blood_conv_all,'blood_conv')
    # step2  using gridsearch to find the best param for lightgbm
    GridSesarch_find_best_param(data_all_proteins,label_all)
    tyep1 = "gridsearch_find_best_param"
    curre_save_path = os.path.join(curre_save_path, tyep1)
    if os.path.exists(curre_save_path) is False:
        os.mkdir(curre_save_path)
    # step3 train model with best param and get feature important
    data_all_proteins, label_all,data_all_val,label_all_val=prepare_lightgbm_input(incident_blood_conv_all,dignosised_blood_conv_all,tyep1)
    # print(data_all_proteins.columns.to_list())
    # train model with 10 fold cv and get feature important of each fold
    train_lightgbm_10fold_classific_feature_important_each_protiens(data_all_proteins,label_all,curre_save_path,tyep1)
    #get model auc with different number of features and get the important of each feature with shap value
    computed_auc_cumulated_by_add_each_feature(data_all_proteins,label_all,curre_save_path,tyep1)
    # plot the cumulated AUC and important of each feature
    plot_cumulated_AUC_and_important_of_each_features(curre_save_path, tyep1,ori_replace)


    #step4 select top k feature with best auc
    camculated_path = "./prediction_result/incident_blood_conv_no_sex_age"
    proteins_list = pd.read_csv(
        os.path.join(camculated_path, "AUC_camculated_lightgbm_each_proteins_incident_blood_conv_no_sex_age.csv"))
    proteins_list['p_delong']=proteins_list['p_delong'].apply(convert_to_float)
    top_n = get_nb_f(proteins_list)
    proteins_list = proteins_list.iloc[:top_n, :]
    proteins_list = proteins_list['Features'].values.tolist()

    #step5 retain the top k feature and get the auc and shap value of each feature
    tyep1="topk_proteins"
    curre_save_path=os.path.join(curre_save_path,tyep1)
    if os.path.exists(curre_save_path) is False:
        os.mkdir(curre_save_path)
    data_all_proteins, label_all,data_all_val,label_all_val=prepare_lightgbm_input(incident_blood,dignosised_blood,tyep1)
    print("data_all_val",data_all_val.shape)
    print("label",label_all_val.shape)
    data_all_proteins=data_all_proteins[proteins_list]
    data_all_val=data_all_val[proteins_list]

    computed_auc_shapvalue_by_selected_Feature(proteins_list, data_all_proteins, label_all, data_all_val, label_all_val,
                                               curre_save_path, tyep1,'-')

    incident_auc_sex1 = pd.read_csv(
        os.path.join(curre_save_path,"incident_auc_scores_tprs_{}_youden.csv".format(tyep1)))
    diagnosed_auc_sex1  = pd.read_csv(
        os.path.join(curre_save_path, "diagnosed_auc_scores_tprs_{}_youden.csv").format(tyep1))
    #draw auc curve of incident and diagnosed with the same feature
    plot_auc_incident_diagnosed(incident_auc_sex1,diagnosed_auc_sex1,tyep1,curre_save_path)


    models_config = [
        {'name': 'Incident',
         'file': './predict_result/incident_10fold_predictions_{}.csv'.format(tyep1), },
        {'name': 'Diagnosed',
         'file': './predict_result/diagnosed_10fold_predictions_{}.csv'.format(tyep1), },
    ]
    #draw DCA and calibration curve of incident and diagnosed with the same feature
    draw_multi_model_DCA_calibration_figure_v124(models_config, "final_main", curre_save_path, "incident_diagnosed")

    all_fold_preds = pd.read_csv(os.path.join(curre_save_path, "internal_10fold_predictions_{}.csv".format(tyep1)))
    all_fold_preds['duration'] = all_fold_preds['duration'] * -1
    print("all_fold_preds.shape", all_fold_preds.shape, all_fold_preds['duration'].max(),
          all_fold_preds['duration'].min())
    # analyze the survival difference between top 10% and bottom 10% predicted risk group and plot the survival curve with HR table
    # survival_analysis_and_plot(all_fold_preds, curre_save_path)




