import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
import joblib
import streamlit as st

title = "Prediction of 30-Day Mortality in Critically Ill Patients with Bone Metastases"

st.set_page_config(    
    page_title=f"{title}",
    page_icon="⭕",
    layout="wide"
)

# 设置标题
st.markdown(f'''
    <h1 style="font-size: 32px; text-align: center; color: black; border-bottom: 2px solid black; margin-bottom: 1rem;">
    {title}
    </h1>''', unsafe_allow_html=True)

# 定义使用的特征变量
var = [
    "Weight", "Charlson_Comorbidity_Index", "SOFA", "Heart_Rate",
    "Resp_Rate", "Lactate", "Hematocrit", "Calcium", "Potassium", 
    "WBC", "Albumin"
]

var1 = """Weight (Kg)：0-200
Charlson Comorbidity Index：0-15
SOFA：0-24
Heart rate (Beats/min)：0-300
Respiratory rate (Breaths/min)：0-50
Lactate (mmol/L)：0-20
Hematocrit (%)：20-70
Ionized calcium (mmol/L)：0-20
Potassium (mmol/L) ：0-20
WBC (×10⁹/L)：0-100
Albumin (g/L)：0-100""".splitlines()

# 导入模型
m1 = joblib.load("preprocessor.pkl")
m2 = joblib.load("tabnet_model.pkl")

var1 = [[i.split("：")[0], float(i.split("：")[1].split("-")[0]), float(i.split("：")[1].split("-")[1])] for i in var1]

# 初始值
dinput = [float(i) for i in "78.4	10	6	137	33	0.9	29	0.92	3.4	5.9	2.9".split("\t")]

# 模型输入
d = {}
with st.expander("**Current input:**", True):
    col = st.columns(4)
    
    k = 0
    for i, j, m in zip(var, dinput, var1):
        d[i] = col[k%4].number_input(m[0], value=j, min_value=m[1], max_value=m[2], format="%0.2f")
        k = k+1
    
    # 输入值
    df = pd.DataFrame([d])

    # 预处理输入数据
    d = m1.transform(df)

    st.dataframe(df, hide_index=True, use_container_width=True)
    
with st.expander("**Predict result:**", True):
    res = m2.predict_proba(d).flatten().tolist()[0]
    if res < 0.3:
        r = "**:green[Low Risk]**"
        r1 = "Low RisK"
    elif res > 0.7:
        r = "**:red[High Risk]**"
        r1 = "High RisK"
    else:
        r1 = "Medium RisK"
        r = "**:orange[Medium Risk]**"

    col = st.columns(2)
    col[0].metric("Predict probability", f"{round(res*100, 2)}%", border=True)
    col[1].metric("Predict Risk level", f"{r1}", border=True)
    
    st.progress(res, f"**Predict probability：{round(res*100, 2)}%, {r}**")

    st.markdown("""
        **Clinical Interpretation:**  
        * :green[Low Risk (<30%): Standard care and routine monitoring are appropriate.]  
        * :orange[Medium Risk (30–70%): Intensified treatment and close monitoring are advised.]  
        * :red[High Risk (>70%): Immediate critical care intervention is strongly recommended.]  
    """)

    st.warning("**Note: This prediction tool supports but does not replace clinical judgment.**")
