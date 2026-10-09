import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, AllChem
from rdkit.Chem.Draw import MolDraw2DSVG
from rdkit.ML.Descriptors import MoleculeDescriptors
import pandas as pd
from autogluon.tabular import TabularPredictor
import tempfile
import base64
from io import BytesIO
from autogluon.tabular import FeatureMetadata
import gc
import re
from tqdm import tqdm 
import numpy as np


# 添加 CSS 样式
st.markdown(
    """
    <style>
    .stApp {
        border: 2px solid #808080;
        border-radius: 20px;
        margin: 50px auto;
        max-width: 40%;
        background-color: #f9f9f9f9;
        padding: 20px;
        box-sizing: border-box;
    }
    .rounded-container h2 {
        margin-top: -80px;
        text-align: center;
        background-color: #e0e0e0e0;
        padding: 10px;
        border-radius: 10px;
    }
    .rounded-container blockquote {
        text-align: left;
        margin: 20px auto;
        background-color: #f0f0f0;
        padding: 10px;
        font-size: 1.1em;
        border-radius: 10px;
    }
    .stMetric {
        font-size: 0.9em;
    }
    .stWrite {
        font-size: 0.9em;
    }
    h3 {
        font-size: 1.2em;
    }
    .dataframe {
        font-size: 0.8em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 页面标题和简介
st.markdown(
    """
    <div class='rounded-container'>
        <h2 style="font-size:22px;">Electrochemical Properties Prediction</h2>
        <blockquote>
            1. This web app predicts electrochemical potentials of solid-state electrolytes.<br>
            2. Select the electrolyte system and target below, then enter a valid chemical formula string.
        </blockquote>
    </div>
    """,
    unsafe_allow_html=True,
)

# 选择体系与预测目标
col1, col2 = st.columns(2)
with col1:
    electrolyte_system = st.selectbox(
        "Select Electrolyte System:",
        ("Li-containing compounds", "Na-containing compounds")
    )
with col2:
    prediction_target = st.selectbox(
        "Select Prediction Target:",
        ("Oxidation potential", "Reduction potential")
    )

# 根据选择动态调整模型路径与示例化学式
if electrolyte_system == "Li-containing compounds":
    system_name = "Li-containing compounds"
    example_formula = "e.g., Ba2Li3(PO3)7, Li7La3Zr2O12, Li10GeP2S12"
    if prediction_target == "Oxidation potential":
        model_path = "./ag_20260729_025205"
    else:
        model_path = "./ag-20260729_071442"
else:
    system_name = "Na-containing compounds"
    example_formula = "e.g., Na5Zr2F13, Na6ZnS4, Na2ZnO2"
    if prediction_target == "Oxidation potential":
        model_path = "./ag-20260901_120910"
    else:
        model_path = "./ag-20260901_120826"

# 各模型对应的特征描述符列表
descriptors_dict = {
    "Li-containing compounds": {
        "Oxidation potential": [
            'mean Electronegativity',
            'MagpieData mode Column',
            'MagpieData avg_dev GSbandgap',
            'MagpieData mode SpaceGroupNumber',
            'MagpieData avg_dev SpaceGroupNumber'
        ],
        "Reduction potential": [
            'mean Electronegativity',
            'MagpieData range NdValence',
            'MagpieData avg_dev GSbandgap',
            'MagpieData avg_dev SpaceGroupNumber',
            'MagpieData avg_dev NpValence',
            'MagpieData avg_dev NUnfilled',
            'MagpieData mean NpUnfilled',
            'MagpieData mean GSvolume_pa'
        ]
    },
    "Na-containing compounds": {
        "Oxidation potential": [
            'avg p valence electrons',
            'vpa_cif',
            'minimum Row',
            'range NpValence'
        ],
        "Reduction potential": [
            'avg p valence electrons',
            'avg d valence electrons',
            'mean NValence',
            'range NUnfilled',
            'range NValence',
            'avg_dev CovalentRadius',
            'avg_dev GSvolume_pa'
        ]
    }
}

required_descriptors = descriptors_dict[electrolyte_system][prediction_target]

# FORMULA 输入区域
formula_input = st.text_input("Enter Chemical Formula of the Material:", placeholder=example_formula)

# 提交按钮
submit_button = st.button("Submit and Predict", key="predict_button")

# 缓存模型加载器（加入 require_py_version_match=False 解决 Python 版本不匹配报错）
@st.cache_resource(show_spinner=False, max_entries=4)
def load_predictor(path):
    return TabularPredictor.load(path, require_py_version_match=False)


# 材料特征计算函数（兼容组分特征与缺失的3D结构特征补零）
def calculate_material_features(formula):
    try:
        from matminer.featurizers.composition import (
            ElementProperty, Meredig, Stoichiometry
        )
        from matminer.featurizers.conversions import StrToComposition

        df = pd.DataFrame({'Formula': [formula]})
        stc = StrToComposition()
        df = stc.featurize_dataframe(df, 'Formula', ignore_errors=True)

        if 'composition' not in df.columns or df['composition'].iloc[0] is None:
            return {'Formula': formula}

        features = {'Formula': formula}

        # 提取 Magpie、Meredig、Stoichiometry 组分特征
        ep = ElementProperty.from_preset('magpie')
        df = ep.featurize_dataframe(df, 'composition', ignore_errors=True)

        mer = Meredig()
        df = mer.featurize_dataframe(df, 'composition', ignore_errors=True)

        sto = Stoichiometry()
        df = sto.featurize_dataframe(df, 'composition', ignore_errors=True)

        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            val = df[col].iloc[0]
            features[col] = float(val) if not pd.isna(val) else 0.0

        return features

    except Exception as e:
        st.warning(f"Feature calculation failed: {e}")
        import traceback
        print(traceback.format_exc())
        return {'Formula': formula}


def filter_selected_features(features_dict, selected_descriptors):
    filtered_features = {}
    for feature_name in selected_descriptors:
        if feature_name == 'Temp':
            filtered_features[feature_name] = 298.0
            continue
        # 如果模型特征中包含结构特征（如 vpa_cif 等），在仅输入化学式时自动补 0.0 容错
        if feature_name in features_dict:
            filtered_features[feature_name] = features_dict[feature_name]
        else:
            filtered_features[feature_name] = 0.0
    return filtered_features


if submit_button:
    if not formula_input:
        st.error("Please enter a valid chemical formula.")
    else:
        with st.spinner("Processing material and making predictions..."):
            try:
                features = calculate_material_features(formula_input)
                st.write(f"✅ Total features extracted: {len(features)}")
                
                selected_features = filter_selected_features(features, required_descriptors)
                feature_df = pd.DataFrame([selected_features])
                
                st.subheader("Material Features")
                st.dataframe(feature_df)
            
                if features:
                    input_data = {
                        "Formula": [formula_input],
                        "Temp": [298.0],
                    }
                    
                    numeric_features = {}
                    for feature_name in required_descriptors:
                        if feature_name == 'Temp':
                            numeric_features[feature_name] = [298.0]
                        elif feature_name in features:
                            numeric_features[feature_name] = [features[feature_name]]
                        else:
                            numeric_features[feature_name] = [0.0]  # 对未提取到的结构特征默认补零
                        
                    input_data.update(numeric_features)
                    input_df = pd.DataFrame(input_data)
                
                try:
                    predictor = load_predictor(model_path)
                    
                    essential_models = ['CatBoost',
                                        'ExtraTreesMSE',
                                        'LightGBM',
                                        'KNeighborsDist',
                                        'WeightedEnsemble_L2',
                                        'XGBoost']
                                        
                    predict_df = input_df.copy()
                    predictions_dict = {}
                    
                    for model in essential_models:
                        try:
                            predictions = predictor.predict(predict_df, model=model)
                            predictions_dict[model] = predictions
                        except Exception as model_error:
                            predictions_dict[model] = "Error"

                    st.write(f"Prediction Results for {electrolyte_system} - {prediction_target}:")
                    st.markdown(
                        "**Note:** WeightedEnsemble_L2 is a meta-model combining predictions from other models.")
                    results_df = pd.DataFrame(predictions_dict)
                    st.dataframe(results_df.iloc[:1,:])
                    
                    del predictor
                    gc.collect()

                except Exception as e:
                    st.error(f"Model loading failed! Please ensure the folder **'{model_path}'** exists in GitHub. Details: {str(e)}")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
