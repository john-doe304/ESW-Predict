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

# 【新增】选择预测目标：氧化电位 还是 还原电位
prediction_target = st.selectbox(
    "Select Prediction Target:",
    ("Oxidation Potential", "Reduction Potential")
)

# 根据选择动态改变标题和简介
if prediction_target == "Oxidation Potential":
    app_title = "Predict Oxidation Potential of Solid Electrolytes"
    app_desc = "This web app predicts oxidation potential of solid electrolytes based on material composition features."
    model_path = "./ag-oxidation-potential-model"  # 对应您的氧化电位模型文件夹
else:
    app_title = "Predict Reduction Potential of Solid Electrolytes"
    app_desc = "This web app predicts reduction potential of solid electrolytes based on material composition features."
    model_path = "./ag-reduction-potential-model"  # 对应您的还原电位模型文件夹

st.markdown(
    f"""
    <div class='rounded-container'>
        <h2 style="font-size:24px;">{app_title}</h2>
        <blockquote>
            1. {app_desc}<br>
            2. Enter a valid chemical formula string below to get the predicted result.
        </blockquote>
    </div>
    """,
    unsafe_allow_html=True,
)

# FORMULA 输入区域
formula_input = st.text_input("Enter Chemical Formula of the Material:", placeholder="e.g., Li7La3Zr2O12, Li10GeP2S12, Li3YCl6")

# 提交按钮
submit_button = st.button("Submit and Predict", key="predict_button")

# 特征列表（请根据您实际训练模型时所用的特征进行调整）
required_descriptors = [
    'MagpieData mean CovalentRadius',
    'MagpieData avg_dev SpaceGroupNumber',
    '0-norm',
    'MagpieData mean MeltingT',
    'MagpieData avg_dev Column',
    'MagpieData mean NValence'
]

# 动态加载对应模型的加载器
@st.cache_resource(show_spinner=False, max_entries=2)
def load_predictor(path):
    """缓存模型加载，避免重复加载导致内存溢出"""
    return TabularPredictor.load(path)


# 材料特征计算函数
def calculate_material_features(formula):
    """计算材料的组成特征"""
    try:
        from matminer.featurizers.composition import (
            ElementProperty, Meredig, Stoichiometry, IonProperty
        )
        from matminer.featurizers.conversions import StrToComposition, CompositionToOxidComposition

        df = pd.DataFrame({'Formula': [formula]})
        stc = StrToComposition()
        df = stc.featurize_dataframe(df, 'Formula', ignore_errors=True)

        if 'composition' not in df.columns or df['composition'].iloc[0] is None:
            return {'Formula': formula}

        features = {'Formula': formula}

        ep = ElementProperty.from_preset('magpie')
        df = ep.featurize_dataframe(df, 'composition', ignore_errors=True)

        mer = Meredig()
        df = mer.featurize_dataframe(df, 'composition', ignore_errors=True)

        sto = Stoichiometry()
        df = sto.featurize_dataframe(df, 'composition', ignore_errors=True)

        cto = CompositionToOxidComposition()
        df = cto.featurize_dataframe(df, 'composition_oxid', ignore_errors=True)
        ion = IonProperty()
        df = ion.featurize_dataframe(df, 'composition_oxid', ignore_errors=True)

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
    """只显示选定的特征"""
    filtered_features = {}
    for feature_name in selected_descriptors:
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
                    }
                    
                    numeric_features = {}
                    for feature_name in required_descriptors:
                        if feature_name in features:
                            numeric_features[feature_name] = [features[feature_name]]
                        else:
                            numeric_features[feature_name] = [0.0]
                        
                    input_data.update(numeric_features)
                    input_df = pd.DataFrame(input_data)
                
                try:
                    # 动态加载当前选择的模型
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

                    st.write(f"Prediction Results for {prediction_target} (Essential Models):")
                    st.markdown(
                        "**Note:** WeightedEnsemble_L2 is a meta-model combining predictions from other models.")
                    results_df = pd.DataFrame(predictions_dict)
                    st.dataframe(results_df.iloc[:1,:])
                    
                    del predictor
                    gc.collect()

                except Exception as e:
                    st.error(f"Model loading failed (Please check if '{model_path}' exists in the repository): {str(e)}")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
