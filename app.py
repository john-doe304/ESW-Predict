import streamlit as st
import os
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

# 尝试导入 materials project 客户端
try:
    from mp_api.client import MPRester
    from pymatgen.core import Structure
    from matminer.featurizers.structure import (
        DensityFeatures, GlobalSymmetryFeatures, StructuralHeterogeneity,
        MaximumPackingEfficiency
    )
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False


# 添加 CSS 样式
st.markdown(
    """
    <style>
    .stApp {
        border: 2px solid #808080;
        border-radius: 20px;
        margin: 50px auto;
        max-width: 45%;
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
            2. Select system and target below, enter a chemical formula, and optionally provide your Materials Project API Key for 3D structure features.
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

# 材料化学式与 MP API Key 输入
col3, col4 = st.columns([2, 1])
with col3:
    if electrolyte_system == "Li-containing compounds":
        example_formula = "e.g., Ba2Li3(PO3)7, Li7La3Zr2O12"
        model_path = "./ag_20260729_025205" if prediction_target == "Oxidation potential" else "./ag-20260729_071442"
    else:
        example_formula = "e.g., Na5Zr2F13, Na2ZnO2"
        model_path = "./ag-20260901_120910" if prediction_target == "Oxidation potential" else "./ag-20260901_120826"
        
    formula_input = st.text_input("Enter Chemical Formula:", placeholder=example_formula)

with col4:
    mp_api_key = st.text_input("MP API Key (Optional):", type="password", help="Enter your Materials Project API key to fetch 3D crystal structure features like vpa_cif.")

# 提交按钮
submit_button = st.button("Submit and Predict", key="predict_button")

# 特征描述符字典配置
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

# 缓存模型加载器
@st.cache_resource(show_spinner=False, max_entries=4)
def load_predictor(path):
    return TabularPredictor.load(path, require_py_version_match=False)


# 从 Materials Project 获取 3D 结构的函数
def get_structure_from_mp(formula, api_key):
    if not api_key or not MP_AVAILABLE:
        return None
    try:
        with MPRester(api_key) as mpr:
            docs = mpr.materials.summary.search(formula=formula, fields=["material_id", "structure", "energy_per_atom"])
            if docs:
                docs = sorted(docs, key=lambda x: x.energy_per_atom)
                return docs[0].structure
    except Exception as e:
        st.warning(f"Could not fetch structure from MP: {e}")
    return None


# 综合特征提取函数（支持组分特征 + 3D结构特征）
def calculate_material_features(formula, api_key):
    try:
        from matminer.featurizers.composition import ElementProperty, Meredig, Stoichiometry
        from matminer.featurizers.conversions import StrToComposition

        df = pd.DataFrame({'Formula': [formula]})
        stc = StrToComposition()
        df = stc.featurize_dataframe(df, 'Formula', ignore_errors=True)

        if 'composition' not in df.columns or df['composition'].iloc[0] is None:
            return {'Formula': formula}

        features = {'Formula': formula}

        # 1. 提取组分特征
        ep = ElementProperty.from_preset('magpie')
        df = ep.featurize_dataframe(df, 'composition', ignore_errors=True)

        mer = Meredig()
        df = mer.featurize_dataframe(df, 'composition', ignore_errors=True)

        sto = Stoichiometry()
        df = sto.featurize_dataframe(df, 'composition', ignore_errors=True)

        # 2. 如果提供了 API Key 尝试提取 3D 结构特征
        structure = get_structure_from_mp(formula, api_key)
        if structure is not None:
            df['structure'] = [structure]
            try:
                df_temp = DensityFeatures().featurize_dataframe(df[['structure']].copy(), 'structure', ignore_errors=True)
                df_temp = df_temp.drop(columns=['structure']).rename(columns={'density': 'density_cif', 'vpa': 'vpa_cif'})
                df = pd.concat([df, df_temp], axis=1)

                df = GlobalSymmetryFeatures().featurize_dataframe(df, 'structure', ignore_errors=True)
                df = StructuralHeterogeneity().featurize_dataframe(df, 'structure', ignore_errors=True)
                df = MaximumPackingEfficiency().featurize_dataframe(df, 'structure', ignore_errors=True)
            except Exception as struct_err:
                st.info(f"Note: Structure feature extraction skipped: {struct_err}")

        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            val = df[col].iloc[0]
            features[col] = float(val) if not pd.isna(val) else 0.0

        return features

    except Exception as e:
        st.warning(f"Feature calculation failed: {e}")
        return {'Formula': formula}


def filter_selected_features(features_dict, selected_descriptors):
    filtered_features = {}
    for feature_name in selected_descriptors:
        if feature_name == 'Temp':
            filtered_features[feature_name] = 298.0
            continue
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
                features = calculate_material_features(formula_input, mp_api_key)
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
                            numeric_features[feature_name] = [0.0]
                        
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
