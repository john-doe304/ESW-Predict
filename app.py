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


# 添加 CSS 样式（还原第一张图的完美卡片边距与阴影效果）
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
    /* 减小指标卡片的字体大小 */
    .stMetric {
        font-size: 0.9em;
    }
    /* 减小特征提取成功信息的字体大小 */
    .stWrite {
        font-size: 0.9em;
    }
    /* 减小子标题的字体大小 */
    h3 {
        font-size: 1.2em;
    }
    /* 减小数据框的字体大小 */
    .dataframe {
        font-size: 0.8em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 1. 恢复第一张图的精美标题和说明样式
st.markdown(
    """
    <div class='rounded-container'>
        <h2 style="font-size:22px;">Electrochemical Properties Prediction</h2>
        <blockquote>
            1. This web app predicts the electrochemical properties of solid electrolytes based on material composition features.<br>
            2. Select the system and target below, then enter a valid chemical formula string.
        </blockquote>
    </div>
    """,
    unsafe_allow_html=True,
)

# 2. 将“选择体系”和“预测目标”放置在输入化学式上方（左右两列布局）
col1, col2 = st.columns(2)
with col1:
    electrolyte_system = st.selectbox(
        "Select Electrolyte System:",
        ("Lithium (Li-ion)", "Sodium (Na-ion)")
    )
with col2:
    prediction_target = st.selectbox(
        "Select Prediction Target:",
        ("Oxidation Potential", "Reduction Potential")
    )

# 根据选择动态调整提示文本、示例与模型路径
if electrolyte_system == "Lithium (Li-ion)":
    system_name = "Li-ion Solid Electrolytes"
    example_formula = "e.g., Ba2Li3(PO3)7, Li7La3Zr2O12, Li10GeP2S12"
    if prediction_target == "Oxidation Potential":
        model_path = "./ag-li-oxidation-model"
    else:
        model_path = "./ag-li-reduction-model"
else:
    system_name = "Na-ion Solid Electrolytes"
    example_formula = "e.g., Na5Zr2F13, Na6ZnS4, Na2ZnO2"
    if prediction_target == "Oxidation Potential":
        model_path = "./ag-na-oxidation-model"
    else:
        model_path = "./ag-na-reduction-model"

# 3. FORMULA 输入区域
formula_input = st.text_input("Enter Chemical Formula of the Material:", placeholder=example_formula)

# 4. 提交按钮
submit_button = st.button("Submit and Predict", key="predict_button")

# 指定的描述符列表
required_descriptors = [
    'MagpieData mean CovalentRadius',
    'Temp',
    'MagpieData avg_dev SpaceGroupNumber',
    '0-norm',
    'MagpieData mean MeltingT',
    'MagpieData avg_dev Column',
    'MagpieData mean NValence'
]

# 缓存模型加载器
@st.cache_resource(show_spinner=False, max_entries=4)
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
    """只选定的特征"""
    filtered_features = {}
    for feature_name in selected_descriptors:
        if feature_name == 'Temp':
            filtered_features[feature_name] = 298.0  # 默认温度
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
                    st.error(f"Model loading failed (Please check if folder '{model_path}' exists in GitHub): {str(e)}")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
