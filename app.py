# -----------------------------------------------------------
#   Electrochemical Properties Predictor (Final Integrated)
#   With Materials Project Crystal Rendering + Multi-System Support
# -----------------------------------------------------------

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import py3Dmol
import traceback
import gc
import re
import tempfile
import os
import random
from io import BytesIO
import base64

# rdkit, mordred, autogluon, matminer 导入防错保护
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Draw, AllChem
    from rdkit.Chem.Draw import MolDraw2DSVG
    from rdkit.ML.Descriptors import MoleculeDescriptors
except Exception:
    rdkit = None

try:
    from autogluon.tabular import TabularPredictor
except Exception:
    TabularPredictor = None

try:
    from matminer.featurizers.composition import ElementProperty, Meredig, Stoichiometry, IonProperty
    from matminer.featurizers.conversions import StrToComposition, CompositionToOxidComposition
except Exception:
    ElementProperty = Meredig = Stoichiometry = IonProperty = StrToComposition = CompositionToOxidComposition = None

try:
    from mp_api.client import MPRester
except Exception:
    MPRester = None

try:
    from pymatgen.core import Structure, Lattice
    from pymatgen.io.cif import CifWriter
except Exception:
    Structure = Lattice = CifWriter = None

st.set_page_config(layout="wide", page_title="Electrochemical Properties Predictor")

# ----------------------------------
# MP 官方配色字典
# ----------------------------------
MP_COLORS = {
    "H": "#FFFFFF", "Li": "#CC80FF", "Be": "#C2FF00", "B": "#FFB5B5", "C": "#909090",
    "N": "#3050F8", "O": "#FF0D0D", "F": "#90E050", "Na": "#AB5CF2", "Mg": "#8AFF00",
    "Al": "#BFA6A6", "Si": "#F0C8A0", "P": "#FF8000", "S": "#FFFF30", "Cl": "#1FF01F",
    "K": "#8F40D4", "Ca": "#FFD478", "Sc": "#E6E6E6", "Ti": "#BFC2C7", "V": "#A6A6AB",
    "Cr": "#8A99C7", "Mn": "#9C7AC7", "Fe": "#E06633", "Co": "#F090A0", "Ni": "#50D050",
    "Cu": "#C88033", "Zn": "#7D80B0", "Ga": "#C28F8F", "Ge": "#4C4CFF", "As": "#BD80E3",
    "Se": "#FFA100", "Br": "#A62929", "Kr": "#5CB8D1", "Rb": "#702EB0", "Sr": "#00FF00",
    "Y": "#94FFFF", "Zr": "#94E0E0", "Nb": "#73C2C9", "Mo": "#54B5B5", "Ru": "#248F8F",
    "Rh": "#0A7D8C", "Pd": "#006985", "Ag": "#C0C0C0", "Cd": "#FFD98F", "In": "#A67573",
    "Sn": "#668080", "Sb": "#9E63B5", "Te": "#D47A00", "I": "#940094", "Xe": "#4DC4FF",
    "Cs": "#57178F", "Ba": "#00C900", "La": "#70D4FF", "Ce": "#FFFFC7", "Pr": "#D9FFC7",
    "Nd": "#C7FFC7", "Pm": "#A3FFC7", "Sm": "#8FFFC7", "Eu": "#61FFC7", "Gd": "#45FFC7",
    "Tb": "#30FFC7", "Dy": "#1FFFC7", "Ho": "#00FF9C", "Er": "#00E675", "Tm": "#00D452",
    "Yb": "#00BF69", "Lu": "#00AB6B", "Hf": "#4DC2FF", "Ta": "#4DA6FF", "W": "#2194D6",
    "Re": "#267DAB", "Os": "#266696", "Ir": "#175487", "Pt": "#D0D0E0", "Au": "#FFD123",
    "Hg": "#B8B8D0", "Tl": "#A6544D", "Pb": "#575961", "Bi": "#9E4FB5", "Po": "#AB5C00",
    "At": "#754F45", "Rn": "#428296", "Fr": "#420066", "Ra": "#00C900", "Ac": "#70ABFA",
    "Th": "#00BAFF", "Pa": "#00A1FF", "U": "#008FFF", "Np": "#0080FF", "Pu": "#006BFF"
}

# 添加 CSS 样式
st.markdown(
    """
    <style>
    .stApp {
        border: 2px solid #808080;
        border-radius: 20px;
        margin: 30px auto;
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
        margin: 15px auto;
        background-color: #f0f0f0;
        padding: 10px;
        font-size: 1.05em;
        border-radius: 10px;
    }
    .stMetric {
        font-size: 0.9em;
    }
    .stWrite {
        font-size: 0.9em;
    }
    h3 {
        font-size: 1.1em;
        margin-bottom: 0.5em;
    }
    .dataframe {
        font-size: 0.8em;
    }
    div[data-testid="column"] {
        padding: 0px !important;
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
            1. This web app predicts electrochemical potentials of solid-state electrolytes[cite: 13].<br>
            2. Select the electrolyte system and target below, then enter a valid chemical formula string.
        </blockquote>
    </div>
    """,
    unsafe_allow_html=True,
)

# 选择体系与预测目标
col_sys, col_tar = st.columns(2)
with col_sys:
    electrolyte_system = st.selectbox(
        "Select Electrolyte System:",
        ("Li-containing compounds", "Na-containing compounds")
    )
with col_tar:
    prediction_target = st.selectbox(
        "Select Prediction Target:",
        ("Oxidation potential", "Reduction potential")
    )

# 根据选择动态调整模型路径、示例化学式与特征描述符
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

# 描述符列表配置
descriptors_dict = {
    "Li-containing compounds": {
        "Oxidation potential": [
            'mean Electronegativity', 'MagpieData mode Column',
            'MagpieData avg_dev GSbandgap', 'MagpieData mode SpaceGroupNumber',
            'MagpieData avg_dev SpaceGroupNumber'
        ],
        "Reduction potential": [
            'mean Electronegativity', 'MagpieData range NdValence',
            'MagpieData avg_dev GSbandgap', 'MagpieData avg_dev SpaceGroupNumber',
            'MagpieData avg_dev NpValence', 'MagpieData avg_dev NUnfilled',
            'MagpieData mean NpUnfilled', 'MagpieData mean GSvolume_pa'
        ]
    },
    "Na-containing compounds": {
        "Oxidation potential": [
            'avg p valence electrons', 'vpa_cif', 'minimum Row', 'range NpValence'
        ],
        "Reduction potential": [
            'avg p valence electrons', 'avg d valence electrons', 'mean NValence',
            'range NUnfilled', 'range NValence', 'avg_dev CovalentRadius', 'avg_dev GSvolume_pa'
        ]
    }
}

required_descriptors = descriptors_dict[electrolyte_system][prediction_target]

# 输入区域（化学式 + 温度 + MP Key 选项）
input_col1, input_col2 = st.columns([2, 1])
with input_col1:
    formula_input = st.text_input("Enter Chemical Formula:", placeholder=example_formula)
    temperature = st.number_input("Temperature (K):", min_value=200, max_value=1000, value=298, step=1)
    submit_button = st.button("Submit and Predict")
with input_col2:
    MP_API_KEY_DEFAULT = "Gd6Y2d9mtjquU8imu8n4GdIiwCvUtZqN"
    mp_key_input = st.text_input("MP API key:", type="password", value=MP_API_KEY_DEFAULT)
    use_placeholder_checkbox = st.checkbox("Use placeholder structure", value=False)


# ------------------------------- 模型加载缓存 -------------------------------
@st.cache_resource(show_spinner=False, max_entries=4)
def load_predictor(path):
    return TabularPredictor.load(path, require_py_version_match=False)


# ------------------------------- MP 结构加载 -------------------------------
def load_structure_from_mp(formula, api_key):
    if MPRester is None:
        return None, "mp-api not installed"
    try:
        with MPRester(api_key) as mpr:
            results = mpr.summary.search(formula=formula, fields=["structure"])
            if not results:
                return None, "No MP entry found"
            doc = results[0]
            try:
                struct = doc.structure.get_primitive_structure()
            except Exception:
                struct = doc.structure
            return struct, "Successfully loaded from MP"
    except Exception as e:
        return None, f"MP error: {e}"


# ------------------------------- 占位晶胞生成 -------------------------------
def generate_placeholder_structure(formula):
    elems = re.findall(r"[A-Z][a-z]?", formula or "")
    elems = list(dict.fromkeys(elems))
    if len(elems) == 0:
        elems = ["Li", "O"]
    coords = []
    n = len(elems)
    for i in range(n):
        coords.append([0.1 + 0.8*((i+1)/(n+1)), 0.1 + 0.6*random.random(), 0.1 + 0.6*random.random()])
    if Lattice is None or Structure is None:
        return None
    lattice = Lattice.cubic(10.0)
    struct = Structure(lattice, elems, coords)
    return struct


# ------------------------------- 结构转 CIF 字符串 -------------------------------
def structure_to_cif_string(structure):
    if CifWriter is None:
        return None
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as tmp:
            fname = tmp.name
        try:
            CifWriter(structure).write_file(fname)
        except Exception:
            structure.to(filename=fname)
        with open(fname, "r", encoding="utf-8") as f:
            cif_str = f.read()
        return cif_str
    finally:
        try:
            if tmp is not None:
                os.unlink(tmp.name)
        except Exception:
            pass


# ------------------------------- py3Dmol 结构渲染 -------------------------------
def render_structure_with_legend(structure, width=420, height=240):
    cif_str = structure_to_cif_string(structure)
    if not cif_str:
        return None

    view = py3Dmol.view(width=280, height=height)
    view.addModel(cif_str, "cif")

    for i, site in enumerate(structure.sites):
        el = str(site.specie)
        color = MP_COLORS.get(el, "#9E9E9E")
        view.setStyle({"index": i}, {
            "sphere": {"radius": 0.45, "color": color},
            "stick": {"radius": 0.18, "color": color}
        })

    view.addUnitCell()
    view.zoomTo()
    structure_html = view._make_html()

    elements = sorted({str(s.specie) for s in structure.sites})
    legend_items = ""
    for el in elements:
        c = MP_COLORS.get(el, "#9E9E9E")
        legend_items += f"""
        <div style="display:flex;align-items:center;margin-bottom:4px;">
            <div style="width:12px;height:12px;background:{c};border:1px solid #333;border-radius:3px;margin-right:6px;"></div>
            <span style="font-size:12px;color:#222;">{el}</span>
        </div>
        """

    legend_html = f"""
    <div style="background:#f5f5f5;border:1px solid #ccc;border-radius:8px;padding:8px;width:100px;">
        <div style="text-align:center;font-weight:600;margin-bottom:6px;font-size:12px;">Colors</div>
        {legend_items}
    </div>
    """

    final_html = f"""
    <div style="display:flex;align-items:flex-start;gap:10px;width:{width}px;">
        <div>{structure_html}</div>
        {legend_html}
    </div>
    """
    return final_html


# ------------------------------- 特征计算函数 -------------------------------
def calculate_material_features(formula):
    try:
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

        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            val = df[col].iloc[0]
            features[col] = float(val) if not pd.isna(val) else 0.0

        return features
    except Exception as e:
        st.warning(f"Feature calculation failed: {e}")
        return {'Formula': formula}


def filter_selected_features(features_dict, selected_descriptors, temperature):
    filtered_features = {}
    filtered_features['Temp'] = float(temperature)
    for feature_name in selected_descriptors:
        if feature_name == 'Temp':
            continue
        if feature_name in features_dict:
            filtered_features[feature_name] = features_dict[feature_name]
        else:
            filtered_features[feature_name] = 0.0
    return filtered_features


# ------------------------------- 提交预测主逻辑 -------------------------------
if submit_button:
    if not formula_input:
        st.error("Please enter a valid chemical formula.")
        st.stop()

    with st.spinner("Processing crystal structure and predicting..."):
        # 1. 结构加载与 3D 渲染
        structure = None
        mp_msg = ""
        if (mp_key_input and not use_placeholder_checkbox) and (MPRester is not None):
            try:
                struct, info = load_structure_from_mp(formula_input, mp_key_input)
                if struct:
                    structure = struct
                    mp_msg = f"Loaded from MP: {info}"
                else:
                    mp_msg = f"MP lookup failed: {info}"
            except Exception as e:
                mp_msg = f"MP exception: {e}"
        else:
            mp_msg = "Placeholder structure selected."

        if structure is None:
            structure = generate_placeholder_structure(formula_input)

        if structure:
            st.subheader("Crystal Structure Preview (Unit Cell)")
            html = render_structure_with_legend(structure)
            if html:
                components.html(html, height=260, scrolling=False)

        # 2. 特征提取与模型预测
        features = calculate_material_features(formula_input)
        selected_features = filter_selected_features(features, required_descriptors, temperature)
        feature_df = pd.DataFrame([selected_features])

        st.subheader("Material Features")
        st.dataframe(feature_df)

        input_data = {"Formula": [formula_input], "Temp": [temperature]}
        for feature_name in required_descriptors:
            if feature_name == 'Temp':
                input_data[feature_name] = [temperature]
            elif feature_name in features:
                input_data[feature_name] = [features[feature_name]]
            else:
                input_data[feature_name] = [0.0]

        input_df = pd.DataFrame(input_data)

        try:
            predictor = load_predictor(model_path)
            essential_models = ['CatBoost', 'ExtraTreesMSE', 'LightGBM', 'KNeighborsDist', 'WeightedEnsemble_L2', 'XGBoost']
            predictions_dict = {}

            for model in essential_models:
                try:
                    predictions = predictor.predict(input_df, model=model)
                    predictions_dict[model] = predictions
                except Exception:
                    predictions_dict[model] = "Error"

            st.subheader(f"Prediction Results for {electrolyte_system} - {prediction_target}:")
            st.markdown("**Note:** WeightedEnsemble_L2 is a meta-model combining predictions from other models.")
            results_df = pd.DataFrame(predictions_dict)
            st.dataframe(results_df.iloc[:1, :])

            del predictor
            gc.collect()

        except Exception as e:
            st.error(f"Model loading failed! Please ensure folder **'{model_path}'** exists in GitHub. Details: {str(e)}")
