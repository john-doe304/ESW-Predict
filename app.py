# -----------------------------------------------------------
#   Electrochemical Properties Predictor (Clean Version)
#   With MP Crystal 3D Rendering & Direct Prediction Output
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

# rdkit, autogluon, matminer 导入防错保护
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

# 添加 CSS 样式 (同步完美外框布局，修复顶部线切断问题)
st.markdown(
    """
    <style>
    .stApp {
        border: 2px solid #808080;
        border-radius: 20px;
        margin: 40px auto;
        max-width: 40%;
        background-color: #f9f9f9f9;
        padding: 25px;
        box-sizing: border-box;
    }
    .rounded-container {
        margin-top: 10px;
        margin-bottom: 25px;
    }
    .rounded-container h2 {
        margin-top: 0px; 
        text-align: center;
        background-color: #e0e0e0e0;
        padding: 12px;
        border-radius: 10px;
    }
    .rounded-container blockquote {
        text-align: left;
        margin: 20px auto;
        background-color: #f0f0f0;
        padding: 12px;
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
        <h2 style="font-size:24px;">Electrochemical Properties Prediction</h2>
        <blockquote>
            1. This web app predicts electrochemical potentials of solid-state electrolytes.<br>
            2. Select the electrolyte system and target below, then enter a valid chemical formula string.
        </blockquote>
    </div>
    """,
    unsafe_allow_html=True,
)

# 选择体系与预测目标 (直接并排)
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
    example_formula = "e.g., Ba2Li3(PO3)7, Li7La3Zr2O12, Li10GeP2S12"
    if prediction_target == "Oxidation potential":
        model_path = "./ag_20260729_025205"
    else:
        model_path = "./ag-20260729_071442"
else:
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

# 输入区域（化学式 + 提交按钮 + MP Key 选项）- UI 布局同步
input_col1, input_col2 = st.columns([2, 1])
with input_col1:
    formula_input = st.text_input("Enter Chemical Formula:", placeholder=example_formula)
    submit_button = st.button("Submit and Predict")
with input_col2:
    MP_API_KEY_DEFAULT = "Gd6Y2d9mtjquU8imu8n4GdIiwCvUtZqN"
    mp_key_input = st.text_input("MP API key:", type="password", value=MP_API_KEY_DEFAULT)
    use_placeholder_checkbox = st.checkbox("Use placeholder structure", value=False)


# ------------------------------- 模型加载缓存 -------------------------------
@st.cache_resource(show_spinner=False, max_entries=4)
def load_predictor(path
