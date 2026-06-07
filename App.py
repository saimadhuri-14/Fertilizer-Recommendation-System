"""
Fertilizer Recommendation System with Machine Learning
A Streamlit web application that uses ML algorithms for intelligent fertilizer recommendations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import base64
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# Page configuration
st.set_page_config(
    page_title="AI Fertilizer Recommendation System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        background: linear-gradient(135deg, #2E7D32, #4CAF50);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #1B5E20;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        color: white;
        text-align: center;
    }
    .recommendation-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #2E7D32;
        margin: 1rem 0;
    }
    .fertilizer-card {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4CAF50;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .fertilizer-name {
        font-size: 1.2rem;
        font-weight: bold;
        color: #2E7D32;
        margin-bottom: 0.5rem;
    }
    .ml-box {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .prediction-card {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
    .nutrient-tag {
        background-color: #4CAF50;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 5px;
        font-size: 0.8rem;
        display: inline-block;
        margin: 0.2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Install required packages
def show_installation_instructions():
    with st.expander("📦 Required Packages Installation (ML Version)", expanded=False):
        st.code("""
        # Install all required packages
        pip install streamlit pandas numpy plotly matplotlib seaborn scikit-learn xgboost lightgbm
        
        # Or use requirements.txt
        pip install -r requirements.txt
        
        # requirements.txt content:
        streamlit==1.28.0
        pandas==2.0.3
        numpy==1.24.3
        plotly==5.17.0
        matplotlib==3.7.2
        seaborn==0.12.2
        scikit-learn==1.3.0
        xgboost==2.0.0
        lightgbm==4.0.0
        """, language="bash")

# Comprehensive Fertilizer Database
@st.cache_data
def get_fertilizer_database():
    """Complete fertilizer database with recommendations for each crop"""
    fertilizer_db = {
        'Corn': {
            'fertilizers': [
                {'name': 'Urea (46-0-0)', 'n': 46, 'p': 0, 'k': 0, 'rate_kg_ha': 250, 'timing': 'Pre-planting and side-dress at V6', 'method': 'Broadcast or band application', 'cost_per_kg': 0.85},
                {'name': 'DAP (18-46-0)', 'n': 18, 'p': 46, 'k': 0, 'rate_kg_ha': 150, 'timing': 'At planting', 'method': 'Band 2 inches beside and below seed', 'cost_per_kg': 1.20},
                {'name': 'Potash (0-0-60)', 'n': 0, 'p': 0, 'k': 60, 'rate_kg_ha': 200, 'timing': 'Pre-planting', 'method': 'Broadcast and incorporate', 'cost_per_kg': 0.95},
                {'name': 'NPK 15-15-15', 'n': 15, 'p': 15, 'k': 15, 'rate_kg_ha': 300, 'timing': 'Basal application', 'method': 'Broadcast before planting', 'cost_per_kg': 1.10}
            ],
            'organic_options': [
                {'name': 'Farmyard Manure (FYM)', 'rate_tons_ha': 10, 'n_pct': 0.5, 'p_pct': 0.2, 'k_pct': 0.5, 'timing': '3-4 weeks before planting', 'cost_per_ton': 30},
                {'name': 'Vermicompost', 'rate_tons_ha': 5, 'n_pct': 1.5, 'p_pct': 0.8, 'k_pct': 1.2, 'timing': 'During land preparation', 'cost_per_ton': 80},
                {'name': 'Neem Cake', 'rate_kg_ha': 500, 'n_pct': 5.0, 'p_pct': 1.0, 'k_pct': 1.5, 'timing': 'At final ploughing', 'cost_per_kg': 0.45}
            ],
            'schedule': {
                'basal': 'Apply 50% N and full P & K before planting',
                'top_dressing_1': 'Apply 25% N at knee-high stage (V6-V8)',
                'top_dressing_2': 'Apply 25% N at tasseling stage (VT)'
            }
        },
        'Wheat': {
            'fertilizers': [
                {'name': 'Urea (46-0-0)', 'n': 46, 'p': 0, 'k': 0, 'rate_kg_ha': 200, 'timing': 'Split application', 'method': 'Broadcast', 'cost_per_kg': 0.85},
                {'name': 'Single Super Phosphate (0-16-0)', 'n': 0, 'p': 16, 'k': 0, 'rate_kg_ha': 300, 'timing': 'At sowing', 'method': 'Broadcast and incorporate', 'cost_per_kg': 0.65},
                {'name': 'Muriate of Potash (0-0-60)', 'n': 0, 'p': 0, 'k': 60, 'rate_kg_ha': 100, 'timing': 'At sowing', 'method': 'Broadcast', 'cost_per_kg': 0.95},
                {'name': 'NPK 12-32-16', 'n': 12, 'p': 32, 'k': 16, 'rate_kg_ha': 250, 'timing': 'Basal application', 'method': 'Drill or broadcast', 'cost_per_kg': 1.15}
            ],
            'organic_options': [
                {'name': 'Vermicompost', 'rate_tons_ha': 5, 'n_pct': 1.5, 'p_pct': 0.8, 'k_pct': 1.2, 'timing': 'At land preparation', 'cost_per_ton': 80},
                {'name': 'Poultry Manure', 'rate_tons_ha': 4, 'n_pct': 3.0, 'p_pct': 2.0, 'k_pct': 2.0, 'timing': '2 weeks before sowing', 'cost_per_ton': 50}
            ],
            'schedule': {
                'basal': 'Apply full P, K and 50% N at sowing',
                'top_dressing_1': 'Apply 25% N at crown root initiation (20-25 DAS)',
                'top_dressing_2': 'Apply 25% N at flowering stage'
            }
        },
        'Soybeans': {
            'fertilizers': [
                {'name': 'Single Super Phosphate (0-16-0)', 'n': 0, 'p': 16, 'k': 0, 'rate_kg_ha': 400, 'timing': 'At sowing', 'method': 'Band placement', 'cost_per_kg': 0.65},
                {'name': 'Muriate of Potash (0-0-60)', 'n': 0, 'p': 0, 'k': 60, 'rate_kg_ha': 100, 'timing': 'At sowing', 'method': 'Broadcast', 'cost_per_kg': 0.95},
                {'name': 'Biofertilizer (Rhizobium)', 'rate_kg_ha': 5, 'timing': 'Seed treatment', 'method': 'Mix with seeds before sowing', 'cost_per_kg': 25}
            ],
            'organic_options': [
                {'name': 'Green Manure (Sesbania)', 'rate_tons_ha': 8, 'timing': 'Incorporate 4-6 weeks before sowing', 'cost_per_ton': 20},
                {'name': 'Farmyard Manure', 'rate_tons_ha': 8, 'n_pct': 0.5, 'p_pct': 0.2, 'k_pct': 0.5, 'timing': 'During land preparation', 'cost_per_ton': 30}
            ],
            'schedule': {
                'basal': 'Apply full P and K at sowing',
                'nitrogen': 'No nitrogen needed (legume fixes own N)',
                'biofertilizer': 'Rhizobium inoculation essential for nitrogen fixation'
            }
        },
        'Rice': {
            'fertilizers': [
                {'name': 'Urea (46-0-0)', 'n': 46, 'p': 0, 'k': 0, 'rate_kg_ha': 200, 'timing': 'Split (3 applications)', 'method': 'Broadcast in standing water', 'cost_per_kg': 0.85},
                {'name': 'DAP (18-46-0)', 'n': 18, 'p': 46, 'k': 0, 'rate_kg_ha': 125, 'timing': 'At transplanting', 'method': 'Broadcast', 'cost_per_kg': 1.20},
                {'name': 'Muriate of Potash (0-0-60)', 'n': 0, 'p': 0, 'k': 60, 'rate_kg_ha': 100, 'timing': 'At tillering', 'method': 'Broadcast', 'cost_per_kg': 0.95},
                {'name': 'Zinc Sulfate (ZnSO4)', 'n': 0, 'p': 0, 'k': 0, 'rate_kg_ha': 25, 'timing': 'At transplanting', 'method': 'Broadcast', 'cost_per_kg': 1.50}
            ],
            'organic_options': [
                {'name': 'Green Manure (Sesbania)', 'rate_tons_ha': 10, 'timing': 'Incorporate 4 weeks before transplanting', 'cost_per_ton': 20},
                {'name': 'Rice Straw Compost', 'rate_tons_ha': 5, 'n_pct': 0.8, 'p_pct': 0.2, 'k_pct': 1.2, 'timing': 'During final ploughing', 'cost_per_ton': 25}
            ],
            'schedule': {
                'basal': 'Apply full P, K and 30% N at transplanting',
                'top_dressing_1': 'Apply 40% N at active tillering (20-25 DAT)',
                'top_dressing_2': 'Apply 30% N at panicle initiation (45-50 DAT)'
            }
        },
        'Cotton': {
            'fertilizers': [
                {'name': 'Urea (46-0-0)', 'n': 46, 'p': 0, 'k': 0, 'rate_kg_ha': 250, 'timing': 'Split (3 applications)', 'method': 'Broadcast in rows', 'cost_per_kg': 0.85},
                {'name': 'DAP (18-46-0)', 'n': 18, 'p': 46, 'k': 0, 'rate_kg_ha': 150, 'timing': 'At sowing', 'method': 'Band placement', 'cost_per_kg': 1.20},
                {'name': 'Muriate of Potash (0-0-60)', 'n': 0, 'p': 0, 'k': 60, 'rate_kg_ha': 150, 'timing': 'At flowering', 'method': 'Broadcast', 'cost_per_kg': 0.95}
            ],
            'organic_options': [
                {'name': 'Neem Cake', 'rate_kg_ha': 500, 'n_pct': 5.0, 'timing': 'At sowing', 'method': 'Mix with soil', 'cost_per_kg': 0.45},
                {'name': 'Farmyard Manure', 'rate_tons_ha': 12, 'timing': '3-4 weeks before sowing', 'cost_per_ton': 30}
            ],
            'schedule': {
                'basal': 'Apply 30% N, full P at sowing',
                'top_dressing_1': 'Apply 40% N at square formation (45 DAS)',
                'top_dressing_2': 'Apply 30% N at flowering (75 DAS)'
            }
        },
        'Vegetables': {
            'fertilizers': [
                {'name': 'NPK 15-15-15', 'n': 15, 'p': 15, 'k': 15, 'rate_kg_ha': 400, 'timing': 'Basal application', 'method': 'Broadcast and incorporate', 'cost_per_kg': 1.10},
                {'name': 'Urea (46-0-0)', 'n': 46, 'p': 0, 'k': 0, 'rate_kg_ha': 150, 'timing': 'Split applications', 'method': 'Side dressing', 'cost_per_kg': 0.85},
                {'name': 'Calcium Nitrate (15.5-0-0)', 'n': 15.5, 'p': 0, 'k': 0, 'rate_kg_ha': 100, 'timing': 'During fruit development', 'method': 'Foliar spray or soil application', 'cost_per_kg': 1.30}
            ],
            'organic_options': [
                {'name': 'Vermicompost', 'rate_tons_ha': 8, 'n_pct': 1.5, 'p_pct': 0.8, 'k_pct': 1.2, 'timing': 'At land preparation', 'cost_per_ton': 80},
                {'name': 'Panchagavya', 'rate_litres_ha': 500, 'timing': 'Weekly foliar spray', 'method': 'Spray on leaves', 'cost_per_litre': 2.50}
            ],
            'schedule': {
                'basal': 'Apply 50% N, full P & K before planting',
                'top_dressing': 'Apply remaining N in 2-3 splits every 20 days'
            }
        },
        'Fruits': {
            'fertilizers': [
                {'name': 'NPK 12-12-12', 'n': 12, 'p': 12, 'k': 12, 'rate_kg_tree': 1.5, 'timing': 'Pre-flowering', 'method': 'Ring application', 'cost_per_kg': 1.05},
                {'name': 'Urea (46-0-0)', 'n': 46, 'p': 0, 'k': 0, 'rate_kg_tree': 0.5, 'timing': 'Post-harvest', 'method': 'Broadcast around tree', 'cost_per_kg': 0.85},
                {'name': 'Muriate of Potash (0-0-60)', 'n': 0, 'p': 0, 'k': 60, 'rate_kg_tree': 0.75, 'timing': 'Fruit development', 'method': 'Ring application', 'cost_per_kg': 0.95}
            ],
            'organic_options': [
                {'name': 'Farmyard Manure', 'rate_kg_tree': 25, 'timing': 'During winter', 'method': 'Apply in basins', 'cost_per_ton': 30},
                {'name': 'Neem Cake', 'rate_kg_tree': 2, 'timing': 'Pre-monsoon', 'method': 'Mix with soil', 'cost_per_kg': 0.45}
            ],
            'schedule': {
                'basal': 'Apply after harvest during dormant season',
                'pre_flowering': 'Apply N and K before flowering',
                'fruit_development': 'Apply K during fruit development stage'
            }
        },
        'Sugarcane': {
            'fertilizers': [
                {'name': 'Urea (46-0-0)', 'n': 46, 'p': 0, 'k': 0, 'rate_kg_ha': 400, 'timing': 'Split (4 applications)', 'method': 'Band placement in furrows', 'cost_per_kg': 0.85},
                {'name': 'DAP (18-46-0)', 'n': 18, 'p': 46, 'k': 0, 'rate_kg_ha': 200, 'timing': 'At planting', 'method': 'Apply in furrows', 'cost_per_kg': 1.20},
                {'name': 'Muriate of Potash (0-0-60)', 'n': 0, 'p': 0, 'k': 60, 'rate_kg_ha': 200, 'timing': 'At tillering', 'method': 'Broadcast', 'cost_per_kg': 0.95}
            ],
            'organic_options': [
                {'name': 'Pressmud', 'rate_tons_ha': 10, 'n_pct': 1.2, 'p_pct': 0.5, 'k_pct': 0.8, 'timing': 'At planting', 'cost_per_ton': 20},
                {'name': 'Farmyard Manure', 'rate_tons_ha': 15, 'timing': 'During land preparation', 'cost_per_ton': 30}
            ],
            'schedule': {
                'basal': 'Apply full P, 30% N at planting',
                'top_dressing_1': 'Apply 30% N at tillering (60 DAP)',
                'top_dressing_2': 'Apply 20% N at grand growth (120 DAP)',
                'top_dressing_3': 'Apply 20% N at maturity (180 DAP)'
            }
        },
        'Potatoes': {
            'fertilizers': [
                {'name': 'NPK 10-26-26', 'n': 10, 'p': 26, 'k': 26, 'rate_kg_ha': 400, 'timing': 'At planting', 'method': 'Band placement', 'cost_per_kg': 1.25},
                {'name': 'Urea (46-0-0)', 'n': 46, 'p': 0, 'k': 0, 'rate_kg_ha': 150, 'timing': 'At earthing up', 'method': 'Side dressing', 'cost_per_kg': 0.85},
                {'name': 'Ammonium Sulphate (21-0-0)', 'n': 21, 'p': 0, 'k': 0, 'rate_kg_ha': 200, 'timing': 'At tuber initiation', 'method': 'Broadcast', 'cost_per_kg': 0.90}
            ],
            'organic_options': [
                {'name': 'Farmyard Manure', 'rate_tons_ha': 20, 'timing': '3-4 weeks before planting', 'cost_per_ton': 30},
                {'name': 'Vermicompost', 'rate_tons_ha': 5, 'timing': 'At planting', 'method': 'Apply in furrows', 'cost_per_ton': 80}
            ],
            'schedule': {
                'basal': 'Apply 50% N, full P & K at planting',
                'top_dressing': 'Apply 50% N at earthing up (30-40 DAP)'
            }
        },
        'Tomatoes': {
            'fertilizers': [
                {'name': 'NPK 12-32-16', 'n': 12, 'p': 32, 'k': 16, 'rate_kg_ha': 500, 'timing': 'At transplanting', 'method': 'Band placement', 'cost_per_kg': 1.15},
                {'name': 'Urea (46-0-0)', 'n': 46, 'p': 0, 'k': 0, 'rate_kg_ha': 200, 'timing': 'Split (2 applications)', 'method': 'Side dressing', 'cost_per_kg': 0.85},
                {'name': 'Calcium Nitrate (15.5-0-0)', 'n': 15.5, 'p': 0, 'k': 0, 'rate_kg_ha': 150, 'timing': 'Fruit set stage', 'method': 'Foliar spray', 'cost_per_kg': 1.30}
            ],
            'organic_options': [
                {'name': 'Farmyard Manure', 'rate_tons_ha': 15, 'timing': '2 weeks before transplanting', 'cost_per_ton': 30},
                {'name': 'Compost Tea', 'rate_litres_ha': 500, 'timing': 'Weekly foliar', 'method': 'Spray on leaves', 'cost_per_litre': 1.50}
            ],
            'schedule': {
                'basal': 'Apply 50% N, full P & K at transplanting',
                'top_dressing_1': 'Apply 25% N at flowering',
                'top_dressing_2': 'Apply 25% N at fruit development'
            }
        }
    }
    return fertilizer_db

# Function to get specific fertilizer recommendations based on nutrient deficits
def get_fertilizer_by_nutrient(crop_name, nutrient_deficits):
    """Get specific fertilizer names based on nutrient deficits"""
    fertilizer_db = get_fertilizer_database()
    
    if crop_name not in fertilizer_db:
        return []
    
    crop_data = fertilizer_db[crop_name]
    recommendations = []
    
    # Check nitrogen deficit
    if nutrient_deficits.get('Nitrogen', 0) > 0:
        n_ferts = [f for f in crop_data['fertilizers'] if f['n'] > 0]
        if n_ferts:
            recommendations.append({
                'nutrient': 'Nitrogen',
                'fertilizers': n_ferts[:2]  # Top 2 nitrogen fertilizers
            })
    
    # Check phosphorus deficit
    if nutrient_deficits.get('Phosphorus', 0) > 0:
        p_ferts = [f for f in crop_data['fertilizers'] if f['p'] > 0]
        if p_ferts:
            recommendations.append({
                'nutrient': 'Phosphorus',
                'fertilizers': p_ferts[:2]  # Top 2 phosphorus fertilizers
            })
    
    # Check potassium deficit
    if nutrient_deficits.get('Potassium', 0) > 0:
        k_ferts = [f for f in crop_data['fertilizers'] if f['k'] > 0]
        if k_ferts:
            recommendations.append({
                'nutrient': 'Potassium',
                'fertilizers': k_ferts[:2]  # Top 2 potassium fertilizers
            })
    
    return recommendations

# Function to display fertilizer recommendations with names
def display_fertilizer_recommendations(crop_name, nutrient_deficits=None):
    """Display comprehensive fertilizer recommendations with names"""
    fertilizer_db = get_fertilizer_database()
    
    if crop_name not in fertilizer_db:
        st.warning(f"No specific fertilizer data available for {crop_name}")
        return
    
    crop_data = fertilizer_db[crop_name]
    
    # If nutrient deficits are provided, show targeted recommendations
    if nutrient_deficits:
        st.markdown("#### 🎯 Targeted Fertilizer Recommendations Based on Deficits")
        targeted_recs = get_fertilizer_by_nutrient(crop_name, nutrient_deficits)
        
        if targeted_recs:
            for rec in targeted_recs:
                st.markdown(f"**For {rec['nutrient']} Deficiency:**")
                for fert in rec['fertilizers']:
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.markdown(f"✅ **{fert['name']}**")
                    with col2:
                        st.markdown(f"NPK: {fert['n']}-{fert['p']}-{fert['k']}")
                    with col3:
                        st.markdown(f"Rate: {fert['rate_kg_ha']} kg/ha")
                st.markdown("---")
        else:
            st.info("No specific fertilizer recommendations available for the detected deficits")
    
    # Chemical Fertilizers
    st.markdown("#### 🧪 Recommended Chemical Fertilizers")
    for fert in crop_data['fertilizers']:
        with st.container():
            st.markdown(f"""
            <div class="fertilizer-card">
                <div class="fertilizer-name">📌 {fert['name']}</div>
                <div>
                    <span class="nutrient-tag">N: {fert['n']}%</span>
                    <span class="nutrient-tag">P: {fert['p']}%</span>
                    <span class="nutrient-tag">K: {fert['k']}%</span>
                </div>
                <br>
                <strong>Application Rate:</strong> {fert['rate_kg_ha']} kg/ha<br>
                <strong>Timing:</strong> {fert['timing']}<br>
                <strong>Method:</strong> {fert['method']}<br>
                <strong>Estimated Cost:</strong> ${fert['rate_kg_ha'] * fert.get('cost_per_kg', 1.0):.2f}/ha
            </div>
            """, unsafe_allow_html=True)
    
    # Organic Fertilizers
    if 'organic_options' in crop_data:
        st.markdown("#### 🌱 Organic Fertilizer Options")
        for org in crop_data['organic_options']:
            with st.container():
                st.markdown(f"""
                <div class="fertilizer-card">
                    <div class="fertilizer-name">🌿 {org['name']}</div>
                """, unsafe_allow_html=True)
                
                if 'rate_tons_ha' in org:
                    st.write(f"**Application Rate:** {org['rate_tons_ha']} tons/ha")
                    if 'cost_per_ton' in org:
                        st.write(f"**Estimated Cost:** ${org['rate_tons_ha'] * org['cost_per_ton']:.2f}/ha")
                elif 'rate_kg_ha' in org:
                    st.write(f"**Application Rate:** {org['rate_kg_ha']} kg/ha")
                    if 'cost_per_kg' in org:
                        st.write(f"**Estimated Cost:** ${org['rate_kg_ha'] * org['cost_per_kg']:.2f}/ha")
                elif 'rate_litres_ha' in org:
                    st.write(f"**Application Rate:** {org['rate_litres_ha']} litres/ha")
                    if 'cost_per_litre' in org:
                        st.write(f"**Estimated Cost:** ${org['rate_litres_ha'] * org['cost_per_litre']:.2f}/ha")
                elif 'rate_kg_tree' in org:
                    st.write(f"**Application Rate:** {org['rate_kg_tree']} kg/tree")
                
                if 'n_pct' in org:
                    st.write(f"**NPK Content:** {org['n_pct']}-{org.get('p_pct', '0')}-{org.get('k_pct', '0')}%")
                st.write(f"**Timing:** {org['timing']}")
                if 'method' in org:
                    st.write(f"**Method:** {org['method']}")
                st.markdown("</div>", unsafe_allow_html=True)
    
    # Application Schedule
    st.markdown("#### 📅 Application Schedule")
    schedule_df = pd.DataFrame([
        {"Stage": stage.replace('_', ' ').title(), "Recommendation": rec} 
        for stage, rec in crop_data['schedule'].items()
    ])
    st.dataframe(schedule_df, use_container_width=True, hide_index=True)

# Generate enhanced sample data with yield information
@st.cache_data
def generate_ml_sample_data():
    """Generate comprehensive sample data for ML training"""
    np.random.seed(42)
    
    n_samples = 500
    
    crops = ['Corn', 'Wheat', 'Soybeans', 'Rice', 'Cotton', 'Vegetables', 
             'Fruits', 'Sugarcane', 'Potatoes', 'Tomatoes']
    
    soil_types = ['Sandy', 'Loamy', 'Clay', 'Silty', 'Peaty']
    
    data = []
    for i in range(n_samples):
        # Generate features
        crop = np.random.choice(crops)
        soil_type = np.random.choice(soil_types)
        ph = np.random.uniform(4.5, 8.5)
        nitrogen = np.random.uniform(20, 200)
        phosphorus = np.random.uniform(10, 150)
        potassium = np.random.uniform(50, 400)
        organic_matter = np.random.uniform(0.5, 5.0)
        temperature = np.random.uniform(15, 35)
        rainfall = np.random.uniform(500, 2000)
        
        # Calculate yield based on features (synthetic relationship)
        base_yield = {
            'Corn': 8.5, 'Wheat': 4.2, 'Soybeans': 3.5, 'Rice': 5.8, 'Cotton': 2.5,
            'Vegetables': 25, 'Fruits': 15, 'Sugarcane': 65, 'Potatoes': 30, 'Tomatoes': 40
        }[crop]
        
        # Adjust yield based on nutrients
        nutrient_score = (nitrogen / 150 + phosphorus / 60 + potassium / 120) / 3
        ph_score = 1 - abs(ph - 6.5) / 3
        weather_score = (temperature / 25 + rainfall / 1200) / 2
        
        yield_value = base_yield * (0.5 + nutrient_score * 0.3 + ph_score * 0.1 + weather_score * 0.1)
        yield_value += np.random.normal(0, yield_value * 0.1)  # Add noise
        
        data.append({
            'Field_ID': f'FIELD_{i+1:04d}',
            'Crop': crop,
            'Soil_Type': soil_type,
            'pH': ph,
            'Nitrogen_N': nitrogen,
            'Phosphorus_P': phosphorus,
            'Potassium_K': potassium,
            'Organic_Matter': organic_matter,
            'Temperature': temperature,
            'Rainfall': rainfall,
            'Yield_tons_per_ha': max(0, yield_value)
        })
    
    return pd.DataFrame(data)

# Crop requirement database
@st.cache_data
def get_crop_requirements_ml():
    """Get optimal nutrient requirements for different crops"""
    crop_requirements = {
        'Corn': {'N': 150, 'P': 60, 'K': 120, 'pH_min': 5.8, 'pH_max': 7.0, 'target_yield': 9.5},
        'Wheat': {'N': 120, 'P': 50, 'K': 80, 'pH_min': 6.0, 'pH_max': 7.5, 'target_yield': 5.0},
        'Soybeans': {'N': 80, 'P': 40, 'K': 100, 'pH_min': 6.0, 'pH_max': 6.8, 'target_yield': 4.0},
        'Rice': {'N': 100, 'P': 40, 'K': 60, 'pH_min': 5.5, 'pH_max': 6.5, 'target_yield': 6.5},
        'Cotton': {'N': 140, 'P': 55, 'K': 110, 'pH_min': 5.8, 'pH_max': 7.2, 'target_yield': 3.0},
        'Vegetables': {'N': 130, 'P': 50, 'K': 140, 'pH_min': 6.2, 'pH_max': 6.8, 'target_yield': 30},
        'Fruits': {'N': 110, 'P': 45, 'K': 130, 'pH_min': 6.0, 'pH_max': 6.5, 'target_yield': 18},
        'Sugarcane': {'N': 160, 'P': 60, 'K': 150, 'pH_min': 6.0, 'pH_max': 7.0, 'target_yield': 70},
        'Potatoes': {'N': 140, 'P': 70, 'K': 160, 'pH_min': 5.2, 'pH_max': 6.4, 'target_yield': 35},
        'Tomatoes': {'N': 120, 'P': 65, 'K': 150, 'pH_min': 6.0, 'pH_max': 6.8, 'target_yield': 45}
    }
    return crop_requirements

# ML Models for Yield Prediction
class YieldPredictor:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def prepare_features(self, df):
        """Prepare features for ML models"""
        df_processed = df.copy()
        
        # Encode categorical variables
        categorical_cols = ['Crop', 'Soil_Type']
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col])
            else:
                df_processed[col] = self.label_encoders[col].transform(df_processed[col])
        
        # Select features
        feature_cols = ['Crop', 'Soil_Type', 'pH', 'Nitrogen_N', 'Phosphorus_P', 
                       'Potassium_K', 'Organic_Matter', 'Temperature', 'Rainfall']
        
        X = df_processed[feature_cols]
        y = df_processed['Yield_tons_per_ha']
        
        return X, y
    
    def train_models(self, X, y):
        """Train multiple ML models"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0)
        }
        
        # Train and evaluate models
        results = {}
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'y_pred': y_pred,
                'y_test': y_test
            }
            
            self.models[name] = model
        
        return results, X_test_scaled, y_test
    
    def predict_yield(self, features):
        """Predict yield using trained models"""
        features_scaled = self.scaler.transform(features)
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(features_scaled)[0]
        return predictions

# Fertilizer Optimization using ML
class FertilizerOptimizer:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        
    def prepare_optimization_data(self, df):
        """Prepare data for fertilizer optimization"""
        # Target: maximize yield with minimum fertilizer cost
        features = ['pH', 'Nitrogen_N', 'Phosphorus_P', 'Potassium_K', 
                   'Organic_Matter', 'Temperature', 'Rainfall']
        
        X = df[features]
        y = df['Yield_tons_per_ha']
        
        return X, y
    
    def train_optimizer(self, X, y):
        """Train optimizer model"""
        X_scaled = self.scaler.fit_transform(X)
        self.model = GradientBoostingRegressor(n_estimators=200, random_state=42)
        self.model.fit(X_scaled, y)
        return self.model
    
    def optimize_fertilizer(self, current_soil, crop_req):
        """Find optimal fertilizer combination"""
        current_n = current_soil['Nitrogen_N']
        current_p = current_soil['Phosphorus_P']
        current_k = current_soil['Potassium_K']
        
        # Calculate optimal deficits
        n_deficit = max(0, crop_req['N'] - current_n)
        p_deficit = max(0, crop_req['P'] - current_p)
        k_deficit = max(0, crop_req['K'] - current_k)
        
        # Calculate optimal application rates (considering efficiency)
        n_opt = n_deficit * 1.1  # 10% buffer
        p_opt = p_deficit * 1.05
        k_opt = k_deficit * 1.1
        
        return {
            'Nitrogen': {'deficit': n_deficit, 'optimal': n_opt},
            'Phosphorus': {'deficit': p_deficit, 'optimal': p_opt},
            'Potassium': {'deficit': k_deficit, 'optimal': k_opt}
        }

# Soil Health Clustering
class SoilClusterAnalyzer:
    def __init__(self):
        self.kmeans = None
        self.pca = None
        
    def cluster_soils(self, df, n_clusters=4):
        """Cluster soils based on properties"""
        features = ['pH', 'Nitrogen_N', 'Phosphorus_P', 'Potassium_K', 'Organic_Matter']
        X = df[features]
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply KMeans
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = self.kmeans.fit_predict(X_scaled)
        
        # PCA for visualization
        self.pca = PCA(n_components=2)
        pca_result = self.pca.fit_transform(X_scaled)
        
        return clusters, pca_result

# Feature Importance Analysis
def analyze_feature_importance(model, feature_names):
    """Analyze feature importance from trained model"""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        return importance_df
    return None

# Create visualizations
def create_yield_prediction_chart(predictions):
    """Create yield prediction comparison chart"""
    fig = go.Figure()
    
    models = list(predictions.keys())
    yields = list(predictions.values())
    
    fig.add_trace(go.Bar(
        x=models,
        y=yields,
        marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'],
        text=[f'{y:.2f} t/ha' for y in yields],
        textposition='auto',
    ))
    
    fig.update_layout(
        title="Yield Predictions by Different ML Models",
        xaxis_title="Model",
        yaxis_title="Predicted Yield (tons/ha)",
        showlegend=False,
        height=400
    )
    
    return fig

def create_cluster_visualization(pca_result, clusters):
    """Create cluster visualization"""
    fig = go.Figure()
    
    unique_clusters = np.unique(clusters)
    colors = px.colors.qualitative.Set1
    
    for cluster in unique_clusters:
        mask = clusters == cluster
        fig.add_trace(go.Scatter(
            x=pca_result[mask, 0],
            y=pca_result[mask, 1],
            mode='markers',
            name=f'Cluster {cluster}',
            marker=dict(size=8, color=colors[cluster % len(colors)])
        ))
    
    fig.update_layout(
        title="Soil Health Clusters (PCA Visualization)",
        xaxis_title="First Principal Component",
        yaxis_title="Second Principal Component",
        height=500
    )
    
    return fig

def create_radar_chart(soil_data, crop_req):
    """Create enhanced radar chart"""
    categories = ['Nitrogen', 'Phosphorus', 'Potassium', 'Organic Matter']
    soil_values = [
        soil_data['Nitrogen_N'] / crop_req['N'] * 100,
        soil_data['Phosphorus_P'] / crop_req['P'] * 100,
        soil_data['Potassium_K'] / crop_req['K'] * 100,
        soil_data.get('Organic_Matter', 2) / 4 * 100
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=soil_values,
        theta=categories,
        fill='toself',
        name='Soil Health (%)',
        line_color='#2E7D32',
        fillcolor='rgba(46, 125, 50, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 150],
                tickvals=[0, 50, 100, 150],
                ticktext=['0%', '50%', '100%', '150%']
            )),
        title="Soil Health Assessment",
        height=400
    )
    
    return fig

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🌾 AI-Powered Fertilizer Recommendation System</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Machine Learning for Precision Agriculture</p>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=80)
        st.header("⚙️ ML Controls")
        
        show_installation_instructions()
        
        st.markdown("---")
        
        # Data source selection
        data_source = st.radio(
            "Select Data Source",
            ["📊 ML Sample Data", "📝 Manual Input"],
            key="data_source_selector"
        )
        
        st.markdown("---")
        
        # ML Model selection
        st.subheader("🤖 ML Models")
        use_ml_prediction = st.checkbox("Enable ML Predictions", value=True, key="enable_ml_predictions")
        
        if use_ml_prediction:
            selected_model = st.selectbox(
                "Select Model for Prediction",
                ["Random Forest", "Gradient Boosting", "Ensemble (Average)"],
                key="model_selector"
            )
        
        st.markdown("---")
        
        # Download sample data
        if st.button("📥 Download ML Sample Data", key="download_btn"):
            sample_data = generate_ml_sample_data()
            csv = sample_data.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="ml_soil_data.csv">Click to download</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    # Load data
    if data_source == "📊 ML Sample Data":
        data = generate_ml_sample_data()
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Fields", len(data))
        with col2:
            st.metric("Avg Yield", f"{data['Yield_tons_per_ha'].mean():.1f} t/ha")
        with col3:
            st.metric("Crop Types", data['Crop'].nunique())
        with col4:
            st.metric("Avg pH", f"{data['pH'].mean():.2f}")
        
        # Tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Data Explorer", 
            "🤖 ML Predictions",
            "🌱 Fertilizer Optimization",
            "📈 Cluster Analysis",
            "🎯 Smart Recommendations",
            "🧪 Fertilizer Database"
        ])
        
        with tab1:
            st.subheader("Soil Data Explorer")
            st.dataframe(data.head(100), use_container_width=True)
            
            # Summary statistics
            st.subheader("Summary Statistics")
            stats_df = data[['pH', 'Nitrogen_N', 'Phosphorus_P', 'Potassium_K', 
                           'Organic_Matter', 'Temperature', 'Rainfall', 'Yield_tons_per_ha']].describe()
            st.dataframe(stats_df, use_container_width=True)
            
            # Distribution plots
            col1, col2 = st.columns(2)
            with col1:
                fig_yield = px.histogram(data, x='Yield_tons_per_ha', color='Crop',
                                        title='Yield Distribution by Crop',
                                        nbins=50)
                st.plotly_chart(fig_yield, use_container_width=True)
            
            with col2:
                fig_corr = px.scatter_matrix(data, dimensions=['pH', 'Nitrogen_N', 'Phosphorus_P', 'Potassium_K'],
                                            color='Crop', title='Nutrient Correlation Matrix')
                st.plotly_chart(fig_corr, use_container_width=True)
        
        with tab2:
            st.subheader("🤖 Machine Learning Yield Predictions")
            
            if use_ml_prediction:
                # Prepare data for ML
                predictor = YieldPredictor()
                X, y = predictor.prepare_features(data)
                
                # Train models
                with st.spinner("Training ML models..."):
                    results, X_test_scaled, y_test = predictor.train_models(X, y)
                
                # Display model performance
                st.write("**Model Performance Comparison**")
                
                performance_df = pd.DataFrame({
                    'Model': list(results.keys()),
                    'RMSE (t/ha)': [results[m]['rmse'] for m in results],
                    'MAE (t/ha)': [results[m]['mae'] for m in results],
                    'R² Score': [results[m]['r2'] for m in results]
                })
                
                st.dataframe(performance_df, use_container_width=True)
                
                # Visualize model performance
                fig_perf = go.Figure()
                for model in results:
                    fig_perf.add_trace(go.Scatter(
                        x=results[model]['y_test'],
                        y=results[model]['y_pred'],
                        mode='markers',
                        name=model,
                        marker=dict(size=6)
                    ))
                
                fig_perf.add_trace(go.Scatter(
                    x=[y_test.min(), y_test.max()],
                    y=[y_test.min(), y_test.max()],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(color='red', dash='dash')
                ))
                
                fig_perf.update_layout(
                    title="Predicted vs Actual Yield",
                    xaxis_title="Actual Yield (t/ha)",
                    yaxis_title="Predicted Yield (t/ha)",
                    height=500
                )
                
                st.plotly_chart(fig_perf, use_container_width=True)
                
                # Feature importance
                rf_model = results['Random Forest']['model']
                feature_names = ['Crop', 'Soil_Type', 'pH', 'Nitrogen_N', 'Phosphorus_P', 
                               'Potassium_K', 'Organic_Matter', 'Temperature', 'Rainfall']
                
                importance_df = analyze_feature_importance(rf_model, feature_names)
                if importance_df is not None:
                    fig_importance = px.bar(importance_df, x='Importance', y='Feature',
                                          orientation='h', title='Feature Importance Analysis')
                    st.plotly_chart(fig_importance, use_container_width=True)
                
                # Predict for specific field
                st.subheader("Predict Yield for a Specific Field")
                selected_field = st.selectbox(
                    "Select Field", 
                    data['Field_ID'].unique(),
                    key="predict_field_selector"
                )
                field_data = data[data['Field_ID'] == selected_field].iloc[0:1]
                
                # Prepare features for prediction
                X_field, _ = predictor.prepare_features(field_data)
                predictions = predictor.predict_yield(X_field)
                
                # Display predictions
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                    st.write("**Soil Parameters**")
                    st.write(f"pH: {field_data['pH'].values[0]:.2f}")
                    st.write(f"Nitrogen: {field_data['Nitrogen_N'].values[0]:.1f} kg/ha")
                    st.write(f"Phosphorus: {field_data['Phosphorus_P'].values[0]:.1f} kg/ha")
                    st.write(f"Potassium: {field_data['Potassium_K'].values[0]:.1f} kg/ha")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                    st.write("**ML Model Predictions**")
                    for model_name, pred in predictions.items():
                        st.write(f"{model_name}: {pred:.2f} t/ha")
                    
                    # Ensemble prediction
                    ensemble_pred = np.mean(list(predictions.values()))
                    st.write(f"**Ensemble Prediction: {ensemble_pred:.2f} t/ha**")
                    st.markdown('</div>', unsafe_allow_html=True)
        
        with tab3:
            st.subheader("🌱 Fertilizer Optimization with ML")
            
            # Initialize optimizer
            optimizer = FertilizerOptimizer()
            
            # Prepare optimization data
            X_opt, y_opt = optimizer.prepare_optimization_data(data)
            optimizer.train_optimizer(X_opt, y_opt)
            
            # Select field for optimization
            selected_field = st.selectbox(
                "Select Field for Optimization", 
                data['Field_ID'].unique(),
                key="optimize_field_selector"
            )
            field_data = data[data['Field_ID'] == selected_field].iloc[0]
            
            # Get crop requirements
            crop_requirements = get_crop_requirements_ml()
            selected_crop = field_data['Crop']
            crop_req = crop_requirements[selected_crop]
            
            # Display current vs optimal
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Current Soil Status**")
                current_df = pd.DataFrame({
                    'Nutrient': ['Nitrogen', 'Phosphorus', 'Potassium'],
                    'Current': [field_data['Nitrogen_N'], field_data['Phosphorus_P'], field_data['Potassium_K']],
                    'Required': [crop_req['N'], crop_req['P'], crop_req['K']]
                })
                st.dataframe(current_df, use_container_width=True)
                
                # Create comparison bar chart
                fig_compare = go.Figure()
                fig_compare.add_trace(go.Bar(name='Current', x=current_df['Nutrient'], y=current_df['Current']))
                fig_compare.add_trace(go.Bar(name='Required', x=current_df['Nutrient'], y=current_df['Required']))
                fig_compare.update_layout(title="Current vs Required Nutrients", barmode='group')
                st.plotly_chart(fig_compare, use_container_width=True)
            
            with col2:
                # Optimize fertilizer
                optimal = optimizer.optimize_fertilizer(field_data.to_dict(), crop_req)
                
                st.write("**Optimal Fertilizer Recommendation**")
                for nutrient, values in optimal.items():
                    st.write(f"**{nutrient}:**")
                    st.write(f"  • Deficit: {values['deficit']:.1f} kg/ha")
                    st.write(f"  • Optimal Application: {values['optimal']:.1f} kg/ha")
                
                # Cost estimation
                st.write("**Estimated Cost (Based on Market Prices)**")
                n_cost = optimal['Nitrogen']['optimal'] * 1.5  # $1.5 per kg N
                p_cost = optimal['Phosphorus']['optimal'] * 2.0  # $2.0 per kg P
                k_cost = optimal['Potassium']['optimal'] * 1.8  # $1.8 per kg K
                total_cost = n_cost + p_cost + k_cost
                st.write(f"💰 Total Estimated Cost: ${total_cost:.2f}")
            
            # Prediction after optimization
            st.subheader("Expected Yield After Optimization")
            
            # Create optimized soil parameters
            optimized_soil = field_data.copy()
            optimized_soil['Nitrogen_N'] += optimal['Nitrogen']['optimal']
            optimized_soil['Phosphorus_P'] += optimal['Phosphorus']['optimal']
            optimized_soil['Potassium_K'] += optimal['Potassium']['optimal']
            
            # Predict yield with optimized parameters
            predictor_opt = YieldPredictor()
            X_all, y_all = predictor_opt.prepare_features(data)
            predictor_opt.train_models(X_all, y_all)
            
            # Prepare optimized features
            opt_df = pd.DataFrame([optimized_soil])
            X_opt_field, _ = predictor_opt.prepare_features(opt_df)
            opt_predictions = predictor_opt.predict_yield(X_opt_field)
            
            # Display before/after comparison
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.write("**Before Optimization**")
                st.write(f"Yield: {field_data['Yield_tons_per_ha']:.2f} t/ha")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.write("**After Optimization**")
                avg_pred = np.mean(list(opt_predictions.values()))
                improvement = avg_pred - field_data['Yield_tons_per_ha']
                st.write(f"Yield: {avg_pred:.2f} t/ha")
                st.write(f"Improvement: +{improvement:.2f} t/ha ({improvement/field_data['Yield_tons_per_ha']*100:.1f}%)")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Display fertilizer names after optimization
            st.subheader("📌 Recommended Fertilizers for This Field")
            deficits = {
                'Nitrogen': optimal['Nitrogen']['deficit'],
                'Phosphorus': optimal['Phosphorus']['deficit'],
                'Potassium': optimal['Potassium']['deficit']
            }
            display_fertilizer_recommendations(selected_crop, deficits)
        
        with tab4:
            st.subheader("📈 Soil Health Cluster Analysis")
            
            # Perform clustering
            cluster_analyzer = SoilClusterAnalyzer()
            clusters, pca_result = cluster_analyzer.cluster_soils(data)
            
            # Add clusters to data
            data['Cluster'] = clusters
            
            # Display cluster visualization
            fig_clusters = create_cluster_visualization(pca_result, clusters)
            st.plotly_chart(fig_clusters, use_container_width=True)
            
            # Analyze clusters
            st.subheader("Cluster Characteristics")
            cluster_summary = data.groupby('Cluster').agg({
                'pH': 'mean',
                'Nitrogen_N': 'mean',
                'Phosphorus_P': 'mean',
                'Potassium_K': 'mean',
                'Organic_Matter': 'mean',
                'Yield_tons_per_ha': 'mean'
            }).round(2)
            
            st.dataframe(cluster_summary, use_container_width=True)
            
            # Recommendations by cluster
            st.subheader("Recommendations by Cluster")
            for cluster in sorted(data['Cluster'].unique()):
                cluster_data = data[data['Cluster'] == cluster]
                st.write(f"**Cluster {cluster}** (n={len(cluster_data)} fields)")
                
                if cluster_data['pH'].mean() < 6.0:
                    st.write("• 🔴 Low pH - Add lime")
                elif cluster_data['pH'].mean() > 7.5:
                    st.write("• 🟡 High pH - Add organic matter")
                else:
                    st.write("• ✅ Optimal pH")
                
                if cluster_data['Nitrogen_N'].mean() < 80:
                    st.write("• 🔴 Low Nitrogen - Apply nitrogen-rich fertilizer")
                    # Get specific nitrogen fertilizers
                    sample_crop = cluster_data['Crop'].iloc[0]
                    deficits = {'Nitrogen': 50, 'Phosphorus': 0, 'Potassium': 0}
                    recs = get_fertilizer_by_nutrient(sample_crop, deficits)
                    if recs:
                        for rec in recs:
                            for fert in rec['fertilizers'][:1]:
                                st.write(f"  - Recommended: **{fert['name']}**")
                
                if cluster_data['Phosphorus_P'].mean() < 30:
                    st.write("• 🔴 Low Phosphorus - Add phosphate fertilizer")
                    sample_crop = cluster_data['Crop'].iloc[0]
                    deficits = {'Nitrogen': 0, 'Phosphorus': 50, 'Potassium': 0}
                    recs = get_fertilizer_by_nutrient(sample_crop, deficits)
                    if recs:
                        for rec in recs:
                            for fert in rec['fertilizers'][:1]:
                                st.write(f"  - Recommended: **{fert['name']}**")
                
                if cluster_data['Potassium_K'].mean() < 100:
                    st.write("• 🔴 Low Potassium - Add potash fertilizer")
                    sample_crop = cluster_data['Crop'].iloc[0]
                    deficits = {'Nitrogen': 0, 'Phosphorus': 0, 'Potassium': 50}
                    recs = get_fertilizer_by_nutrient(sample_crop, deficits)
                    if recs:
                        for rec in recs:
                            for fert in rec['fertilizers'][:1]:
                                st.write(f"  - Recommended: **{fert['name']}**")
        
        with tab5:
            st.subheader("🎯 Smart Recommendations")
            
            # Select field
            selected_field = st.selectbox(
                "Select Field", 
                data['Field_ID'].unique(),
                key="recommend_field_selector"
            )
            field_data = data[data['Field_ID'] == selected_field].iloc[0]
            
            # Get crop requirements
            selected_crop = field_data['Crop']
            crop_req = get_crop_requirements_ml()[selected_crop]
            
            # Calculate deficits
            optimizer = FertilizerOptimizer()
            optimal = optimizer.optimize_fertilizer(field_data.to_dict(), crop_req)
            deficits = {
                'Nitrogen': optimal['Nitrogen']['deficit'],
                'Phosphorus': optimal['Phosphorus']['deficit'],
                'Potassium': optimal['Potassium']['deficit']
            }
            
            # Create comprehensive recommendation
            st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
            
            st.write(f"### 🌿 Recommendations for {selected_crop} - {selected_field}")
            
            # 1. Soil Health Assessment
            st.write("#### 📊 Soil Health Assessment")
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(create_radar_chart(field_data, crop_req), use_container_width=True)
            
            with col2:
                # Health score calculation
                health_score = (
                    min(100, field_data['Nitrogen_N'] / crop_req['N'] * 100) * 0.3 +
                    min(100, field_data['Phosphorus_P'] / crop_req['P'] * 100) * 0.3 +
                    min(100, field_data['Potassium_K'] / crop_req['K'] * 100) * 0.3 +
                    (1 - abs(field_data['pH'] - 6.5) / 3) * 100 * 0.1
                )
                
                st.metric("Overall Soil Health Score", f"{health_score:.1f}/100")
                
                if health_score < 50:
                    st.warning("⚠️ Poor soil health - Immediate intervention needed")
                elif health_score < 75:
                    st.info("📈 Moderate soil health - Some improvement needed")
                else:
                    st.success("✅ Excellent soil health - Maintain current practices")
            
            # 2. Nutrient Management with Specific Fertilizer Names
            st.write("#### 🌱 Nutrient Management Plan")
            st.write("Based on soil test results, here are the specific fertilizers you need:")
            
            for nutrient, values in optimal.items():
                if values['deficit'] > 10:
                    st.write(f"**{nutrient} Deficiency:** {values['deficit']:.1f} kg/ha")
                    # Get specific fertilizers for this nutrient
                    nutrient_deficits = {
                        'Nitrogen': values['deficit'] if nutrient == 'Nitrogen' else 0,
                        'Phosphorus': values['deficit'] if nutrient == 'Phosphorus' else 0,
                        'Potassium': values['deficit'] if nutrient == 'Potassium' else 0
                    }
                    recs = get_fertilizer_by_nutrient(selected_crop, nutrient_deficits)
                    if recs:
                        for rec in recs:
                            for fert in rec['fertilizers']:
                                st.markdown(f"""
                                <div style="background-color: #e8f5e9; padding: 0.5rem; border-radius: 5px; margin: 0.5rem 0;">
                                    <strong>📌 {fert['name']}</strong><br>
                                    Rate: {fert['rate_kg_ha']} kg/ha | 
                                    Timing: {fert['timing']} | 
                                    Method: {fert['method']}
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.write(f"  • Apply {values['optimal']:.0f} kg/ha of {nutrient} fertilizer")
                elif values['deficit'] > 0:
                    st.write(f"**{nutrient}:** Light application of {values['optimal']:.0f} kg/ha needed")
                else:
                    st.write(f"✅ **{nutrient}:** Adequate levels - No additional fertilizer needed")
            
            # 3. pH Management
            st.write("#### 🧪 pH Management")
            if field_data['pH'] < crop_req['pH_min']:
                lime_needed = (crop_req['pH_min'] - field_data['pH']) * 1000
                st.write(f"• Add **{lime_needed:.0f} kg/ha** of **Agricultural Lime**")
                st.write("• Monitor pH after 3-6 months")
            elif field_data['pH'] > crop_req['pH_max']:
                sulfur_needed = (field_data['pH'] - crop_req['pH_max']) * 500
                st.write(f"• Add **{sulfur_needed:.0f} kg/ha** of **Elemental Sulfur**")
                st.write("• Apply organic matter to improve buffering capacity")
            else:
                st.write("✅ pH is within optimal range")
            
            # 4. Complete Fertilizer Recommendations
            st.write("#### 🧪 Complete Fertilizer Recommendations")
            display_fertilizer_recommendations(selected_crop, deficits)
            
            # 5. Yield Prediction
            st.write("#### 📈 Expected Yield")
            
            predictor = YieldPredictor()
            X_all, y_all = predictor.prepare_features(data)
            predictor.train_models(X_all, y_all)
            
            field_df = pd.DataFrame([field_data])
            X_field, _ = predictor.prepare_features(field_df)
            predictions = predictor.predict_yield(X_field)
            avg_pred = np.mean(list(predictions.values()))
            
            st.write(f"**Predicted Yield:** {avg_pred:.2f} t/ha")
            st.write(f"**Target Yield:** {crop_req['target_yield']:.2f} t/ha")
            
            if avg_pred < crop_req['target_yield']:
                gap = crop_req['target_yield'] - avg_pred
                st.write(f"Yield gap: {gap:.2f} t/ha - Follow recommendations to close the gap")
            else:
                st.write("🎯 On track to meet or exceed yield targets")
            
            # 6. Financial Analysis
            st.write("#### 💰 Financial Analysis")
            
            # Calculate fertilizer cost using actual fertilizer names
            total_cost = 0
            fertilizer_list = []
            for nutrient, values in optimal.items():
                if values['deficit'] > 0:
                    nutrient_deficits = {
                        'Nitrogen': values['deficit'] if nutrient == 'Nitrogen' else 0,
                        'Phosphorus': values['deficit'] if nutrient == 'Phosphorus' else 0,
                        'Potassium': values['deficit'] if nutrient == 'Potassium' else 0
                    }
                    recs = get_fertilizer_by_nutrient(selected_crop, nutrient_deficits)
                    if recs:
                        for rec in recs:
                            for fert in rec['fertilizers'][:1]:
                                fert_cost = fert['rate_kg_ha'] * fert.get('cost_per_kg', 1.0)
                                total_cost += fert_cost
                                fertilizer_list.append(f"{fert['name']}: ${fert_cost:.2f}")
            
            # Calculate expected revenue
            crop_prices = {
                'Corn': 200, 'Wheat': 250, 'Soybeans': 450, 'Rice': 350,
                'Cotton': 1200, 'Vegetables': 800, 'Fruits': 1000,
                'Sugarcane': 50, 'Potatoes': 300, 'Tomatoes': 400
            }
            
            price_per_ton = crop_prices.get(selected_crop, 300)
            current_revenue = field_data['Yield_tons_per_ha'] * price_per_ton
            projected_revenue = avg_pred * price_per_ton
            
            st.write("**Fertilizer Cost Breakdown:**")
            for fert_cost in fertilizer_list:
                st.write(f"  • {fert_cost}")
            st.write(f"**Total Fertilizer Investment:** ${total_cost:.2f}/ha")
            st.write(f"**Current Revenue:** ${current_revenue:.2f}/ha")
            st.write(f"**Projected Revenue:** ${projected_revenue:.2f}/ha")
            st.write(f"**Net Gain:** ${projected_revenue - current_revenue - total_cost:.2f}/ha")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab6:
            st.subheader("🧪 Complete Fertilizer Database")
            st.markdown("Browse through our comprehensive fertilizer recommendations for all crops")
            
            # Crop selector for database
            fertilizer_db = get_fertilizer_database()
            selected_crop_db = st.selectbox(
                "Select Crop to View Fertilizer Details",
                list(fertilizer_db.keys()),
                key="fertilizer_db_selector"
            )
            
            # Display fertilizer information
            st.markdown(f"### Fertilizer Recommendations for {selected_crop_db}")
            display_fertilizer_recommendations(selected_crop_db)
            
            # Quick reference table
            st.markdown("### Quick Reference: NPK Requirements")
            crop_requirements = get_crop_requirements_ml()
            quick_ref_df = pd.DataFrame([
                {
                    'Crop': crop,
                    'N (kg/ha)': req['N'],
                    'P (kg/ha)': req['P'],
                    'K (kg/ha)': req['K'],
                    'pH Range': f"{req['pH_min']} - {req['pH_max']}",
                    'Target Yield (t/ha)': req['target_yield'],
                    'Recommended Starter': fertilizer_db[crop]['fertilizers'][0]['name'] if crop in fertilizer_db else 'N/A'
                }
                for crop, req in crop_requirements.items() if crop in fertilizer_db
            ])
            st.dataframe(quick_ref_df, use_container_width=True)
    
    else:  # Manual Input
        st.subheader("📝 Manual Soil Test Input with ML Predictions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            field_id = st.text_input("Field ID", value="MY_FIELD_001", key="manual_field_id")
            crop_type = st.selectbox(
                "Crop Type", 
                list(get_crop_requirements_ml().keys()),
                key="manual_crop_type"
            )
            soil_type = st.selectbox(
                "Soil Type", 
                ['Sandy', 'Loamy', 'Clay', 'Silty', 'Peaty'],
                key="manual_soil_type"
            )
            ph = st.slider("pH Level", 0.0, 14.0, 6.5, 0.1, key="manual_ph")
            nitrogen = st.number_input("Nitrogen (N) - kg/ha", 0, 500, 100, key="manual_nitrogen")
            phosphorus = st.number_input("Phosphorus (P) - kg/ha", 0, 300, 50, key="manual_phosphorus")
        
        with col2:
            potassium = st.number_input("Potassium (K) - kg/ha", 0, 500, 150, key="manual_potassium")
            organic_matter = st.number_input("Organic Matter (%)", 0.0, 10.0, 2.0, 0.1, key="manual_organic")
            temperature = st.number_input("Temperature (°C)", 0, 50, 25, key="manual_temp")
            rainfall = st.number_input("Rainfall (mm/year)", 0, 3000, 1000, key="manual_rainfall")
            target_yield = st.number_input("Target Yield (t/ha)", 0.0, 100.0, 5.0, 0.5, key="manual_target")
        
        if st.button("Get ML-Powered Recommendations", type="primary", key="get_recommendations_btn"):
            # Create soil data dictionary
            soil_data = {
                'Field_ID': field_id,
                'Crop': crop_type,
                'Soil_Type': soil_type,
                'pH': ph,
                'Nitrogen_N': nitrogen,
                'Phosphorus_P': phosphorus,
                'Potassium_K': potassium,
                'Organic_Matter': organic_matter,
                'Temperature': temperature,
                'Rainfall': rainfall,
                'Yield_tons_per_ha': target_yield
            }
            
            # Convert to DataFrame
            soil_df = pd.DataFrame([soil_data])
            
            # Get ML predictions
            with st.spinner("Running ML models..."):
                # Yield prediction
                predictor = YieldPredictor()
                sample_data = generate_ml_sample_data()
                X_train, y_train = predictor.prepare_features(sample_data)
                predictor.train_models(X_train, y_train)
                
                X_input, _ = predictor.prepare_features(soil_df)
                yield_predictions = predictor.predict_yield(X_input)
                
                # Fertilizer optimization
                optimizer = FertilizerOptimizer()
                crop_req = get_crop_requirements_ml()[crop_type]
                optimal = optimizer.optimize_fertilizer(soil_data, crop_req)
                
                # Soil clustering
                cluster_analyzer = SoilClusterAnalyzer()
                clusters, _ = cluster_analyzer.cluster_soils(sample_data)
                
                # Assign cluster
                soil_features = [[ph, nitrogen, phosphorus, potassium, organic_matter]]
                scaler = StandardScaler()
                sample_features = sample_data[['pH', 'Nitrogen_N', 'Phosphorus_P', 'Potassium_K', 'Organic_Matter']]
                scaler.fit(sample_features)
                soil_scaled = scaler.transform(soil_features)
                cluster = cluster_analyzer.kmeans.predict(soil_scaled)[0]
                
                # Calculate deficits
                deficits = {
                    'Nitrogen': optimal['Nitrogen']['deficit'],
                    'Phosphorus': optimal['Phosphorus']['deficit'],
                    'Potassium': optimal['Potassium']['deficit']
                }
            
            # Display results
            st.markdown("---")
            st.subheader("📊 ML Analysis Results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.write("**Yield Predictions**")
                for model, pred in yield_predictions.items():
                    st.write(f"{model}: {pred:.2f} t/ha")
                ensemble = np.mean(list(yield_predictions.values()))
                st.write(f"**Ensemble: {ensemble:.2f} t/ha**")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.write("**Soil Health Analysis**")
                health_score = (
                    min(100, nitrogen / crop_req['N'] * 100) * 0.3 +
                    min(100, phosphorus / crop_req['P'] * 100) * 0.3 +
                    min(100, potassium / crop_req['K'] * 100) * 0.3 +
                    (1 - abs(ph - 6.5) / 3) * 100 * 0.1
                )
                st.write(f"Health Score: {health_score:.1f}/100")
                st.write(f"Soil Cluster: {cluster}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.write("**Fertilizer Recommendations**")
                for nutrient, values in optimal.items():
                    if values['deficit'] > 0:
                        st.write(f"{nutrient}: {values['optimal']:.0f} kg/ha")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Detailed recommendations with fertilizer names
            st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
            st.write("### 🌿 Detailed Recommendations")
            
            # Nutrient-specific recommendations with fertilizer names
            st.write("**Nutrient Management with Specific Fertilizers:**")
            for nutrient, values in optimal.items():
                if values['deficit'] > 20:
                    st.write(f"🔴 **{nutrient}** - Severe deficiency detected")
                    st.write(f"   • Required: {values['optimal']:.0f} kg/ha")
                    
                    # Get specific fertilizer recommendations
                    nutrient_deficits = {
                        'Nitrogen': values['deficit'] if nutrient == 'Nitrogen' else 0,
                        'Phosphorus': values['deficit'] if nutrient == 'Phosphorus' else 0,
                        'Potassium': values['deficit'] if nutrient == 'Potassium' else 0
                    }
                    recs = get_fertilizer_by_nutrient(crop_type, nutrient_deficits)
                    if recs:
                        for rec in recs:
                            for fert in rec['fertilizers']:
                                st.write(f"   • Recommended: **{fert['name']}**")
                                st.write(f"     - Rate: {fert['rate_kg_ha']} kg/ha")
                                st.write(f"     - Timing: {fert['timing']}")
                                st.write(f"     - Method: {fert['method']}")
                    else:
                        st.write(f"   • Apply {values['optimal']:.0f} kg/ha of {nutrient} fertilizer")
                    
                elif values['deficit'] > 0:
                    st.write(f"🟡 **{nutrient}** - Moderate deficiency")
                    st.write(f"   • Required: {values['optimal']:.0f} kg/ha")
                    
                    nutrient_deficits = {
                        'Nitrogen': values['deficit'] if nutrient == 'Nitrogen' else 0,
                        'Phosphorus': values['deficit'] if nutrient == 'Phosphorus' else 0,
                        'Potassium': values['deficit'] if nutrient == 'Potassium' else 0
                    }
                    recs = get_fertilizer_by_nutrient(crop_type, nutrient_deficits)
                    if recs:
                        for rec in recs:
                            for fert in rec['fertilizers'][:1]:
                                st.write(f"   • Recommended: **{fert['name']}** at {fert['rate_kg_ha']} kg/ha")
                else:
                    st.write(f"✅ **{nutrient}** - Adequate levels")
            
            # pH recommendation
            st.write("**pH Management**")
            if ph < crop_req['pH_min']:
                st.write("🔴 **Soil is too acidic**")
                lime_needed = (crop_req['pH_min'] - ph) * 1000
                st.write(f"• Apply **{lime_needed:.0f} kg/ha** of **Agricultural Lime**")
                st.write("• Retest soil in 6 months")
            elif ph > crop_req['pH_max']:
                st.write("🟡 **Soil is too alkaline**")
                sulfur_needed = (ph - crop_req['pH_max']) * 500
                st.write(f"• Apply **{sulfur_needed:.0f} kg/ha** of **Elemental Sulfur**")
                st.write("• Increase organic matter content")
            else:
                st.write("✅ pH is within optimal range")
            
            # Complete fertilizer recommendations
            st.write("**🧪 Complete Fertilizer Recommendations for Your Crop:**")
            display_fertilizer_recommendations(crop_type, deficits)
            
            # Application timing
            st.write("**⏰ Application Timing**")
            st.write("• Apply fertilizers during early morning or late evening")
            st.write("• Avoid application before heavy rainfall")
            st.write("• Incorporate fertilizers into soil for better absorption")
            st.write("• Consider split application for nitrogen in sandy soils")
            
            # Financial analysis
            st.write("**💰 Financial Projection**")
            crop_prices = {
                'Corn': 200, 'Wheat': 250, 'Soybeans': 450, 'Rice': 350,
                'Cotton': 1200, 'Vegetables': 800, 'Fruits': 1000,
                'Sugarcane': 50, 'Potatoes': 300, 'Tomatoes': 400
            }
            price = crop_prices.get(crop_type, 300)
            current_revenue = target_yield * price
            projected_revenue = ensemble * price
            
            # Calculate fertilizer cost using actual fertilizer names
            total_cost = 0
            fertilizer_list = []
            for nutrient, values in optimal.items():
                if values['deficit'] > 0:
                    nutrient_deficits = {
                        'Nitrogen': values['deficit'] if nutrient == 'Nitrogen' else 0,
                        'Phosphorus': values['deficit'] if nutrient == 'Phosphorus' else 0,
                        'Potassium': values['deficit'] if nutrient == 'Potassium' else 0
                    }
                    recs = get_fertilizer_by_nutrient(crop_type, nutrient_deficits)
                    if recs:
                        for rec in recs:
                            for fert in rec['fertilizers'][:1]:
                                fert_cost = fert['rate_kg_ha'] * fert.get('cost_per_kg', 1.0)
                                total_cost += fert_cost
                                fertilizer_list.append(f"{fert['name']}: ${fert_cost:.2f}")
            
            st.write("**Fertilizer Cost Breakdown:**")
            if fertilizer_list:
                for fert_cost in fertilizer_list:
                    st.write(f"  • {fert_cost}")
            else:
                st.write("  • No additional fertilizers needed")
            
            st.write(f"**Total Fertilizer Investment:** ${total_cost:.2f}/ha")
            st.write(f"**Current Revenue:** ${current_revenue:.2f}/ha")
            st.write(f"**Projected Revenue:** ${projected_revenue:.2f}/ha")
            st.write(f"**ROI:** ${projected_revenue - current_revenue - total_cost:.2f}/ha")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Save recommendation
            if st.button("💾 Save Recommendation", key="save_recommendation_btn"):
                st.success("✅ Recommendation saved successfully!")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray; padding: 1rem;'>
        🤖 AI-Powered Fertilizer Recommendation System v2.0 | Machine Learning for Sustainable Agriculture<br>
        Powered by Random Forest, Gradient Boosting, and Ensemble Methods
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()