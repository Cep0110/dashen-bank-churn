import os
import base64
import joblib
import tempfile
import numpy as np
import pandas as pd
import gradio as gr
import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
from typing import List, Tuple, Optional

# --- SCIPY/SKLEARN COMPATIBILITY HACK ---
import sklearn.compose._column_transformer as ct
if not hasattr(ct, '_RemainderColsList'):
    class _RemainderColsList(list):
        pass
    ct._RemainderColsList = _RemainderColsList

import sklearn.impute as impute
if hasattr(impute, 'SimpleImputer'):
    orig_init = impute.SimpleImputer.__init__
    def patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        if not hasattr(self, '_fill_dtype'):
            self._fill_dtype = object
    impute.SimpleImputer.__init__ = patched_init
    
    # Also patch the class itself to handle objects already loaded
    if not hasattr(impute.SimpleImputer, '_fill_dtype'):
        impute.SimpleImputer._fill_dtype = object
# ----------------------------------------

DASHEN_BLUE = "#003d7a"
DASHEN_ACCENT = "#0066cc"
DASHEN_LIGHT = "#004da6"
DASHEN_PALE = "#e8f0ff"
LOGO_PATH = "dashen_logo.png"
MODEL_BUNDLE_PATH = "final_churn_bundle.joblib"

DEFAULT_FEATURES = [
    "CustomerID", "Gender", "Age", "TenureYears", "Balance_ETB",
    "City", "Branch", "AccountType", "HasMobileBanking", "HasPartner",
    "HasCreditCard", "IsActiveMember", "MonthlyServiceFee_ETB", "NumberOfTransactions"
]

CATEGORICAL_FIELDS = {
    "Gender": ["Male", "Female"],
    "HasCreditCard": ["Yes", "No"],
    "IsActiveMember": ["Yes", "No"],
    "AccountType": ["Savings", "Current", "Fixed"],
    "HasMobileBanking": ["Yes", "No"],
    "HasPartner": ["Yes", "No"],
    "City": [
        "Addis Ababa","Bahir Dar","Hawassa","Nekemte","Wolkite","Dire Dawa","Mekelle","Bekoji","Halaba","Dilla",
        "Adama","Asella","Shashemene","Metu","Dessie","Mojo","Bishoftu","Jimma","Jijiga","Arba Minch",
        "Wolaita Sodo","Wondogenet","Gondar","Gambella","Yirgachefe","Ziway","Lalibela","Woldiya",
        "Debre Markos","Harar","Mizan Aman","Debre Birhan","Ambo"
    ],
    "Branch": [
        "Head Office","Kotebe Zero Hulet Branch","Bahir Dar Stadium Branch","Tabor Branch","Nekemte Main Branch","Wolkite Branch",
        "CMC Branch","Bahir Dar Main Branch","Bekoji Branch","Halaba Kulito Branch","Dilla Branch","Aba Geda Branch",
        "Asella Branch","Bilal Branch","Mojo Branch","Shashemene Branch","Metu Branch","Balcha Branch",
        "Misrak Gerji Branch","Dessie Main Branch","Jijiga Branch","Bishoftu Branch","Jimma Arju Branch",
        "18 Mazoriya Branch","Arba Minch Branch","Melka Jebdu Branch","Ayder Branch","Bole Branch","93 Mazoriya Branch",
        "Wonji Branch","Gambella New Land Sub Branch","Yeka Abado Branch","Piazza Branch","Wolaita Sodo Branch",
        "Wondogenet Branch","Azezo Branch","Giorgis Branch","Ambo Branch","Hawassa Tabor Branch","Jijiga Taiwan Branch",
        "Airport Branch","Kazanchis Branch","Dire Dawa Main Branch","Megenagna Branch","Sar Bet Branch",
        "Ayat Branch","Mekane Selam Branch"
    ]
}

CITY_TYPE_MAP = {
    'Addis Ababa': 'Federal Capital',
    'Bahir Dar': 'Regional Capital', 'Hawassa': 'Regional Capital', 
    'Dire Dawa': 'Regional Capital', 'Mekelle': 'Regional Capital', 
    'Jijiga': 'Regional Capital', 'Gambella': 'Regional Capital',
    'Nekemte': 'Zonal Capital', 'Wolkite': 'Zonal Capital', 
    'Dilla': 'Zonal Capital', 'Adama': 'Zonal Capital', 
    'Asella': 'Zonal Capital', 'Metu': 'Zonal Capital', 
    'Dessie': 'Zonal Capital', 'Jimma': 'Zonal Capital', 
    'Arba Minch': 'Zonal Capital', 'Wolaita Sodo': 'Zonal Capital', 
    'Gondar': 'Zonal Capital',
    'Shashemene': 'Zonal Town',
    'Bekoji': 'District Town', 'Halaba': 'District Town', 
    'Wondogenet': 'District Town', 'Lalibela': 'District Town', 
    'Ambo': 'District Town'
}

BRANCH_TYPE_MAP = {
    'Head Office': 'Main', 'Bahir Dar Main Branch': 'Main', 
    'Nekemte Main Branch': 'Main', 'Dire Dawa Main Branch': 'Main', 
    'Dessie Main Branch': 'Main',
    'Bahir Dar Stadium Branch': 'Sub', 'Bilal Branch': 'Sub', 
    'Giorgis Branch': 'Sub', 'Tabor Branch': 'Sub', 
    'Hawassa Tabor Branch': 'Sub', 'Melka Jebdu Branch': 'Sub', 
    'Halaba Kulito Branch': 'Sub', 'Jijiga Taiwan Branch': 'Sub', 
    'Gambella New Land Sub Branch': 'Sub'
}

BRANCH_TO_CITY = {
    "Head Office": "Addis Ababa", "Kotebe Zero Hulet Branch": "Addis Ababa", "CMC Branch": "Addis Ababa",
    "Bahir Dar Stadium Branch": "Bahir Dar", "Bahir Dar Main Branch": "Bahir Dar",
    "Tabor Branch": "Hawassa", "Hawassa Tabor Branch": "Hawassa",
    "Nekemte Main Branch": "Nekemte", "Wolkite Branch": "Wolkite", "Bekoji Branch": "Bekoji",
    "Halaba Kulito Branch": "Halaba", "Dilla Branch": "Dilla", "Aba Geda Branch": "Adama",
    "Asella Branch": "Asella", "Bilal Branch": "Addis Ababa", "Mojo Branch": "Mojo",
    "Shashemene Branch": "Shashemene", "Metu Branch": "Metu", "Balcha Branch": "Addis Ababa",
    "Misrak Gerji Branch": "Addis Ababa", "Dessie Main Branch": "Dessie", "Jijiga Branch": "Jijiga",
    "Jijiga Taiwan Branch": "Jijiga", "Bishoftu Branch": "Bishoftu", "Jimma Arju Branch": "Jimma",
    "18 Mazoriya Branch": "Addis Ababa", "Arba Minch Branch": "Arba Minch", "Melka Jebdu Branch": "Dire Dawa",
    "Ayder Branch": "Mekelle", "Bole Branch": "Addis Ababa", "93 Mazoriya Branch": "Addis Ababa",
    "Wonji Branch": "Adama", "Gambella New Land Sub Branch": "Gambella", "Yeka Abado Branch": "Addis Ababa",
    "Piazza Branch": "Addis Ababa", "Wolaita Sodo Branch": "Wolaita Sodo", "Wondogenet Branch": "Wondogenet",
    "Azezo Branch": "Gondar", "Giorgis Branch": "Addis Ababa", "Ambo Branch": "Ambo",
    "Airport Branch": "Addis Ababa", "Kazanchis Branch": "Addis Ababa", "Dire Dawa Main Branch": "Dire Dawa",
    "Megenagna Branch": "Addis Ababa", "Sar Bet Branch": "Addis Ababa", "Ayat Branch": "Addis Ababa",
    "Mekane Selam Branch": "Mekane Selam"
}

CITY_TO_REGION = {
    'Addis Ababa': 'Addis Ababa',
    'Bahir Dar': 'Amhara', 'Gondar': 'Amhara', 'Dessie': 'Amhara', 'Woldiya': 'Amhara', 'Debre Markos': 'Amhara', 'Debre Birhan': 'Amhara', 'Lalibela': 'Amhara',
    'Hawassa': 'Sidama', 'Yirgachefe': 'Sidama', 'Dilla': 'SNNPR', 'Arba Minch': 'SNNPR', 'Wolaita Sodo': 'SNNPR', 'Wondogenet': 'SNNPR', 'Mizan Aman': 'SNNPR', 'Wolkite': 'SNNPR',
    'Nekemte': 'Oromia', 'Adama': 'Oromia', 'Asella': 'Oromia', 'Shashemene': 'Oromia', 'Metu': 'Oromia', 'Mojo': 'Oromia', 'Bishoftu': 'Oromia', 'Jimma': 'Oromia', 'Ziway': 'Oromia', 'Ambo': 'Oromia',
    'Dire Dawa': 'Dire Dawa', 'Mekelle': 'Tigray', 'Jijiga': 'Somali', 'Gambella': 'Gambella', 'Harar': 'Harari'
}

NUMERIC_FIELDS = {
    "Age": (18, 100, 35),
    "TenureYears": (0, 40, 5),
    "Balance_ETB": (0, 500000, 50000),
    "NumberOfTransactions": (0, 500, 100),
    "MonthlyServiceFee_ETB": (0, 5000, 200),
}

# -----------------------------
# Globals & Model Loading
# -----------------------------
model = None
model_expected_features: Optional[List[str]] = None
model_threshold: float = 0.5
model_loaded_msg = "No model loaded."

def fix_imputer_attributes(obj):
    if hasattr(obj, "named_steps"):
        for step in obj.named_steps.values():
            fix_imputer_attributes(step)
    if hasattr(obj, "transformers_"):
        for _, trans, _ in obj.transformers_:
            fix_imputer_attributes(trans)
    if hasattr(obj, "transformer_"):
        fix_imputer_attributes(obj.transformer_)
    if hasattr(obj, "remainder"):
        fix_imputer_attributes(obj.remainder)
    if isinstance(obj, impute.SimpleImputer):
        if not hasattr(obj, "_fill_dtype"):
            obj._fill_dtype = object

def load_initial_model():
    global model, model_expected_features, model_threshold, model_loaded_msg
    # Try the bundle first, then the pickle file
    for path in [MODEL_BUNDLE_PATH, "model.pkl"]:
        if os.path.exists(path):
            try:
                bundle = joblib.load(path)
                if isinstance(bundle, dict):
                    model = bundle.get("model") or bundle.get("final_model")
                    model_expected_features = bundle.get("features")
                    # Try to get threshold from metadata or direct key
                    meta = bundle.get("repro_metadata", {})
                    # Set to 0.40 threshold as requested
                    model_threshold = 0.40
                else:
                    model = bundle
                if model is not None:
                    fix_imputer_attributes(model)
                    model_loaded_msg = f"✅ Model loaded from: {path} (Threshold: {model_threshold})"
                    return
            except Exception as e:
                model_loaded_msg = f"❌ Error loading {path}: {str(e)}"
    
    if model is None and not model_loaded_msg.startswith("❌"):
        model_loaded_msg = "⚠️ No model files found in root directory."

load_initial_model()

def load_logo_b64(path: str) -> str:
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

logo_b64 = load_logo_b64(LOGO_PATH)

# -----------------------------
# Logic & Helpers
# -----------------------------
def normalize_input(col: str, value):
    if value is None or (isinstance(value, str) and value.strip() == ""):
        return np.nan
    if isinstance(value, (int, float)):
        return value
    v = str(value).strip()
    if v.lower() in {"yes", "y", "true", "t", "1"}:
        if any(k in col.lower() for k in ("has", "is", "active", "creditcard", "partner")):
            return 1
    if v.lower() in {"no", "n", "false", "f", "0"}:
        if any(k in col.lower() for k in ("has", "is", "active", "creditcard", "partner")):
            return 0
    try:
        return float(v.replace(",", "").replace("ETB", "").strip())
    except:
        return v

def calculate_retention_budget(churn_prob: float, balance: float, transaction_count: float) -> float:
    if churn_prob < 0.3:
        return 0
    base_cost = 500
    risk_multiplier = 1 + (churn_prob - 0.3) * 5
    value_factor = min(balance / 10000, 2.0)
    activity_factor = min(transaction_count / 200, 1.5)
    total = base_cost * risk_multiplier * value_factor * activity_factor
    return min(total, 5000)

def generate_recommendations(ui_data: dict) -> str:
    recommendations = []
    
    age = ui_data.get("Age", 35)
    is_active = ui_data.get("IsActiveMember", 1)
    balance = ui_data.get("Balance_ETB", 50000)
    tenure = ui_data.get("TenureYears", 5)
    transactions = ui_data.get("NumberOfTransactions", 100)
    has_partner = ui_data.get("HasPartner", 1)
    has_mobile = ui_data.get("HasMobileBanking", 1)
    
    if is_active == 0:
        recommendations.append("• Increase engagement: Activate dormant customer with personalized offers")
    
    if balance < 10000:
        recommendations.append("• Boost balance: Offer savings incentives or credit products")
    
    if tenure < 2:
        recommendations.append("• Strengthen loyalty: Introduce onboarding bonuses and welcome benefits")
    
    if transactions < 50:
        recommendations.append("• Promote activity: Encourage daily transactions with rewards program")
    
    if age > 50:
        recommendations.append("• Senior engagement: Tailor retirement/investment products for mature customers")
    
    if has_mobile == 0:
        recommendations.append("• Enable digital banking: Promote mobile app with tutorials and incentives")
    
    if has_partner == 0:
        recommendations.append("• Cross-sell products: Bundle accounts for family members")
    
    if not recommendations:
        recommendations.append("• Customer is well-positioned; maintain regular check-ins")
    
    return "\n".join(recommendations)

def predict_churn(*values) -> Tuple[str, str, str, str, str, str]:
    for i, val in enumerate(values):
        if val is None or (isinstance(val, str) and val.strip() == ""):
            return "⚠️", "<div style='color:#f59e0b; text-align:center; padding:20px;'>Please fill all fields</div>", "—", "—", "—", ""

    # 13-digit Customer ID validation (CustomerID is the first element)
    cust_id = str(values[0]).strip()
    if len(cust_id) != 13 or not cust_id.isdigit():
        return "⚠️", "<div style='color:#ef4444; text-align:center; padding:20px;'><b>Validation Error:</b> Customer ID must be exactly 13 digits.</div>", "—", "—", "—", ""

    # Map inputs to field names (exclude CityType and BranchType from raw inputs if they are UI-only)
    ui_data = {feat: normalize_input(feat, val) for feat, val in zip(DEFAULT_FEATURES, values)}

    # Advanced Feature Engineering (matching Strategic Report)
    city = ui_data.get("City", "Addis Ababa")
    branch = ui_data.get("Branch", "Head Office")
    ui_data["Region"] = CITY_TO_REGION.get(city, "Unknown")
    ui_data["CityType_Detailed"] = CITY_TYPE_MAP.get(city, "District Town")
    ui_data["BranchType_Detailed"] = BRANCH_TYPE_MAP.get(branch, "Standard")
    
    balance = ui_data.get("Balance_ETB", 0) or 0
    fee = ui_data.get("MonthlyServiceFee_ETB", 0) or 0
    ui_data["Fee_Sensitivity"] = fee / (balance + 1)
    
    tenure = ui_data.get("TenureYears", 0) or 0
    # Tenure_Group (bins=[-1, 2, 7, 15, 100])
    if tenure <= 2: ui_data["Tenure_Group"] = "Danger Zone (0-2y)"
    elif tenure <= 7: ui_data["Tenure_Group"] = "Established (2-7y)"
    elif tenure <= 15: ui_data["Tenure_Group"] = "Legacy (7-15y)"
    else: ui_data["Tenure_Group"] = "Veteran (15y+)"

    # TenureGroup (bins=[-1,0,2,5,10,20,100])
    if tenure <= 0: ui_data["TenureGroup"] = "Missing"
    elif tenure <= 2: ui_data["TenureGroup"] = "New"
    elif tenure <= 5: ui_data["TenureGroup"] = "Early"
    elif tenure <= 10: ui_data["TenureGroup"] = "Mid"
    elif tenure <= 20: ui_data["TenureGroup"] = "Loyal"
    else: ui_data["TenureGroup"] = "Veteran"

    # Churn Rates (Placeholder - using global mean ~0.2)
    ui_data["Branch_ChurnRate"] = 0.20
    ui_data["City_ChurnRate"] = 0.20

    if model is None:
        score = 0
        if ui_data.get("Age", 35) > 50: score += 0.2
        if ui_data.get("IsActiveMember") == 0: score += 0.3
        if ui_data.get("Balance_ETB", 0) < 5000: score += 0.2
        
        prob = min(0.95, max(0.05, score))
    else:
        try:
            if model_expected_features:
                row = {f: ui_data.get(f, np.nan) for f in model_expected_features}
                df = pd.DataFrame([row], columns=model_expected_features)
            else:
                df = pd.DataFrame([ui_data])

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(df)
                prob = float(probs[0, 1])
                print(f"DEBUG: Prob={prob:.4f}, Threshold={model_threshold:.4f}")
            else:
                pred = model.predict(df)[0]
                prob = 1.0 if pred == 1 else 0.0
        except Exception as e:
            return "Error", f"<div style='color:#ef4444; text-align:center; padding:20px;'>{str(e)[:100]}</div>", "—", "—", "—", ""

    risk_level = "HIGH" if prob >= model_threshold else "LOW"
    color = "#dc2626" if risk_level == "HIGH" else "#16a34a"
    bg = "#fef2f2" if risk_level == "HIGH" else "#f0fdf4"
    icon = "🔴" if risk_level == "HIGH" else "🟢"
    
    action_text = "Action Required: Contact customer immediately" if risk_level == "HIGH" else "Customer is stable and engaged"
    
    decision_html = f"""
    <div style='padding:25px; border-radius:12px; background:{bg}; border:3px solid {color}; text-align:center;'>
        <h1 style='color:{color}; margin:0; font-size:2.5em;'>{icon} {risk_level}</h1>
        <p style='margin:15px 0 0 0; font-size:1.1em; color:{color};'>{action_text}</p>
    </div>
    """
    
    balance = ui_data.get("Balance_ETB", 0) or 0
    transactions = ui_data.get("NumberOfTransactions", 100) or 100
    retention_budget = calculate_retention_budget(prob, balance, transactions)
    
    if retention_budget == 0:
        budget_text = "No intervention needed"
    else:
        budget_text = f"{retention_budget:,.2f} ETB"
    
    recommendations = generate_recommendations(ui_data)
    
    return f"{prob:.1%}", decision_html, budget_text, risk_level, prob, recommendations

APP_CSS = f"""
    .header {{ background: linear-gradient(135deg, {DASHEN_BLUE} 0%, {DASHEN_LIGHT} 100%); color: white; padding: 30px; border-radius: 15px; margin-bottom: 30px; box-shadow: 0 12px 30px rgba(0,20,60,0.3); }}
    .input-card {{ background: linear-gradient(to bottom, #ffffff, {DASHEN_PALE}); padding: 25px; border-radius: 15px; border: 2px solid {DASHEN_ACCENT}; box-shadow: 0 4px 15px rgba(0,102,204,0.1); }}
    .output-card {{ background: #ffffff; padding: 30px; border-radius: 15px; box-shadow: 0 10px 35px rgba(0,20,60,0.15); border-left: 6px solid {DASHEN_ACCENT}; }}
    .recommendations-card {{ background: linear-gradient(to right, {DASHEN_PALE}, #ffffff); padding: 25px; border-radius: 12px; border-left: 5px solid {DASHEN_ACCENT}; margin-top: 15px; box-shadow: 0 6px 20px rgba(0,102,204,0.08); }}
    .predict-btn {{ background: linear-gradient(135deg, {DASHEN_BLUE} 0%, {DASHEN_ACCENT} 100%) !important; color: white !important; font-weight: bold !important; height: 55px !important; border-radius: 10px !important; font-size: 1.1em !important; box-shadow: 0 6px 20px rgba(0,61,122,0.3) !important; border: none !important; }}
    .predict-btn:hover {{ box-shadow: 0 8px 25px rgba(0,61,122,0.4) !important; }}
    .footer {{ text-align: center; color: {DASHEN_BLUE}; margin-top: 40px; font-size: 0.95em; font-weight: 500; }}
    .stat-box {{ background: {DASHEN_ACCENT}; color: white; padding: 15px; border-radius: 10px; text-align: center; font-weight: bold; }}
    .title-text {{ color: {DASHEN_BLUE}; font-weight: bold; }}
    .recommend-title {{ color: {DASHEN_ACCENT}; font-weight: 700; font-size: 1.1em; margin-bottom: 12px; }}
    #customer-id-input input {{ color: white !important; -webkit-text-fill-color: white !important; }}
"""

with gr.Blocks(title="Dashen Bank Churn Predictor") as demo:

    with gr.Column(elem_classes="header"):
        with gr.Row():
            with gr.Column(scale=1, min_width=120):
                if logo_b64:
                    gr.HTML(f'<img src="data:image/png;base64,{logo_b64}" width="100" style="filter: drop-shadow(0 2px 6px rgba(0,0,0,0.4));">')
            with gr.Column(scale=4):
                gr.HTML(f"""
                    <h1 style='margin:0; font-size: 2.8em; color:white; text-shadow: 0 2px 4px rgba(0,0,0,0.2);'>Dashen Bank</h1>
                    <h2 style='margin:5px 0 0 0; font-size: 1.8em; color:#e8f0ff; font-weight:600;'>Churn Predictor</h2>
                    <p style='margin:10px 0 0 0; font-size:1.2em; font-weight: bold; color: white; text-shadow: 0 1px 2px rgba(0,0,0,0.3);'>Real-time Customer Risk Assessment & Retention Planning</p>
                """)

    with gr.Row(equal_height=False):
        with gr.Column(scale=3):
            with gr.Column(elem_classes="input-card"):
                gr.Markdown(f"## 👤 Customer Information", elem_classes="title-text")
                inputs = {}
                
                with gr.Row():
                    with gr.Column():
                        for feat in DEFAULT_FEATURES[:7]:
                            if feat == "CustomerID":
                                inputs[feat] = gr.Textbox(label=feat, placeholder="Enter 13-digit ID...", elem_id="customer-id-input")
                            elif feat == "Branch":
                                branch_input = gr.Dropdown(choices=CATEGORICAL_FIELDS[feat], label=feat, value=CATEGORICAL_FIELDS[feat][0])
                                inputs[feat] = branch_input
                            elif feat == "City":
                                city_input = gr.Dropdown(choices=CATEGORICAL_FIELDS[feat], label=feat, value=CATEGORICAL_FIELDS[feat][0])
                                inputs[feat] = city_input
                            elif feat in CATEGORICAL_FIELDS:
                                inputs[feat] = gr.Dropdown(choices=CATEGORICAL_FIELDS[feat], label=feat, value=CATEGORICAL_FIELDS[feat][0])
                            elif feat in NUMERIC_FIELDS:
                                lo, hi, default = NUMERIC_FIELDS[feat]
                                inputs[feat] = gr.Number(label=feat, value=default, precision=0)
                            else:
                                inputs[feat] = gr.Textbox(label=feat, placeholder=f"Enter {feat}...")
                
                with gr.Row():
                    with gr.Column():
                        for feat in DEFAULT_FEATURES[7:]:
                            if feat in CATEGORICAL_FIELDS:
                                inputs[feat] = gr.Dropdown(choices=CATEGORICAL_FIELDS[feat], label=feat, value=CATEGORICAL_FIELDS[feat][0])
                            elif feat in NUMERIC_FIELDS:
                                lo, hi, default = NUMERIC_FIELDS[feat]
                                inputs[feat] = gr.Number(label=feat, value=default, precision=0)
                            else:
                                if any(k in feat.lower() for k in ("has", "is", "active")):
                                    inputs[feat] = gr.Dropdown(choices=["Yes", "No"], label=feat, value="Yes")
                                else:
                                    inputs[feat] = gr.Textbox(label=feat, placeholder=f"Enter {feat}...")

                with gr.Row():
                    city_type_display = gr.Textbox(label="City Type (Auto)", value="Federal Capital", interactive=False)
                    branch_type_display = gr.Textbox(label="Branch Tier (Auto)", value="Main", interactive=False)

                def on_branch_change(branch):
                    city = BRANCH_TO_CITY.get(branch, "Addis Ababa")
                    city_type = CITY_TYPE_MAP.get(city, "District Town")
                    branch_type = BRANCH_TYPE_MAP.get(branch, "Standard")
                    return city, city_type, branch_type

                branch_input.change(
                    on_branch_change,
                    inputs=[branch_input],
                    outputs=[city_input, city_type_display, branch_type_display]
                )

                input_list = [inputs[f] for f in DEFAULT_FEATURES]

                with gr.Row():
                    reset_btn = gr.Button("Reset", variant="secondary", scale=1)
                    predict_btn = gr.Button("Assess Churn Risk", variant="primary", scale=2, elem_classes="predict-btn")

        with gr.Column(scale=2):
            with gr.Column(elem_classes="output-card"):
                gr.Markdown(f"## 📊 Risk Assessment", elem_classes="title-text")
                
                decision_out = gr.HTML(
                    f"<div style='text-align:center; padding:50px 20px; color:#9ca3af;'>"
                    f"<p style='font-size:1.2em;'>Adjust customer data and click<br><strong>Assess Churn Risk</strong></p>"
                    f"</div>"
                )
                
                with gr.Group():
                    with gr.Row():
                        prob_out = gr.Textbox(label="Churn Risk Score", interactive=False, scale=1, visible=True)
                        risk_level_out = gr.Textbox(label="Risk Category", interactive=False, scale=1, visible=True)
                
                budget_out = gr.Textbox(label="Recommended Retention Investment", interactive=False, visible=False)
                
                with gr.Column(elem_classes="recommendations-card"):
                    gr.Markdown(f"**💡 Strategic Actions**", elem_classes="recommend-title")
                    recommendations_out = gr.Textbox(
                        label="",
                        value="Customer profile analysis will appear here",
                        interactive=False,
                        lines=5
                    )
                
                with gr.Row():
                    model_status = gr.Textbox(label="System Status", value=model_loaded_msg, interactive=False, scale=1)

    with gr.Accordion("Advanced Options", open=False):
        with gr.Row():
            model_file = gr.File(label="Update Model Bundle (.joblib)")
            model_update_status = gr.Textbox(label="Update Status", interactive=False)
            def update_model(file):
                global model
                try:
                    new_bundle = joblib.load(file.name)
                    model = new_bundle.get("model") if isinstance(new_bundle, dict) else new_bundle
                    if model is not None:
                        fix_imputer_attributes(model)
                    return "Model updated successfully"
                except Exception as e:
                    return f"Error: {str(e)[:80]}"
            model_file.upload(update_model, model_file, model_update_status)

    gr.HTML(f"""
        <div class="footer">
            <p>2026 Dashen Bank PLC | Digital Innovation Division</p>
            <p style="font-size: 0.9em; margin-top:5px;">Enterprise Risk Assessment System v1.0</p>
        </div>
    """)

    prob_hidden = gr.State(0.0)

    predict_btn.click(
        predict_churn,
        inputs=input_list,
        outputs=[prob_out, decision_out, budget_out, risk_level_out, prob_hidden, recommendations_out]
    )

    def reset_inputs():
        res = []
        for feat in DEFAULT_FEATURES:
            if feat in CATEGORICAL_FIELDS:
                res.append(CATEGORICAL_FIELDS[feat][0])
            elif feat in NUMERIC_FIELDS:
                res.append(NUMERIC_FIELDS[feat][2])
            elif any(k in feat.lower() for k in ("has", "is", "active")):
                res.append("Yes")
            else:
                res.append("")
        return res
    
    reset_btn.click(reset_inputs, outputs=input_list)

if __name__ == "__main__":
    demo.launch(share=True, css=APP_CSS)
