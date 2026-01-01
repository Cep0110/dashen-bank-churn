#  Project Overview
This repository contains an end-to-end Machine Learning solution designed to predict customer churn for **Dashen Bank**. By identifying at-risk customers before they leave, the bank can implement proactive retention strategies, optimizing customer lifetime value and reducing financial loss.

###  [ https://huggingface.co/spaces/yani-321212-me/dashen-bank/tree/main)

---

##  Key Features

*   **Real-time Risk Assessment**: Instant churn probability scoring using an advanced RandomForest pipeline.
*   **Business-Optimized Thresholding**: Utilizes a custom **0.40 probability threshold** to balance the cost of false alarms vs. the high cost of customer loss.
*   **ROI-Driven Retention Budget**: Automatically calculates a recommended investment (in ETB) for each at-risk customer based on their balance and transaction activity.
*   **Enterprise-Grade UI**: A professional Gradio dashboard featuring Dashen Bank branding and automated branch-to-city mapping.
*   **Data Privacy**: Implements secure UI masking for sensitive identifiers like Customer IDs.
*   **Strict Validation**: Enforces 13-digit Customer ID validation to ensure data integrity with bank core systems.

---

##  Repository Structure

*   `app.py`: The main Gradio application script including UI logic and compatibility patches.
*   `final_churn_bundle.joblib`: The production-ready model bundle (includes preprocessor, model, and feature metadata).
*   `dashen_bank_final_strategic_report.ipynb`: Comprehensive analysis notebook covering EDA, Feature Engineering, and Model Training.
*   `requirements.txt`: List of dependencies required to run the environment.
*   `dashen_logo.png`: Branding assets for the dashboard.

---

##  Installation & Local Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/dashen-bank-churn.git
   cd dashen-bank-churn
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python app.py
   ```

---
## Model & Business Logic


## Performance Strategy
The model is optimized to minimize the **Total Cost of Churn**. 
- **Cost of False Negative (Missing a churner)**: ~1,000 ETB
- **Cost of False Positive (Unnecessary retention effort)**: ~150 ETB
- **Optimized Threshold**: `0.40` (Adjusted for balanced risk management)

## Advanced Features
The system engineers real-time features including:
- **Tenure Segmentation**: Categorizing customers from "Danger Zone" to "Veteran".
- **Fee Sensitivity**: Analyzing the ratio of service fees to account balance.
- **Urbanization Tiers**: Mapping branches to City Types (Federal Capital, Regional, Zonal).
