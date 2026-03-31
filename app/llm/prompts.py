# app/llm/prompts.py

INVESTIGATOR_SYSTEM_PROMPT = """
You are a senior fraud analyst with 15+ years of experience investigating credit card fraud.

Your task is to analyze the provided ML signals and write a clear, professional case note.
Use only the information given to you. Do not invent any facts or numbers.

Output MUST follow this exact format with these exact section headers:

RISK LEVEL: HIGH
PRIMARY SIGNALS: 
CONTEXT: 
RECOMMENDED ACTION: 

Keep each section short and actionable. Write in natural sentences.
"""

def get_investigator_user_prompt(data: dict) -> str:
    """Format the ML output into a user prompt for the LLM"""
    
    # Format SHAP contributions
    shap_text = "\n".join([
        f"{feature}: {impact:.3f}" 
        for feature, impact in data.get("shap_contributions", [])[:6]
    ])
    
    return f"""
Transaction details:
- Amount: {data.get('amount', 0)}
- Hour of day: {data.get('hour', 0)}

XGBoost Results:
- Fraud Probability: {data.get('xgb_fraud_probability', 0):.4f}

Isolation Forest Results:
- Anomaly Score: {data.get('isolation_forest_anomaly_score', 0):.4f}

Top SHAP Contributions:
{shap_text}

Analyze this transaction and respond **strictly** in the following format:

RISK LEVEL: HIGH
PRIMARY SIGNALS: 
CONTEXT: 
RECOMMENDED ACTION: 
"""

# Role 2: Risk Override Prompts
OVERRIDE_SYSTEM_PROMPT = """
You are a Senior Fraud Decision Specialist. Your role is to determine if a borderline transaction should be ESCALATED to high-risk or CLEARED to low-risk.

You will be presented with a transaction that the ML model is uncertain about (Risk Score: 0.4 - 0.7).
Your decision must be BINARY: Either 'ESCALATE' or 'CLEAR'.

ANALYSIS CRITERIA:
- **ESCALATE**: If the context (unusual hour, account age, etc.) significantly increases the risk profile beyond the base ML signals.
- **CLEAR**: If the context provides a plausible explanation for the anomaly or suggests the user's intent is consistent with a safe transaction.

OUTPUT FORMAT (Strict):
DECISION: [ESCALATE | CLEAR]
REASON: [A concise, one-sentence justification for your decision]
"""

def get_override_user_prompt(ml_output: dict, context: dict) -> str:
    """Format the ML data and extra context for the Override decision."""
    
    return f"""
Borderline Prediction Alert:
ML Fraud Probability: {ml_output.get('xgb_fraud_probability', 0):.2%}
Anomaly Score: {ml_output.get('isolation_forest_anomaly_score', 0):.4f}

Additional Human Context:
- Account Age (Days): {context.get('account_age_days', 'N/A')}
- Is International: {context.get('is_international', 'N/A')}
- Transactions in Last Hour: {context.get('tx_count_last_hour', 'N/A')}
- Time of Transaction (Local): {context.get('local_time', 'N/A')}

Based on the above, provide your final decision.
"""