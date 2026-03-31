from app.core.config import settings
from .client import get_groq_client
from .prompts import INVESTIGATOR_SYSTEM_PROMPT, get_investigator_user_prompt

def investigate_transaction(ml_output: dict) -> dict:
    """Main function to get case note from LLM"""
    
    client = get_groq_client()
    
    # Prepare prompt
    user_prompt = get_investigator_user_prompt(ml_output)
    
    # Call LLM
    response = client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=[
            {"role": "system", "content": INVESTIGATOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.15,
        max_tokens=650
    )
    
    case_note = response.choices[0].message.content.strip()
    
    return {
        "case_note": case_note
    }

if __name__ == "__main__":
    # Sample data for direct testing
    sample_ml_output = {
        "amount": 120.50,
        "hour": 14,
        "xgb_fraud_probability": 0.85,
        "isolation_forest_anomaly_score": -0.15,
        "shap_contributions": [("V14", 0.5), ("V12", 0.3), ("Amount", 0.1)]
    }
    
    print("Testing LLM Investigation locally...\n")
    try:
        result = investigate_transaction(sample_ml_output)
        print("INVESTIGATION RESULT:")
        print(result["case_note"])
    except Exception as e:
        print(f"Error during investigation: {e}")