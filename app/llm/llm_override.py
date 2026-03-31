import sqlite3
import json
import os
from datetime import datetime
from app.core.config import settings
from app.llm.client import get_groq_client
from app.llm.prompts import OVERRIDE_SYSTEM_PROMPT, get_override_user_prompt

class RiskOverrideManager:
    """
    Role 2: Risk Override Manager
    Evaluates borderline ML predictions (0.4 - 0.7) using LLM reasoning 
    and logs decisions to a SQLite audit trail.
    """
    
    def __init__(self):
        self.db_path = settings.DATABASE_PATH
        self.client = get_groq_client()
        self._init_db()

    def _init_db(self):
        """Ensure the audit table exists using the schema definition."""
        schema_path = os.path.join(settings.PROJECT_ROOT, "app", "schemas", "llm_db", "schema.sql")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                if os.path.exists(schema_path):
                    with open(schema_path, "r") as f:
                        conn.executescript(f.read())
                else:
                    # Fallback if schema file is missing
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS llm_decisions (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            ml_score REAL NOT NULL,
                            verdict TEXT NOT NULL,
                            reason TEXT NOT NULL,
                            context_signals TEXT,
                            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                        );
                    """)
        except Exception as e:
            print(f"Database initialization failed: {e}")

    def should_override(self, probability: float) -> bool:
        """Check if the probability falls into the 'uncertain zone'."""
        return settings.UNCERTAIN_ZONE_LOW <= probability <= settings.UNCERTAIN_ZONE_HIGH

    def process_override(self, ml_output: dict, context: dict) -> dict:
        """
        Runs the LLM override analysis and logs the decision.
        Returns a dictionary with the decision and reason.
        """
        prob = ml_output.get("xgb_fraud_probability", 0.0)
        
        # 1. Double check threshold (Safety gate)
        if not self.should_override(prob):
            return {
                "decision": "NO_OVERRIDE_REQUIRED",
                "reason": "Probability outside uncertain zone.",
                "was_overridden": False
            }

        # 2. Get LLM Decision
        user_prompt = get_override_user_prompt(ml_output, context)
        
        try:
            response = self.client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=[
                    {"role": "system", "content": OVERRIDE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0  # High precision for binary decisions
            )
            
            content = response.choices[0].message.content.strip()
            
            # Simple parsing for DECISION: and REASON:
            decision = "CLEAR"
            reason = "No reason provided by LLM."
            
            for line in content.split("\n"):
                if line.startswith("DECISION:"):
                    decision = line.replace("DECISION:", "").strip().upper()
                elif line.startswith("REASON:"):
                    reason = line.replace("REASON:", "").strip()

            # 3. Log to Database
            self._log_decision(prob, decision, reason, context)

            return {
                "decision": decision,
                "reason": reason,
                "was_overridden": True
            }

        except Exception as e:
            print(f"LLM Override failed: {e}")
            return {
                "decision": "ERROR",
                "reason": str(e),
                "was_overridden": False
            }

    def _log_decision(self, score: float, verdict: str, reason: str, context: dict):
        """Persistent audit logging of the LLM decision."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO llm_decisions (ml_score, verdict, reason, context_signals) VALUES (?, ?, ?, ?)",
                    (score, verdict, reason, json.dumps(context))
                )
        except Exception as e:
            print(f"Failed to log decision: {e}")

if __name__ == "__main__":
    # Test Block
    manager = RiskOverrideManager()
    
    test_ml = {"xgb_fraud_probability": 0.55, "isolation_forest_anomaly_score": -0.1}
    test_ctx = {
        "account_age_days": 10,
        "is_international": True,
        "tx_count_last_hour": 5,
        "local_time": "02:30 AM"
    }
    
    print(f"Testing Override for score: {test_ml['xgb_fraud_probability']}")
    result = manager.process_override(test_ml, test_ctx)
    print(json.dumps(result, indent=2))
