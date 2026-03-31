import sqlite3
import json
import re
from app.core.config import settings
from app.llm.client import get_groq_client

# -- ROLE 3: NATURAL LANGUAGE QUERY ENGINE --

QUERY_SYSTEM_PROMPT = """
You are a Database Expert and Fraud Analyst. Your role is to translate a user's plain-English question into a single, valid, read-only SQL query.

DATABASE SCHEMA:
The table is named 'llm_decisions'.
Columns:
- id (INTEGER): Primary Key
- ml_score (REAL): The XGBoost fraud probability (0.0 to 1.0)
- verdict (TEXT): The LLM's final decision ('ESCALATE' or 'CLEAR')
- reason (TEXT): The textual justification for the decision
- context_signals (TEXT): JSON string containing 'account_age_days', 'is_international', etc.
- timestamp (DATETIME): Automatically set when the record is created.

RULES:
1. ONLY return the SQL query if the user asks for data. 
2. Use the 'run_readonly_query' tool to execute the SQL.
3. Once you receive the tool results, PROVIDE a human-friendly interpretation of the data. 
4. DO NOT attempt to write anything but SELECT statements.
"""

def run_readonly_query(sql_query: str):
    """
    Executes a read-only SQL query against the llm_decisions database.
    Includes a strict security filter to allow only SELECT statements.
    """
    # Security Filter: Regex to ensure the query starts with SELECT and contains no forbidden words
    clean_query = sql_query.strip().lower()
    
    # Must start with select
    if not clean_query.startswith("select"):
        return {"error": "SECURITY_VIOLATION: Only SELECT statements are allowed."}
    
    # Forbidden keywords for write operations
    forbidden = ["drop", "delete", "update", "insert", "alter", "truncate", "create", "base"]
    if any(word in clean_query for word in forbidden):
        return {"error": f"SECURITY_VIOLATION: Forbidden keyword detected."}

    try:
        with sqlite3.connect(settings.DATABASE_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(sql_query)
            rows = cursor.fetchall()
            
            # Convert rows to list of dicts for LLM processing
            results = [dict(row) for row in rows]
            return results
    except Exception as e:
        return {"error": str(e)}

class NaturalLanguageQueryEngine:
    """
    Orchestrates the conversion of English -> SQL -> Data -> English.
    Uses Groq's tool-calling for a reliable self-correcting loop.
    """
    
    def __init__(self):
        self.client = get_groq_client()
        self.tools = [{
            "type": "function",
            "function": {
                "name": "run_readonly_query",
                "description": "Execute a SQL SELECT query against the llm_decisions table.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sql_query": {
                            "type": "string",
                            "description": "The exact SQL query to execute (e.g., SELECT count(*) FROM llm_decisions)"
                        }
                    },
                    "required": ["sql_query"]
                }
            }
        }]

    def get_query_interpretation(self, user_question: str) -> str:
        """
        Processes a natural language question and returns a interpreted response.
        """
        messages = [
            {"role": "system", "content": QUERY_SYSTEM_PROMPT},
            {"role": "user", "content": user_question}
        ]

        try:
            # 1. Initial Call: Generate SQL (LLM decides to call a tool)
            response = self.client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
                temperature=0.0
            )
            
            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls

            if tool_calls:
                # 2. Execution Phase: Run the tool
                messages.append(response_message)
                
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    print(f"DEBUG: Executing SQL --> {function_args.get('sql_query')}")
                    
                    if function_name == "run_readonly_query":
                        tool_result = run_readonly_query(function_args.get("sql_query"))
                        
                        messages.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": json.dumps(tool_result)
                        })

                # 3. Final Step: LLM interprets the result
                second_response = self.client.chat.completions.create(
                    model=settings.LLM_MODEL,
                    messages=messages,
                    temperature=0.7
                )
                return second_response.choices[0].message.content.strip()
            
            else:
                return response_message.content.strip()

        except Exception as e:
            return f"Query engine error: {str(e)}"

if __name__ == "__main__":
    # Test Block
    engine = NaturalLanguageQueryEngine()
    
    questions = [
        "How many total records are there in the database?",
        "What is the average ML score for transactions marked as 'ESCALATE'?",
        "Show me all transactions that happened between a score of 0.5 and 0.6"
    ]
    
    print("🚀 Natural Language Query Engine Test\n")
    for q in questions:
        print(f"QUESTION: {q}")
        print(f"ANSWER: {engine.get_query_interpretation(q)}\n" + "-"*30)
