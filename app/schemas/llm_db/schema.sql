-- app/schemas/llm_db/schema.sql

CREATE TABLE IF NOT EXISTS llm_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ml_score REAL NOT NULL,
    verdict TEXT NOT NULL CHECK(verdict IN ('ESCALATE', 'CLEAR')),
    reason TEXT NOT NULL,
    context_signals TEXT, -- JSON representation of extra signals
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
