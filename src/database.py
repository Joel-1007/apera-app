import sqlite3
import json
from datetime import datetime

DB_NAME = "apera_production.db"

def init_db():
    """Initialize SQLite database for persistent logging"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    # Create table with all necessary columns
    c.execute('''CREATE TABLE IF NOT EXISTS audit_logs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  session_id TEXT,
                  user_query TEXT,
                  response TEXT,
                  intent TEXT,
                  is_toxic INTEGER,
                  toxicity_score REAL,
                  rating TEXT)''')
    conn.commit()
    conn.close()

def log_interaction(session_id, query, response, meta, safety):
    """Log a chat interaction to the database"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    is_toxic = 1 if not safety.get('is_safe', True) else 0
    # Extract score if available, else default to high if toxic
    tox_score = 0.0 
    if is_toxic: tox_score = 0.95
    
    # Safe extraction of intent
    intent = meta.get('intent', 'UNKNOWN') if meta else 'UNKNOWN'
    
    c.execute("INSERT INTO audit_logs (timestamp, session_id, user_query, response, intent, is_toxic, toxicity_score, rating) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
              (str(datetime.now()), session_id, query, str(response), intent, is_toxic, tox_score, 'none'))
    conn.commit()
    conn.close()

def update_feedback(query, rating):
    """Updates the rating (positive/negative) for a specific query"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    # Update the most recent matching query that has no rating
    c.execute("UPDATE audit_logs SET rating = ? WHERE user_query = ? AND rating = 'none'", (rating, query))
    conn.commit()
    conn.close()

def get_toxicity_trends():
    """Returns data for Toxicity Trend Graphs"""
    conn = sqlite3.connect(DB_NAME)
    # Group by Hour (simple string slicing for YYYY-MM-DD HH)
    query = """
        SELECT substr(timestamp, 1, 13) as hour, sum(is_toxic) as toxic_count
        FROM audit_logs 
        GROUP BY hour 
        ORDER BY hour ASC
    """
    try:
        data = conn.execute(query).fetchall()
    except:
        data = []
    conn.close()
    return [{"time": d[0], "toxic_count": d[1]} for d in data]

def get_session_logs(session_id=None):
    """Returns data for Session-level Audit View"""
    conn = sqlite3.connect(DB_NAME)
    if session_id:
        data = conn.execute("SELECT * FROM audit_logs WHERE session_id=? ORDER BY id DESC", (session_id,)).fetchall()
    else:
        data = conn.execute("SELECT * FROM audit_logs ORDER BY id DESC LIMIT 100").fetchall()
    conn.close()
    
    # Convert tuple to dict for JSON response
    logs = []
    for d in data:
        logs.append({
            "timestamp": d[1],
            "session": d[2],
            "query": d[3],
            "intent": d[5],
            "toxic": d[6],
            "rating": d[8]
        })
    return logs
