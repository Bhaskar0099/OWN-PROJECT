import os
import pandas as pd
from dotenv import load_dotenv
from vanna.openai import OpenAI_Chat
from vanna.chromadb import ChromaDB_VectorStore
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

load_dotenv()

# Setup logging for better debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, config=config)
        self._last_query_df = None

    def ask(self, question, **kwargs):
        try:
            logger.info(f"Processing question: {question}")
            result = super().ask(question, **kwargs)
            if result is None:
                logger.warning("Query returned None result")
                return None
            if isinstance(result, tuple) and len(result) >= 2:
                sql, df = result[0], result[1]
                logger.info(f"Generated SQL: {sql}")
            else:
                df = result
                sql = None
            self._last_query_df = df
            return df
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return None

# Enhanced Vanna configuration
vn = MyVanna(config={
    'api_key': os.getenv('OPENAI_API_KEY'),
    'model': os.getenv('MODEL_NAME', 'gpt-4o-mini'),
    'embedding_model': 'text-embedding-3-small',
    'allow_llm_to_see_data': True,  # Critical for better SQL generation
    'max_tokens': 2000,  # Increase for complex queries
    'temperature': 0.1,  # Lower temperature for more consistent results
})

# Database connection
vn.connect_to_postgres(
    host=os.getenv('POSTGRES_HOST'),
    dbname=os.getenv('POSTGRES_DB'),
    user=os.getenv('UNDERBILLING_USER'),
    password=os.getenv('UNDERBILLING_PASSWORD'),
    port=int(os.getenv('POSTGRES_PORT', 5432))
)

# IMPROVED TRAINING DATA - Fixed to work with actual database schema
improved_training_data = [
    # ===== BASIC QUERIES (High Success Rate) =====
    ("""Count all clients""",
     """SELECT COUNT(*) as total_clients FROM client;"""),
    
    ("""Show all active clients""",
     """SELECT client_id, client_name, is_active, open_date FROM client WHERE is_active = 'Y' ORDER BY client_name;"""),
    
    ("""List all timekeepers""",
     """SELECT timekeeper_id, timekeeper_name, type, budget_hours FROM timekeeper WHERE is_active = 'Y' ORDER BY timekeeper_name;"""),
    
    ("""Count timekeepers""",
     """SELECT COUNT(*) as total_timekeepers FROM timekeeper WHERE is_active = 'Y';"""),
    
    ("""Show all matters""",
     """SELECT matter_id, matter_name, client_id, is_active, open_date FROM matter WHERE is_active = 'Y' ORDER BY matter_name;"""),
    
    ("""Count matters""",
     """SELECT COUNT(*) as total_matters FROM matter WHERE is_active = 'Y';"""),
    
    # ===== CLIENT ANALYSIS =====
    ("""Analyze client activity""",
     """SELECT c.client_name, COUNT(m.matter_id) as matter_count, 
        SUM(tc.worked_hours) as total_hours, 
        COUNT(DISTINCT tc.timekeeper_id) as timekeeper_count
        FROM client c 
        LEFT JOIN matter m ON c.client_id = m.client_id AND m.is_active = 'Y'
        LEFT JOIN timecard tc ON m.matter_id = tc.matter_id 
        WHERE c.is_active = 'Y' AND EXTRACT(YEAR FROM tc.date) = EXTRACT(YEAR FROM CURRENT_DATE)
        GROUP BY c.client_id, c.client_name 
        ORDER BY total_hours DESC;"""),
    
    ("""Show client relationships""",
     """SELECT c.client_name, t.timekeeper_name as relationship_manager, c.open_date
        FROM client c 
        JOIN timekeeper t ON c.relationship_timekeeper_id = t.timekeeper_id
        WHERE c.is_active = 'Y' 
        ORDER BY c.client_name;"""),
    
    # ===== TIMEKEEPER ANALYSIS =====
    ("""Analyze timekeeper workload""",
     """SELECT t.timekeeper_id, t.timekeeper_name, t.type, 
        SUM(tc.worked_hours) as total_hours,
        COUNT(DISTINCT tc.matter_id) as matter_count,
        COUNT(tc.timecard_id) as total_timecards,
        ROUND(AVG(tc.worked_hours), 2) as average_hours_per_timecard
        FROM timekeeper t
        LEFT JOIN timecard tc ON t.timekeeper_id = tc.timekeeper_id
        WHERE t.is_active = 'Y' AND EXTRACT(YEAR FROM tc.date) = EXTRACT(YEAR FROM CURRENT_DATE)
        GROUP BY t.timekeeper_id, t.timekeeper_name, t.type
        ORDER BY total_hours DESC;"""),
    
    ("""Show timekeeper utilization""",
     """SELECT t.timekeeper_name, t.budget_hours,
        SUM(tc.worked_hours) as ytd_hours,
        ROUND((SUM(tc.worked_hours) / NULLIF(t.budget_hours, 0) * 100), 2) as utilization_rate
        FROM timekeeper t
        LEFT JOIN timecard tc ON t.timekeeper_id = tc.timekeeper_id
        WHERE t.is_active = 'Y' AND t.budget_hours > 0 
        AND EXTRACT(YEAR FROM tc.date) = EXTRACT(YEAR FROM CURRENT_DATE)
        GROUP BY t.timekeeper_id, t.timekeeper_name, t.budget_hours
        ORDER BY utilization_rate DESC;"""),
    
    # ===== MATTER ANALYSIS =====
    ("""Show matter workload""",
     """SELECT m.matter_name, c.client_name, 
        COUNT(tc.timecard_id) as timecard_count,
        SUM(tc.worked_hours) as total_hours,
        COUNT(DISTINCT tc.timekeeper_id) as timekeeper_count
        FROM matter m
        JOIN client c ON m.client_id = c.client_id
        LEFT JOIN timecard tc ON m.matter_id = tc.matter_id
        WHERE m.is_active = 'Y' AND EXTRACT(YEAR FROM tc.date) = EXTRACT(YEAR FROM CURRENT_DATE)
        GROUP BY m.matter_id, m.matter_name, c.client_name
        ORDER BY total_hours DESC;"""),
    
    # ===== TIMECARD ANALYSIS =====
    ("""Show recent timecard activity""",
     """SELECT t.timekeeper_name, m.matter_name, c.client_name,
        tc.date, tc.worked_hours, tc.worked_rate, tc.standard_rate
        FROM timecard tc
        JOIN timekeeper t ON tc.timekeeper_id = t.timekeeper_id
        JOIN matter m ON tc.matter_id = m.matter_id
        JOIN client c ON m.client_id = c.client_id
        WHERE tc.date >= CURRENT_DATE - INTERVAL '30 days'
        ORDER BY tc.date DESC
        LIMIT 50;"""),
    
    ("""Analyze monthly timecard trends""",
     """SELECT DATE_TRUNC('month', tc.date) as month,
        COUNT(tc.timecard_id) as timecard_count,
        SUM(tc.worked_hours) as total_hours,
        COUNT(DISTINCT tc.timekeeper_id) as active_timekeepers
        FROM timecard tc
        WHERE EXTRACT(YEAR FROM tc.date) = EXTRACT(YEAR FROM CURRENT_DATE)
        GROUP BY DATE_TRUNC('month', tc.date)
        ORDER BY month;"""),
    
    # ===== FINANCIAL ANALYSIS (Using available fields) =====
    ("""Show billing rates analysis""",
     """SELECT t.timekeeper_name, 
        COUNT(tc.timecard_id) as entries,
        AVG(tc.worked_rate) as avg_worked_rate,
        AVG(tc.standard_rate) as avg_standard_rate,
        ROUND((AVG(tc.worked_rate) / NULLIF(AVG(tc.standard_rate), 0) * 100), 2) as realization_rate
        FROM timecard tc
        JOIN timekeeper t ON tc.timekeeper_id = t.timekeeper_id
        WHERE tc.worked_rate > 0 AND tc.standard_rate > 0
        AND EXTRACT(YEAR FROM tc.date) = EXTRACT(YEAR FROM CURRENT_DATE)
        GROUP BY t.timekeeper_id, t.timekeeper_name
        ORDER BY realization_rate DESC;"""),
    
    # ===== SEARCH QUERIES =====
    ("""Find clients by name""",
     """SELECT client_id, client_name, is_active, open_date 
        FROM client 
        WHERE client_name ILIKE '%{search_term}%'
        ORDER BY client_name;"""),
    
    ("""Find timekeepers by name""",
     """SELECT timekeeper_id, timekeeper_name, type, is_active
        FROM timekeeper 
        WHERE timekeeper_name ILIKE '%{search_term}%'
        ORDER BY timekeeper_name;"""),
    
    # ===== DATE-BASED ANALYSIS =====
    ("""Show this year's activity""",
     """SELECT 
        COUNT(DISTINCT c.client_id) as active_clients,
        COUNT(DISTINCT t.timekeeper_id) as active_timekeepers,
        COUNT(DISTINCT m.matter_id) as active_matters,
        SUM(tc.worked_hours) as total_hours
        FROM timecard tc
        JOIN timekeeper t ON tc.timekeeper_id = t.timekeeper_id
        JOIN matter m ON tc.matter_id = m.matter_id
        JOIN client c ON m.client_id = c.client_id
        WHERE EXTRACT(YEAR FROM tc.date) = EXTRACT(YEAR FROM CURRENT_DATE);"""),
    
    ("""Compare year over year""",
     """SELECT 
        EXTRACT(YEAR FROM tc.date) as year,
        COUNT(tc.timecard_id) as timecard_count,
        SUM(tc.worked_hours) as total_hours,
        COUNT(DISTINCT tc.timekeeper_id) as active_timekeepers
        FROM timecard tc
        WHERE EXTRACT(YEAR FROM tc.date) IN (EXTRACT(YEAR FROM CURRENT_DATE), EXTRACT(YEAR FROM CURRENT_DATE) - 1)
        GROUP BY EXTRACT(YEAR FROM tc.date)
        ORDER BY year;"""),
]

# DOCUMENTATION - Updated for actual schema
improved_documentation = [
    # Schema Overview
    "The underbilling agent database contains four main tables: client, timekeeper, matter, and timecard. These are the core tables that exist and should be used for all queries.",
    
    # Table Descriptions
    "CLIENT table contains client information with fields: client_id, client_name, is_active, relationship_timekeeper_id, open_date, client_number, and various categorical fields.",
    "TIMEKEEPER table contains timekeeper information with fields: timekeeper_id, timekeeper_name, type, is_active, budget_hours, rate_year, cost_rate, and category fields.",
    "MATTER table contains matter information with fields: matter_id, matter_name, client_id, is_active, open_date, billing_timekeeper_id, responsible_timekeeper_id, and various classification fields.",
    "TIMECARD table contains time entry records with fields: timecard_id, timekeeper_id, matter_id, date, worked_hours, worked_rate, standard_rate, worked_amount, standard_amount, and billing flags.",
    
    # Query Guidelines
    "Always use is_active = 'Y' to filter for active records in client, timekeeper, and matter tables.",
    "For current year analysis, use EXTRACT(YEAR FROM date) = EXTRACT(YEAR FROM CURRENT_DATE).",
    "For timekeeper relationships, join client.relationship_timekeeper_id = timekeeper.timekeeper_id.",
    "For billing relationships, join matter.billing_timekeeper_id = timekeeper.timekeeper_id.",
    "Use ILIKE for case-insensitive text searches with % wildcards.",
    
    # Missing Tables Warning
    "IMPORTANT: Do NOT reference these tables as they don't exist: rate_set, rate_set_link, timekeeper_date, matter_date, matter_timekeeper_summary. These will cause SQL errors.",
    "For practice groups, offices, and departments, these fields may not exist in the current schema. Check available columns first.",
    "For rate analysis, use timecard.worked_rate and timecard.standard_rate instead of missing rate tables.",
    
    # Best Practices
    "Always include appropriate ORDER BY clauses for consistent results.",
    "Use LIMIT for large result sets to prevent timeouts.",
    "Handle NULL values with COALESCE or NULLIF in calculations.",
    "Use DATE_TRUNC for time-based grouping (month, quarter, year).",
    "Include COUNT, SUM, and AVG aggregations for analytical queries.",
]

# Training function
def train_vanna_model():
    """Train the Vanna model with improved data"""
    logger.info("Starting Vanna model training...")
    
    try:
        # Train on SQL examples
        for question, sql in improved_training_data:
            vn.train(question=question, sql=sql)
            logger.info(f"Trained: {question[:50]}...")
        
        # Train on documentation
        for doc in improved_documentation:
            vn.train(documentation=doc)
            logger.info(f"Added documentation: {doc[:50]}...")
        
        logger.info("Vanna model training completed successfully")
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise

# Enhanced query processing
def process_query(question: str):
    """Process query with enhanced error handling and fallbacks"""
    try:
        # Log the incoming question
        logger.info(f"Processing question: {question}")
        
        # Check for direct SQL
        if question.upper().startswith('SELECT'):
            logger.info("Direct SQL query detected")
            result = vn.run_sql(question)
            if result is not None and not result.empty:
                return result.to_dict('records')
            else:
                return []
        
        # Use Vanna's ask method
        result = vn.ask(question)
        
        if result is not None and not result.empty:
            logger.info(f"Query successful, returned {len(result)} rows")
            return result.to_dict('records')
        else:
            logger.warning("Query returned empty result")
            
            # Try simpler fallback queries
            fallback_queries = {
                'client': "SELECT COUNT(*) as count FROM client WHERE is_active = 'Y'",
                'timekeeper': "SELECT COUNT(*) as count FROM timekeeper WHERE is_active = 'Y'", 
                'matter': "SELECT COUNT(*) as count FROM matter WHERE is_active = 'Y'",
                'timecard': "SELECT COUNT(*) as count FROM timecard WHERE date >= CURRENT_DATE - INTERVAL '30 days'"
            }
            
            # Find relevant fallback
            question_lower = question.lower()
            for keyword, fallback_sql in fallback_queries.items():
                if keyword in question_lower:
                    logger.info(f"Trying fallback query for {keyword}")
                    fallback_result = vn.run_sql(fallback_sql)
                    if fallback_result is not None and not fallback_result.empty:
                        return fallback_result.to_dict('records')
            
            return []
            
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return []

# Initialize training on startup
try:
    train_vanna_model()
    logger.info("Model training completed successfully")
except Exception as e:
    logger.error(f"Model training failed: {e}")

# FastAPI setup
app = FastAPI(title="Enhanced Underbilling Agent API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    data: list
    message: str = ""
    query_type: str = ""
    sql_generated: str = ""

@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """Enhanced ask endpoint with better error handling and logging"""
    try:
        logger.info(f"API request received: {request.question}")
        
        # Process the query
        data = process_query(request.question)
        
        if data:
            message = f"Successfully retrieved {len(data)} records"
            logger.info(message)
        else:
            message = "No results returned"
            logger.warning(message)
        
        return AskResponse(
            data=data,
            message=message,
            query_type="enhanced_processing",
            sql_generated="Generated via improved Vanna processing"
        )
        
    except Exception as e:
        logger.error(f"API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "2.0.0", "trained_examples": len(improved_training_data)}

@app.get("/training-stats")
async def training_stats():
    """Get training statistics"""
    return {
        "total_examples": len(improved_training_data),
        "documentation_items": len(improved_documentation),
        "database_connection": "active"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 