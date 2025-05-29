import os
import sqlite3
import pandas as pd
from dotenv import load_dotenv
from vanna.openai import OpenAI_Chat
from vanna.chromadb import ChromaDB_VectorStore
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import tempfile

load_dotenv()

def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'data', 'updated_synthetic_legal_billing_data copy1.csv')
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, 'legal_billing.db')
    df = pd.read_csv(csv_path)
    conn = sqlite3.connect(db_path)
    df.to_sql('legal_billing', conn, if_exists='replace', index=False)
    conn.close()
    return db_path

class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, config=config)
        self._last_query_df = None

    def ask(self, question, **kwargs):
        sql, df, _ = super().ask(question, **kwargs)
        self._last_query_df = df
        return df

vn = MyVanna(config={
    'api_key': os.getenv('OPENAI_API_KEY'),
    'model': os.getenv('MODEL_NAME', 'gpt-4o-mini'),
    'embedding_model': 'text-embedding-3-small'
})

db_path = load_data()
vn.connect_to_sqlite(db_path)

schema_sql = "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL"
df_ddl = vn.run_sql(schema_sql)
for ddl in df_ddl['sql'].dropna():
    vn.train(ddl=ddl)

    profitability_risk_sql_training = [
    (
        """
        SELECT practice_area,
               ROUND(AVG(discount_amount / NULLIF(original_amount, 0) * 100), 2) AS avg_discount_pct,
               ROUND(AVG(realized_amount / NULLIF(billed_amount, 0) * 100), 2) AS avg_realization_pct,
               ROUND(AVG(collection_days), 2) AS avg_collection_days
        FROM legal_billing
        GROUP BY practice_area
        HAVING avg_discount_pct > 25 OR avg_realization_pct < 80
        ORDER BY avg_discount_pct DESC, avg_realization_pct ASC;
        """,
        "Show practice areas with high profitability risks based on discounts or realization rates."
    ),
    (
        """
        SELECT practice_area,
               SUM(discount_amount) AS total_discount,
               SUM(billed_amount) AS total_billed,
               ROUND(SUM(discount_amount) / NULLIF(SUM(billed_amount) + SUM(discount_amount), 0) * 100, 2) AS discount_pct
        FROM legal_billing
        GROUP BY practice_area
        HAVING discount_pct > 25
        ORDER BY discount_pct DESC;
        """,
        "Which practice areas have an average discount percentage greater than 25%?"
    ),
    (
        """
        SELECT practice_area,
               ROUND(AVG(realized_amount / NULLIF(billed_amount, 0) * 100), 2) AS avg_realization_pct
        FROM legal_billing
        GROUP BY practice_area
        HAVING avg_realization_pct < 80
        ORDER BY avg_realization_pct ASC;
        """,
        "List practice areas with average realization rates below 80%."
    ),
    (
        """
        SELECT practice_area,
               ROUND(AVG(collection_days), 2) AS avg_collection_days
        FROM legal_billing
        GROUP BY practice_area
        ORDER BY avg_collection_days DESC;
        """,
        "What is the average collection time for each practice area?"
    ),
    (
        """
        SELECT practice_area,
               SUM(write_off_amount) AS total_write_off,
               ROUND(AVG(write_off_amount / NULLIF(billed_amount, 0) * 100), 2) AS avg_write_off_pct
        FROM legal_billing
        GROUP BY practice_area
        HAVING avg_write_off_pct > 10
        ORDER BY total_write_off DESC;
        """,
        "Identify practice areas with high write-off amounts relative to billed amounts."
    ),
    (
        """
        SELECT practice_area,
               COUNT(*) AS matter_count,
               ROUND(AVG(discount_amount), 2) AS avg_discount
        FROM legal_billing
        WHERE matter_date >= date('now', '-1 year')
        GROUP BY practice_area
        ORDER BY avg_discount DESC;
        """,
        "Show the number of matters and average discount for each practice area in the last year."
    ),
    (
        """
        SELECT practice_area,
               ROUND(AVG(realized_amount / NULLIF(billed_amount, 0) * 100), 2) AS avg_realization_pct,
               ROUND(AVG(collection_days), 2) AS avg_collection_days
        FROM legal_billing
        WHERE lawyer_role = 'Partner'
        GROUP BY practice_area
        HAVING avg_realization_pct < 80;
        """,
        "Which practice areas have low realization rates for matters handled by Partners?"
    ),
    (
        """
        SELECT client_id,
               practice_area,
               ROUND(SUM(discount_amount) / NULLIF(SUM(original_amount), 0) * 100, 2) AS discount_pct
        FROM legal_billing
        GROUP BY client_id, practice_area
        HAVING discount_pct > 25
        ORDER BY discount_pct DESC;
        """,
        "List clients with high discount percentages by practice area."
    )
]

    # Documentation training examples
    profitability_risk_documentation = [
    "Profitability risks in legal billing are identified by high discount percentages (discount_amount / original_amount > 25%), low realization rates (realized_amount / billed_amount < 80%), or extended collection times (high collection_days).",
    "The 'practice_area' column categorizes matters into areas like Litigation, Corporate, IP, Tax, or Employment, and is used to group data for profitability analysis.",
    "The 'original_amount' is the initial amount before discounts, used to calculate discount percentages.",
    "The 'discount_amount' is the amount reduced from the original_amount, critical for identifying high-discount risks.",
    "The 'billed_amount' is the amount invoiced to the client (original_amount - discount_amount), used in realization rate calculations.",
    "The 'realized_amount' is the amount actually collected, used to compute realization rates (realized_amount / billed_amount).",
    "The 'collection_days' column indicates the number of days to collect payment, with higher values signaling potential cash flow risks.",
    "The 'write_off_amount' is the uncollected amount written off, indicating financial loss in a practice area.",
    "The 'matter_date' is the billing date, useful for time-based filtering (e.g., last year’s data).",
    "The 'client_id' identifies the client, allowing analysis of client-specific profitability risks.",
    "The 'lawyer_role' indicates the lawyer’s role (e.g., Partner, Associate), useful for role-based profitability analysis.",
    "The 'firm_specific_code' is a unique matter identifier used internally by the firm.",
    "When analyzing profitability, focus on patterns across practice areas, such as consistently high discounts or low realization rates, to identify systemic risks.",
    "Compare discount percentages and realization rates against thresholds (25% for discounts, 80% for realization) to flag risky practice areas.",
    "Long collection days may indicate client payment issues or inefficiencies in billing processes.",
]

for sql, question in profitability_risk_sql_training:
    vn.train(sql=sql, question=question)

for doc in profitability_risk_documentation:
    vn.train(documentation=doc)

app = FastAPI(title="Underbilling Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    question: str

@app.post("/ask")
def ask(request: AskRequest):
    df = vn.ask(request.question)
    if df is not None and not df.empty:
        return {"data": df.to_dict(orient="records")}
    return {"data": [], "message": "No results returned"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8084)