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

class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, config=config)
        self._last_query_df = None

    def ask(self, question, **kwargs):
        result = super().ask(question, **kwargs)
        if result is None:
            return None
        if isinstance(result, tuple) and len(result) >= 2:
            sql, df = result[0], result[1]
        else:
            df = result
            sql = None
        self._last_query_df = df
        return df

vn = MyVanna(config={
    'api_key': os.getenv('OPENAI_API_KEY'),
    'model': os.getenv('MODEL_NAME', 'gpt-4o-mini'),
    'embedding_model': 'text-embedding-3-small'
})

vn.connect_to_postgres(
    host=os.getenv('POSTGRES_HOST'),
    dbname=os.getenv('POSTGRES_DB'),
    user=os.getenv('PROFITABILITY_USER'),
    password=os.getenv('PROFITABILITY_PASSWORD'),
    port=int(os.getenv('POSTGRES_PORT', 5432))
)

profitability_risk_sql_training = [
    (
        """
        SELECT "Matter Practice Group",
               ROUND(SUM("Discount Amount") / NULLIF(SUM("Billed Amount") + SUM("Discount Amount"), 0) * 100, 2) AS discount_pct,
               ROUND(SUM("Collected Amount") / NULLIF(SUM("Billed Amount"), 0) * 100, 2) AS realization_pct
        FROM legal_billing
        GROUP BY "Matter Practice Group"
        HAVING discount_pct > 25 OR realization_pct < 80
        ORDER BY discount_pct DESC, realization_pct ASC;
        """,
        "Which matter practice groups have high discount percentages or low realization rates?"
    ),
    (
        """
        SELECT "Client ID", "Client Name",
               ROUND(SUM("Discount Amount") / NULLIF(SUM("Billed Amount"), 0) * 100, 2) AS client_discount_pct
        FROM legal_billing
        GROUP BY "Client ID", "Client Name"
        HAVING client_discount_pct > 25
        ORDER BY client_discount_pct DESC;
        """,
        "List clients with discount percentages higher than 25%."
    ),
    (
        """
        SELECT "Matter Office", 
               ROUND(SUM("Write Off Amount") / NULLIF(SUM("Billed Amount"), 0) * 100, 2) AS write_off_pct
        FROM legal_billing
        GROUP BY "Matter Office"
        HAVING write_off_pct > 10
        ORDER BY write_off_pct DESC;
        """,
        "Which offices have high write-off percentages?"
    ),
    (
        """
        SELECT "Matter Department", 
               ROUND(SUM("Collected Amount") / NULLIF(SUM("Billed Amount"), 0) * 100, 2) AS realization_rate
        FROM legal_billing
        GROUP BY "Matter Department"
        ORDER BY realization_rate ASC;
        """,
        "Show departments with lowest realization rates."
    ),
    (
        """
        SELECT "Timekeeper Role",
               ROUND(AVG("Effective Rate"), 2) AS avg_effective_rate
        FROM legal_billing
        JOIN timekeeper_details USING ("Timekeeper ID")
        GROUP BY "Timekeeper Role"
        ORDER BY avg_effective_rate DESC;
        """,
        "What is the average effective rate by timekeeper role?"
    ),
    (
        """
        SELECT "Matter Type",
               COUNT(*) AS matter_count,
               ROUND(AVG("Billed Hours"), 2) AS avg_billed_hours,
               ROUND(AVG("Worked Hours"), 2) AS avg_worked_hours
        FROM legal_billing
        GROUP BY "Matter Type"
        ORDER BY matter_count DESC;
        """,
        "Which matter types are most common, and what are their average billed and worked hours?"
    ),
    (
        """
        SELECT "Billing Attorney",
               ROUND(AVG("Effective Rate"), 2) AS avg_effective_rate
        FROM legal_billing
        GROUP BY "Billing Attorney"
        ORDER BY avg_effective_rate DESC
        LIMIT 10;
        """,
        "Who are the top 10 billing attorneys with the highest average effective rates?"
    ),
    (
        """
        SELECT "Matter ID", "Matter Name", 
               "Billed Amount", "Collected Amount", 
               ROUND("Collected Amount" / NULLIF("Billed Amount", 0) * 100, 2) AS realization_pct
        FROM legal_billing
        ORDER BY realization_pct ASC
        LIMIT 10;
        """,
        "Which matters have the lowest realization rates?"
    )
]

    # Documentation training examples
profitability_risk_documentation = [
    "Profitability is affected by high discount percentages and low realization rates, derived from billed and collected amounts.",
    "The 'Discount Amount' column reflects revenue lost due to fee reductions.",
    "The 'Billed Amount' is the amount invoiced to the client before collections or write-offs.",
    "The 'Collected Amount' shows what was actually received, helping calculate realization rate.",
    "The 'Write Off Amount' indicates uncollected revenue that was written off.",
    "The 'Effective Rate' reflects the true billing rate after discounts and adjustments, used to evaluate profitability per timekeeper or attorney.",
    "Grouping by 'Matter Practice Group' or 'Matter Department' helps identify which areas are more or less profitable.",
    "The 'Timekeeper Role' and 'Timekeeper Practice Group' provide insight into performance by staff type or group.",
    "Client-level analysis using 'Client ID' and 'Client Name' helps identify accounts with frequent underperformance.",
    "Attorney-specific fields like 'Billing Attorney' or 'Responsible Attorney' can highlight who is managing profitable (or unprofitable) matters.",
    "Realization Rate = Collected Amount / Billed Amount, and it's a key metric for profitability.",
    "Discount Percentage = Discount Amount / (Billed Amount + Discount Amount), showing how much is written off up front."
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
    uvicorn.run(app, host="0.0.0.0", port=8000)