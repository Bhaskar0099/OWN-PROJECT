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
    user=os.getenv('POSTGRES_USER'),
    password=os.getenv('POSTGRES_PASSWORD'),
    port=int(os.getenv('POSTGRES_PORT', 5432))
)

underbilling_sql_training = [
    (
        "SELECT lb.\"Timekeeper Name\", SUM(lb.\"Write Off Amount\") AS total_write_off "
        "FROM legal_billing lb "
        "JOIN timekeeper_details td ON lb.\"Timekeeper ID\" = td.\"Timekeeper ID\" "
        "WHERE lb.\"Invoice Date\" >= current_date - interval '3 months' "
        "GROUP BY lb.\"Timekeeper Name\" "
        "HAVING SUM(lb.\"Write Off Amount\") > 1000 "
        "ORDER BY total_write_off DESC;",
        "Which timekeepers have a total write-off amount greater than $1000 in the last quarter?"
    ),
    (
        "SELECT td.\"Timekeeper Role\", AVG(lb.\"Realization Rate\") AS avg_realization "
        "FROM legal_billing lb "
        "JOIN timekeeper_details td ON lb.\"Timekeeper ID\" = td.\"Timekeeper ID\" "
        "GROUP BY td.\"Timekeeper Role\";",
        "What is the average realization rate for each timekeeper role?"
    ),
    (
        "SELECT lb.\"Matter Name\", lb.\"Billed Hours\", lb.\"Worked Hours\" "
        "FROM legal_billing lb "
        "WHERE lb.\"Billed Hours\" < 0.5 * lb.\"Worked Hours\";",
        "List matters where billed hours are less than 50% of worked hours."
    ),
    (
        "SELECT lb.\"Client Name\", AVG(lb.\"Discount Amount\") AS avg_discount "
        "FROM legal_billing lb "
        "GROUP BY lb.\"Client Name\" "
        "HAVING AVG(lb.\"Discount Amount\") > 500;",
        "Which clients have an average discount amount greater than $500?"
    ),
    (
        "SELECT lb.\"Timekeeper Name\", COUNT(*) AS underbilled_matters "
        "FROM legal_billing lb "
        "WHERE lb.\"Underbilling Flag\" = 'Yes' "
        "GROUP BY lb.\"Timekeeper Name\";",
        "How many matters have an underbilling flag set to 'Yes' for each timekeeper?"
    ),
    (
        "SELECT lb.\"Matter Type\", AVG(lb.\"Write Off Amount\" + lb.\"Discount Amount\") AS avg_underbilling "
        "FROM legal_billing lb "
        "GROUP BY lb.\"Matter Type\";",
        "What is the average underbilling amount for each matter type?"
    ),
    (
        "SELECT lb.\"Timekeeper Name\", AVG(lb.\"Utilization Rate\") AS avg_utilization "
        "FROM legal_billing lb "
        "GROUP BY lb.\"Timekeeper Name\" "
        "HAVING AVG(lb.\"Utilization Rate\") < 50;",
        "Which timekeepers have an average utilization rate below 50?"
    ),
    (
        "SELECT lb.\"Timekeeper Name\", "
        "COUNT(CASE WHEN lb.\"Underbilling Flag\" = 'Yes' THEN 1 END) AS underbilling_count, "
        "AVG(lb.\"Realization Rate\") AS avg_realization "
        "FROM legal_billing lb "
        "GROUP BY lb.\"Timekeeper Name\" "
        "HAVING COUNT(CASE WHEN lb.\"Underbilling Flag\" = 'Yes' THEN 1 END) > 5 "
        "AND AVG(lb.\"Realization Rate\") < 80;",
        "List timekeepers who have both a high number of underbilling flags and a low average realization rate."
    ),
    (
        "SELECT lb.\"Matter Name\", lb.\"Effective Rate\", lb.\"Standard Rate\", lb.\"Time Entry Hours\" "
        "FROM legal_billing lb "
        "WHERE lb.\"Effective Rate\" < 0.8 * lb.\"Standard Rate\" "
        "AND lb.\"Time Entry Hours\" > 5;",
        "Identify matters where the effective rate is less than 80% of the standard rate and the time entry hours are greater than 5."
    ),
    (
        "SELECT lb.\"Matter Name\", lb.\"Billed Amount\", lb.\"Collected Amount\" "
        "FROM legal_billing lb "
        "WHERE lb.\"Collected Amount\" < 0.9 * lb.\"Billed Amount\";",
        "Which matters have a collected amount less than 90% of the billed amount?"
    ),
    (
        "SELECT lb.\"Timekeeper Name\", "
        "AVG(lb.\"Billed Hours\" / NULLIF(lb.\"Time Entry Hours\", 0)) AS avg_ratio "
        "FROM legal_billing lb "
        "GROUP BY lb.\"Timekeeper Name\";",
        "What is the average ratio of billed hours to time entry hours for each timekeeper?"
    )
]

underbilling_documentation = [
    "Underbilling occurs when the billed amount is less than the expected amount based on the time worked or standard rates. This can be due to discounts, write-offs, or other adjustments.",
    "The 'Write Off Amount' column indicates the amount that was not billed to the client, which can be a sign of underbilling.",
    "The 'Realization Rate' is the percentage of the billed amount that was actually collected, which can indicate underbilling if it’s consistently low.",
    "The 'Underbilling Flag' is a direct indicator of whether underbilling was detected for a particular matter.",
    "The 'Timekeeper Role' can be used to analyze if certain roles (e.g., Partner, Paralegal) are more prone to underbilling.",
    "The 'Matter Type' might influence underbilling likelihood, as some types (e.g., Litigation) may have more variable billing.",
    "The 'Client ID' and 'Client Name' can help identify if certain clients are associated with more underbilling.",
    "The 'Effective Rate' is calculated as the billed amount divided by the billed hours, which can be compared to the 'Standard Rate' to detect underbilling.",
    "The 'Utilization Rate' shows how much of a timekeeper’s available time is billed; low rates might suggest underbilling.",
    "The 'Time Entry Description' provides context for why certain time entries might be underbilled, such as non-billable activities."
]

for sql, question in underbilling_sql_training:
    vn.train(sql=sql, question=question)

for doc in underbilling_documentation:
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