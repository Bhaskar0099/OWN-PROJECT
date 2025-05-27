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
    csv_path = os.path.join(base_dir, 'data', 'updated_synthetic_legal_billing_data copy.csv')
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

underbilling_sql_training = [
    ("SELECT \"Timekeeper Name\", SUM(\"Write Off Amount\") as total_write_off FROM legal_billing WHERE \"Invoice Date\" >= date('now', '-3 months') GROUP BY \"Timekeeper Name\" HAVING total_write_off > 1000 ORDER BY total_write_off DESC;", "Which timekeepers have a total write-off amount greater than $1000 in the last quarter?"),
    ("SELECT \"Timekeeper Role\", AVG(\"Realization Rate\") as avg_realization FROM legal_billing GROUP BY \"Timekeeper Role\";", "What is the average realization rate for each timekeeper role?"),
    ("SELECT \"Matter Name\", \"Billed Hours\", \"Worked Hours\" FROM legal_billing WHERE \"Billed Hours\" < 0.5 * \"Worked Hours\";", "List matters where billed hours are less than 50% of worked hours."),
    ("SELECT \"Client Name\", AVG(\"Discount Amount\") as avg_discount FROM legal_billing GROUP BY \"Client Name\" HAVING avg_discount > 500;", "Which clients have an average discount amount greater than $500?"),
    ("SELECT \"Timekeeper Name\", COUNT(*) as underbilled_matters FROM legal_billing WHERE \"Underbilling Flag\" = 'Yes' GROUP BY \"Timekeeper Name\";", "How many matters have an underbilling flag set to 'Yes' for each timekeeper?"),
    ("SELECT \"Matter Type\", AVG(\"Write Off Amount\" + \"Discount Amount\") as avg_underbilling FROM legal_billing GROUP BY \"Matter Type\";", "What is the average underbilling amount for each matter type?"),
    ("SELECT \"Timekeeper Name\", AVG(\"Utilization Rate\") as avg_utilization FROM legal_billing GROUP BY \"Timekeeper Name\" HAVING avg_utilization < 50;", "Which timekeepers have an average utilization rate below 50%?"),
    ("SELECT \"Timekeeper Name\", COUNT(CASE WHEN \"Underbilling Flag\" = 'Yes' THEN 1 END) as underbilling_count, AVG(\"Realization Rate\") as avg_realization FROM legal_billing GROUP BY \"Timekeeper Name\" HAVING underbilling_count > 5 AND avg_realization < 80;", "List timekeepers who have both a high number of underbilling flags and a low average realization rate."),
    ("SELECT \"Matter Name\", \"Effective Rate\", \"Standard Rate\", \"Time Entry Hours\" FROM legal_billing WHERE \"Effective Rate\" < 0.8 * \"Standard Rate\" AND \"Time Entry Hours\" > 5;", "Identify matters where the effective rate is less than 80% of the standard rate and the time entry hours are greater than 5."),
    ("SELECT \"Matter Name\", \"Billed Amount\", \"Collected Amount\" FROM legal_billing WHERE \"Collected Amount\" < 0.9 * \"Billed Amount\";", "Which matters have a collected amount less than 90% of the billed amount?"),
    ("SELECT \"Timekeeper Name\", AVG(\"Billed Hours\" / NULLIF(\"Time Entry Hours\", 0)) as avg_ratio FROM legal_billing GROUP BY \"Timekeeper Name\";", "What is the average ratio of billed hours to time entry hours for each timekeeper?")
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

legal_billing_terms = [
    "MT# or Matter# means \"Matter ID\" in legal billing data",
    "MT Name means \"Matter Name\" in legal billing data",
    "MT Off or Loc means \"Matter Office\" in legal billing data",
    "MT Dept means \"Matter Department\" in legal billing data",
    "MT PG or Matter PG means \"Matter Practice Group\" in legal billing data",
    "MT Type means \"Matter Type\" in legal billing data",
    "Batty or billing atty means \"Billing Attorney\" in legal billing data",
    "Supatty or supatty means \"Supervising Attorney\" in legal billing data",
    "Respaty or respatty means \"Responsible Attorney\" in legal billing data",
    "Orig atty or orig means \"Originating Attorney\" in legal billing data",
    "Pro atty or proatty means \"Proliferating Attorney\" in legal billing data",
    "Task Description means \"Time Entry Description\" in legal billing data",
    "When users say 'Active', search for 'Open' in \"Matter Status\""
]

for sql, question in underbilling_sql_training:
    vn.train(sql=sql, question=question)

for doc in underbilling_documentation + legal_billing_terms:
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
