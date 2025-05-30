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
    user=os.getenv('UNDERBILLING_USER'),
    password=os.getenv('UNDERBILLING_PASSWORD'),
    port=int(os.getenv('POSTGRES_PORT', 5432))
)

get_tables_sql = """
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public' AND table_type = 'BASE TABLE';
"""
df_tables = vn.run_sql(get_tables_sql)

# Step 3: Loop through each table, get its columns, print, and train
for table in df_tables['table_name']:
    print(f"\n📘 Table: {table}")
    
    get_columns_sql = f"""
    SELECT column_name, data_type, is_nullable
    FROM information_schema.columns
    WHERE table_schema = 'public' AND table_name = '{table}';
    """
    df_columns = vn.run_sql(get_columns_sql)

    # Print column details
    for _, row in df_columns.iterrows():
        print(f" - Column: {row['column_name']}, Type: {row['data_type']}, Nullable: {row['is_nullable']}")

    # Step 4: Construct a DDL string
    ddl_parts = []
    for _, row in df_columns.iterrows():
        col = row['column_name']
        dtype = row['data_type'].upper()
        nullable = 'NOT NULL' if row['is_nullable'] == 'NO' else ''
        ddl_parts.append(f'"{col}" {dtype} {nullable}'.strip())

    ddl_statement = f'CREATE TABLE {table} (\n  ' + ',\n  '.join(ddl_parts) + '\n);'

    # Step 5: Train Vanna with the generated DDL
    vn.train(ddl=ddl_statement)

underbilling_sql_training = [
    (
        'SELECT t."timekeeper_name", ROUND(SUM(mts."standard_amount" - mts."worked_amount")::numeric, 2) AS total_underbilling '
        'FROM "matter_timekeeper_summary" mts '
        'JOIN "timekeeper" t ON mts."timekeeper_id" = t."timekeeper_id" '
        'WHERE mts."year" = 2025 '
        'AND t."is_active" = \'Y\' '
        'AND mts."is_error" = \'N\' '
        'GROUP BY t."timekeeper_name" '
        'ORDER BY total_underbilling DESC '
        'LIMIT 5;',
        "Which timekeepers have the highest total underbilling amount in 2025?"
    ),
    (
        'SELECT m."matter_name", ROUND(SUM(mts."standard_amount" - mts."worked_amount")::numeric, 2) AS total_underbilling '
        'FROM "matter_timekeeper_summary" mts '
        'JOIN "matter" m ON mts."matter_id" = m."matter_id" '
        'WHERE mts."year" = 2025 '
        'AND m."is_active" = \'Y\' '
        'AND mts."is_error" = \'N\' '
        'GROUP BY m."matter_name" '
        'ORDER BY total_underbilling DESC '
        'LIMIT 5;',
        "Which matters have the highest underbilling amount in 2025?"
    ),
    (
        'SELECT t."timekeeper_name", ROUND(AVG(mts."worked_amount" / NULLIF(mts."standard_amount", 0) * 100)::numeric, 2) AS avg_realization_rate '
        'FROM "matter_timekeeper_summary" mts '
        'JOIN "timekeeper" t ON mts."timekeeper_id" = t."timekeeper_id" '
        'WHERE mts."year" = 2025 '
        'AND t."is_active" = \'Y\' '
        'AND mts."is_error" = \'N\' '
        'GROUP BY t."timekeeper_name" '
        'HAVING AVG(mts."worked_amount" / NULLIF(mts."standard_amount", 0) * 100) < 80 '
        'ORDER BY avg_realization_rate ASC;',
        "Which timekeepers have an average realization rate below 80% in 2025?"
    ),
    (
        'SELECT c."client_name", ROUND(AVG(rd."deviation_amount")::numeric, 2) AS avg_discount_amount '
        'FROM "rate_detail" rd '
        'JOIN "rate_component" rc ON rd."rate_component_id" = rc."rate_component_id" '
        'JOIN "rate_set_link" rsl ON rc."rate_component_id" = rsl."rate_component_id" '
        'JOIN "rate_set" rs ON rsl."rate_set_id" = rs."rate_set_id" '
        'JOIN "matter" m ON rs."rate_set_id" = m."rate_set_id" '
        'JOIN "client" c ON m."client_id" = c."client_id" '
        'WHERE rd."start_date" <= \'2025-12-31\' '
        'AND (rd."end_date" >= \'2025-01-01\' OR rd."end_date" IS NULL) '
        'AND c."is_active" = \'Y\' '
        'GROUP BY c."client_name" '
        'HAVING AVG(rd."deviation_amount") > 1000 '
        'ORDER BY avg_discount_amount DESC;',
        "Which clients have an average discount amount greater than $1,000 in 2025?"
    ),
    (
        'SELECT m."matter_name", '
        'ROUND((SUM(CASE WHEN tc."is_nonbillable" = \'Y\' THEN tc."worked_hours" ELSE 0 END) / NULLIF(SUM(tc."worked_hours"), 0) * 100)::numeric, 2) AS non_billable_percentage '
        'FROM "timecard" tc '
        'JOIN "matter" m ON tc."matter_id" = m."matter_id" '
        'WHERE tc."date" >= \'2025-01-01\' '
        'AND tc."date" < \'2026-01-01\' '
        'AND tc."is_active" = \'Y\' '
        'AND m."is_active" = \'Y\' '
        'GROUP BY m."matter_name" '
        'HAVING (SUM(CASE WHEN tc."is_nonbillable" = \'Y\' THEN tc."worked_hours" ELSE 0 END) / NULLIF(SUM(tc."worked_hours"), 0)) > 0.3 '
        'ORDER BY non_billable_percentage DESC;',
        "Which matters have more than 30% of their timecard hours marked as non-billable in 2025?"
    ),
    (
        'SELECT t."timekeeper_name", COUNT(tc."timecard_id") AS underbilled_timecards '
        'FROM "timecard" tc '
        'JOIN "timekeeper" t ON tc."timekeeper_id" = t."timekeeper_id" '
        'WHERE tc."date" >= \'2025-01-01\' '
        'AND tc."date" < \'2026-01-01\' '
        'AND tc."worked_rate" < 0.75 * tc."standard_rate" '
        'AND tc."is_active" = \'Y\' '
        'AND tc."is_nonbillable" = \'N\' '
        'AND tc."is_no_charge" = \'N\' '
        'AND t."is_active" = \'Y\' '
        'GROUP BY t."timekeeper_name" '
        'ORDER BY underbilled_timecards DESC '
        'LIMIT 5;',
        "Which timekeepers have the most timecards with worked rates below 75% of standard rates in 2025?"
    ),
    (
        'SELECT m."type" AS matter_type, ROUND(AVG(mts."standard_amount" - mts."worked_amount")::numeric, 2) AS avg_underbilling '
        'FROM "matter_timekeeper_summary" mts '
        'JOIN "matter" m ON mts."matter_id" = m."matter_id" '
        'WHERE mts."year" = 2025 '
        'AND m."is_active" = \'Y\' '
        'AND mts."is_error" = \'N\' '
        'GROUP BY m."type" '
        'ORDER BY avg_underbilling DESC;',
        "Which matter types have the highest average underbilling amount in 2025?"
    ),
    (
        'SELECT c."client_name", ROUND(SUM(mts."standard_amount" - mts."worked_amount")::numeric, 2) AS total_underbilling '
        'FROM "matter_timekeeper_summary" mts '
        'JOIN "matter" m ON mts."matter_id" = m."matter_id" '
        'JOIN "client" c ON m."client_id" = c."client_id" '
        'WHERE mts."year" = 2025 '
        'AND c."is_active" = \'Y\' '
        'AND mts."is_error" = \'N\' '
        'GROUP BY c."client_name" '
        'ORDER BY total_underbilling DESC '
        'LIMIT 5;',
        "Which clients have the highest total underbilling amount in 2025?"
    ),
    (
        'SELECT m."matter_name", ROUND(AVG(rd."deviation_percent" * 100)::numeric, 2) AS avg_discount_percent '
        'FROM "rate_detail" rd '
        'JOIN "rate_component" rc ON rd."rate_component_id" = rc."rate_component_id" '
        'JOIN "rate_set_link" rsl ON rc."rate_component_id" = rsl."rate_component_id" '
        'JOIN "rate_set" rs ON rsl."rate_set_id" = rs."rate_set_id" '
        'JOIN "matter" m ON rs."rate_set_id" = m."rate_set_id" '
        'WHERE rd."start_date" <= \'2025-12-31\' '
        'AND (rd."end_date" >= \'2025-01-01\' OR rd."end_date" IS NULL) '
        'AND m."is_active" = \'Y\' '
        'GROUP BY m."matter_name" '
        'HAVING AVG(rd."deviation_percent" * 100) > 20 '
        'ORDER BY avg_discount_percent DESC;',
        "Which matters have a high discount percentage greater than 20% in 2025?"
    ),
    (
        'SELECT t."timekeeper_name", COUNT(*) AS error_timecards '
        'FROM "timecard" tc '
        'JOIN "timekeeper" t ON tc."timekeeper_id" = t."timekeeper_id" '
        'WHERE tc."worked_amount" > tc."standard_amount" '
        'AND tc."date" >= \'2025-01-01\' '
        'AND tc."date" < \'2026-01-01\' '
        'AND tc."is_active" = \'Y\' '
        'AND tc."is_nonbillable" = \'N\' '
        'AND t."is_active" = \'Y\' '
        'GROUP BY t."timekeeper_name" '
        'ORDER BY error_timecards DESC;',
        "Which timekeepers have errors in their timecards where the worked amount exceeds the standard amount in 2025?"
    )
]

underbilling_documentation = [
    "Underbilling occurs when the billed amount is less than the expected amount based on time worked or standard rates, often due to lower rates, discounts, non-billable hours, or no-charge entries.",
    "In the \"timecard\" table, \"worked\"_amount\" (based on \"worked\"_rate\") is the actual billed amount, while \"standard\"_amount\" (based on \"standard\"_rate\") is the expected amount. If \"worked\" is less than \"standard\", underbilling is likely.",
    "The \"deviation_amount\" and \"deviation_percent\" in \"rate_detail\" represent discounts applied to rates, reducing billed amounts and contributing to underbilling.",
    "Realization rate, calculated as \"worked_amount\" / \"standard\" / \"standard_amount\" * amount\", from \"matter_timekeeper_summary\", is low (e.g., < 80%) when underbilling occurs.",
    "Timecards with \"is\"_nonbillable\" = 'Y'\' or \"_is\"_no_charge\" = 'Y'\")' in \"timecard\" represent hours worked but not billed, a form of underbilling.",
    "The \"type\"_\" column in \"timekeeper\", indicating roles like Partner or Associate, is useful for analyzing underbilling trends.",
    "The \"type\" column in \"matter\" categorizes matters (e.g., Litigation, Patent Filing), helping identify underbilling patterns by matter type.",
    "The \"client\"_name\" in \"client\" links to matters, aiding in spotting clients with frequent underbilling issues.",
    "Comparing \"worked\"_rate\" to \"standard\"_rate\" in \"timecard\" to \"reveals rate reductions leading to underbilling.",
    "Data errors, such as \"worked\"_amount\" exceeding \"_standard\"amount\" in \"amount\", in \"timecard\" can obscure underbilling and need investigation."
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

# Use a thread pool executor to run blocking vn.ask() calls asynchronously
executor = ThreadPoolExecutor(max_workers=4)

@app.post("/ask")
async def ask(request: AskRequest):
    loop = asyncio.get_event_loop()
    try:
        # Run the blocking vn.ask() in a thread to avoid blocking the event loop
        df = await loop.run_in_executor(executor, vn.ask, request.question)
        if df is None:
            return {"data": [], "message": "No results returned"}
        if df.empty:
            return {"data": [], "message": "No results returned"}
        # Limit output size to avoid large payloads
        limited_df = df.head(100)
        return {"data": limited_df.to_dict(orient="records")}
    except Exception as e:
        # Log the error and return a 500 response with error message
        print(f"Exception in /ask endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)