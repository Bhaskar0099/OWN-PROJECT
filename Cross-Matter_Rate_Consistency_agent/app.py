import os
import uuid
import pandas as pd
from dotenv import load_dotenv
from vanna.openai import OpenAI_Chat
from vanna.chromadb import ChromaDB_VectorStore

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import plotly.express as px
import plotly.io as pio
import plotly.basedatatypes as _bdt
import plotly.graph_objects as go

import asyncio
from concurrent.futures import ThreadPoolExecutor

import openai

# ───────────────────────────────────────────────────────────────────
# Prevent Plotly from auto‐opening a browser:
pio.renderers.default = "svg"

# Monkey‐patch BaseFigure.show → no‐op, so that any fig.show( ) Vanna tries to do will do nothing
_bdt.BaseFigure.show = lambda *args, **kwargs: None
# ───────────────────────────────────────────────────────────────────

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, config=config)
        self._last_query_df = None

    def ask(self, question, **kwargs):
        """
        Call the parent .ask(…) which may return:
     • None  
     • a single DataFrame  
     • a tuple like (sql_string, DataFrame)  
     • or a tuple like (sql_string, DataFrame, Figure)  
        """
        result = super().ask(question, **kwargs)
        if result is None:
            return None

        # If Vanna returned a tuple, unpack it intelligently
        if isinstance(result, tuple):
            # If it’s exactly (sql, df, fig), grab df and fig
            if len(result) == 3 and isinstance(result[2], _bdt.BaseFigure):
                sql, df, fig = result
                self._last_query_df = df
                return (df, fig)

            # If it’s (sql, df), no figure
            if len(result) >= 2 and isinstance(result[1], pd.DataFrame):
                sql, df = result[0], result[1]
                self._last_query_df = df
                return (df, None)

            # Otherwise fall back: if result[0] itself is a DataFrame
            if isinstance(result[0], pd.DataFrame):
                df = result[0]
                self._last_query_df = df
                return (df, None)

            # If it’s something else we don’t recognize, just return None
            return None

        # If Vanna returned just a DataFrame
        if isinstance(result, pd.DataFrame):
            self._last_query_df = result
            return (result, None)

        # Otherwise, unknown return type
        return None

vn = MyVanna(config={
    'api_key': os.getenv('OPENAI_API_KEY'),
    'model': os.getenv('MODEL_NAME', 'gpt-4o-mini'),
    'embedding_model': 'text-embedding-3-small',
    'allow_llm_to_see_data': True,
})

vn.connect_to_postgres(
    host=os.getenv('POSTGRES_HOST'),
    dbname=os.getenv('POSTGRES_DB'),
    user=os.getenv('UNDERBILLING_USER'),
    password=os.getenv('UNDERBILLING_PASSWORD'),
    port=int(os.getenv('POSTGRES_PORT', 5432))
)


cross_matter_rate_consistency_sql_training = [
    (
        'SELECT c."client_name", m."matter_name", ROUND(rd."deviation_percent" * 100::numeric, 2) AS discount_percent '
        'FROM "rate_detail" rd '
        'JOIN "rate_component" rc ON rd."rate_component_id" = rc."rate_component_id" '
        'JOIN "rate_set_link" rsl ON rc."rate_component_id" = rsl."rate_component_id" '
        'JOIN "rate_set" rs ON rsl."rate_set_id" = rs."rate_set_id" '
        'JOIN "matter" m ON rs."rate_set_id" = m."rate_set_id" '
        'JOIN "client" c ON m."client_id" = c."client_id" '
        'WHERE rd."start_date" <= \'2025-12-31\' '
        'AND (rd."end_date" >= \'2025-01-01\' OR rd."end_date" IS NULL) '
        'AND m."is_active" = \'Y\' '
        'AND c."is_active" = \'Y\' '
        'AND rd."deviation_percent" > 0.20 '
        'ORDER BY c."client_name", discount_percent DESC;',
        "Which matters have discounts greater than 20% for active clients in 2025?"
    ),
    (
        'SELECT c."client_name", ROUND(AVG(rd."deviation_percent" * 100)::numeric, 2) AS avg_discount_percent '
        'FROM "rate_detail" rd '
        'JOIN "rate_component" rc ON rd."rate_component_id" = rc."rate_component_id" '
        'JOIN "rate_set_link" rsl ON rc."rate_component_id" = rsl."rate_component_id" '
        'JOIN "rate_set" rs ON rsl."rate_set_id" = rs."rate_set_id" '
        'JOIN "matter" m ON rs."rate_set_id" = m."rate_set_id" '
        'JOIN "client" c ON m."client_id" = c."client_id" '
        'WHERE rd."start_date" <= \'2025-12-31\' '
        'AND (rd."end_date" >= \'2025-01-01\' OR rd."end_date" IS NULL) '
        'AND m."is_active" = \'Y\' '
        'AND c."is_active" = \'Y\' '
        'GROUP BY c."client_name" '
        'ORDER BY avg_discount_percent DESC '
        'LIMIT 5;',
        "Which clients have the highest average discount percentage across their matters in 2025?"
    ),
    (
        'WITH client_avg AS ('
        'SELECT m."client_id", AVG(rd."deviation_percent" * 100) AS avg_discount '
        'FROM "rate_detail" rd '
        'JOIN "rate_component" rc ON rd."rate_component_id" = rc."rate_component_id" '
        'JOIN "rate_set_link" rsl ON rc."rate_component_id" = rsl."rate_component_id" '
        'JOIN "rate_set" rs ON rsl."rate_set_id" = rs."rate_set_id" '
        'JOIN "matter" m ON rs."rate_set_id" = m."rate_set_id" '
        'WHERE rd."start_date" <= \'2025-12-31\' '
        'AND (rd."end_date" >= \'2025-01-01\' OR rd."end_date" IS NULL) '
        'AND m."is_active" = \'Y\' '
        'GROUP BY m."client_id" '
        ') '
        'SELECT c."client_name", m."matter_name", ROUND(rd."deviation_percent" * 100::numeric, 2) AS discount_percent, '
        'ROUND(ABS(rd."deviation_percent" * 100 - ca.avg_discount)::numeric, 2) AS deviation_from_avg '
        'FROM "rate_detail" rd '
        'JOIN "rate_component" rc ON rd."rate_component_id" = rc."rate_component_id" '
        'JOIN "rate_set_link" rsl ON rc."rate_component_id" = rsl."rate_component_id" '
        'JOIN "rate_set" rs ON rsl."rate_set_id" = rs."rate_set_id" '
        'JOIN "matter" m ON rs."rate_set_id" = m."rate_set_id" '
        'JOIN "client" c ON m."client_id" = c."client_id" '
        'JOIN client_avg ca ON m."client_id" = ca."client_id" '
        'WHERE rd."start_date" <= \'2025-12-31\' '
        'AND (rd."end_date" >= \'2025-01-01\' OR rd."end_date" IS NULL) '
        'AND m."is_active" = \'Y\' '
        'AND c."is_active" = \'Y\' '
        'AND ABS(rd."deviation_percent" * 100 - ca.avg_discount) > 10 '
        'ORDER BY deviation_from_avg DESC;',
        "Which matters have discounts deviating by more than 10% from their client’s average discount in 2025?"
    ),
    (
        'SELECT rd."practice_group", ROUND(AVG(rd."deviation_percent" * 100)::numeric, 2) AS avg_discount_percent '
        'FROM "rate_detail" rd '
        'JOIN "rate_component" rc ON rd."rate_component_id" = rc."rate_component_id" '
        'JOIN "rate_set_link" rsl ON rc."rate_component_id" = rsl."rate_component_id" '
        'JOIN "rate_set" rs ON rsl."rate_set_id" = rs."rate_set_id" '
        'JOIN "matter" m ON rs."rate_set_id" = m."rate_set_id" '
        'WHERE rd."start_date" <= \'2025-12-31\' '
        'AND (rd."end_date" >= \'2025-01-01\' OR rd."end_date" IS NULL) '
        'AND m."is_active" = \'Y\' '
        'GROUP BY rd."practice_group" '
        'ORDER BY avg_discount_percent DESC;',
        "Which practice areas have the highest average discount percentage in 2025?"
    ),
    (
        'SELECT c."client_name", rd."practice_group", m."matter_name", ROUND(rd."deviation_percent" * 100::numeric, 2) AS discount_percent '
        'FROM "rate_detail" rd '
        'JOIN "rate_component" rc ON rd."rate_component_id" = rc."rate_component_id" '
        'JOIN "rate_set_link" rsl ON rc."rate_component_id" = rsl."rate_component_id" '
        'JOIN "rate_set" rs ON rsl."rate_set_id" = rs."rate_set_id" '
        'JOIN "matter" m ON rs."rate_set_id" = m."rate_set_id" '
        'JOIN "client" c ON m."client_id" = c."client_id" '
        'WHERE rd."start_date" <= \'2025-12-31\' '
        'AND (rd."end_date" >= \'2025-01-01\' OR rd."end_date" IS NULL) '
        'AND m."is_active" = \'Y\' '
        'AND c."is_active" = \'Y\' '
        'AND rd."deviation_percent" < 0.05 '
        'ORDER BY c."client_name", rd."practice_group", discount_percent;',
        "Which matters have discounts below 5% for active clients, grouped by practice area in 2025?"
    ),
    (
        'WITH practice_avg AS ('
        'SELECT rd."practice_group", AVG(rd."deviation_percent" * 100) AS avg_discount '
        'FROM "rate_detail" rd '
        'JOIN "rate_component" rc ON rd."rate_component_id" = rc."rate_component_id" '
        'JOIN "rate_set_link" rsl ON rc."rate_component_id" = rsl."rate_component_id" '
        'JOIN "rate_set" rs ON rsl."rate_set_id" = rs."rate_set_id" '
        'JOIN "matter" m ON rs."rate_set_id" = m."rate_set_id" '
        'WHERE rd."start_date" <= \'2025-12-31\' '
        'AND (rd."end_date" >= \'2025-01-01\' OR rd."end_date" IS NULL) '
        'AND m."is_active" = \'Y\' '
        'GROUP BY rd."practice_group" '
        ') '
        'SELECT c."client_name", rd."practice_group", m."matter_name", '
        'ROUND(rd."deviation_percent" * 100::numeric, 2) AS discount_percent, '
        'ROUND(ABS(rd."deviation_percent" * 100 - pa.avg_discount)::numeric, 2) AS deviation_from_practice_avg '
        'FROM "rate_detail" rd '
        'JOIN "rate_component" rc ON rd."rate_component_id" = rc."rate_component_id" '
        'JOIN "rate_set_link" rsl ON rc."rate_component_id" = rsl."rate_component_id" '
        'JOIN "rate_set" rs ON rsl."rate_set_id" = rs."rate_set_id" '
        'JOIN "matter" m ON rs."rate_set_id" = m."rate_set_id" '
        'JOIN "client" c ON m."client_id" = c."client_id" '
        'JOIN practice_avg pa ON rd."practice_group" = pa."practice_group" '
        'WHERE rd."start_date" <= \'2025-12-31\' '
        'AND (rd."end_date" >= \'2025-01-01\' OR rd."end_date" IS NULL) '
        'AND m."is_active" = \'Y\' '
        'AND c."is_active" = \'Y\' '
        'AND ABS(rd."deviation_percent" * 100 - pa.avg_discount) > 15 '
        'ORDER BY deviation_from_practice_avg DESC;',
        "Which matters have discounts deviating by more than 15% from their practice area’s average in 2025?"
    ),
    (
        'SELECT c."client_name", COUNT(DISTINCT m."matter_id") AS matter_count, '
        'ROUND(AVG(rd."deviation_percent" * 100)::numeric, 2) AS avg_discount_percent '
        'FROM "rate_detail" rd '
        'JOIN "rate_component" rc ON rd."rate_component_id" = rc."rate_component_id" '
        'JOIN "rate_set_link" rsl ON rc."rate_component_id" = rsl."rate_component_id" '
        'JOIN "rate_set" rs ON rsl."rate_set_id" = rs."rate_set_id" '
        'JOIN "matter" m ON rs."rate_set_id" = m."rate_set_id" '
        'JOIN "client" c ON m."client_id" = c."client_id" '
        'WHERE rd."start_date" <= \'2025-12-31\' '
        'AND (rd."end_date" >= \'2025-01-01\' OR rd."end_date" IS NULL) '
        'AND m."is_active" = \'Y\' '
        'AND c."is_active" = \'Y\' '
        'GROUP BY c."client_name" '
        'HAVING COUNT(DISTINCT m."matter_id") > 5 '
        'ORDER BY avg_discount_percent DESC;',
        "Which clients with more than 5 active matters have the highest average discount in 2025?"
    ),
    (
        'SELECT c."client_name", m."matter_name", rd."rate_currency" '
        'FROM "rate_detail" rd '
        'JOIN "rate_component" rc ON rd."rate_component_id" = rc."rate_component_id" '
        'JOIN "rate_set_link" rsl ON rc."rate_component_id" = rsl."rate_component_id" '
        'JOIN "rate_set" rs ON rsl."rate_set_id" = rs."rate_set_id" '
        'JOIN "matter" m ON rs."rate_set_id" = m."rate_set_id" '
        'JOIN "client" c ON m."client_id" = c."client_id" '
        'WHERE rd."start_date" <= \'2025-12-31\' '
        'AND (rd."end_date" >= \'2025-01-01\' OR rd."end_date" IS NULL) '
        'AND m."is_active" = \'Y\' '
        'AND c."is_active" = \'Y\' '
        'AND rd."rate_currency" != ('
        'SELECT MODE() WITHIN GROUP (ORDER BY rd2."rate_currency") '
        'FROM "rate_detail" rd2 '
        'JOIN "rate_component" rc2 ON rd2."rate_component_id" = rc2."rate_component_id" '
        'JOIN "rate_set_link" rsl2 ON rc2."rate_component_id" = rsl2."rate_component_id" '
        'JOIN "rate_set" rs2 ON rsl2."rate_set_id" = rs2."rate_set_id" '
        'JOIN "matter" m2 ON rs2."rate_set_id" = m2."rate_set_id" '
        'WHERE m2."client_id" = m."client_id" '
        'AND rd2."start_date" <= \'2025-12-31\' '
        'AND (rd2."end_date" >= \'2025-01-01\' OR rd2."end_date" IS NULL) '
        'AND m2."is_active" = \'Y\' '
        ') '
        'ORDER BY c."client_name", m."matter_name";',
        "Which matters have rate currencies different from their client’s most common currency in 2025?"
    ),
    (
        'SELECT rd."practice_group", COUNT(DISTINCT m."matter_id") AS matter_count, '
        'ROUND(STDDEV(rd."deviation_percent" * 100)::numeric, 2) AS discount_variability '
        'FROM "rate_detail" rd '
        'JOIN "rate_component" rc ON rd."rate_component_id" = rc."rate_component_id" '
        'JOIN "rate_set_link" rsl ON rc."rate_component_id" = rsl."rate_component_id" '
        'JOIN "rate_set" rs ON rsl."rate_set_id" = rs."rate_set_id" '
        'JOIN "matter" m ON rs."rate_set_id" = m."rate_set_id" '
        'WHERE rd."start_date" <= \'2025-12-31\' '
        'AND (rd."end_date" >= \'2025-01-01\' OR rd."end_date" IS NULL) '
        'AND m."is_active" = \'Y\' '
        'GROUP BY rd."practice_group" '
        'HAVING STDDEV(rd."deviation_percent" * 100) > 5 '
        'ORDER BY discount_variability DESC;',
        "Which practice areas have the highest variability in discount percentages in 2025?"
    ),
    (
        'SELECT c."client_name", m."matter_name", rs."rate_set_name", '
        'ROUND(rd."deviation_percent" * 100::numeric, 2) AS discount_percent '
        'FROM "rate_detail" rd '
        'JOIN "rate_component" rc ON rd."rate_component_id" = rc."rate_component_id" '
        'JOIN "rate_set_link" rsl ON rc."rate_component_id" = rsl."rate_component_id" '
        'JOIN "rate_set" rs ON rsl."rate_set_id" = rs."rate_set_id" '
        'JOIN "matter" m ON rs."rate_set_id" = m."rate_set_id" '
        'JOIN "client" c ON m."client_id" = c."client_id" '
        'WHERE rd."start_date" <= \'2025-12-31\' '
        'AND (rd."end_date" >= \'2025-01-01\' OR rd."end_date" IS NULL) '
        'AND m."is_active" = \'Y\' '
        'AND c."is_active" = \'Y\' '
        'AND rs."rate_set_name" != ('
        'SELECT MODE() WITHIN GROUP (ORDER BY rs2."rate_set_name") '
        'FROM "rate_set" rs2 '
        'JOIN "matter" m2 ON rs2."rate_set_id" = m2."rate_set_id" '
        'WHERE m2."client_id" = m."client_id" '
        'AND m2."is_active" = \'Y\' '
        ') '
        'ORDER BY c."client_name", discount_percent DESC;',
        "Which matters use a different rate set than their client’s most common rate set in 2025?"
    ),
    (
        'WITH client_practice_avg AS ('
        'SELECT m."client_id", rd."practice_group", AVG(rd."deviation_percent" * 100) AS avg_discount '
        'FROM "rate_detail" rd '
        'JOIN "rate_component" rc ON rd."rate_component_id" = rc."rate_component_id" '
        'JOIN "rate_set_link" rsl ON rc."rate_component_id" = rsl."rate_component_id" '
        'JOIN "rate_set" rs ON rsl."rate_set_id" = rs."rate_set_id" '
        'JOIN "matter" m ON rs."rate_set_id" = m."rate_set_id" '
        'WHERE rd."start_date" <= \'2025-12-31\' '
        'AND (rd."end_date" >= \'2025-01-01\' OR rd."end_date" IS NULL) '
        'AND m."is_active" = \'Y\' '
        'GROUP BY m."client_id", rd."practice_group" '
        ') '
        'SELECT c."client_name", rd."practice_group", m."matter_name", '
        'ROUND(rd."deviation_percent" * 100::numeric, 2) AS discount_percent, '
        'ROUND(ABS(rd."deviation_percent" * 100 - cpa.avg_discount)::numeric, 2) AS deviation_from_avg '
        'FROM "rate_detail" rd '
        'JOIN "rate_component" rc ON rd."rate_component_id" = rc."rate_component_id" '
        'JOIN "rate_set_link" rsl ON rc."rate_component_id" = rsl."rate_component_id" '
        'JOIN "rate_set" rs ON rsl."rate_set_id" = rs."rate_set_id" '
        'JOIN "matter" m ON rs."rate_set_id" = m."rate_set_id" '
        'JOIN "client" c ON m."client_id" = c."client_id" '
        'JOIN client_practice_avg cpa ON m."client_id" = cpa."client_id" AND rd."practice_group" = cpa."practice_group" '
        'WHERE rd."start_date" <= \'2025-12-31\' '
        'AND (rd."end_date" >= \'2025-01-01\' OR rd."end_date" IS NULL) '
        'AND m."is_active" = \'Y\' '
        'AND c."is_active" = \'Y\' '
        'AND ABS(rd."deviation_percent" * 100 - cpa.avg_discount) > 10 '
        'ORDER BY deviation_from_avg DESC;',
        "Which matters have discounts deviating by more than 10% from their client’s practice area average in 2025?"
    ),
    (
        'SELECT c."client_name", COUNT(DISTINCT rd."practice_group") AS practice_group_count, '
        'ROUND(AVG(rd."deviation_percent" * 100)::numeric, 2) AS avg_discount_percent '
        'FROM "rate_detail" rd '
        'JOIN "rate_component" rc ON rd."rate_component_id" = rc."rate_component_id" '
        'JOIN "rate_set_link" rsl ON rc."rate_component_id" = rsl."rate_component_id" '
        'JOIN "rate_set" rs ON rsl."rate_set_id" = rs."rate_set_id" '
        'JOIN "matter" m ON rs."rate_set_id" = m."rate_set_id" '
        'JOIN "client" c ON m."client_id" = c."client_id" '
        'WHERE rd."start_date" <= \'2025-12-31\' '
        'AND (rd."end_date" >= \'2025-01-01\' OR rd."end_date" IS NULL) '
        'AND m."is_active" = \'Y\' '
        'AND c."is_active" = \'Y\' '
        'GROUP BY c."client_name" '
        'HAVING COUNT(DISTINCT rd."practice_group") > 2 '
        'ORDER BY avg_discount_percent DESC;',
        "Which clients with matters in more than 2 practice areas have the highest average discount in 2025?"
    ),
    (
        'SELECT rd."practice_group", c."client_name", m."matter_name", '
        'ROUND(rd."deviation_percent" * 100::numeric, 2) AS discount_percent '
        'FROM "rate_detail" rd '
        'JOIN "rate_component" rc ON rd."rate_component_id" = rc."rate_component_id" '
        'JOIN "rate_set_link" rsl ON rc."rate_component_id" = rsl."rate_component_id" '
        'JOIN "rate_set" rs ON rsl."rate_set_id" = rs."rate_set_id" '
        'JOIN "matter" m ON rs."rate_set_id" = m."rate_set_id" '
        'JOIN "client" c ON m."client_id" = c."client_id" '
        'WHERE rd."start_date" <= \'2025-12-31\' '
        'AND (rd."end_date" >= \'2025-01-01\' OR rd."end_date" IS NULL) '
        'AND m."is_active" = \'Y\' '
        'AND c."is_active" = \'Y\' '
        'AND rd."deviation_percent" > 0.30 '
        'ORDER BY rd."practice_group", discount_percent DESC;',
        "Which matters in each practice area have discounts above 30% in 2025?"
    ),
    (
        'SELECT c."client_name", m."matter_number", m."matter_name", '
        'ROUND(rd."deviation_percent" * 100::numeric, 2) AS discount_percent '
        'FROM "rate_detail" rd '
        'JOIN "rate_component" rc ON rd."rate_component_id" = rc."rate_component_id" '
        'JOIN "rate_set_link" rsl ON rc."rate_component_id" = rsl."rate_component_id" '
        'JOIN "rate_set" rs ON rsl."rate_set_id" = rs."rate_set_id" '
        'JOIN "matter" m ON rs."rate_set_id" = m."rate_set_id" '
        'JOIN "client" c ON m."client_id" = c."client_id" '
        'WHERE rd."start_date" <= \'2025-12-31\' '
        'AND (rd."end_date" >= \'2025-01-01\' OR rd."end_date" IS NULL) '
        'AND m."is_active" = \'Y\' '
        'AND c."is_active" = \'Y\' '
        'AND rd."deviation_percent" = 0 '
        'ORDER BY c."client_name", m."matter_number";',
        "Which matters have no discounts applied for active clients in 2025?"
    ),
    (
        'SELECT rd."practice_group", ROUND(MIN(rd."deviation_percent" * 100)::numeric, 2) AS min_discount, '
        'ROUND(MAX(rd."deviation_percent" * 100)::numeric, 2) AS max_discount '
        'FROM "rate_detail" rd '
        'JOIN "rate_component" rc ON rd."rate_component_id" = rc."rate_component_id" '
        'JOIN "rate_set_link" rsl ON rc."rate_component_id" = rsl."rate_component_id" '
        'JOIN "rate_set" rs ON rsl."rate_set_id" = rs."rate_set_id" '
        'JOIN "matter" m ON rs."rate_set_id" = m."rate_set_id" '
        'WHERE rd."start_date" <= \'2025-12-31\' '
        'AND (rd."end_date" >= \'2025-01-01\' OR rd."end_date" IS NULL) '
        'AND m."is_active" = \'Y\' '
        'GROUP BY rd."practice_group" '
        'ORDER BY max_discount DESC;',
        "What are the minimum and maximum discount percentages for each practice area in 2025?"
    ),
    (
        'WITH client_discount_rank AS ('
        'SELECT c."client_name", m."matter_name", rd."deviation_percent" * 100 AS discount_percent, '
        'RANK() OVER (PARTITION BY m."client_id" ORDER BY rd."deviation_percent" DESC) AS discount_rank '
        'FROM "rate_detail" rd '
        'JOIN "rate_component" rc ON rd."rate_component_id" = rc."rate_component_id" '
        'JOIN "rate_set_link" rsl ON rc."rate_component_id" = rsl."rate_component_id" '
        'JOIN "rate_set" rs ON rsl."rate_set_id" = rs."rate_set_id" '
        'JOIN "matter" m ON rs."rate_set_id" = m."rate_set_id" '
        'JOIN "client" c ON m."client_id" = c."client_id" '
        'WHERE rd."start_date" <= \'2025-12-31\' '
        'AND (rd."end_date" >= \'2025-01-01\' OR rd."end_date" IS NULL) '
        'AND m."is_active" = \'Y\' '
        'AND c."is_active" = \'Y\' '
        ') '
        'SELECT client_name, matter_name, ROUND(discount_percent::numeric, 2) AS discount_percent '
        'FROM client_discount_rank '
        'WHERE discount_rank = 1 '
        'ORDER BY discount_percent DESC;',
        "Which matter has the highest discount for each client in 2025?"
    ),
    (
        'SELECT c."client_name", rd."practice_group", '
        'ROUND(STDDEV(rd."deviation_percent" * 100)::numeric, 2) AS discount_variability '
        'FROM "rate_detail" rd '
        'JOIN "rate_component" rc ON rd."rate_component_id" = rc."rate_component_id" '
        'JOIN "rate_set_link" rsl ON rc."rate_component_id" = rsl."rate_component_id" '
        'JOIN "rate_set" rs ON rsl."rate_set_id" = rs."rate_set_id" '
        'JOIN "matter" m ON rs."rate_set_id" = m."rate_set_id" '
        'JOIN "client" c ON m."client_id" = c."client_id" '
        'WHERE rd."start_date" <= \'2025-12-31\' '
        'AND (rd."end_date" >= \'2025-01-01\' OR rd."end_date" IS NULL) '
        'AND m."is_active" = \'Y\' '
        'AND c."is_active" = \'Y\' '
        'GROUP BY c."client_name", rd."practice_group" '
        'HAVING STDDEV(rd."deviation_percent" * 100) > 7 '
        'ORDER BY discount_variability DESC;',
        "Which clients have high discount variability (>7%) within practice areas in 2025?"
    ),
    (
        'SELECT c."client_name", m."matter_name", rd."practice_group", '
        'ROUND(rd."deviation_percent" * 100::numeric, 2) AS discount_percent '
        'FROM "rate_detail" rd '
        'JOIN "rate_component" rc ON rd."rate_component_id" = rc."rate_component_id" '
        'JOIN "rate_set_link" rsl ON rc."rate_component_id" = rsl."rate_component_id" '
        'JOIN "rate_set" rs ON rsl."rate_set_id" = rs."rate_set_id" '
        'JOIN "matter" m ON rs."rate_set_id" = m."rate_set_id" '
        'JOIN "client" c ON m."client_id" = c."client_id" '
        'WHERE rd."start_date" <= \'2025-12-31\' '
        'AND (rd."end_date" >= \'2025-01-01\' OR rd."end_date" IS NULL) '
        'AND m."is_active" = \'Y\' '
        'AND c."is_active" = \'Y\' '
        'AND rd."deviation_percent" > ('
        'SELECT AVG(rd2."deviation_percent") + 0.10 '
        'FROM "rate_detail" rd2 '
        'JOIN "rate_component" rc2 ON rd2."rate_component_id" = rc2."rate_component_id" '
        'JOIN "rate_set_link" rsl2 ON rc2."rate_component_id" = rsl2."rate_component_id" '
        'JOIN "rate_set" rs2 ON rsl2."rate_set_id" = rs2."rate_set_id" '
        'JOIN "matter" m2 ON rs2."rate_set_id" = m2."rate_set_id" '
        'WHERE m2."client_id" = m."client_id" '
        'AND rd2."start_date" <= \'2025-12-31\' '
        'AND (rd2."end_date" >= \'2025-01-01\' OR rd2."end_date" IS NULL) '
        'AND m2."is_active" = \'Y\' '
        ') '
        'ORDER BY discount_percent DESC;',
        "Which matters have discounts more than 10% above their client’s average discount in 2025?"
    ),
    (
        'SELECT rd."practice_group", COUNT(DISTINCT c."client_id") AS client_count, '
        'ROUND(AVG(rd."deviation_percent" * 100)::numeric, 2) AS avg_discount_percent '
        'FROM "rate_detail" rd '
        'JOIN "rate_component" rc ON rd."rate_component_id" = rc."rate_component_id" '
        'JOIN "rate_set_link" rsl ON rc."rate_component_id" = rsl."rate_component_id" '
        'JOIN "rate_set" rs ON rsl."rate_set_id" = rs."rate_set_id" '
        'JOIN "matter" m ON rs."rate_set_id" = m."rate_set_id" '
        'JOIN "client" c ON m."client_id" = c."client_id" '
        'WHERE rd."start_date" <= \'2025-12-31\' '
        'AND (rd."end_date" >= \'2025-01-01\' OR rd."end_date" IS NULL) '
        'AND m."is_active" = \'Y\' '
        'AND c."is_active" = \'Y\' '
        'GROUP BY rd."practice_group" '
        'HAVING COUNT(DISTINCT c."client_id") > 3 '
        'ORDER BY avg_discount_percent DESC;',
        "Which practice areas serve more than 3 clients and have the highest average discount in 2025?"
    ),
    (
        'SELECT c."client_name", m."matter_name", rd."practice_group", '
        'ROUND(rd."deviation_percent" * 100::numeric, 2) AS discount_percent '
        'FROM "rate_detail" rd '
        'JOIN "rate_component" rc ON rd."rate_component_id" = rc."rate_component_id" '
        'JOIN "rate_set_link" rsl ON rc."rate_component_id" = rsl."rate_component_id" '
        'JOIN "rate_set" rs ON rsl."rate_set_id" = rs."rate_set_id" '
        'JOIN "matter" m ON rs."rate_set_id" = m."rate_set_id" '
        'JOIN "client" c ON m."client_id" = c."client_id" '
        'WHERE rd."start_date" <= \'2025-12-31\' '
        'AND (rd."end_date" >= \'2025-01-01\' OR rd."end_date" IS NULL) '
        'AND m."is_active" = \'Y\' '
        'AND c."is_active" = \'Y\' '
        'AND rd."deviation_percent" < ('
        'SELECT AVG(rd2."deviation_percent") - 0.10 '
        'FROM "rate_detail" rd2 '
        'JOIN "rate_component" rc2 ON rd2."rate_component_id" = rc2."rate_component_id" '
        'JOIN "rate_set_link" rsl2 ON rc2."rate_component_id" = rsl2."rate_component_id" '
        'JOIN "rate_set" rs2 ON rsl2."rate_set_id" = rs2."rate_set_id" '
        'JOIN "matter" m2 ON rs2."rate_set_id" = m2."rate_set_id" '
        'WHERE m2."client_id" = m."client_id" '
        'AND rd2."start_date" <= \'2025-12-31\' '
        'AND (rd2."end_date" >= \'2025-01-01\' OR rd2."end_date" IS NULL) '
        'AND m2."is_active" = \'Y\' '
        ') '
        'ORDER BY discount_percent ASC;',
        "Which matters have discounts more than 10% below their client’s average discount in 2025?"
    ),
    (
        'SELECT c."client_name", rd."practice_group", m."matter_name", '
        'ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY rd."deviation_percent" * 100)::numeric, 2) AS median_discount '
        'FROM "rate_detail" rd '
        'JOIN "rate_component" rc ON rd."rate_component_id" = rc."rate_component_id" '
        'JOIN "rate_set_link" rsl ON rc."rate_component_id" = rsl."rate_component_id" '
        'JOIN "rate_set" rs ON rsl."rate_set_id" = rs."rate_set_id" '
        'JOIN "matter" m ON rs."rate_set_id" = m."rate_set_id" '
        'JOIN "client" c ON m."client_id" = c."client_id" '
        'WHERE rd."start_date" <= \'2025-12-31\' '
        'AND (rd."end_date" >= \'2025-01-01\' OR rd."end_date" IS NULL) '
        'AND m."is_active" = \'Y\' '
        'AND c."is_active" = \'Y\' '
        'GROUP BY c."client_name", rd."practice_group", m."matter_name" '
        'HAVING PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY rd."deviation_percent" * 100) > 25 '
        'ORDER BY median_discount DESC;',
        "Which matters have a median discount above 25% within their client and practice area in 2025?"
    ),
    (
        'SELECT c."client_name", m."matter_name", rd."practice_group", '
        'ROUND(rd."deviation_percent" * 100::numeric, 2) AS discount_percent, '
        'rs."rate_set_code" '
        'FROM "rate_detail" rd '
        'JOIN "rate_component" rc ON rd."rate_component_id" = rc."rate_component_id" '
        'JOIN "rate_set_link" rsl ON rc."rate_component_id" = rsl."rate_component_id" '
        'JOIN "rate_set" rs ON rsl."rate_set_id" = rs."rate_set_id" '
        'JOIN "matter" m ON rs."rate_set_id" = m."rate_set_id" '
        'JOIN "client" c ON m."client_id" = c."client_id" '
        'WHERE rd."start_date" <= \'2025-12-31\' '
        'AND (rd."end_date" >= \'2025-01-01\' OR rd."end_date" IS NULL) '
        'AND m."is_active" = \'Y\' '
        'AND c."is_active" = \'Y\' '
        'AND rsl."start_date" <= EXTRACT(EPOCH FROM \'2025-12-31\'::TIMESTAMP) '
        'AND (rsl."end_date" >= EXTRACT(EPOCH FROM \'2025-01-01\'::TIMESTAMP) OR rsl."end_date" IS NULL) '
        'AND rd."deviation_percent" > 0.25 '
        'ORDER BY c."client_name", rd."practice_group", discount_percent DESC;',
        "Which matters with active rate set links have discounts above 25% in 2025?"
    ),
    (
        'SELECT rd."practice_group", c."client_name", COUNT(DISTINCT m."matter_id") AS matter_count '
        'FROM "rate_detail" rd '
        'JOIN "rate_component" rc ON rd."rate_component_id" = rc."rate_component_id" '
        'JOIN "rate_set_link" rsl ON rc."rate_component_id" = rsl."rate_component_id" '
        'JOIN "rate_set" rs ON rsl."rate_set_id" = rs."rate_set_id" '
        'JOIN "matter" m ON rs."rate_set_id" = m."rate_set_id" '
        'JOIN "client" c ON m."client_id" = c."client_id" '
        'WHERE rd."start_date" <= \'2025-12-31\' '
        'AND (rd."end_date" >= \'2025-01-01\' OR rd."end_date" IS NULL) '
        'AND m."is_active" = \'Y\' '
        'AND c."is_active" = \'Y\' '
        'AND rd."deviation_percent" > 0.20 '
        'GROUP BY rd."practice_group", c."client_name" '
        'HAVING COUNT(DISTINCT m."matter_id") > 2 '
        'ORDER BY matter_count DESC;',
        "Which clients have more than 2 matters with discounts above 20% in each practice area in 2025?"
    ),
    (
        'WITH client_discount_stats AS ('
        'SELECT m."client_id", AVG(rd."deviation_percent" * 100) AS avg_discount, '
        'STDDEV(rd."deviation_percent" * 100) AS stddev_discount '
        'FROM "rate_detail" rd '
        'JOIN "rate_component" rc ON rd."rate_component_id" = rc."rate_component_id" '
        'JOIN "rate_set_link" rsl ON rc."rate_component_id" = rsl."rate_component_id" '
        'JOIN "rate_set" rs ON rsl."rate_set_id" = rs."rate_set_id" '
        'JOIN "matter" m ON rs."rate_set_id" = m."rate_set_id" '
        'WHERE rd."start_date" <= \'2025-12-31\' '
        'AND (rd."end_date" >= \'2025-01-01\' OR rd."end_date" IS NULL) '
        'AND m."is_active" = \'Y\' '
        'GROUP BY m."client_id" '
        ') '
        'SELECT c."client_name", m."matter_name", rd."practice_group", '
        'ROUND(rd."deviation_percent" * 100::numeric, 2) AS discount_percent, '
        'ROUND((rd."deviation_percent" * 100 - cds.avg_discount) / NULLIF(cds.stddev_discount, 0)::numeric, 2) AS z_score '
        'FROM "rate_detail" rd '
        'JOIN "rate_component" rc ON rd."rate_component_id" = rc."rate_component_id" '
        'JOIN "rate_set_link" rsl ON rc."rate_component_id" = rsl."rate_component_id" '
        'JOIN "rate_set" rs ON rsl."rate_set_id" = rs."rate_set_id" '
        'JOIN "matter" m ON rs."rate_set_id" = m."rate_set_id" '
        'JOIN "client" c ON m."client_id" = c."client_id" '
        'JOIN client_discount_stats cds ON m."client_id" = cds."client_id" '
        'WHERE rd."start_date" <= \'2025-12-31\' '
        'AND (rd."end_date" >= \'2025-01-01\' OR rd."end_date" IS NULL) '
        'AND m."is_active" = \'Y\' '
        'AND c."is_active" = \'Y\' '
        'AND ABS((rd."deviation_percent" * 100 - cds.avg_discount) / NULLIF(cds.stddev_discount, 0)) > 2 '
        'ORDER BY z_score DESC;',
        "Which matters have discounts statistically significant (z-score > 2) from their client’s average in 2025?"
    ),
    (
        'SELECT c."client_name", m."matter_name", rd."practice_group", '
        'ROUND(rd."deviation_percent" * 100::numeric, 2) AS discount_percent '
        'FROM "rate_detail" rd '
        'JOIN "rate_component" rc ON rd."rate_component_id" = rc."rate_component_id" '
        'JOIN "rate_set_link" rsl ON rc."rate_component_id" = rsl."rate_component_id" '
        'JOIN "rate_set" rs ON rsl."rate_set_id" = rs."rate_set_id" '
        'JOIN "matter" m ON rs."rate_set_id" = m."rate_set_id" '
        'JOIN "client" c ON m."client_id" = c."client_id" '
        'WHERE rd."start_date" <= \'2025-12-31\' '
        'AND (rd."end_date" >= \'2025-01-01\' OR rd."end_date" IS NULL) '
        'AND m."is_active" = \'Y\' '
        'AND c."is_active" = \'Y\' '
        'AND rc."start_date" <= \'2025-12-31\' '
        'AND (rc."end_date" >= \'2025-01-01\' OR rc."end_date" IS NULL) '
        'AND rd."deviation_percent" > 0.15 '
        'ORDER BY c."client_name", rd."practice_group", discount_percent DESC;',
        "Which matters with active rate components have discounts above 15% in 2025?"
    )
]

cross_matter_rate_consistency_documentation = [
    "The Cross-Matter Rate Consistency Agent identifies inconsistencies in discount percentages (deviation_percent) across matters for the same client, flagging matters where discounts deviate by more than 10% from the client’s average to ensure consistent billing practices.",
    "In rate_detail, deviation_percent represents the discount applied to a matter’s rate, expressed as a decimal (e.g., 0.15 for 15%), which is critical for comparing rate consistency.",
    "The rate_detail table links to matter via rate_component, rate_set_link, and rate_set, using rate_component_id and rate_set_id to trace discounts to specific matters.",
    "The practice_group column in rate_detail categorizes matters by practice area (e.g., Litigation, Corporate), enabling analysis of discount consistency and profitability risks by practice area.",
    "High discounts in certain practice areas may pose profitability risks, especially in high-margin areas, which the agent assesses by comparing deviation_percent across practice_group.",
    "The client table’s client_name and client_id link to matters, allowing grouping of matters by client for discount comparison.",
    "The matter table’s matter_name, matter_number, and client_id provide context for reporting outlier matters with inconsistent discounts.",
    "The is_active = 'Y' flag in client and matter ensures analysis focuses on current clients and matters, relevant for 2025 data.",
    "The start_date and end_date in rate_detail, rate_component, and rate_set_link filter for active rates in 2025, ensuring temporal relevance.",
    "The rate_currency in rate_detail helps identify currency inconsistencies across a client’s matters, which could affect discount comparisons.",
    "The rate_set table’s rate_set_code and rate_set_name provide context for the rate structures applied to matters, useful for diagnosing discount outliers.",
    "Discount variability (e.g., standard deviation of deviation_percent) within a client or practice_group indicates inconsistent billing practices, potentially impacting profitability.",
    "Statistical measures like z-scores or percentiles of deviation_percent help identify significant discount outliers for a client, enhancing outlier detection.",
    "Comparing a matter’s deviation_percent to its practice_group’s average discount reveals practice-area-specific inconsistencies, critical for profitability analysis.",
    "The agent recommends aligning discounts for outlier matters unless unique reasons (e.g., strategic client relationships) are identified, using matter_name and client_name for reporting."
]


cross_matter_rate_consistency_ddl_statements = [
    # rate_detail table
    'Table: rate_detail - Column: rate_detail_id, Type: text, Nullable: YES - Column: rate_component_id, Type: text, Nullable: YES - Column: start_date, Type: timestamp without time zone, Nullable: YES - Column: end_date, Type: timestamp without time zone, Nullable: YES - Column: deviation_percent, Type: double precision, Nullable: YES - Column: rate_currency, Type: text, Nullable: YES - Column: practice_group, Type: text, Nullable: YES',
    'Adding ddl: CREATE TABLE rate_detail ("rate_detail_id" TEXT, "rate_component_id" TEXT, "start_date" TIMESTAMP WITHOUT TIME ZONE, "end_date" TIMESTAMP WITHOUT TIME ZONE, "deviation_percent" DOUBLE PRECISION, "rate_currency" TEXT, "practice_group" TEXT);',

    # rate_component table
    'Table: rate_component - Column: rate_component_id, Type: text, Nullable: YES - Column: start_date, Type: timestamp without time zone, Nullable: YES - Column: end_date, Type: timestamp without time zone, Nullable: YES',
    'Adding ddl: CREATE TABLE rate_component ("rate_component_id" TEXT, "start_date" TIMESTAMP WITHOUT TIME ZONE, "end_date" TIMESTAMP WITHOUT TIME ZONE);',

    # rate_set_link table
    'Table: rate_set_link - Column: rate_set_link_id, Type: text, Nullable: YES - Column: rate_set_id, Type: text, Nullable: YES - Column: rate_component_id, Type: text, Nullable: YES - Column: start_date, Type: double precision, Nullable: YES - Column: end_date, Type: double precision, Nullable: YES',
    'Adding ddl: CREATE TABLE rate_set_link ("rate_set_link_id" TEXT, "rate_set_id" TEXT, "rate_component_id" TEXT, "start_date" DOUBLE PRECISION, "end_date" DOUBLE PRECISION);',

    # rate_set table
    'Table: rate_set - Column: rate_set_id, Type: text, Nullable: YES - Column: rate_set_code, Type: text, Nullable: YES - Column: rate_set_name, Type: text, Nullable: YES',
    'Adding ddl: CREATE TABLE rate_set ("rate_set_id" TEXT, "rate_set_code" TEXT, "rate_set_name" TEXT);',

    # matter table
    'Table: matter - Column: matter_id, Type: text, Nullable: YES - Column: matter_number, Type: text, Nullable: YES - Column: matter_name, Type: text, Nullable: YES - Column: client_id, Type: text, Nullable: YES - Column: is_active, Type: text, Nullable: YES - Column: rate_set_id, Type: text, Nullable: YES',
    'Adding ddl: CREATE TABLE matter ("matter_id" TEXT, "matter_number" TEXT, "matter_name" TEXT, "client_id" TEXT, "is_active" TEXT, "rate_set_id" TEXT);',

    # client table
    'Table: client - Column: client_id, Type: text, Nullable: YES - Column: client_name, Type: text, Nullable: YES - Column: is_active, Type: text, Nullable: YES',
    'Adding ddl: CREATE TABLE client ("client_id" TEXT, "client_name" TEXT, "is_active" TEXT);'
]

for sql, question in cross_matter_rate_consistency_sql_training:
    vn.train(sql=sql, question=question)

for doc in cross_matter_rate_consistency_documentation:
    vn.train(documentation=doc)

for ddl in cross_matter_rate_consistency_ddl_statements:
    vn.train(ddl=ddl)

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

# In-memory cache for the interactive HTML strings, keyed by UUID
charts_cache: dict[str, str] = {}

# ThreadPoolExecutor so vn.ask(...) doesn’t block the event loop
executor = ThreadPoolExecutor(max_workers=4)

@app.get("/chart/{chart_id}", response_class=HTMLResponse)
async def serve_chart(chart_id: str):
    html_str = charts_cache.get(chart_id)
    if html_str is None:
        raise HTTPException(status_code=404, detail="Chart not found")
    return HTMLResponse(content=html_str, media_type="text/html")

@app.post("/ask")
async def ask(request: Request, payload: AskRequest):
    loop = asyncio.get_event_loop()
    try:
        # Run the blocking vn.ask(...) in a separate thread
        out = await loop.run_in_executor(executor, vn.ask, payload.question)

        # If Vanna returned None (no DataFrame), short-circuit
        if out is None:
            return {"data": [], "chart_url": None, "message": "No results returned"}

        df, fig_from_vanna = out  # df is DataFrame, fig_from_vanna may be a Plotly Figure or None

        if df is None or df.empty:
            return {"data": [], "chart_url": None, "message": "No results returned"}

        # ─────── LIMIT THE ROWS HERE ───────
        limited_df = df.head(10)
        # ────────────────────────────────────

        # Convert up to 100 rows to a JSON‐serializable list of dicts
        records = limited_df.to_dict(orient="records")

        # If Vanna already gave us a Figure, use it.
        if isinstance(fig_from_vanna, _bdt.BaseFigure) or isinstance(fig_from_vanna, go.Figure):
            fig = fig_from_vanna

            # But make sure the figure’s data matches our limited_df
            # (In many cases, Vanna’s own code already filtered down to an appropriate subset,
            #  so you might not need to re-build it. If you do want to “rebind” it to limited_df,”
            #  you’d have to replicate the same chart type logic Vanna chose. In practice,
            #  most of the time Vanna’s Figure was already built from a narrower subset.)

            # We’ll assume Vanna’s fig is “final.” Convert to HTML now:
            html_str = fig.to_html(include_plotlyjs="cdn")

        else:
            # Vanna did NOT produce a Figure. Build a fallback bar chart on limited_df:
            if len(limited_df.columns) >= 2:
                x_col = limited_df.columns[0]
                y_col = limited_df.columns[1]
                fig = px.bar(limited_df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
                html_str = fig.to_html(include_plotlyjs="cdn")
            else:
                # Not enough columns to make a chart
                html_str = None

        if html_str:
            chart_id = str(uuid.uuid4())
            charts_cache[chart_id] = html_str
            chart_url = str(request.url_for("serve_chart", chart_id=chart_id))
        else:
            chart_url = None

        summary = None
        try:
            # Construct the chat message
            system_msg = (
                "You are a data‐analysis assistant. "
                "When given a user’s question and a JSON array of row objects, "
                "your task is to produce a concise, narrative summary: "
                "focus on overall patterns, counts, and high-level insights rather than listing each row in detail."
            )

            user_msg = (
                f"Here is the user’s question: {payload.question}\n\n"
                f"Below are up to 100 rows of raw results (in JSON array format):\n{records}\n\n"
                "Please read these rows and respond with a brief narrative explanation. "
                "Highlight any notable trends, how many times something occurred, and what it implies, "
                "without enumerating each individual entry."
            )

            response = await loop.run_in_executor(
                executor,
                lambda: openai.chat.completions.create(
                    model="gpt-4.1-nano",
                    temperature=0.0,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                )
            )

            summary = response.choices[0].message.content.strip()

        except Exception as summary_err:
            # If summarization fails (e.g. rate limit), we can log it and continue
            print(f"Warning: summarization error: {summary_err}")
            summary = None

        return {
            "data": records,
            "chart_url": chart_url,
            "summary": summary
        }

    except Exception as e:
        print(f"Exception in /ask endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)