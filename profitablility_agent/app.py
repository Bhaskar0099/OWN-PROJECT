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

profitability_risks_sql_training = [
    # Core Metric: Average Discount %
    (
        """
        SELECT md.practice, ROUND(AVG(rd.deviation_percent * 100)::numeric, 2) AS avg_discount_percent
        FROM matter_date md
        JOIN matter m ON md.matter_id = m.matter_id
        JOIN rate_set rs ON m.rate_set_id = rs.rate_set_id
        JOIN rate_set_link rsl ON rs.rate_set_id = rsl.rate_set_id
        JOIN rate_detail rd ON rsl.rate_component_id = rd.rate_component_id
        WHERE md.start_date <= '2025-12-31'
          AND (md.end_date >= '2025-01-01' OR md.end_date IS NULL)
          AND rd.start_date <= '2025-12-31'
          AND (rd.end_date >= '2025-01-01' OR rd.end_date IS NULL)
          AND m.is_active = 'Y'
        GROUP BY md.practice
        ORDER BY avg_discount_percent DESC;
        """,
        "What is the average discount percentage for each practice area in 2025?"
    ),
    # Core Metric: Average Realization %
    (
        """
        SELECT md.practice, ROUND(AVG(tc.worked_amount / NULLIF(tc.standard_amount, 0) * 100)::numeric, 2) AS avg_realization_percent
        FROM timecard tc
        JOIN matter m ON tc.matter_id = m.matter_id
        JOIN matter_date md ON m.matter_id = md.matter_id
        WHERE tc.date >= '2025-01-01'
          AND tc.date < '2026-01-01'
          AND tc.is_active = 'Y'
          AND tc.is_nonbillable = 'N'
          AND md.start_date <= tc.date
          AND (md.end_date >= tc.date OR md.end_date IS NULL)
          AND m.is_active = 'Y'
        GROUP BY md.practice
        ORDER BY avg_realization_percent ASC;
        """,
        "What is the average realization percentage for each practice area in 2025?"
    ),
    # Risk Detection: High Discounts
    (
        """
        SELECT md.practice, ROUND(AVG(rd.deviation_percent * 100)::numeric, 2) AS avg_discount_percent
        FROM matter_date md
        JOIN matter m ON md.matter_id = m.matter_id
        JOIN rate_set rs ON m.rate_set_id = rs.rate_set_id
        JOIN rate_set_link rsl ON rs.rate_set_id = rsl.rate_set_id
        JOIN rate_detail rd ON rsl.rate_component_id = rd.rate_component_id
        WHERE md.start_date <= '2025-12-31'
          AND (md.end_date >= '2025-01-01' OR md.end_date IS NULL)
          AND rd.start_date <= '2025-12-31'
          AND (rd.end_date >= '2025-01-01' OR rd.end_date IS NULL)
          AND m.is_active = 'Y'
        GROUP BY md.practice
        HAVING AVG(rd.deviation_percent * 100) > 25
        ORDER BY avg_discount_percent DESC;
        """,
        "Which practice areas have an average discount percentage greater than 25% in 2025?"
    ),
    # Risk Detection: Low Realization
    (
        """
        SELECT md.practice, ROUND(AVG(tc.worked_amount / NULLIF(tc.standard_amount, 0) * 100)::numeric, 2) AS avg_realization_percent
        FROM timecard tc
        JOIN matter m ON tc.matter_id = m.matter_id
        JOIN matter_date md ON m.matter_id = md.matter_id
        WHERE tc.date >= '2025-01-01'
          AND tc.date < '2026-01-01'
          AND tc.is_active = 'Y'
          AND tc.is_nonbillable = 'N'
          AND md.start_date <= tc.date
          AND (md.end_date >= tc.date OR md.end_date IS NULL)
          AND m.is_active = 'Y'
        GROUP BY md.practice
        HAVING AVG(tc.worked_amount / NULLIF(tc.standard_amount, 0) * 100) < 80
        ORDER BY avg_realization_percent ASC;
        """,
        "Which practice areas have an average realization percentage less than 80% in 2025?"
    ),
    # Comprehensive Summary
    (
        """
        WITH practice_metrics AS (
          SELECT md.practice,
                 ROUND(AVG(rd.deviation_percent * 100)::numeric, 2) AS avg_discount_percent,
                 ROUND(AVG(tc.worked_amount / NULLIF(tc.standard_amount, 0) * 100)::numeric, 2) AS avg_realization_percent
          FROM matter_date md
          JOIN matter m ON md.matter_id = m.matter_id
          LEFT JOIN rate_set rs ON m.rate_set_id = rs.rate_set_id
          LEFT JOIN rate_set_link rsl ON rs.rate_set_id = rsl.rate_set_id
          LEFT JOIN rate_detail rd ON rsl.rate_component_id = rd.rate_component_id
            AND rd.start_date <= '2025-12-31'
            AND (rd.end_date >= '2025-01-01' OR rd.end_date IS NULL)
          LEFT JOIN timecard tc ON m.matter_id = tc.matter_id
            AND tc.date >= '2025-01-01'
            AND tc.date < '2026-01-01'
            AND tc.is_active = 'Y'
            AND tc.is_nonbillable = 'N'
            AND md.start_date <= tc.date
            AND (md.end_date >= tc.date OR md.end_date IS NULL)
          WHERE m.is_active = 'Y'
            AND md.start_date <= '2025-12-31'
            AND (md.end_date >= '2025-01-01' OR md.end_date IS NULL)
          GROUP BY md.practice
        )
        SELECT practice,
               avg_discount_percent,
               avg_realization_percent,
               CASE
                 WHEN avg_discount_percent > 25 OR avg_realization_percent < 80 THEN 'At Risk'
                 ELSE 'No Risk'
               END AS risk_status,
               CASE
                 WHEN avg_discount_percent > 25 AND avg_realization_percent < 80 THEN 'Reduce discounting and revisit client negotiations'
                 WHEN avg_discount_percent > 25 THEN 'Reduce discounting'
                 WHEN avg_realization_percent < 80 THEN 'Revisit client negotiations'
                 ELSE 'Maintain current practices'
               END AS recommendation
        FROM practice_metrics
        ORDER BY practice;
        """,
        "Provide a summary of practice areas with their average discount percentage, realization percentage, risk status, and recommendation for 2025."
    ),
    # Trend Analysis: Discount Comparison
    (
        """
        SELECT md.practice,
               ROUND(AVG(rd.deviation_percent * 100)::numeric, 2) AS avg_discount_percent_2025,
               (SELECT ROUND(AVG(rd2.deviation_percent * 100)::numeric, 2)
                FROM rate_detail rd2
                JOIN rate_set_link rsl2 ON rd2.rate_component_id = rsl2.rate_component_id
                JOIN rate_set rs2 ON rsl2.rate_set_id = rs2.rate_set_id
                JOIN matter m2 ON rs2.rate_set_id = m2.rate_set_id
                JOIN matter_date md2 ON m2.matter_id = md2.matter_id
                WHERE md2.practice = md.practice
                  AND rd2.start_date <= '2024-12-31'
                  AND (rd2.end_date >= '2024-01-01' OR rd2.end_date IS NULL)
                  AND m2.is_active = 'Y') AS avg_discount_percent_2024
        FROM matter_date md
        JOIN matter m ON md.matter_id = m.matter_id
        JOIN rate_set rs ON m.rate_set_id = rs.rate_set_id
        JOIN rate_set_link rsl ON rs.rate_set_id = rsl.rate_set_id
        JOIN rate_detail rd ON rsl.rate_component_id = rd.rate_component_id
        WHERE md.start_date <= '2025-12-31'
          AND (md.end_date >= '2025-01-01' OR md.end_date IS NULL)
          AND rd.start_date <= '2025-12-31'
          AND (rd.end_date >= '2025-01-01' OR rd.end_date IS NULL)
          AND m.is_active = 'Y'
        GROUP BY md.practice
        ORDER BY avg_discount_percent_2025 DESC;
        """,
        "How do average discount percentages for practice areas in 2025 compare to 2024?"
    ),
    # Trend Analysis: Realization Comparison
    (
        """
        SELECT md.practice,
               ROUND(AVG(tc.worked_amount / NULLIF(tc.standard_amount, 0) * 100)::numeric, 2) AS avg_realization_percent_2025,
               (SELECT ROUND(AVG(tc2.worked_amount / NULLIF(tc2.standard_amount, 0) * 100)::numeric, 2)
                FROM timecard tc2
                JOIN matter m2 ON tc2.matter_id = m2.matter_id
                JOIN matter_date md2 ON m2.matter_id = md2.matter_id
                WHERE md2.practice = md.practice
                  AND tc2.date >= '2024-01-01'
                  AND tc2.date < '2025-01-01'
                  AND tc2.is_active = 'Y'
                  AND tc2.is_nonbillable = 'N'
                  AND md2.start_date <= tc2.date
                  AND (md2.end_date >= tc2.date OR md2.end_date IS NULL)
                  AND m2.is_active = 'Y') AS avg_realization_percent_2024
        FROM timecard tc
        JOIN matter m ON tc.matter_id = m.matter_id
        JOIN matter_date md ON m.matter_id = md.matter_id
        WHERE tc.date >= '2025-01-01'
          AND tc.date < '2026-01-01'
          AND tc.is_active = 'Y'
          AND tc.is_nonbillable = 'N'
          AND md.start_date <= tc.date
          AND (md.end_date >= tc.date OR md.end_date IS NULL)
          AND m.is_active = 'Y'
        GROUP BY md.practice
        ORDER BY avg_realization_percent_2025 ASC;
        """,
        "How do average realization percentages for practice areas in 2025 compare to 2024?"
    ),
    # Variability: Realization
    (
        """
        SELECT md.practice, ROUND(STDDEV(tc.worked_amount / NULLIF(tc.standard_amount, 0) * 100)::numeric, 2) AS realization_variability
        FROM timecard tc
        JOIN matter m ON tc.matter_id = m.matter_id
        JOIN matter_date md ON m.matter_id = m.matter_id
        WHERE tc.date >= '2025-01-01'
          AND tc.date < '2026-01-01'
          AND tc.is_active = 'Y'
          AND tc.is_nonbillable = 'N'
          AND md.start_date <= tc.date
          AND (md.end_date >= tc.date OR md.end_date IS NULL)
          AND m.is_active = 'Y'
        GROUP BY md.practice
        ORDER BY realization_variability DESC;
        """,
        "Which practice areas have the highest variability in realization percentages in 2025?"
    ),
    # Variability: Discounts
    (
        """
        SELECT md.practice, ROUND(STDDEV(rd.deviation_percent * 100)::numeric, 2) AS discount_variability
        FROM matter_date md
        JOIN matter m ON md.matter_id = m.matter_id
        JOIN rate_set rs ON m.rate_set_id = rs.rate_set_id
        JOIN rate_set_link rsl ON rs.rate_set_id = rsl.rate_set_id
        JOIN rate_detail rd ON rsl.rate_component_id = rd.rate_component_id
        WHERE md.start_date <= '2025-12-31'
          AND (md.end_date >= '2025-01-01' OR md.end_date IS NULL)
          AND rd.start_date <= '2025-12-31'
          AND (rd.end_date >= '2025-01-01' OR rd.end_date IS NULL)
          AND m.is_active = 'Y'
        GROUP BY md.practice
        ORDER BY discount_variability DESC;
        """,
        "Which practice areas have the highest variability in discount percentages in 2025?"
    ),
    # Client-Specific: Discounts
    (
        """
        SELECT md.practice, c.client_name, ROUND(AVG(rd.deviation_percent * 100)::numeric, 2) AS avg_discount_percent
        FROM matter_date md
        JOIN matter m ON md.matter_id = m.matter_id
        JOIN client c ON m.client_id = c.client_id
        JOIN rate_set rs ON m.rate_set_id = rs.rate_set_id
        JOIN rate_set_link rsl ON rs.rate_set_id = rsl.rate_set_id
        JOIN rate_detail rd ON rsl.rate_component_id = rd.rate_component_id
        WHERE md.start_date <= '2025-12-31'
          AND (md.end_date >= '2025-01-01' OR md.end_date IS NULL)
          AND rd.start_date <= '2025-12-31'
          AND (rd.end_date >= '2025-01-01' OR rd.end_date IS NULL)
          AND m.is_active = 'Y'
          AND c.is_active = 'Y'
        GROUP BY md.practice, c.client_name
        HAVING AVG(rd.deviation_percent * 100) > 25
        ORDER BY avg_discount_percent DESC;
        """,
        "Which clients within each practice area have average discount percentages greater than 25% in 2025?"
    ),
    # Client-Specific: Realization
    (
        """
        SELECT md.practice, c.client_name, ROUND(AVG(tc.worked_amount / NULLIF(tc.standard_amount, 0) * 100)::numeric, 2) AS avg_realization_percent
        FROM timecard tc
        JOIN matter m ON tc.matter_id = m.matter_id
        JOIN matter_date md ON m.matter_id = md.matter_id
        JOIN client c ON m.client_id = c.client_id
        WHERE tc.date >= '2025-01-01'
          AND tc.date < '2026-01-01'
          AND tc.is_active = 'Y'
          AND tc.is_nonbillable = 'N'
          AND md.start_date <= tc.date
          AND (md.end_date >= tc.date OR md.end_date IS NULL)
          AND m.is_active = 'Y'
          AND c.is_active = 'Y'
        GROUP BY md.practice, c.client_name
        HAVING AVG(tc.worked_amount / NULLIF(tc.standard_amount, 0) * 100) < 80
        ORDER BY avg_realization_percent ASC;
        """,
        "Which clients within each practice area have average realization percentages less than 80% in 2025?"
    ),
    # Additional Factor: Non-Billable Hours
    (
        """
        SELECT md.practice,
               ROUND((SUM(CASE WHEN tc.is_nonbillable = 'Y' THEN tc.worked_amount ELSE 0 END) / NULLIF(SUM(tc.worked_amount), 0) * 100)::numeric, 2) AS non_billable_percent
        FROM timecard tc
        JOIN matter m ON tc.matter_id = m.matter_id
        JOIN matter_date md ON m.matter_id = md.matter_id
        WHERE tc.date >= '2025-01-01'
          AND tc.date < '2026-01-01'
          AND tc.is_active = 'Y'
          AND md.start_date <= tc.date
          AND (md.end_date >= tc.date OR md.end_date IS NULL)
          AND m.is_active = 'Y'
        GROUP BY md.practice
        ORDER BY non_billable_percent DESC;
        """,
        "Which practice areas have the highest percentage of non-billable amounts in 2025?"
    ),
    # Additional Factor: Underbilling
    (
        """
        SELECT md.practice, ROUND(SUM(mts.standard_amount - mts.worked_amount)::numeric, 2) AS total_underbilling
        FROM matter_timekeeper_summary mts
        JOIN matter m ON mts.matter_id = m.matter_id
        JOIN matter_date md ON m.matter_id = md.matter_id
        WHERE mts.year = 2025
          AND mts.is_error = 'N'
          AND m.is_active = 'Y'
          AND md.start_date <= '2025-12-31'
          AND (md.end_date >= '2025-01-01' OR md.end_date IS NULL)
        GROUP BY md.practice
        ORDER BY total_underbilling DESC;
        """,
        "Which practice areas have the highest total underbilling amount in 2025?"
    ),
    # Additional Factor: Currency Mismatches
    (
        """
        SELECT md.practice, COUNT(DISTINCT tc.timecard_id) AS currency_mismatch_count
        FROM timecard tc
        JOIN matter m ON tc.matter_id = m.matter_id
        JOIN matter_date md ON m.matter_id = md.matter_id
        WHERE tc.date >= '2025-01-01'
          AND tc.date < '2026-01-01'
          AND tc.is_active = 'Y'
          AND tc.is_nonbillable = 'N'
          AND md.start_date <= tc.date
          AND (md.end_date >= tc.date OR md.end_date IS NULL)
          AND m.is_active = 'Y'
          AND tc.worked_amount IS NOT NULL
          AND m.matter_currency != mts.currency
        JOIN matter_timekeeper_summary mts ON m.matter_id = mts.matter_id
        GROUP BY md.practice
        ORDER BY currency_mismatch_count DESC;
        """,
        "Which practice areas have the most timecards with currency mismatches in 2025?"
    ),
    # Matter Type Analysis
    (
        """
        SELECT md.practice, m.type, ROUND(AVG(tc.worked_amount / NULLIF(tc.standard_amount, 0) * 100)::numeric, 2) AS avg_realization_percent
        FROM timecard tc
        JOIN matter m ON tc.matter_id = m.matter_id
        JOIN matter_date md ON m.matter_id = md.matter_id
        WHERE tc.date >= '2025-01-01'
          AND tc.date < '2026-01-01'
          AND tc.is_active = 'Y'
          AND tc.is_nonbillable = 'N'
          AND md.start_date <= tc.date
          AND (md.end_date >= tc.date OR md.end_date IS NULL)
          AND m.is_active = 'Y'
        GROUP BY md.practice, m.type
        ORDER BY avg_realization_percent ASC;
        """,
        "What is the average realization percentage by practice area and matter type in 2025?"
    ),
    # Matter Category Analysis
    (
        """
        SELECT md.practice, m.category, ROUND(AVG(rd.deviation_percent * 100)::numeric, 2) AS avg_discount_percent
        FROM matter_date md
        JOIN matter m ON md.matter_id = m.matter_id
        JOIN rate_set rs ON m.rate_set_id = rs.rate_set_id
        JOIN rate_set_link rsl ON rs.rate_set_id = rsl.rate_set_id
        JOIN rate_detail rd ON rsl.rate_component_id = rd.rate_component_id
        WHERE md.start_date <= '2025-12-31'
          AND (md.end_date >= '2025-01-01' OR md.end_date IS NULL)
          AND rd.start_date <= '2025-12-31'
          AND (rd.end_date >= '2025-01-01' OR rd.end_date IS NULL)
          AND m.is_active = 'Y'
        GROUP BY md.practice, m.category
        ORDER BY avg_discount_percent DESC;
        """,
        "What is the average discount percentage by practice area and matter category in 2025?"
    ),
    # Quarterly Breakdown
    (
        """
        SELECT md.practice,
               EXTRACT(QUARTER FROM tc.date) AS quarter,
               ROUND(AVG(tc.worked_amount / NULLIF(tc.standard_amount, 0) * 100)::numeric, 2) AS avg_realization_percent
        FROM timecard tc
        JOIN matter m ON tc.matter_id = m.matter_id
        JOIN matter_date md ON m.matter_id = md.matter_id
        WHERE tc.date >= '2025-01-01'
          AND tc.date < '2026-01-01'
          AND tc.is_active = 'Y'
          AND tc.is_nonbillable = 'N'
          AND md.start_date <= tc.date
          AND (md.end_date >= tc.date OR md.end_date IS NULL)
          AND m.is_active = 'Y'
        GROUP BY md.practice, EXTRACT(QUARTER FROM tc.date)
        ORDER BY md.practice, quarter;
        """,
        "What is the average realization percentage by practice area and quarter in 2025?"
    ),
    # High-Risk Matter Count
    (
        """
        WITH matter_risks AS (
          SELECT md.practice, m.matter_id,
                 AVG(rd.deviation_percent * 100) AS avg_discount_percent,
                 AVG(tc.worked_amount / NULLIF(tc.standard_amount, 0) * 100) AS avg_realization_percent
          FROM matter_date md
          JOIN matter m ON md.matter_id = m.matter_id
          LEFT JOIN rate_set rs ON m.rate_set_id = rs.rate_set_id
          LEFT JOIN rate_set_link rsl ON rs.rate_set_id = rsl.rate_set_id
          LEFT JOIN rate_detail rd ON rsl.rate_component_id = rd.rate_component_id
            AND rd.start_date <= '2025-12-31'
            AND (rd.end_date >= '2025-01-01' OR rd.end_date IS NULL)
          LEFT JOIN timecard tc ON m.matter_id = tc.matter_id
            AND tc.date >= '2025-01-01'
            AND tc.date < '2026-01-01'
            AND tc.is_active = 'Y'
            AND tc.is_nonbillable = 'N'
            AND md.start_date <= tc.date
            AND (md.end_date >= tc.date OR md.end_date IS NULL)
          WHERE m.is_active = 'Y'
            AND md.start_date <= '2025-12-31'
            AND (md.end_date >= '2025-01-01' OR md.end_date IS NULL)
          GROUP BY md.practice, m.matter_id
          HAVING AVG(rd.deviation_percent * 100) > 25 OR AVG(tc.worked_amount / NULLIF(tc.standard_amount, 0) * 100) < 80
        )
        SELECT practice, COUNT(DISTINCT matter_id) AS high_risk_matter_count
        FROM matter_risks
        GROUP BY practice
        ORDER BY high_risk_matter_count DESC;
        """,
        "How many matters in each practice area are at risk due to high discounts or low realization in 2025?"
    ),
    # Error Detection
    (
        """
        SELECT md.practice, COUNT(DISTINCT mts.tk_mat_sum_id) AS error_entries
        FROM matter_timekeeper_summary mts
        JOIN matter m ON mts.matter_id = m.matter_id
        JOIN matter_date md ON m.matter_id = md.matter_id
        WHERE mts.year = 2025
          AND mts.is_error = 'Y'
          AND m.is_active = 'Y'
          AND md.start_date <= '2025-12-31'
          AND (md.end_date >= '2025-01-01' OR md.end_date IS NULL)
        GROUP BY md.practice
        ORDER BY error_entries DESC;
        """,
        "Which practice areas have the most erroneous billing entries in 2025?"
    ),
    # Median Realization
    (
        """
        SELECT md.practice,
               ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY (tc.worked_amount / NULLIF(tc.standard_amount, 0) * 100))::numeric, 2) AS median_realization_percent
        FROM timecard tc
        JOIN matter m ON tc.matter_id = m.matter_id
        JOIN matter_date md ON m.matter_id = md.matter_id
        WHERE tc.date >= '2025-01-01'
          AND tc.date < '2026-01-01'
          AND tc.is_active = 'Y'
          AND tc.is_nonbillable = 'N'
          AND md.start_date <= tc.date
          AND (md.end_date >= tc.date OR md.end_date IS NULL)
          AND m.is_active = 'Y'
        GROUP BY md.practice
        ORDER BY median_realization_percent ASC;
        """,
        "What is the median realization percentage for each practice area in 2025?"
    )
]

profitability_risks_documentation = [
    "Profitability risks in practice areas stem from high discounts (rate_detail.deviation_percent), low realization percentages (timecard.worked_amount / standard_amount), high non-billable amounts (timecard.is_nonbillable = 'Y'), and billing errors (matter_timekeeper_summary.is_error), all reducing revenue.",
    "The matter_date table’s practice column groups matters by practice area (e.g., Litigation, Corporate), with start_date and end_date ensuring accurate time-based analysis.",
    "rate_detail.deviation_percent measures discounts as a percentage reduction from standard rates, impacting profitability when high (e.g., > 25%).",
    "Realization percentage, calculated as (timecard.worked_amount / standard_amount * 100), shows how much of the standard billing is realized; values below 80% indicate significant profitability risks.",
    "matter_timekeeper_summary provides aggregated worked_amount and standard_amount by matter and year, offering an alternative source for realization calculations, with is_error filtering out unreliable data.",
    "Year-over-year comparisons (e.g., 2025 vs. 2024 discounts) reveal trends in discounting or realization, helping identify worsening profitability risks.",
    "Variability in realization or discounts (using STDDEV) indicates inconsistent billing practices within a practice area, signaling operational or client-specific risks.",
    "Client-specific analysis (client.client_name) within practice areas pinpoints clients driving high discounts or low realization, guiding targeted negotiations.",
    "Non-billable amounts (timecard.is_nonbillable = 'Y') reduce billable revenue; high percentages in a practice area suggest inefficiencies or client-imposed constraints.",
    "Underbilling (matter_timekeeper_summary.standard_amount - worked_amount) represents lost revenue potential, a direct profitability risk when significant.",
    "Currency mismatches between timecard.worked_amount and matter.matter_currency or matter_timekeeper_summary.currency can lead to billing errors, affecting profitability.",
    "Matter type (matter.type) and category (matter.category) provide additional granularity, revealing profitability risks within specific matter classifications.",
    "Quarterly breakdowns of realization percentages identify seasonal or temporal profitability risks within practice areas.",
    "Counting high-risk matters (based on discount > 25% or realization < 80%) quantifies the extent of profitability issues within each practice area.",
    "Erroneous entries (matter_timekeeper_summary.is_error = 'Y') can skew profitability metrics and require correction to ensure accurate risk assessment.",
    "Median realization percentages (using PERCENTILE_CONT) provide a robust measure of central tendency, less sensitive to outliers than averages.",
    "The summary query integrates discount %, realization %, risk status, and recommendations, fulfilling the agent’s core requirement to flag and address profitability risks.",
    "Recommendations like 'reduce discounting' or 'revisit client negotiations' are tailored to specific risks, enhancing actionable insights.",
    "All queries filter for is_active = 'Y' (matter, timecard, client) to focus on current data, ensuring relevance to ongoing operations.",
    "The agent’s inability to assess write-offs (due to missing write_off_amount) limits its scope but does not affect the accuracy of discount and realization-based risk detection."
]

profitability_risks_ddl_statements = [
    # matter_date table
    'Table: matter_date - Column: matter_id, Type: text, Nullable: YES - Column: practice, Type: text, Nullable: YES - Column: start_date, Type: timestamp without time zone, Nullable: YES - Column: end_date, Type: timestamp without time zone, Nullable: YES',
    'Adding ddl: CREATE TABLE matter_date ("matter_id" TEXT, "practice" TEXT, "start_date" TIMESTAMP WITHOUT TIME ZONE, "end_date" TIMESTAMP WITHOUT TIME ZONE);',

    # matter table
    'Table: matter - Column: matter_id, Type: text, Nullable: YES - Column: client_id, Type: text, Nullable: YES - Column: is_active, Type: text, Nullable: YES - Column: matter_currency, Type: text, Nullable: YES - Column: type, Type: text, Nullable: YES - Column: category, Type: text, Nullable: YES',
    'Adding ddl: CREATE TABLE matter ("matter_id" TEXT, "client_id" TEXT, "is_active" TEXT, "matter_currency" TEXT, "type" TEXT, "category" TEXT);',

    # timecard table
    'Table: timecard - Column: timecard_id, Type: text, Nullable: YES - Column: matter_id, Type: text, Nullable: YES - Column: date, Type: timestamp without time zone, Nullable: YES - Column: worked_amount, Type: bigint, Nullable: YES - Column: standard_amount, Type: bigint, Nullable: YES - Column: is_active, Type: text, Nullable: YES - Column: is_nonbillable, Type: text, Nullable: YES',
    'Adding ddl: CREATE TABLE timecard ("timecard_id" TEXT, "matter_id" TEXT, "date" TIMESTAMP WITHOUT TIME ZONE, "worked_amount" BIGINT, "standard_amount" BIGINT, "is_active" TEXT, "is_nonbillable" TEXT);',

    # matter_timekeeper_summary table
    'Table: matter_timekeeper_summary - Column: tk_mat_sum_id, Type: text, Nullable: YES - Column: matter_id, Type: text, Nullable: YES - Column: year, Type: bigint, Nullable: YES - Column: worked_amount, Type: bigint, Nullable: YES - Column: standard_amount, Type: bigint, Nullable: YES - Column: currency, Type: text, Nullable: YES - Column: is_error, Type: text, Nullable: YES',
    'Adding ddl: CREATE TABLE matter_timekeeper_summary ("tk_mat_sum_id" TEXT, "matter_id" TEXT, "year" BIGINT, "worked_amount" BIGINT, "standard_amount" BIGINT, "currency" TEXT, "is_error" TEXT);',

    # rate_detail table
    'Table: rate_detail - Column: rate_detail_id, Type: text, Nullable: YES - Column: rate_component_id, Type: text, Nullable: YES - Column: deviation_percent, Type: double precision, Nullable: YES - Column: start_date, Type: timestamp without time zone, Nullable: YES - Column: end_date, Type: timestamp without time zone, Nullable: YES',
    'Adding ddl: CREATE TABLE rate_detail ("rate_detail_id" TEXT, "rate_component_id" TEXT, "deviation_percent" DOUBLE PRECISION, "start_date" TIMESTAMP WITHOUT TIME ZONE, "end_date" TIMESTAMP WITHOUT TIME ZONE);',

    # rate_set_link table
    'Table: rate_set_link - Column: rate_set_link_id, Type: text, Nullable: YES - Column: rate_set_id, Type: text, Nullable: YES - Column: rate_component_id, Type: text, Nullable: YES',
    'Adding ddl: CREATE TABLE rate_set_link ("rate_set_link_id" TEXT, "rate_set_id" TEXT, "rate_component_id" TEXT);',

    # rate_set table
    'Table: rate_set - Column: rate_set_id, Type: text, Nullable: YES',
    'Adding ddl: CREATE TABLE rate_set ("rate_set_id" TEXT);',

    # client table
    'Table: client - Column: client_id, Type: text, Nullable: YES - Column: client_name, Type: text, Nullable: YES - Column: is_active, Type: text, Nullable: YES',
    'Adding ddl: CREATE TABLE client ("client_id" TEXT, "client_name" TEXT, "is_active" TEXT);'
]

for sql, question in profitability_risks_sql_training:
    vn.train(sql=sql, question=question)

for doc in profitability_risks_documentation:
    vn.train(documentation=doc)

for ddl in profitability_risks_ddl_statements:
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