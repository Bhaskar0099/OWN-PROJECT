import os
import uuid
import json
import argparse
import pandas as pd
from dotenv import load_dotenv

import plotly.express as px
import plotly.io as pio
import plotly.basedatatypes as _bdt
import plotly.graph_objects as go

import openai
from vanna.openai import OpenAI_Chat
from vanna.chromadb import ChromaDB_VectorStore

# ───────────────────────────────────────────────────────────────────
# Prevent Plotly from auto‐opening a browser:
pio.renderers.default = "svg"
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
        Call parent .ask(…) which may return:
          • None
          • a DataFrame
          • (sql, DataFrame)
          • (sql, DataFrame, Figure)
        Normalize to (DataFrame, Figure) or None.
        """
        result = super().ask(question, **kwargs)
        if result is None:
            return None

        if isinstance(result, tuple):
            # (sql, df, fig)
            if len(result) == 3 and isinstance(result[2], _bdt.BaseFigure):
                sql, df, fig = result
                self._last_query_df = df
                return df, fig
            # (sql, df)
            if len(result) >= 2 and isinstance(result[1], pd.DataFrame):
                sql, df = result[0], result[1]
                self._last_query_df = df
                return df, None
            # (df,)
            if isinstance(result[0], pd.DataFrame):
                df = result[0]
                self._last_query_df = df
                return df, None
            return None

        if isinstance(result, pd.DataFrame):
            self._last_query_df = result
            return result, None

        return None


def summarize(question: str, records: list[dict]) -> str:
    """Generate a concise narrative summary via gpt-4.1-nano."""
    system_msg = (
        "You are a data-analysis assistant. "
        "When given a user’s question and a JSON array of row objects, "
        "your task is to produce a concise, narrative summary: "
        "focus on overall patterns, counts, and high-level insights rather than listing each row."
    )
    user_msg = (
        f"Here is the user’s question: {question}\n\n"
        f"Below are up to 100 rows of raw results:\n{json.dumps(records, indent=2)}"
    )
    resp = openai.chat.completions.create(
        model="gpt-4.1-nano",
        temperature=0.0,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ]
    )
    return resp.choices[0].message.content.strip()


# ───────────────────────────────────────────────────────────────────
# Training data: underbilling_sql_training, underbilling_documentation, ddl_statements
# ───────────────────────────────────────────────────────────────────

underbilling_sql_training = [
    (
        'SELECT DISTINCT t."timekeeper_name" '
        'FROM "timecard" tc '
        'JOIN "timekeeper" t ON tc."timekeeper_id" = t."timekeeper_id" '
        'WHERE tc."is_nonbillable" = \'Y\' '
        'AND EXTRACT(YEAR FROM tc."date") = 2025 '
        'AND tc."is_active" = \'Y\' '
        'AND t."is_active" = \'Y\' '
        'ORDER BY t."timekeeper_name";',
        "Which timekeepers have non-billable hours recorded in 2025?"
    ),
    (
        'SELECT m."matter_name", SUM(tc."worked_hours") AS no_charge_hours '
        'FROM "timecard" tc '
        'JOIN "matter" m ON tc."matter_id" = m."matter_id" '
        'WHERE tc."is_no_charge" = \'Y\' '
        'AND EXTRACT(YEAR FROM tc."date") = 2025 '
        'AND tc."is_active" = \'Y\' '
        'AND m."is_active" = \'Y\' '
        'GROUP BY m."matter_name" '
        'ORDER BY no_charge_hours DESC '
        'LIMIT 5;',
        "Which matters have the most no-charge hours in 2025?"
    ),
    (
        'SELECT c."client_name", ROUND(SUM(mts."standard_amount" - mts."worked_amount")::numeric, 2) AS total_underbilling '
        'FROM "matter_timekeeper_summary" mts '
        'JOIN "matter" m ON mts."matter_id" = m."matter_id" '
        'JOIN "client" c ON m."client_id" = c."client_id" '
        'WHERE mts."year" = 2025 '
        'AND mts."is_error" = \'N\' '
        'AND c."is_active" = \'Y\' '
        'GROUP BY c."client_name" '
        'ORDER BY total_underbilling DESC '
        'LIMIT 5;',
        "Which clients have the highest total underbilling amount in 2025?"
    ),
    (
        'SELECT t."timekeeper_name", ROUND(AVG(tc."worked_rate" / NULLIF(tc."standard_rate", 0) * 100)::numeric, 2) AS realization_rate '
        'FROM "timecard" tc '
        'JOIN "timekeeper" t ON tc."timekeeper_id" = t."timekeeper_id" '
        'WHERE EXTRACT(YEAR FROM tc."date") = 2025 '
        'AND tc."is_active" = \'Y\' '
        'AND t."is_active" = \'Y\' '
        'AND tc."is_nonbillable" = \'N\' '
        'GROUP BY t."timekeeper_name" '
        'HAVING AVG(tc."worked_rate" / NULLIF(tc."standard_rate", 0) * 100) < 85 '
        'ORDER BY realization_rate ASC;',
        "Which timekeepers have an average realization rate below 85% for billable timecards in 2025?"
    ),
    (
        'SELECT m."matter_name", ROUND(AVG(rd."deviation_amount")::numeric, 2) AS avg_discount '
        'FROM "rate_detail" rd '
        'JOIN "rate_component" rc ON rd."rate_component_id" = rc."rate_component_id" '
        'JOIN "rate_set_link" rsl ON rc."rate_component_id" = rsl."rate_component_id" '
        'JOIN "rate_set" rs ON rsl."rate_set_id" = rs."rate_set_id" '
        'JOIN "matter" m ON rs."rate_set_id" = m."rate_set_id" '
        'WHERE rd."start_date" <= \'2025-12-31\' '
        'AND (rd."end_date" >= \'2025-01-01\' OR rd."end_date" IS NULL) '
        'AND m."is_active" = \'Y\' '
        'GROUP BY m."matter_name" '
        'HAVING AVG(rd."deviation_amount") > 500 '
        'ORDER BY avg_discount DESC;',
        "Which matters have an average discount amount greater than $500 in 2025?"
    ),
    (
        'SELECT t."timekeeper_name", COUNT(tc."timecard_id") AS low_rate_timecards '
        'FROM "timecard" tc '
        'JOIN "timekeeper" t ON tc."timekeeper_id" = t."timekeeper_id" '
        'WHERE tc."worked_rate" < 0.8 * tc."standard_rate" '
        'AND EXTRACT(YEAR FROM tc."date") = 2025 '
        'AND tc."is_active" = \'Y\' '
        'AND tc."is_nonbillable" = \'N\' '
        'AND tc."is_no_charge" = \'N\' '
        'AND t."is_active" = \'Y\' '
        'GROUP BY t."timekeeper_name" '
        'ORDER BY low_rate_timecards DESC '
        'LIMIT 5;',
        "Which timekeepers have the most timecards with rates below 80% of standard rates in 2025?"
    ),
    (
        'SELECT m."type", ROUND(SUM(mts."standard_amount" - mts."worked_amount")::numeric, 2) AS total_underbilling '
        'FROM "matter_timekeeper_summary" mts '
        'JOIN "matter" m ON mts."matter_id" = m."matter_id" '
        'WHERE mts."year" = 2025 '
        'AND mts."is_error" = \'N\' '
        'AND m."is_active" = \'Y\' '
        'GROUP BY m."type" '
        'ORDER BY total_underbilling DESC;',
        "Which matter types have the highest total underbilling in 2025?"
    ),
    (
        'SELECT m."matter_name", '
        'ROUND((SUM(CASE WHEN tc."is_nonbillable" = \'Y\' THEN tc."worked_hours" ELSE 0 END) / NULLIF(SUM(tc."worked_hours"), 0) * 100)::numeric, 2) AS non_billable_percent '
        'FROM "timecard" tc '
        'JOIN "matter" m ON tc."matter_id" = m."matter_id" '
        'WHERE EXTRACT(YEAR FROM tc."date") = 2025 '
        'AND tc."is_active" = \'Y\' '
        'AND m."is_active" = \'Y\' '
        'GROUP BY m."matter_name" '
        'HAVING SUM(CASE WHEN tc."is_nonbillable" = \'Y\' THEN tc."worked_hours" ELSE 0 END) / NULLIF(SUM(tc."worked_hours"), 0) > 0.25 '
        'ORDER BY non_billable_percent DESC;',
        "Which matters have more than 25% non-billable hours in 2025?"
    ),
    (
        'SELECT c."client_name", COUNT(DISTINCT tc."timecard_id") AS error_timecards '
        'FROM "timecard" tc '
        'JOIN "matter" m ON tc."matter_id" = m."matter_id" '
        'JOIN "client" c ON m."client_id" = c."client_id" '
        'WHERE tc."worked_amount" > tc."standard_amount" '
        'AND EXTRACT(YEAR FROM tc."date") = 2025 '
        'AND tc."is_active" = \'Y\' '
        'AND c."is_active" = \'Y\' '
        'GROUP BY c."client_name" '
        'ORDER BY error_timecards DESC;',
        "Which clients have timecards with worked amounts exceeding standard amounts in 2025?"
    ),
    (
        'SELECT t."timekeeper_name", ROUND(AVG(mts."standard_amount" - mts."worked_amount")::numeric, 2) AS avg_underbilling '
        'FROM "matter_timekeeper_summary" mts '
        'JOIN "timekeeper" t ON mts."timekeeper_id" = t."timekeeper_id" '
        'WHERE mts."year" = 2025 '
        'AND mts."is_error" = \'N\' '
        'AND t."is_active" = \'Y\' '
        'GROUP BY t."timekeeper_name" '
        'ORDER BY avg_underbilling DESC '
        'LIMIT 5;',
        "Which timekeepers have the highest average underbilling amount per matter in 2025?"
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
        'HAVING AVG(rd."deviation_percent" * 100) > 15 '
        'ORDER BY avg_discount_percent DESC;',
        "Which matters have an average discount percentage above 15% in 2025?"
    ),
    (
        'SELECT t."type" AS timekeeper_type, ROUND(SUM(tc."worked_hours")::numeric, 2) AS non_billable_hours '
        'FROM "timecard" tc '
        'JOIN "timekeeper" t ON tc."timekeeper_id" = t."timekeeper_id" '
        'WHERE tc."is_nonbillable" = \'Y\' '
        'AND EXTRACT(YEAR FROM tc."date") = 2025 '
        'AND tc."is_active" = \'Y\' '
        'AND t."is_active" = \'Y\' '
        'GROUP BY t."type" '
        'ORDER BY non_billable_hours DESC;',
        "Which timekeeper types have the most non-billable hours in 2025?"
    ),
    (
        'SELECT m."matter_name", COUNT(tc."timecard_id") AS currency_mismatch '
        'FROM "timecard" tc '
        'JOIN "matter" m ON tc."matter_id" = m."matter_id" '
        'WHERE tc."worked_currency" != m."matter_currency" '
        'AND EXTRACT(YEAR FROM tc."date") = 2025 '
        'AND tc."is_active" = \'Y\' '
        'AND m."is_active" = \'Y\' '
        'GROUP BY m."matter_name" '
        'ORDER BY currency_mismatch DESC;',
        "Which matters have timecards with currency mismatches in 2025?"
    ),
    (
        'SELECT c."client_name", ROUND(SUM(tc."worked_hours")::numeric, 2) AS no_charge_hours '
        'FROM "timecard" tc '
        'JOIN "matter" m ON tc."matter_id" = m."matter_id" '
        'JOIN "client" c ON m."client_id" = c."client_id" '
        'WHERE tc."is_no_charge" = \'Y\' '
        'AND EXTRACT(YEAR FROM tc."date") = 2025 '
        'AND tc."is_active" = \'Y\' '
        'AND c."is_active" = \'Y\' '
        'GROUP BY c."client_name" '
        'ORDER BY no_charge_hours DESC '
        'LIMIT 5;',
        "Which clients have the most no-charge hours across their matters in 2025?"
    ),
    (
        'SELECT t."timekeeper_name", '
        'ROUND(SUM(mts."standard_amount" - mts."worked_amount") / NULLIF(SUM(mts."worked_hours"), 0)::numeric, 2) AS underbilling_per_hour '
        'FROM "matter_timekeeper_summary" mts '
        'JOIN "timekeeper" t ON mts."timekeeper_id" = t."timekeeper_id" '
        'WHERE mts."year" = 2025 '
        'AND mts."is_error" = \'N\' '
        'AND t."is_active" = \'Y\' '
        'GROUP BY t."timekeeper_name" '
        'ORDER BY underbilling_per_hour DESC '
        'LIMIT 5;',
        "Which timekeepers have the highest underbilling amount per hour in 2025?"
    ),
    (
        'SELECT m."matter_name", COUNT(DISTINCT mts."tk_mat_sum_id") AS underbilled_entries '
        'FROM "matter_timekeeper_summary" mts '
        'JOIN "matter" m ON mts."matter_id" = m."matter_id" '
        'WHERE mts."worked_amount" < mts."standard_amount" '
        'AND mts."year" = 2025 '
        'AND mts."is_error" = \'N\' '
        'AND m."is_active" = \'Y\' '
        'GROUP BY m."matter_name" '
        'ORDER BY underbilled_entries DESC '
        'LIMIT 5;',
        "Which matters have the most underbilled entries in 2025?"
    ),
    (
        'SELECT t."timekeeper_name", '
        'ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY (tc."worked_rate" / NULLIF(tc."standard_rate", 0)))::numeric, 2) AS median_rate_ratio '
        'FROM "timecard" tc '
        'JOIN "timekeeper" t ON tc."timekeeper_id" = t."timekeeper_id" '
        'WHERE EXTRACT(YEAR FROM tc."date") = 2025 '
        'AND tc."is_active" = \'Y\' '
        'AND tc."is_nonbillable" = \'N\' '
        'AND t."is_active" = \'Y\' '
        'GROUP BY t."timekeeper_name" '
        'HAVING PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY (tc."worked_rate" / NULLIF(tc."standard_rate", 0))) < 0.9 '
        'ORDER BY median_rate_ratio ASC;',
        "Which timekeepers have a median worked-to-standard rate ratio below 90% in 2025?"
    ),
    (
        'SELECT m."category", ROUND(AVG(mts."standard_amount" - mts."worked_amount")::numeric, 2) AS avg_underbilling '
        'FROM "matter_timekeeper_summary" mts '
        'JOIN "matter" m ON mts."matter_id" = m."matter_id" '
        'WHERE mts."year" = 2025 '
        'AND mts."is_error" = \'N\' '
        'AND m."is_active" = \'Y\' '
        'GROUP BY m."category" '
        'ORDER BY avg_underbilling DESC;',
        "Which matter categories have the highest average underbilling in 2025?"
    ),
    (
        'SELECT c."client_name", '
        'ROUND((SUM(CASE WHEN tc."is_nonbillable" = \'Y\' OR tc."is_no_charge" = \'Y\' THEN tc."worked_hours" ELSE 0 END) / NULLIF(SUM(tc."worked_hours"), 0) * 100)::numeric, 2) AS non_billed_percent '
        'FROM "timecard" tc '
        'JOIN "matter" m ON tc."matter_id" = m."matter_id" '
        'JOIN "client" c ON m."client_id" = c."client_id" '
        'WHERE EXTRACT(YEAR FROM tc."date") = 2025 '
        'AND tc."is_active" = \'Y\' '
        'AND c."is_active" = \'Y\' '
        'GROUP BY c."client_name" '
        'HAVING SUM(CASE WHEN tc."is_nonbillable" = \'Y\' OR tc."is_no_charge" = \'Y\' THEN tc."worked_hours" ELSE 0 END) / NULLIF(SUM(tc."worked_hours"), 0) > 0.2 '
        'ORDER BY non_billed_percent DESC;',
        "Which clients have more than 20% of their hours as non-billable or no-charge in 2025?"
    ),
    (
        'SELECT m."matter_name", '
        'ROUND(SUM(CASE WHEN tc."worked_amount" < tc."standard_amount" THEN (tc."standard_amount" - tc."worked_amount") ELSE 0 END)::numeric, 2) AS underbilling_amount '
        'FROM "timecard" tc '
        'JOIN "matter" m ON tc."matter_id" = m."matter_id" '
        'WHERE EXTRACT(YEAR FROM tc."date") = 2025 '
        'AND tc."is_active" = \'Y\' '
        'AND m."is_active" = \'Y\' '
        'AND tc."is_nonbillable" = \'N\' '
        'AND tc."is_no_charge" = \'N\' '
        'GROUP BY m."matter_name" '
        'ORDER BY underbilling_amount DESC '
        'LIMIT 5;',
        "Which matters have the highest underbilling amount for billable timecards in 2025?"
    ),
    # New Queries to Address Drawbacks
    (
        'SELECT EXTRACT(MONTH FROM tc."date") AS month, ROUND(SUM(tc."standard_amount" - tc."worked_amount")::numeric, 2) AS monthly_underbilling '
        'FROM "timecard" tc '
        'JOIN "matter" m ON tc."matter_id" = m."matter_id" '
        'WHERE EXTRACT(YEAR FROM tc."date") = 2025 '
        'AND tc."is_active" = \'Y\' '
        'AND m."is_active" = \'Y\' '
        'AND tc."is_nonbillable" = \'N\' '
        'AND tc."is_no_charge" = \'N\' '
        'GROUP BY EXTRACT(MONTH FROM tc."date") '
        'ORDER BY month;',
        "What is the monthly breakdown of underbilling amounts for billable timecards in 2025?"
    ),
    (
        'SELECT m."matter_name", SUM(tc."worked_hours") AS total_hours '
        'FROM "timecard" tc '
        'JOIN "matter" m ON tc."matter_id" = m."matter_id" '
        'WHERE m."is_flat_fees" = \'Y\' '
        'AND EXTRACT(YEAR FROM tc."date") = 2025 '
        'AND tc."is_active" = \'Y\' '
        'AND m."is_active" = \'Y\' '
        'GROUP BY m."matter_name" '
        'HAVING SUM(tc."worked_hours") > 100 '
        'ORDER BY total_hours DESC '
        'LIMIT 5;',
        "Which flat-fee matters have more than 100 hours worked in 2025?"
    ),
    (
        'SELECT m."matter_name", COUNT(tc."timecard_id") AS null_rate_timecards '
        'FROM "timecard" tc '
        'JOIN "matter" m ON tc."matter_id" = m."matter_id" '
        'WHERE (tc."worked_rate" IS NULL OR tc."standard_rate" IS NULL) '
        'AND EXTRACT(YEAR FROM tc."date") = 2025 '
        'AND tc."is_active" = \'Y\' '
        'AND m."is_active" = \'Y\' '
        'GROUP BY m."matter_name" '
        'ORDER BY null_rate_timecards DESC;',
        "Which matters have timecards with missing worked or standard rates in 2025?"
    ),
    (
        'SELECT m."matter_name", COUNT(tc."timecard_id") AS currency_mismatch '
        'FROM "timecard" tc '
        'JOIN "matter" m ON tc."matter_id" = m."matter_id" '
        'WHERE tc."standard_rate_currency" != m."matter_currency" '
        'AND EXTRACT(YEAR FROM tc."date") = 2025 '
        'AND tc."is_active" = \'Y\' '
        'AND m."is_active" = \'Y\' '
        'GROUP BY m."matter_name" '
        'ORDER BY currency_mismatch DESC;',
        "Which matters have timecards with standard rate currency mismatches in 2025?"
    ),
    (
        'SELECT rd."arrangement", ROUND(AVG(rd."deviation_amount")::numeric, 2) AS avg_discount '
        'FROM "rate_detail" rd '
        'JOIN "rate_component" rc ON rd."rate_component_id" = rc."rate_component_id" '
        'JOIN "rate_set_link" rsl ON rc."rate_component_id" = rsl."rate_component_id" '
        'JOIN "rate_set" rs ON rsl."rate_set_id" = rs."rate_set_id" '
        'JOIN "matter" m ON rs."rate_set_id" = m."rate_set_id" '
        'WHERE rd."start_date" <= \'2025-12-31\' '
        'AND (rd."end_date" >= \'2025-01-01\' OR rd."end_date" IS NULL) '
        'AND m."is_active" = \'Y\' '
        'GROUP BY rd."arrangement" '
        'ORDER BY avg_discount DESC;',
        "Which rate arrangements have the highest average discount amounts in 2025?"
    ),
    (
        'SELECT t."timekeeper_name", '
        'ROUND(SUM(tc."worked_hours" * t."cost_rate")::numeric, 2) AS total_cost '
        'FROM "timecard" tc '
        'JOIN "timekeeper" t ON tc."timekeeper_id" = t."timekeeper_id" '
        'WHERE tc."is_nonbillable" = \'Y\' '
        'AND EXTRACT(YEAR FROM tc."date") = 2025 '
        'AND tc."is_active" = \'Y\' '
        'AND t."is_active" = \'Y\' '
        'GROUP BY t."timekeeper_name" '
        'ORDER BY total_cost DESC '
        'LIMIT 5;',
        "Which timekeepers have the highest cost for non-billable hours in 2025?"
    ),
    (
        'SELECT c."type" AS client_type, ROUND(SUM(mts."standard_amount" - mts."worked_amount")::numeric, 2) AS total_underbilling '
        'FROM "matter_timekeeper_summary" mts '
        'JOIN "matter" m ON mts."matter_id" = m."matter_id" '
        'JOIN "client" c ON m."client_id" = c."client_id" '
        'WHERE mts."year" = 2025 '
        'AND mts."is_error" = \'N\' '
        'AND c."is_active" = \'Y\' '
        'GROUP BY c."type" '
        'ORDER BY total_underbilling DESC;',
        "Which client types have the highest total underbilling in 2025?"
    ),
    (
        'SELECT m."matter_name", '
        'ROUND(STDDEV(tc."worked_rate" / NULLIF(tc."standard_rate", 0))::numeric, 2) AS rate_ratio_stddev '
        'FROM "timecard" tc '
        'JOIN "matter" m ON tc."matter_id" = m."matter_id" '
        'WHERE EXTRACT(YEAR FROM tc."date") = 2025 '
        'AND tc."is_active" = \'Y\' '
        'AND m."is_active" = \'Y\' '
        'AND tc."is_nonbillable" = \'N\' '
        'GROUP BY m."matter_name" '
        'HAVING STDDEV(tc."worked_rate" / NULLIF(tc."standard_rate", 0)) > 0.1 '
        'ORDER BY rate_ratio_stddev DESC '
        'LIMIT 5;',
        "Which matters have the highest variability in worked-to-standard rate ratios in 2025?"
    ),
    (
        'SELECT t."timekeeper_name", '
        'ROUND(SUM(CASE WHEN tc."worked_amount" < tc."standard_amount" THEN (tc."standard_amount" - tc."worked_amount") ELSE 0 END)::numeric, 2) AS underbilling_2025, '
        'ROUND(LAG(SUM(CASE WHEN tc."worked_amount" < tc."standard_amount" THEN (tc."standard_amount" - tc."worked_amount") ELSE 0 END)) OVER (PARTITION BY t."timekeeper_name" ORDER BY EXTRACT(YEAR FROM tc."date"))::numeric, 2) AS underbilling_2024 '
        'FROM "timecard" tc '
        'JOIN "timekeeper" t ON tc."timekeeper_id" = t."timekeeper_id" '
        'WHERE EXTRACT(YEAR FROM tc."date") IN (2024, 2025) '
        'AND tc."is_active" = \'Y\' '
        'AND t."is_active" = \'Y\' '
        'AND tc."is_nonbillable" = \'N\' '
        'AND tc."is_no_charge" = \'N\' '
        'GROUP BY t."timekeeper_name", EXTRACT(YEAR FROM tc."date") '
        'HAVING EXTRACT(YEAR FROM tc."date") = 2025 '
        'ORDER BY underbilling_2025 DESC '
        'LIMIT 5;',
        "Which timekeepers have the highest underbilling in 2025 compared to 2024?"
    ),
    (
        'SELECT m."matter_name", COUNT(DISTINCT rd."rate_detail_id") AS missing_rate_details '
        'FROM "matter" m '
        'LEFT JOIN "rate_set" rs ON m."rate_set_id" = rs."rate_set_id" '
        'LEFT JOIN "rate_set_link" rsl ON rs."rate_set_id" = rsl."rate_set_id" '
        'LEFT JOIN "rate_detail" rd ON rsl."rate_component_id" = rd."rate_component_id" '
        'WHERE EXTRACT(YEAR FROM rd."start_date") = 2025 '
        'AND m."is_active" = \'Y\' '
        'AND rd."rate_detail_id" IS NULL '
        'GROUP BY m."matter_name" '
        'ORDER BY missing_rate_details DESC '
        'LIMIT 5;',
        "Which matters lack corresponding rate details in 2025?"
    )
]

underbilling_documentation = [
    "Underbilling occurs when the billed amount (worked_amount) is less than the expected amount (standard_amount) in timecard or matter_timekeeper_summary, due to lower rates, discounts, non-billable hours, no-charge entries, or flat-fee arrangements.",
    "In the timecard table, worked_amount is calculated using worked_rate, while standard_amount uses standard_rate. A worked_amount lower than standard_amount indicates underbilling.",
    "The is_nonbillable = 'Y' flag in timecard marks hours that are not billed, contributing to underbilling by reducing revenue.",
    "The is_no_charge = 'Y' flag in timecard indicates hours worked but not charged to the client, a direct form of underbilling.",
    "In rate_detail, deviation_amount and deviation_percent quantify discounts applied to rates, reducing worked_amount and causing underbilling.",
    "The matter_timekeeper_summary table aggregates worked_amount and standard_amount by timekeeper and matter, useful for calculating total or average underbilling.",
    "The timekeeper table’s type column (e.g., Partner, Associate) helps analyze underbilling trends by role.",
    "The matter table’s type and category columns categorize matters (e.g., Litigation, Corporate), enabling underbilling analysis by matter characteristics.",
    "The client table’s client_name links to matters, allowing identification of clients with high underbilling.",
    "Comparing worked_rate to standard_rate in timecard reveals rate reductions that lead to underbilling.",
    "Data errors, such as worked_amount exceeding standard_amount or null worked_rate in timecard, can mask or cause underbilling and require investigation.",
    "The rate_set, rate_set_link, and rate_component tables connect matters to rate structures, with rate_detail providing discount details via deviation_amount.",
    "Realization rate (worked_amount / standard_amount * 100 or worked_rate / standard_rate * 100) in matter_timekeeper_summary or timecard indicates underbilling when below 100%.",
    "Currency mismatches between worked_currency or standard_rate_currency in timecard and matter_currency in matter may cause billing errors, potentially leading to underbilling.",
    "The is_active = 'Y' flag in timecard, matter, and client ensures analysis focuses on current, relevant data.",
    "The is_error = 'N' flag in matter_timekeeper_summary filters out erroneous entries to ensure accurate underbilling calculations.",
    "The is_flat_fees = 'Y' flag in matter or is_flat_fee = 'Y' in timecard indicates fixed-fee arrangements, where underbilling occurs if hours worked exceed expected value.",
    "The timekeeper table’s cost_rate and cost_rate_currency allow cost-based underbilling analysis, e.g., non-billable hours’ financial impact.",
    "The client table’s type column (e.g., Corporate, Individual) enables underbilling analysis by client characteristics.",
    "The rate_detail table’s arrangement column categorizes billing arrangements, useful for analyzing discount patterns.",
    "Statistical measures like standard deviation of worked_rate / standard_rate in timecard reveal variability in billing practices.",
    "Year-over-year comparisons using timecard or matter_timekeeper_summary data identify trends in underbilling.",
    "Missing rate_detail entries for matters indicate potential billing configuration errors, leading to underbilling."
]

ddl_statements = [
    # timecard table
    'Table: timecard - Column: timecard_id, Type: text, Nullable: YES - Column: timekeeper_id, Type: text, Nullable: YES - Column: matter_id, Type: text, Nullable: YES - Column: date, Type: timestamp without time zone, Nullable: YES - Column: time_type, Type: text, Nullable: YES - Column: is_flat_fee, Type: text, Nullable: YES - Column: is_nonbillable, Type: text, Nullable: YES - Column: is_no_charge, Type: text, Nullable: YES - Column: is_active, Type: text, Nullable: YES - Column: worked_hours, Type: double precision, Nullable: YES - Column: worked_rate, Type: bigint, Nullable: YES - Column: worked_amount, Type: bigint, Nullable: YES - Column: worked_currency, Type: text, Nullable: YES - Column: standard_rate, Type: bigint, Nullable: YES - Column: standard_amount, Type: bigint, Nullable: YES - Column: standard_rate_currency, Type: text, Nullable: YES',
    'Adding ddl: CREATE TABLE timecard ("timecard_id" TEXT, "timekeeper_id" TEXT, "matter_id" TEXT, "date" TIMESTAMP WITHOUT TIME ZONE, "time_type" TEXT, "is_flat_fee" TEXT, "is_nonbillable" TEXT, "is_no_charge" TEXT, "is_active" TEXT, "worked_hours" DOUBLE PRECISION, "worked_rate" BIGINT, "worked_amount" BIGINT, "worked_currency" TEXT, "standard_rate" BIGINT, "standard_amount" BIGINT, "standard_rate_currency" TEXT);',

    # matter_timekeeper_summary table
    'Table: matter_timekeeper_summary - Column: tk_mat_sum_id, Type: text, Nullable: YES - Column: matter_id, Type: text, Nullable: YES - Column: timekeeper_id, Type: text, Nullable: YES - Column: year, Type: bigint, Nullable: YES - Column: worked_hours, Type: bigint, Nullable: YES - Column: worked_amount, Type: bigint, Nullable: YES - Column: standard_amount, Type: bigint, Nullable: YES - Column: currency, Type: text, Nullable: YES - Column: is_error, Type: text, Nullable: YES - Column: error_message, Type: double precision, Nullable: YES',
    'Adding ddl: CREATE TABLE matter_timekeeper_summary ("tk_mat_sum_id" TEXT, "matter_id" TEXT, "timekeeper_id" TEXT, "year" BIGINT, "worked_hours" BIGINT, "worked_amount" BIGINT, "standard_amount" BIGINT, "currency" TEXT, "is_error" TEXT, "error_message" DOUBLE PRECISION);',

    # timekeeper table
    'Table: timekeeper - Column: timekeeper_id, Type: text, Nullable: YES - Column: timekeeper_number, Type: bigint, Nullable: YES - Column: timekeeper_name, Type: text, Nullable: YES - Column: timekeeper_sort_name, Type: text, Nullable: YES - Column: is_active, Type: text, Nullable: YES - Column: rate_year, Type: double precision, Nullable: YES - Column: type, Type: text, Nullable: YES - Column: attribute, Type: text, Nullable: YES - Column: category, Type: text, Nullable: YES - Column: budget_hours, Type: double precision, Nullable: YES - Column: cost_rate, Type: double precision, Nullable: YES - Column: cost_rate_currency, Type: text, Nullable: YES - Column: is_current, Type: text, Nullable: YES - Column: is_error, Type: text, Nullable: YES - Column: error_message, Type: text, Nullable: YES',
    'Adding ddl: CREATE TABLE timekeeper ("timekeeper_id" TEXT, "timekeeper_number" BIGINT, "timekeeper_name" TEXT, "timekeeper_sort_name" TEXT, "is_active" TEXT, "rate_year" DOUBLE PRECISION, "type" TEXT, "attribute" TEXT, "category" TEXT, "budget_hours" DOUBLE PRECISION, "cost_rate" DOUBLE PRECISION, "cost_rate_currency" TEXT, "is_current" TEXT, "is_error" TEXT, "error_message" TEXT);',

    # matter table
    'Table: matter - Column: matter_id, Type: text, Nullable: YES - Column: matter_number, Type: text, Nullable: YES - Column: matter_name, Type: text, Nullable: YES - Column: client_id, Type: text, Nullable: YES - Column: related_matter_id, Type: text, Nullable: YES - Column: is_active, Type: text, Nullable: YES - Column: open_date, Type: timestamp without time zone, Nullable: YES - Column: matter_currency, Type: text, Nullable: YES - Column: rate_set_id, Type: text, Nullable: YES - Column: notification_days, Type: double precision, Nullable: YES - Column: rate_review_date, Type: timestamp without time zone, Nullable: YES - Column: billing_timekeeper_id, Type: text, Nullable: YES - Column: responsible_timekeeper_id, Type: text, Nullable: YES - Column: supervising_timekeeper_id, Type: text, Nullable: YES - Column: type, Type: text, Nullable: YES - Column: attribute, Type: text, Nullable: YES - Column: category, Type: text, Nullable: YES - Column: is_proforma_adjustments, Type: text, Nullable: YES - Column: is_flat_fees, Type: text, Nullable: YES - Column: is_ebilled, Type: text, Nullable: YES - Column: is_current, Type: text, Nullable: YES - Column: is_error, Type: text, Nullable: YES - Column: error_message, Type: double precision, Nullable: YES',
    'Adding ddl: CREATE TABLE matter ("matter_id" TEXT, "matter_number" TEXT, "matter_name" TEXT, "client_id" TEXT, "related_matter_id" TEXT, "is_active" TEXT, "open_date" TIMESTAMP WITHOUT TIME ZONE, "matter_currency" TEXT, "rate_set_id" TEXT, "notification_days" DOUBLE PRECISION, "rate_review_date" TIMESTAMP WITHOUT TIME ZONE, "billing_timekeeper_id" TEXT, "responsible_timekeeper_id" TEXT, "supervising_timekeeper_id" TEXT, "type" TEXT, "attribute" TEXT, "category" TEXT, "is_proforma_adjustments" TEXT, "is_flat_fees" TEXT, "is_ebilled" TEXT, "is_current" TEXT, "is_error" TEXT, "error_message" DOUBLE PRECISION);',

    # client table
    'Table: client - Column: client_id, Type: text, Nullable: YES - Column: client_number, Type: bigint, Nullable: YES - Column: client_name, Type: text, Nullable: YES - Column: related_client_id, Type: text, Nullable: YES - Column: is_active, Type: text, Nullable: YES - Column: open_date, Type: timestamp without time zone, Nullable: YES - Column: rate_set_id, Type: text, Nullable: YES - Column: notification_days, Type: double precision, Nullable: YES - Column: rate_review_date, Type: timestamp without time zone, Nullable: YES - Column: year_end_month, Type: double precision, Nullable: YES - Column: type, Type: text, Nullable: YES - Column: attribute, Type: text, Nullable: YES - Column: category, Type: text, Nullable: YES - Column: relationship_timekeeper_id, Type: text, Nullable: YES - Column: is_proforma_adjustments, Type: text, Nullable: YES - Column: is_ebilled, Type: text, Nullable: YES - Column: is_current, Type: text, Nullable: YES - Column: is_error, Type: text, Nullable: YES - Column: error_message, Type: text, Nullable: YES',
    'Adding ddl: CREATE TABLE client ("client_id" TEXT, "client_number" BIGINT, "client_name" TEXT, "related_client_id" TEXT, "is_active" TEXT, "open_date" TIMESTAMP WITHOUT TIME ZONE, "rate_set_id" TEXT, "notification_days" DOUBLE PRECISION, "rate_review_date" TIMESTAMP WITHOUT TIME ZONE, "year_end_month" DOUBLE PRECISION, "type" TEXT, "attribute" TEXT, "category" TEXT, "relationship_timekeeper_id" TEXT, "is_proforma_adjustments" TEXT, "is_ebilled" TEXT, "is_current" TEXT, "is_error" TEXT, "error_message" TEXT);',

    # rate_set table
    'Table: rate_set - Column: rate_set_id, Type: text, Nullable: YES - Column: rate_set_code, Type: text, Nullable: YES - Column: rate_set_name, Type: text, Nullable: YES',
    'Adding ddl: CREATE TABLE rate_set ("rate_set_id" TEXT, "rate_set_code" TEXT, "rate_set_name" TEXT);',

    # rate_set_link table
    'Table: rate_set_link - Column: rate_set_link_id, Type: text, Nullable: YES - Column: rate_set_id, Type: text, Nullable: YES - Column: rate_component_id, Type: text, Nullable: YES - Column: rate_type, Type: text, Nullable: YES - Column: start_date, Type: double precision, Nullable: YES - Column: end_date, Type: double precision, Nullable: YES - Column: ordinal, Type: bigint, Nullable: YES',
    'Adding ddl: CREATE TABLE rate_set_link ("rate_set_link_id" TEXT, "rate_set_id" TEXT, "rate_component_id" TEXT, "rate_type" TEXT, "start_date" DOUBLE PRECISION, "end_date" DOUBLE PRECISION, "ordinal" BIGINT);',

    # rate_component table
    'Table: rate_component - Column: rate_component_id, Type: text, Nullable: YES - Column: rate_type, Type: text, Nullable: YES - Column: rate_num, Type: bigint, Nullable: YES - Column: source_code, Type: text, Nullable: YES - Column: source_name, Type: text, Nullable: YES - Column: start_date, Type: timestamp without time zone, Nullable: YES - Column: end_date, Type: timestamp without time zone, Nullable: YES - Column: frozen_date, Type: timestamp without time zone, Nullable: YES - Column: is_step_up, Type: text, Nullable: YES - Column: is_skip_client_exceptions, Type: text, Nullable: YES - Column: default_rate_amount, Type: double precision, Nullable: YES - Column: default_currency, Type: text, Nullable: YES - Column: is_match_currency, Type: text, Nullable: YES - Column: is_error, Type: text, Nullable: YES - Column: error_message, Type: double precision, Nullable: YES',
    'Adding ddl: CREATE TABLE rate_component ("rate_component_id" TEXT, "rate_type" TEXT, "rate_num" BIGINT, "source_code" TEXT, "source_name" TEXT, "start_date" TIMESTAMP WITHOUT TIME ZONE, "end_date" TIMESTAMP WITHOUT TIME ZONE, "frozen_date" TIMESTAMP WITHOUT TIME ZONE, "is_step_up" TEXT, "is_skip_client_exceptions" TEXT, "default_rate_amount" DOUBLE PRECISION, "default_currency" TEXT, "is_match_currency" TEXT, "is_error" TEXT, "error_message" DOUBLE PRECISION);',

    # rate_detail table
    'Table: rate_detail - Column: rate_detail_id, Type: text, Nullable: YES - Column: rate_component_id, Type: text, Nullable: YES - Column: is_virtual, Type: text, Nullable: YES - Column: start_date, Type: timestamp without time zone, Nullable: YES - Column: end_date, Type: timestamp without time zone, Nullable: YES - Column: ordinal, Type: bigint, Nullable: YES - Column: rate_id, Type: text, Nullable: YES - Column: rate_amount, Type: double precision, Nullable: YES - Column: deviation_amount, Type: double precision, Nullable: YES - Column: deviation_percent, Type: double precision, Nullable: YES - Column: rate_currency, Type: text, Nullable: YES - Column: rounding_method, Type: text, Nullable: YES - Column: matter_rate_min, Type: double precision, Nullable: YES - Column: matter_rate_max, Type: double precision, Nullable: YES - Column: timekeeper_id, Type: text, Nullable: YES - Column: title, Type: text, Nullable: YES - Column: officet, Type: text, Nullable: YES - Column: officem, Type: text, Nullable: YES - Column: rate_class, Type: text, Nullable: YES - Column: departmentt, Type: text, Nullable: YES - Column: departmentm, Type: text, Nullable: YES - Column: sectiont, Type: text, Nullable: YES - Column: sectionm, Type: text, Nullable: YES - Column: rate_year_min, Type: double precision, Nullable: YES - Column: rate_year_max, Type: double precision, Nullable: YES - Column: arrangement, Type: text, Nullable: YES - Column: practice_group, Type: text, Nullable: YES',
    'Adding ddl: CREATE TABLE rate_detail ("rate_detail_id" TEXT, "rate_component_id" TEXT, "is_virtual" TEXT, "start_date" TIMESTAMP WITHOUT TIME ZONE, "end_date" TIMESTAMP WITHOUT TIME ZONE, "ordinal" BIGINT, "rate_id" TEXT, "rate_amount" DOUBLE PRECISION, "deviation_amount" DOUBLE PRECISION, "deviation_percent" DOUBLE PRECISION, "rate_currency" TEXT, "rounding_method" TEXT, "matter_rate_min" DOUBLE PRECISION, "matter_rate_max" DOUBLE PRECISION, "timekeeper_id" TEXT, "title" TEXT, "officet" TEXT, "officem" TEXT, "rate_class" TEXT, "departmentt" TEXT, "departmentm" TEXT, "sectiont" TEXT, "sectionm" TEXT, "rate_year_min" DOUBLE PRECISION, "rate_year_max" DOUBLE PRECISION, "arrangement" TEXT, "practice_group" TEXT);',
]

def main():
    parser = argparse.ArgumentParser(description="Underbilling CLI")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a JSON response instead of human-readable output",
    )
    args = parser.parse_args()

    # 1) Instantiate and configure Vanna
    vn = MyVanna(config={
        'api_key': os.getenv('OPENAI_API_KEY'),
        'model': os.getenv('MODEL_NAME', 'gpt-4o-mini'),
        'embedding_model': 'text-embedding-3-small',
        'allow_llm_to_see_data': True,
    })

    # 2) Train Vanna
    for sql, question in underbilling_sql_training:
        vn.train(sql=sql, question=question)
    for doc in underbilling_documentation:
        vn.train(documentation=doc)
    for ddl in ddl_statements:
        vn.train(ddl=ddl)

    # 3) Connect to PostgreSQL
    vn.connect_to_postgres(
        host=os.getenv('POSTGRES_HOST'),
        dbname=os.getenv('POSTGRES_DB'),
        user=os.getenv('UNDERBILLING_USER'),
        password=os.getenv('UNDERBILLING_PASSWORD'),
        port=int(os.getenv('POSTGRES_PORT', 5432))
    )

    # Ensure charts directory exists
    os.makedirs("charts", exist_ok=True)

    print("Underbilling Agent-CLI ready. Type your question (or ‘quit’ to exit).")
    while True:
        question = input("\n> ").strip()
        if not question or question.lower() in ("quit", "exit"):
            break

        result = vn.ask(question)
        if result is None:
            if not args.json:
                print("No results returned.\n")
            continue

        df, fig = result
        if df is None or df.empty:
            if not args.json:
                print("No results returned.\n")
            continue

        # 4) Save or generate chart
        chart_id = uuid.uuid4().hex
        chart_filename = f"{chart_id}.html"
        chart_path = os.path.join("charts", chart_filename)
        if isinstance(fig, (_bdt.BaseFigure, go.Figure)):
            chosen_fig = fig
        elif len(df.columns) >= 2:
            x, y = df.columns[0], df.columns[1]
            chosen_fig = px.bar(df.head(10), x=x, y=y, title=f"{y} by {x}")
        else:
            chosen_fig = None

        chart_url = None
        if chosen_fig:
            chosen_fig.write_html(chart_path, include_plotlyjs="cdn")
            chart_url = f"http://127.0.0.1:8000/{chart_filename}"

        # 5) Prepare JSON-friendly data
        records = df.head(100).copy()
        for col in records.select_dtypes(include=["datetime64[ns]"]):
            records[col] = records[col].dt.strftime("%Y-%m-%dT%H:%M:%S")
        data = records.to_dict(orient="records")

        # 6) Generate summary
        try:
            summary = summarize(question, data)
        except Exception as e:
            summary = f"Could not generate summary: {e}"

        # 7) Emit output
        payload = {
            "data": data,
            "chart_url": chart_url,
            "summary": summary
        }

        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            # human-readable fallback
            print("\nRESULTS (top 10 rows):")
            print(df.head(10).to_string(index=False))
            if chart_url:
                print(f"\nChart URL: {chart_url}")
            print(f"\nSUMMARY:\n{summary}")

    print("\nGoodbye!")


if __name__ == "__main__":
    main()
