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