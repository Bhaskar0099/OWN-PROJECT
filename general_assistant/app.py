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
    def _init_(self, config=None):
        ChromaDB_VectorStore._init_(self, config=config)
        OpenAI_Chat._init_(self, config=config)
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
# Training data for the General Query Assistant
# This dataset includes SQL queries paired with natural-language questions, covering various categories.
# All column names match the YAML configuration exactly.

general_query_sql_training = [
    # Lists Category
    (
        'SELECT c.client_id, c.client_name, t.timekeeper_name AS relationship_manager, '
        'COUNT(DISTINCT m.matter_id) AS active_matter_count, '
        'COALESCE(SUM(mts.worked_amount), 0) AS total_fees_paid, '
        'STRING_AGG(DISTINCT m.type, \', \') AS legal_service_types, '
        'MIN(c.open_date) AS relationship_start_date, '
        'ROUND(EXTRACT(DAY FROM CURRENT_DATE - MIN(c.open_date)) / 365.0, 2) AS relationship_years '
        'FROM client c '
        'JOIN timekeeper t ON c.relationship_timekeeper_id = t.timekeeper_id '
        'LEFT JOIN matter m ON c.client_id = m.client_id AND m.is_active = \'Y\' '
        'LEFT JOIN matter_timekeeper_summary mts ON m.matter_id = mts.matter_id '
        'WHERE t.timekeeper_name = \'Lynn Bond\' '
        'AND c.is_active = \'Y\' '
        'AND mts.year = EXTRACT(YEAR FROM CURRENT_DATE) '
        'GROUP BY c.client_id, c.client_name, t.timekeeper_name '
        'ORDER BY c.client_name;',
        "List all clients for Lynn, including client names, contact information, active matters, total fees paid to date, and any outstanding balances. Also, provide a summary of the types of legal services provided to each client and the duration of Lynn's relationship with each client."
    ),
    (
        'SELECT c.client_id, c.client_name, m.matter_id, m.matter_name, m.open_date, '
        'md.office, md.department, md.practice_group, COALESCE(SUM(mts.worked_hours), 0) AS hours_worked, '
        'STRING_AGG(DISTINCT m.type, \', \') AS legal_service_types '
        'FROM client c '
        'JOIN matter m ON c.client_id = m.client_id '
        'JOIN matter_date md ON m.matter_id = md.matter_id '
        'LEFT JOIN matter_timekeeper_summary mts ON m.matter_id = mts.matter_id '
        'WHERE m.open_date >= CURRENT_DATE - INTERVAL \'21 days\' '
        'AND m.is_active = \'Y\' '
        'AND c.is_active = \'Y\' '
        'AND md.end_date IS NULL '
        'GROUP BY c.client_id, c.client_name, m.matter_id, m.matter_name, m.open_date, md.office, md.department, md.practice_group '
        'ORDER BY c.client_name, m.open_date;',
        "Retrieve a detailed list of all new clients and matters opened in the last 21 days for the billing attorney, including client names, matter details, office, department, practice group, total hours worked, and a summary of legal service types provided."
    ),
    (
        'SELECT t.timekeeper_name AS billing_attorney, COUNT(DISTINCT c.client_id) AS client_count, '
        'COUNT(DISTINCT m.matter_id) AS matter_count, '
        'COALESCE(SUM(mts.worked_hours), 0) AS total_hours_worked, '
        'STRING_AGG(DISTINCT md.practice_group, \', \') AS practice_groups '
        'FROM matter m '
        'JOIN timekeeper t ON m.billing_timekeeper_id = t.timekeeper_id '
        'JOIN client c ON m.client_id = c.client_id '
        'JOIN matter_date md ON m.matter_id = md.matter_id '
        'LEFT JOIN matter_timekeeper_summary mts ON m.matter_id = mts.matter_id '
        'WHERE t.timekeeper_name = \'Lynn Bond\' '
        'AND m.is_active = \'Y\' '
        'AND c.is_active = \'Y\' '
        'AND md.end_date IS NULL '
        'AND mts.year = EXTRACT(YEAR FROM CURRENT_DATE) '
        'GROUP BY t.timekeeper_name '
        'ORDER BY matter_count DESC;',
        "Provide a comprehensive report on the client and matter load for billing attorney Lynn Bond, including the number of clients, active matters, total hours worked, and a breakdown of practice groups involved."
    ),
    (
        'SELECT md.office, md.department, md.practice_group, '
        'COUNT(DISTINCT c.client_id) AS client_count, COUNT(DISTINCT m.matter_id) AS matter_count, '
        'COALESCE(SUM(mts.worked_hours), 0) AS total_hours, '
        'ROUND((COUNT(DISTINCT m.matter_id) - LAG(COUNT(DISTINCT m.matter_id)) OVER (PARTITION BY md.office ORDER BY md.department)) / '
        'NULLIF(LAG(COUNT(DISTINCT m.matter_id)) OVER (PARTITION BY md.office ORDER BY md.department), 0)::numeric * 100, 2) AS matter_trend_percent '
        'FROM matter m '
        'JOIN matter_date md ON m.matter_id = md.matter_id '
        'JOIN client c ON m.client_id = c.client_id '
        'LEFT JOIN matter_timekeeper_summary mts ON m.matter_id = mts.matter_id '
        'WHERE m.is_active = \'Y\' '
        'AND c.is_active = \'Y\' '
        'AND md.end_date IS NULL '
        'AND mts.year = EXTRACT(YEAR FROM CURRENT_DATE) '
        'GROUP BY md.office, md.department, md.practice_group '
        'ORDER BY client_count DESC, matter_count DESC;',
        "Generate an in-depth load analysis report for clients and matters, grouped by office, department, and practice group, including client and matter counts, total hours worked, and percentage trends in matter counts compared to previous periods."
    ),
    (
        'SELECT c.client_id, c.client_name, c.category, COUNT(m.matter_id) AS matter_count, '
        'COALESCE(SUM(mts.worked_amount), 0) AS total_fees, '
        'STRING_AGG(DISTINCT m.type, \', \') AS service_types '
        'FROM client c '
        'LEFT JOIN matter m ON c.client_id = m.client_id AND m.is_active = \'Y\' '
        'LEFT JOIN matter_timekeeper_summary mts ON m.matter_id = mts.matter_id '
        'WHERE (c.client_name ILIKE \'%Retail%\' OR c.category ILIKE \'%Retail%\' OR c.type ILIKE \'%Retail%\') '
        'AND c.is_active = \'Y\' '
        'AND mts.year = EXTRACT(YEAR FROM CURRENT_DATE) '
        'GROUP BY c.client_id, c.client_name, c.category '
        'ORDER BY matter_count DESC;',
        "Compile a detailed list of clients in the retail sector, identified by client name, category, or type containing 'Retail', including matter counts, total fees paid, and a summary of legal service types provided."
    ),
    (
        'SELECT c2.client_id, c2.client_name, t.timekeeper_name AS relationship_manager, '
        'COUNT(m.matter_id) AS matter_count, '
        'COALESCE(SUM(mts.worked_amount), 0) AS total_fees, '
        'MIN(c2.open_date) AS relationship_start '
        'FROM client c1 '
        'JOIN client c2 ON c1.related_client_id = c2.related_client_id '
        'JOIN timekeeper t ON c2.relationship_timekeeper_id = t.timekeeper_id '
        'LEFT JOIN matter m ON c2.client_id = m.client_id AND m.is_active = \'Y\' '
        'LEFT JOIN matter_timekeeper_summary mts ON m.matter_id = mts.matter_id '
        'WHERE c1.client_name ILIKE \'%Pepsi%\' '
        'AND c1.client_id != c2.client_id '
        'AND c2.is_active = \'Y\' '
        'AND mts.year = EXTRACT(YEAR FROM CURRENT_DATE) '
        'GROUP BY c2.client_id, c2.client_name, t.timekeeper_name '
        'ORDER BY matter_count DESC;',
        "Identify all clients related to Pepsi, including their client names, relationship managers, active matter counts, total fees paid, and the start date of their relationship with the firm."
    ),
    (
        'SELECT c.client_id, c.client_name, t.timekeeper_name AS relationship_manager, '
        'COUNT(m.matter_id) AS matter_count, '
        'COALESCE(SUM(mts.worked_amount), 0) AS total_fees, '
        'STRING_AGG(DISTINCT m.type, \', \') AS service_types '
        'FROM client c '
        'JOIN timekeeper t ON c.relationship_timekeeper_id = t.timekeeper_id '
        'LEFT JOIN matter m ON c.client_id = m.client_id AND m.is_active = \'Y\' '
        'LEFT JOIN matter_timekeeper_summary mts ON m.matter_id = mts.matter_id '
        'WHERE c.is_active = \'Y\' '
        'AND mts.year = EXTRACT(YEAR FROM CURRENT_DATE) '
        'GROUP BY c.client_id, c.client_name, t.timekeeper_name '
        'ORDER BY c.client_name;',
        "Provide a comprehensive list of all active clients, including client names, relationship managers, active matter counts, total fees paid, and a summary of legal service types provided."
    ),

    # Flat Fees Category
    (
        'SELECT c.client_id, c.client_name, t.timekeeper_name AS billing_attorney, '
        'COUNT(CASE WHEN m.is_flat_fees = \'N\' THEN m.matter_id END) AS non_flat_fee_matter_count, '
        'COUNT(CASE WHEN m.is_flat_fees = \'Y\' THEN m.matter_id END) AS flat_fee_matter_count, '
        'COALESCE(SUM(mts.worked_amount), 0) AS total_fees, '
        'STRING_AGG(DISTINCT md.practice_group, \', \') AS practice_groups '
        'FROM client c '
        'JOIN matter m ON c.client_id = m.client_id '
        'JOIN timekeeper t ON m.billing_timekeeper_id = t.timekeeper_id '
        'JOIN matter_date md ON m.matter_id = md.matter_id '
        'LEFT JOIN matter_timekeeper_summary mts ON m.matter_id = mts.matter_id '
        'WHERE m.is_active = \'Y\' '
        'AND c.is_active = \'Y\' '
        'AND md.end_date IS NULL '
        'AND mts.year = EXTRACT(YEAR FROM CURRENT_DATE) '
        'GROUP BY c.client_id, c.client_name, t.timekeeper_name '
        'HAVING COUNT(CASE WHEN m.is_flat_fees = \'Y\' THEN m.matter_id END) > 0 '
        'ORDER BY flat_fee_matter_count DESC;',
        "Generate a detailed report of clients with flat fee matters, including client names, billing attorneys, counts of flat fee and non-flat fee matters, total fees paid, and a breakdown of practice groups involved."
    ),
    (
        'SELECT DATE_TRUNC(\'month\', m.open_date) AS month, '
        'COUNT(m.matter_id) AS flat_fee_matter_count, '
        'COUNT(m.matter_id) - LAG(COUNT(m.matter_id)) OVER (ORDER BY DATE_TRUNC(\'month\', m.open_date)) AS change_from_previous, '
        'ROUND((COUNT(m.matter_id) - LAG(COUNT(m.matter_id)) OVER (ORDER BY DATE_TRUNC(\'month\', m.open_date))) / '
        'NULLIF(LAG(COUNT(m.matter_id)) OVER (ORDER BY DATE_TRUNC(\'month\', m.open_date)), 0)::numeric * 100, 2) AS percent_change '
        'FROM matter m '
        'WHERE m.is_flat_fees = \'Y\' '
        'AND m.open_date >= CURRENT_DATE - INTERVAL \'2 years\' '
        'GROUP BY DATE_TRUNC(\'month\', m.open_date) '
        'ORDER BY month;',
        "Analyze the trend of flat fee matters opened over the past two years, including monthly counts, changes from the previous month, percentage changes, and any significant increases or decreases in the data."
    ),

    # Productivity Analysis Category
    (
        'SELECT md.office, md.department, md.practice_group, DATE_TRUNC(\'month\', tc.date) AS month, '
        'SUM(tc.worked_hours) AS total_hours, '
        'SUM(CASE WHEN EXTRACT(YEAR FROM tc.date) = EXTRACT(YEAR FROM CURRENT_DATE) - 1 THEN tc.worked_hours ELSE 0 END) AS prev_year_hours, '
        'ROUND((SUM(tc.worked_hours) - SUM(CASE WHEN EXTRACT(YEAR FROM tc.date) = EXTRACT(YEAR FROM CURRENT_DATE) - 1 THEN tc.worked_hours ELSE 0 END)) / '
        'NULLIF(SUM(CASE WHEN EXTRACT(YEAR FROM tc.date) = EXTRACT(YEAR FROM CURRENT_DATE) - 1 THEN tc.worked_hours ELSE 0 END), 0)::numeric * 100, 2) AS yoy_change '
        'FROM timecard tc '
        'JOIN matter m ON tc.matter_id = m.matter_id '
        'JOIN matter_date md ON m.matter_id = md.matter_id '
        'WHERE EXTRACT(YEAR FROM tc.date) IN (EXTRACT(YEAR FROM CURRENT_DATE), EXTRACT(YEAR FROM CURRENT_DATE) - 1) '
        'AND md.end_date IS NULL '
        'GROUP BY md.office, md.department, md.practice_group, DATE_TRUNC(\'month\', tc.date) '
        'ORDER BY md.office, month;',
        "Conduct a detailed analysis of hours worked year-to-date, grouped by office, department, and practice group, including monthly totals, comparisons with the previous year, and percentage changes to identify trends."
    ),
    (
        'SELECT c.client_id, c.client_name, t.timekeeper_name AS relationship_manager, '
        'COUNT(DISTINCT m.matter_id) AS matter_count, SUM(tc.worked_hours) AS total_hours, '
        'STRING_AGG(DISTINCT md.practice_group, \', \') AS practice_groups '
        'FROM timecard tc '
        'JOIN matter m ON tc.matter_id = m.matter_id '
        'JOIN client c ON m.client_id = c.client_id '
        'JOIN timekeeper t ON c.relationship_timekeeper_id = t.timekeeper_id '
        'JOIN matter_date md ON m.matter_id = md.matter_id '
        'WHERE EXTRACT(YEAR FROM tc.date) = EXTRACT(YEAR FROM CURRENT_DATE) '
        'AND md.end_date IS NULL '
        'GROUP BY c.client_id, c.client_name, t.timekeeper_name '
        'ORDER BY total_hours DESC '
        'LIMIT 10;',
        "Identify the top 10 clients with the most hours worked year-to-date, including client names, relationship managers, matter counts, total hours, and a breakdown of practice groups involved."
    ),
    (
        'SELECT DATE_TRUNC(\'month\', tc.date) AS month, SUM(tc.worked_hours) AS total_hours, '
        'SUM(CASE WHEN EXTRACT(YEAR FROM tc.date) = EXTRACT(YEAR FROM CURRENT_DATE) - 1 THEN tc.worked_hours ELSE 0 END) AS prev_year_hours, '
        'ROUND((SUM(tc.worked_hours) - SUM(CASE WHEN EXTRACT(YEAR FROM tc.date) = EXTRACT(YEAR FROM CURRENT_DATE) - 1 THEN tc.worked_hours ELSE 0 END)) / '
        'NULLIF(SUM(CASE WHEN EXTRACT(YEAR FROM tc.date) = EXTRACT(YEAR FROM CURRENT_DATE) - 1 THEN tc.worked_hours ELSE 0 END), 0)::numeric * 100, 2) AS yoy_change '
        'FROM timecard tc '
        'JOIN matter m ON tc.matter_id = m.matter_id '
        'JOIN client c ON m.client_id = c.client_id '
        'WHERE c.client_name ILIKE \'%Apple%\' '
        'AND EXTRACT(YEAR FROM tc.date) IN (EXTRACT(YEAR FROM CURRENT_DATE), EXTRACT(YEAR FROM CURRENT_DATE) - 1) '
        'GROUP BY DATE_TRUNC(\'month\', tc.date) '
        'ORDER BY month;',
        "Analyze the year-to-date trend of hours worked for matters associated with clients containing 'Apple' in their name, including monthly totals, comparisons with the previous year, and percentage changes to identify trends or anomalies."
    ),
    (
        'SELECT c.client_id, c.client_name, COUNT(DISTINCT m.matter_id) AS matter_count, '
        'SUM(CASE WHEN EXTRACT(YEAR FROM tc.date) = EXTRACT(YEAR FROM CURRENT_DATE) THEN tc.worked_hours ELSE 0 END) AS ytd_hours, '
        'SUM(CASE WHEN EXTRACT(YEAR FROM tc.date) = EXTRACT(YEAR FROM CURRENT_DATE) - 1 THEN tc.worked_hours ELSE 0 END) AS lytd_hours, '
        'RANK() OVER (ORDER BY SUM(CASE WHEN EXTRACT(YEAR FROM tc.date) = EXTRACT(YEAR FROM CURRENT_DATE) THEN tc.worked_hours ELSE 0 END) DESC) AS ytd_rank, '
        'RANK() OVER (ORDER BY SUM(CASE WHEN EXTRACT(YEAR FROM tc.date) = EXTRACT(YEAR FROM CURRENT_DATE) - 1 THEN tc.worked_hours ELSE 0 END) DESC) AS lytd_rank '
        'FROM timecard tc '
        'JOIN matter m ON tc.matter_id = m.matter_id '
        'JOIN client c ON m.client_id = c.client_id '
        'WHERE EXTRACT(YEAR FROM tc.date) IN (EXTRACT(YEAR FROM CURRENT_DATE), EXTRACT(YEAR FROM CURRENT_DATE) - 1) '
        'GROUP BY c.client_id, c.client_name '
        'ORDER BY ytd_hours DESC;',
        "Provide a year-over-year analysis of client hours worked, including client names, matter counts, YTD and last YTD hours, and rankings for both years to highlight changes in client activity."
    ),
    (
        'SELECT c.client_id, c.client_name, DATE_TRUNC(\'month\', tc.date) AS month, '
        'SUM(tc.worked_hours) AS total_hours, '
        'STRING_AGG(DISTINCT md.practice_group, \', \') AS practice_groups '
        'FROM timecard tc '
        'JOIN matter m ON tc.matter_id = m.matter_id '
        'JOIN client c ON m.client_id = c.client_id '
        'JOIN matter_date md ON m.matter_id = md.matter_id '
        'WHERE c.relationship_timekeeper_id = \'user_id\' '
        'AND EXTRACT(YEAR FROM tc.date) = EXTRACT(YEAR FROM CURRENT_DATE) '
        'AND md.end_date IS NULL '
        'GROUP BY c.client_id, c.client_name, DATE_TRUNC(\'month\', tc.date) '
        'ORDER BY c.client_name, month;',
        "Analyze the monthly hours worked for clients managed by the relationship manager, including client names, total hours, and practice groups involved, to identify trends in client activity."
    ),

    # Realization Category
    (
        'SELECT c.client_id, c.client_name, '
        'ROUND((SUM(mts.worked_amount) / NULLIF(SUM(mts.standard_amount), 0) * 100)::numeric, 2) AS realization_rate, '
        'SUM(mts.worked_amount) AS total_worked_amount, SUM(mts.standard_amount) AS total_standard_amount '
        'FROM matter_timekeeper_summary mts '
        'JOIN matter m ON mts.matter_id = m.matter_id '
        'JOIN client c ON m.client_id = c.client_id '
        'WHERE c.relationship_timekeeper_id = (SELECT timekeeper_id FROM timekeeper WHERE timekeeper_name = \'Lynn Bond\') '
        'AND mts.year = EXTRACT(YEAR FROM CURRENT_DATE) '
        'GROUP BY c.client_id, c.client_name '
        'HAVING SUM(mts.worked_amount) / NULLIF(SUM(mts.standard_amount), 0) * 100 < 70 '
        'ORDER BY realization_rate;',
        "List all clients managed by Lynn with a realization rate below 70%, including client names, realization rates, total worked and standard amounts, to identify underperforming engagements."
    ),
    (
        'SELECT c.client_id, c.client_name, '
        'ROUND((SUM(mts.worked_amount) / NULLIF(SUM(mts.standard_amount), 0) * 100)::numeric, 2) AS realization_rate, '
        'SUM(mts.worked_amount) AS total_worked_amount, SUM(mts.standard_amount) AS total_standard_amount '
        'FROM matter_timekeeper_summary mts '
        'JOIN matter m ON mts.matter_id = m.matter_id '
        'JOIN client c ON m.client_id = c.client_id '
        'JOIN timekeeper t ON c.relationship_timekeeper_id = t.timekeeper_id '
        'WHERE t.timekeeper_name = \'Lynn Bond\' '
        'AND mts.year = EXTRACT(YEAR FROM CURRENT_DATE) '
        'GROUP BY c.client_id, c.client_name '
        'HAVING SUM(mts.worked_amount) / NULLIF(SUM(mts.standard_amount), 0) * 100 < 70 '
        'ORDER BY realization_rate;',
        "Provide a detailed report of clients managed by Lynn Bond with realization rates below 70%, including client names, realization rates, and total worked and standard amounts to assess billing efficiency."
    ),
    (
        'SELECT c.client_id, c.client_name, '
        'ROUND((SUM(mts.worked_amount) / NULLIF(SUM(mts.standard_amount), 0) * 100)::numeric, 2) AS realization_rate, '
        'SUM(mts.worked_amount) AS total_worked_amount, SUM(mts.standard_amount) AS total_standard_amount '
        'FROM matter_timekeeper_summary mts '
        'JOIN matter m ON mts.matter_id = m.matter_id '
        'JOIN client c ON m.client_id = c.client_id '
        'WHERE mts.year = EXTRACT(YEAR FROM CURRENT_DATE) '
        'GROUP BY c.client_id, c.client_name '
        'HAVING SUM(mts.worked_amount) / NULLIF(SUM(mts.standard_amount), 0) * 100 > 200 '
        'ORDER BY realization_rate DESC;',
        "Identify clients with realization rates exceeding 200%, including client names, realization rates, and total worked and standard amounts, to investigate potential billing anomalies."
    ),
    (
        'SELECT c.client_id, c.client_name, '
        'ROUND((SUM(mts.standard_amount - mts.worked_amount))::numeric, 2) AS total_gap, '
        'SUM(mts.worked_amount) AS worked_amount, SUM(mts.standard_amount) AS standard_amount, '
        'ROUND((SUM(mts.worked_amount) / NULLIF(SUM(mts.standard_amount), 0) * 100)::numeric, 2) AS realization_rate '
        'FROM matter_timekeeper_summary mts '
        'JOIN matter m ON mts.matter_id = m.matter_id '
        'JOIN client c ON m.client_id = c.client_id '
        'WHERE mts.year = EXTRACT(YEAR FROM CURRENT_DATE) '
        'GROUP BY c.client_id, c.client_name '
        'ORDER BY total_gap DESC '
        'LIMIT 10;',
        "List the top 10 clients with the widest gap between standard and billed rates, including client names, total gaps, worked and standard amounts, and realization rates, to prioritize billing reviews."
    ),

    # E-billed Category
    (
        'SELECT c.client_id, c.client_name, t.timekeeper_name AS relationship_manager, '
        'COUNT(DISTINCT m.billing_timekeeper_id) AS billing_attorney_count, '
        'COUNT(DISTINCT m.matter_id) AS matter_count, COALESCE(SUM(mts.worked_hours), 0) AS total_hours, '
        'STRING_AGG(DISTINCT md.practice_group, \', \') AS practice_groups '
        'FROM client c '
        'JOIN matter m ON c.client_id = m.client_id '
        'JOIN timekeeper t ON c.relationship_timekeeper_id = t.timekeeper_id '
        'JOIN matter_date md ON m.matter_id = md.matter_id '
        'LEFT JOIN matter_timekeeper_summary mts ON m.matter_id = mts.matter_id '
        'WHERE c.is_ebilled = \'Y\' '
        'AND c.is_active = \'Y\' '
        'AND md.end_date IS NULL '
        'AND mts.year = EXTRACT(YEAR FROM CURRENT_DATE) '
        'GROUP BY c.client_id, c.client_name, t.timekeeper_name '
        'ORDER BY matter_count DESC;',
        "Compile a detailed report of e-billed clients, including client names, relationship managers, number of billing attorneys, matter counts, total hours worked, and practice groups involved."
    ),
    (
        'SELECT c.client_id, c.client_name, DATE_TRUNC(\'month\', tc.date) AS month, '
        'COUNT(DISTINCT m.matter_id) AS matter_count, SUM(tc.worked_hours) AS total_hours, '
        'STRING_AGG(DISTINCT md.practice_group, \', \') AS practice_groups '
        'FROM timecard tc '
        'JOIN matter m ON tc.matter_id = m.matter_id '
        'JOIN client c ON m.client_id = c.client_id '
        'JOIN matter_date md ON m.matter_id = md.matter_id '
        'WHERE c.relationship_timekeeper_id = \'user_id\' '
        'AND c.is_ebilled = \'Y\' '
        'AND EXTRACT(YEAR FROM tc.date) = EXTRACT(YEAR FROM CURRENT_DATE) '
        'AND md.end_date IS NULL '
        'GROUP BY c.client_id, c.client_name, DATE_TRUNC(\'month\', tc.date) '
        'ORDER BY c.client_name, month;',
        "Analyze the monthly trend of hours worked for e-billed clients managed by the relationship manager, including client names, matter counts, total hours, and practice groups, to track billing activity."
    ),

    # Rate Review Category
    (
        'SELECT c.client_id, c.client_name, t.timekeeper_name AS relationship_manager, '
        'c.last_review_date, c.rate_review_date, c.notification_days, '
        'COUNT(DISTINCT m.matter_id) AS matter_count, COUNT(DISTINCT rs.rate_set_id) AS rate_set_count, '
        'STRING_AGG(DISTINCT rs.rate_set_code, \', \') AS rate_set_codes '
        'FROM client c '
        'JOIN timekeeper t ON c.relationship_timekeeper_id = t.timekeeper_id '
        'LEFT JOIN matter m ON c.client_id = m.client_id AND m.is_active = \'Y\' '
        'LEFT JOIN rate_set rs ON m.rate_set_id = rs.rate_set_id '
        'WHERE (c.last_review_date IS NULL OR c.last_review_date < CURRENT_DATE - INTERVAL \'1 year\' - c.notification_days * INTERVAL \'1 day\') '
        'AND c.is_active = \'Y\' '
        'GROUP BY c.client_id, c.client_name, t.timekeeper_name, c.last_review_date, c.rate_review_date, c.notification_days '
        'ORDER BY c.client_name;',
        "List all clients that have not had their rates reviewed, including client names, relationship managers, last review dates, rate review dates, notification days, matter counts, rate set counts, and rate set codes."
    ),
    (
        'SELECT c.client_id, c.client_name, t.timekeeper_name AS relationship_manager, '
        'c.last_review_date, rs.rate_set_code, '
        'COUNT(DISTINCT m.matter_id) AS matter_count, '
        'COUNT(DISTINCT CASE WHEN rsl.rate_type <> \'GR\' THEN m.matter_id END) AS exception_rate_matter_count '
        'FROM client c '
        'JOIN timekeeper t ON c.relationship_timekeeper_id = t.timekeeper_id '
        'JOIN matter m ON c.client_id = m.client_id '
        'JOIN rate_set rs ON m.rate_set_id = rs.rate_set_id '
        'JOIN rate_set_link rsl ON rs.rate_set_id = rsl.rate_set_id '
        'WHERE rsl.rate_type <> \'GR\' '
        'AND (c.last_review_date IS NULL OR c.last_review_date < CURRENT_DATE - INTERVAL \'1 year\') '
        'AND c.is_active = \'Y\' '
        'AND m.is_active = \'Y\' '
        'GROUP BY c.client_id, c.client_name, t.timekeeper_name, c.last_review_date, rs.rate_set_code '
        'ORDER BY matter_count DESC;',
        "Provide a detailed report of clients with exception rates that have not been reviewed in the last year, including client names, relationship managers, last review dates, rate set codes, total matter counts, and counts of matters with exception rates."
    ),

    # Utilization and Budget Category
    (
        'SELECT t.timekeeper_id, t.timekeeper_name, tkd.title, tkd.office, '
        'SUM(CASE WHEN tc.date >= DATE_TRUNC(\'month\', CURRENT_DATE - INTERVAL \'1 month\') THEN tc.worked_hours ELSE 0 END) AS last_month_hours, '
        'ROUND((SUM(CASE WHEN tc.date >= DATE_TRUNC(\'month\', CURRENT_DATE - INTERVAL \'1 month\') THEN tc.worked_hours ELSE 0 END) / NULLIF(t.budget_hours / 12, 0) * 100)::numeric, 2) AS last_month_utilization, '
        'SUM(tc.worked_hours) AS ytd_hours, '
        'ROUND((SUM(tc.worked_hours) / NULLIF(t.budget_hours, 0) * 100)::numeric, 2) AS ytd_utilization '
        'FROM timekeeper t '
        'JOIN timekeeper_date tkd ON t.timekeeper_id = tkd.timekeeper_id '
        'JOIN timecard tc ON t.timekeeper_id = tc.timekeeper_id '
        'WHERE EXTRACT(YEAR FROM tc.date) = EXTRACT(YEAR FROM CURRENT_DATE) '
        'AND t.is_active = \'Y\' '
        'AND tkd.end_date IS NULL '
        'AND tc.is_nonbillable = \'N\' '
        'GROUP BY t.timekeeper_id, t.timekeeper_name, tkd.title, tkd.office, t.budget_hours '
        'HAVING (SUM(CASE WHEN tc.date >= DATE_TRUNC(\'month\', CURRENT_DATE - INTERVAL \'1 month\') THEN tc.worked_hours ELSE 0 END) / NULLIF(t.budget_hours / 12, 0) * 100) < 80 '
        'ORDER BY last_month_utilization;',
        "Identify timekeepers with low utilization rates (below 80%) in the last month, including their names, titles, offices, last month and YTD hours, and utilization percentages, to assess resource allocation."
    ),
    (
        'SELECT DATE_TRUNC(\'month\', tc.date) AS month, SUM(tc.worked_hours) AS total_hours, '
        't.budget_hours / 12 AS monthly_budget_hours, '
        'ROUND((SUM(tc.worked_hours) / NULLIF(t.budget_hours / 12, 0) * 100)::numeric, 2) AS utilization_percent '
        'FROM timecard tc '
        'JOIN timekeeper t ON tc.timekeeper_id = t.timekeeper_id '
        'WHERE tc.timekeeper_id = \'user_id\' '
        'AND tc.is_active = \'Y\' '
        'AND EXTRACT(YEAR FROM tc.date) = EXTRACT(YEAR FROM CURRENT_DATE) '
        'GROUP BY DATE_TRUNC(\'month\', tc.date), t.budget_hours '
        'ORDER BY month;',
        "Analyze the monthly utilization trend for the current user, including total hours worked, monthly budget hours, and utilization percentages, to compare actual performance against budget."
    ),

    # Timecard Analysis Category
    (
        'SELECT t.timekeeper_name, tkd.title, tkd.office, '
        'ROUND(AVG((tc.posted_date - tc.date))::numeric, 2) AS avg_days_late, '
        'COUNT(CASE WHEN tc.posted_date - tc.date > 5 THEN tc.timecard_id END) AS late_timecards, '
        'ROUND((COUNT(CASE WHEN tc.posted_date - tc.date > 5 THEN tc.timecard_id END)::numeric / NULLIF(COUNT(tc.timecard_id), 0)) * 100, 2) AS late_percentage '
        'FROM timecard tc '
        'JOIN timekeeper t ON tc.timekeeper_id = t.timekeeper_id '
        'JOIN timekeeper_date tkd ON t.timekeeper_id = tkd.timekeeper_id '
        'WHERE EXTRACT(YEAR FROM tc.date) = EXTRACT(YEAR FROM CURRENT_DATE) '
        'AND tc.is_active = \'Y\' '
        'AND tkd.end_date IS NULL '
        'GROUP BY t.timekeeper_name, tkd.title, tkd.office '
        'HAVING COUNT(CASE WHEN tc.posted_date - tc.date > 5 THEN tc.timecard_id END) > 0 '
        'ORDER BY avg_days_late DESC;',
        "Identify timekeepers who regularly enter timecards more than 5 days late, including their names, titles, offices, average days late, number of late timecards, and percentage of late entries, to improve time entry compliance."
    ),
    (
        'SELECT t.timekeeper_name, tkd.title, '
        'SUM(CASE WHEN tc.is_nonbillable = \'Y\' AND tc.date >= DATE_TRUNC(\'month\', CURRENT_DATE - INTERVAL \'1 month\') THEN tc.worked_hours ELSE 0 END) AS non_billable_hours_mtd, '
        'SUM(CASE WHEN tc.is_nonbillable = \'N\' AND tc.date >= DATE_TRUNC(\'month\', CURRENT_DATE - INTERVAL \'1 month\') THEN tc.worked_hours ELSE 0 END) AS billable_hours_mtd, '
        'SUM(CASE WHEN tc.is_nonbillable = \'Y\' AND EXTRACT(YEAR FROM tc.date) = EXTRACT(YEAR FROM CURRENT_DATE) THEN tc.worked_hours ELSE 0 END) AS non_billable_hours_ytd, '
        'SUM(CASE WHEN tc.is_nonbillable = \'N\' AND EXTRACT(YEAR FROM tc.date) = EXTRACT(YEAR FROM CURRENT_DATE) THEN tc.worked_hours ELSE 0 END) AS billable_hours_ytd, '
        'ROUND(SUM(CASE WHEN tc.is_nonbillable = \'Y\' THEN tc.worked_hours ELSE 0 END)::numeric / NULLIF(SUM(tc.worked_hours), 0) * 100, 2) AS non_billable_percentage '
        'FROM timecard tc '
        'JOIN timekeeper t ON tc.timekeeper_id = t.timekeeper_id '
        'JOIN timekeeper_date tkd ON t.timekeeper_id = tkd.timekeeper_id '
        'WHERE t.type = \'Litigation\' '
        'AND tc.is_active = \'Y\' '
        'AND tkd.end_date IS NULL '
        'GROUP BY t.timekeeper_name, tkd.title '
        'ORDER BY non_billable_hours_ytd DESC;',
        "Analyze the non-billable versus billable hours split for the litigation team, including timekeeper names, titles, monthly and YTD hours for both categories, and the percentage of non-billable hours, to optimize resource allocation."
    ),

    # Matter Load Category
    (
        'SELECT t.timekeeper_id, t.timekeeper_name, tkd.title, tkd.office, '
        'DATE_TRUNC(\'month\', tc.date) AS month, '
        'SUM(CASE WHEN EXTRACT(YEAR FROM tc.date) = EXTRACT(YEAR FROM CURRENT_DATE) THEN tc.worked_hours ELSE 0 END) AS current_year_hours, '
        'SUM(CASE WHEN EXTRACT(YEAR FROM tc.date) = EXTRACT(YEAR FROM CURRENT_DATE) - 1 THEN tc.worked_hours ELSE 0 END) AS prev_year_hours, '
        'ROUND((SUM(CASE WHEN EXTRACT(YEAR FROM tc.date) = EXTRACT(YEAR FROM CURRENT_DATE) THEN tc.worked_hours ELSE 0 END) - '
        'SUM(CASE WHEN EXTRACT(YEAR FROM tc.date) = EXTRACT(YEAR FROM CURRENT_DATE) - 1 THEN tc.worked_hours ELSE 0 END)) / '
        'NULLIF(SUM(CASE WHEN EXTRACT(YEAR FROM tc.date) = EXTRACT(YEAR FROM CURRENT_DATE) - 1 THEN tc.worked_hours ELSE 0 END), 0)::numeric * 100, 2) AS yoy_change '
        'FROM timecard tc '
        'JOIN timekeeper t ON tc.timekeeper_id = t.timekeeper_id '
        'JOIN timekeeper_date tkd ON t.timekeeper_id = tkd.timekeeper_id '
        'WHERE tkd.title = \'Paralegal\' '
        'AND tkd.office = \'New York\' '
        'AND EXTRACT(YEAR FROM tc.date) IN (EXTRACT(YEAR FROM CURRENT_DATE), EXTRACT(YEAR FROM CURRENT_DATE) - 1) '
        'AND t.is_active = \'Y\' '
        'AND tkd.end_date IS NULL '
        'GROUP BY t.timekeeper_id, t.timekeeper_name, tkd.title, tkd.office, DATE_TRUNC(\'month\', tc.date) '
        'ORDER BY month, yoy_change DESC;',
        "Analyze and display the trend of collected hours for Paralegals based in New York. Include a comparison with the previous year's data, identify any significant increases or decreases, and provide possible reasons for these changes. Also, highlight any patterns or anomalies in the data."
    )
]
general_query_documentation = [  
    # Lists Category
    "Query: 'List my clients.' - Purpose: Retrieves a list of active clients managed by the current user (relationship manager). "
    "Tables: client (c), timekeeper (t), matter (m), rate_set (rs). "
    "Columns: c.client_id, c.client_name, t.timekeeper_name, m.matter_id, rs.rate_set_id. "
    "Expected Output: Client ID, name, relationship manager name, count of active matters, and count of rate sets. "
    "Notes: Assumes 'user_id' is the current user's timekeeper ID; uses LEFT JOIN to include clients without matters.",
    
    "Query: 'My new clients / matters.' - Purpose: Lists clients and their matters opened in the last 21 days with hours worked. "
    "Tables: client (c), matter (m), matter_date (md), matter_timekeeper_summary (mts). "
    "Columns: c.client_id, c.client_name, m.matter_id, m.matter_name, m.open_date, md.office, md.department, md.practice_group, mts.worked_hours. "
    "Expected Output: Client and matter details with office, department, practice group, and total hours worked. "
    "Notes: Filters for active clients/matters; assumes current date is June 12, 2025.",
    
    "Query: 'Client / Matter load by Lynn Bond.' - Purpose: Shows the number of clients and matters for a specific billing attorney. "
    "Tables: matter (m), timekeeper (t), client (c). "
    "Columns: t.timekeeper_name, c.client_id, m.matter_id. "
    "Expected Output: Billing attorney name, client count, matter count. "
    "Notes: Replace 'Lynn Bond' with any billing attorney's name as needed.",
    
    "Query: 'Load Analysis.' - Purpose: Analyzes client and matter counts by office, department, and practice group. "
    "Tables: matter (m), matter_date (md), client (c). "
    "Columns: md.office, md.department, md.practice_group, c.client_id, m.matter_id. "
    "Expected Output: Office, department, practice group with client and matter counts, sorted by counts. "
    "Notes: Only includes active clients and matters.",
    
    "Query: 'Retail Clients.' - Purpose: Lists active clients related to retail based on name or category. "
    "Tables: client (c), matter (m). "
    "Columns: c.client_id, c.client_name, c.category, m.matter_id. "
    "Expected Output: Client ID, name, category, and matter count. "
    "Notes: Uses ILIKE for case-insensitive matching.",
    
    "Query: 'Related clients for Pepsi.' - Purpose: Finds clients related to Pepsi via related_client_id. "
    "Tables: client (c1, c2), timekeeper (t), matter (m). "
    "Columns: c2.client_id, c2.client_name, t.timekeeper_name, m.matter_id. "
    "Expected Output: Related client details with matter counts. "
    "Notes: Excludes Pepsi itself from results.",
    
    "Query: 'List all clients.' - Purpose: Retrieves all active clients with their relationship managers and matter counts. "
    "Tables: client (c), timekeeper (t), matter (m). "
    "Columns: c.client_id, c.client_name, t.timekeeper_name, m.matter_id. "
    "Expected Output: Client ID, name, relationship manager, and matter count. "
    "Notes: Comprehensive list for all active clients.",

    # Flat Fees Category
    "Query: 'Clients with flat fee matters.' - Purpose: Identifies clients with active flat fee matters. "
    "Tables: client (c), matter (m), timekeeper (t). "
    "Columns: c.client_id, c.client_name, t.timekeeper_name, m.is_flat_fees, m.matter_id. "
    "Expected Output: Client ID, name, billing attorney, counts of flat fee and non-flat fee matters. "
    "Notes: Filters for clients with at least one flat fee matter.",
    
    "Query: 'Flat Fee Trend.' - Purpose: Tracks the number of flat fee matters opened monthly over the past 2 years. "
    "Tables: matter (m). "
    "Columns: m.open_date, m.matter_id, m.is_flat_fees. "
    "Expected Output: Month and count of flat fee matters. "
    "Notes: Assumes current date is June 12, 2025.",

    # Productivity Analysis Category
    "Query: 'Load Analysis by Hours worked.' - Purpose: Analyzes hours worked by office, department, and practice group YTD. "
    "Tables: matter_timekeeper_summary (mts), matter (m), matter_date (md). "
    "Columns: md.office, md.department, md.practice_group, mts.date, mts.worked_hours. "
    "Expected Output: Office, department, practice group, month, and total hours. "
    "Notes: Filters for current year (2025).",
    
    "Query: 'Clients with most hours.' - Purpose: Lists top 10 clients by hours worked YTD. "
    "Tables: matter_timekeeper_summary (mts), matter (m), client (c), timekeeper (t). "
    "Columns: c.client_id, c.client_name, t.timekeeper_name, m.matter_id, mts.worked_hours. "
    "Expected Output: Client ID, name, relationship manager, matter count, total hours. "
    "Notes: Limits to top 10 by hours.",
    
    "Query: 'Trend Apple YTD.' - Purpose: Shows monthly hours worked for Apple-related matters YTD. "
    "Tables: matter_timekeeper_summary (mts), matter (m), client (c). "
    "Columns: mts.date, mts.worked_hours, c.client_name. "
    "Expected Output: Month and total hours. "
    "Notes: Uses ILIKE for Apple name matching.",
    
    "Query: 'Client Hours Trend YOY.' - Purpose: Compares client hours worked YTD vs. last year. "
    "Tables: matter_timekeeper_summary (mts), matter (m), client (c). "
    "Columns: c.client_id, c.client_name, m.matter_id, mts.worked_hours, mts.year. "
    "Expected Output: Client ID, name, matter count, YTD hours, last year’s hours. "
    "Notes: Assumes 2025 and 2024 data.",
    
    "Query: 'Trend my clients.' - Purpose: Tracks monthly hours worked for the user's clients YTD. "
    "Tables: matter_timekeeper_summary (mts), matter (m), client (c). "
    "Columns: c.client_id, c.client_name, mts.date, mts.worked_hours. "
    "Expected Output: Client ID, name, month, total hours. "
    "Notes: Uses 'user_id' for filtering.",

    # Realization Category
    "Query: 'List my clients with realization < 70%.' - Purpose: Identifies user’s clients with low realization rates YTD. "
    "Tables: matter_timekeeper_summary (mts), matter (m), client (c). "
    "Columns: c.client_id, c.client_name, mts.worked_amount, mts.standard_amount. "
    "Expected Output: Client ID, name, realization rate (percentage). "
    "Notes: Realization = (worked_amount / standard_amount) * 100.",
    
    "Query: 'List clients for Lynn Bond with realization < 70%.' - Purpose: Lists clients with low realization for a specific manager. "
    "Tables: matter_timekeeper_summary (mts), matter (m), client (c), timekeeper (t). "
    "Columns: c.client_id, c.client_name, mts.worked_amount, mts.standard_amount, t.timekeeper_name. "
    "Expected Output: Client ID, name, realization rate. "
    "Notes: Filters for 'Lynn Bond'.",
    
    "Query: 'List clients with realization > 200%.' - Purpose: Identifies clients with unusually high realization rates. "
    "Tables: matter_timekeeper_summary (mts), matter (m), client (c). "
    "Columns: c.client_id, c.client_name, mts.worked_amount, mts.standard_amount. "
    "Expected Output: Client ID, name, realization rate. "
    "Notes: Highlights potential overbilling.",
    
    "Query: 'Which clients have the widest gap between standard and billed rate?' - Purpose: Finds top 10 clients with the largest billing gaps. "
    "Tables: matter_timekeeper_summary (mts), matter (m), client (c). "
    "Columns: c.client_id, c.client_name, mts.standard_amount, mts.worked_amount. "
    "Expected Output: Client ID, name, total gap, worked amount, standard amount. "
    "Notes: Gap = standard_amount - worked_amount.",

    # E-billed Category
    "Query: 'E-billed clients.' - Purpose: Lists active e-billed clients with matter and billing details. "
    "Tables: client (c), matter (m), timekeeper (t), matter_timekeeper_summary (mts). "
    "Columns: c.client_id, c.client_name, t.timekeeper_name, m.billing_timekeeper_id, m.matter_id, mts.worked_hours. "
    "Expected Output: Client ID, name, relationship manager, billing attorney count, matter count, total hours. "
    "Notes: Filters for is_ebilled = 'Y'.",
    
    "Query: 'Trend my clients e-billed.' - Purpose: Tracks monthly trends for user’s e-billed clients YTD. "
    "Tables: matter_timekeeper_summary (mts), matter (m), client (c). "
    "Columns: c.client_id, c.client_name, mts.date, m.matter_id, mts.worked_hours. "
    "Expected Output: Client ID, name, month, matter count, total hours. "
    "Notes: Combines e-billing and user-specific filters.",

    # Rate Review Category
    "Query: 'Clients that have not been reviewed.' - Purpose: Identifies clients overdue for rate review. "
    "Tables: client (c), timekeeper (t), matter (m), rate_set (rs). "
    "Columns: c.client_id, c.client_name, t.timekeeper_name, c.last_review_date, c.rate_review_date, c.notification_days, m.matter_id, rs.rate_set_id. "
    "Expected Output: Client ID, name, relationship manager, review dates, notification days, matter and rate set counts. "
    "Notes: Uses notification_days for overdue calculation.",
    
    "Query: 'Clients with exception rates that haven't been reviewed.' - Purpose: Lists clients with non-standard rates overdue for review. "
    "Tables: client (c), timekeeper (t), rate_set (rs), matter (m). "
    "Columns: c.client_id, c.client_name, t.timekeeper_name, c.last_review_date, rs.rate_set_code, m.matter_id. "
    "Expected Output: Client ID, name, relationship manager, last review date, rate set code, matter count. "
    "Notes: Excludes 'GR' rate type.",

    # Utilization and Budget Category
    "Query: 'Timekeepers with low utilization.' - Purpose: Identifies timekeepers with utilization below 80% last month. "
    "Tables: timekeeper (t), timekeeper_date (tkd), matter_timekeeper_summary (mts). "
    "Columns: t.timekeeper_id, t.timekeeper_name, tkd.title, tkd.office, mts.worked_hours, t.budget_hours. "
    "Expected Output: Timekeeper ID, name, title, office, last month hours, utilization, YTD hours, utilization. "
    "Notes: Utilization = (worked_hours / budget_hours) * 100.",
    
    "Query: 'Trend my utilization.' - Purpose: Tracks the user’s monthly utilization YTD against budget. "
    "Tables: timecard (tc), timekeeper (t). "
    "Columns: tc.date, tc.worked_hours, t.budget_hours. "
    "Expected Output: Month, total hours, monthly budget hours. "
    "Notes: Assumes 'user_id' is the user’s timekeeper ID.",

    # Timecard Analysis Category
    "Query: 'Late Time Entries.' - Purpose: Identifies timekeepers with average late entries over 5 days. "
    "Tables: timecard (tc), timekeeper (t), timekeeper_date (tkd). "
    "Columns: t.timekeeper_name, tkd.title, tkd.office, tc.posted_date, tc.date. "
    "Expected Output: Timekeeper name, title, office, average days late. "
    "Notes: Filters for active timecards.",
    
    "Query: 'Non-Billable vs. Billable Split.' - Purpose: Analyzes billable vs. non-billable hours for Litigation team last month and YTD. "
    "Tables: timecard (tc), timekeeper (t), timekeeper_date (tkd). "
    "Columns: t.timekeeper_name, tkd.title, tc.is_nonbillable, tc.worked_hours, tc.date. "
    "Expected Output: Timekeeper name, title, non-billable and billable hours (MTD and YTD). "
    "Notes: Filters for Litigation type.",

    # Matter Load Category
    "Query: 'Matter Load by paralegal.' - Purpose: Shows active matter counts for paralegals. "
    "Tables: matter (m), matter_timekeeper_summary (mts), timekeeper (t), timekeeper_date (tkd). "
    "Columns: t.timekeeper_name, tkd.title, m.matter_id. "
    "Expected Output: Timekeeper name, title, active matter count. "
    "Notes: Filters for 'Paralegal' title."
]

# DDL statements are powerful because they specify table names, column names, types, and potentially relationships
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
    for sql, question in general_query_sql_training:
        vn.train(sql=sql, question=question)
    for doc in general_query_documentation:
        vn.train(documentation=doc)
    
    # DDL statements are powerful because they specify table names, column names, types, and potentially relationships
    vn.train(ddl="""
        -- Client management tables
        CREATE TABLE IF NOT EXISTS client (
            client_id TEXT PRIMARY KEY,
            client_name TEXT NOT NULL,
            email TEXT,
            phone TEXT,
            address TEXT,
            category TEXT,
            type TEXT,
            is_active VARCHAR(1) DEFAULT 'Y',
            is_ebilled VARCHAR(1) DEFAULT 'N',
            open_date DATE,
            close_date DATE,
            relationship_timekeeper_id TEXT,
            related_client_id TEXT,
            last_review_date DATE,
            rate_review_date DATE,
            notification_days INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (relationship_timekeeper_id) REFERENCES timekeeper(timekeeper_id)
        );

        -- Matter management tables
        CREATE TABLE IF NOT EXISTS matter (
            matter_id TEXT PRIMARY KEY,
            client_id TEXT NOT NULL,
            matter_name TEXT NOT NULL,
            status TEXT,
            type TEXT,
            is_active VARCHAR(1) DEFAULT 'Y',
            billing_timekeeper_id TEXT,
            open_date DATE,
            close_date DATE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (client_id) REFERENCES client(client_id),
            FOREIGN KEY (billing_timekeeper_id) REFERENCES timekeeper(timekeeper_id)
        );

        -- Matter date and organizational details
        CREATE TABLE IF NOT EXISTS matter_date (
            matter_date_id TEXT PRIMARY KEY,
            matter_id TEXT NOT NULL,
            start_date DATE,
            end_date DATE,
            office TEXT,
            department TEXT,
            practice_group TEXT,
            section TEXT,
            FOREIGN KEY (matter_id) REFERENCES matter(matter_id)
        );

        -- Timekeeper (attorney/staff) management
        CREATE TABLE IF NOT EXISTS timekeeper (
            timekeeper_id TEXT PRIMARY KEY,
            timekeeper_name TEXT NOT NULL,
            email TEXT,
            title TEXT,
            rate DECIMAL(10,2),
            department TEXT,
            office TEXT,
            section TEXT,
            is_active VARCHAR(1) DEFAULT 'Y',
            budget_hours DECIMAL(10,2),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Timekeeper historical data
        CREATE TABLE IF NOT EXISTS timekeeper_date (
            timekeeper_date_id TEXT PRIMARY KEY,
            timekeeper_id TEXT NOT NULL,
            start_date DATE,
            end_date DATE,
            title TEXT,
            department TEXT,
            office TEXT,
            section TEXT,
            rate_class TEXT,
            FOREIGN KEY (timekeeper_id) REFERENCES timekeeper(timekeeper_id)
        );

        -- Time tracking and billing
        CREATE TABLE IF NOT EXISTS timecard (
            timecard_id TEXT PRIMARY KEY,
            timekeeper_id TEXT NOT NULL,
            matter_id TEXT NOT NULL,
            date DATE NOT NULL,
            time_type TEXT,
            is_flat_fee VARCHAR(1) DEFAULT 'N',
            is_nonbillable VARCHAR(1) DEFAULT 'N',
            is_no_charge VARCHAR(1) DEFAULT 'N',
            is_active VARCHAR(1) DEFAULT 'Y',
            worked_hours DECIMAL(10,2),
            worked_rate DECIMAL(10,2),
            worked_amount DECIMAL(10,2),
            posted_date DATE,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (timekeeper_id) REFERENCES timekeeper(timekeeper_id),
            FOREIGN KEY (matter_id) REFERENCES matter(matter_id)
        );

        -- Matter timekeeper summary for reporting
        CREATE TABLE IF NOT EXISTS matter_timekeeper_summary (
            summary_id TEXT PRIMARY KEY,
            matter_id TEXT NOT NULL,
            timekeeper_id TEXT,
            year INTEGER,
            month INTEGER,
            date DATE,
            worked_hours DECIMAL(10,2),
            worked_amount DECIMAL(10,2),
            standard_amount DECIMAL(10,2),
            billed_hours DECIMAL(10,2),
            billed_amount DECIMAL(10,2),
            FOREIGN KEY (matter_id) REFERENCES matter(matter_id),
            FOREIGN KEY (timekeeper_id) REFERENCES timekeeper(timekeeper_id)
        );

        -- Rate management
        CREATE TABLE IF NOT EXISTS rate_set (
            rate_set_id TEXT PRIMARY KEY,
            rate_set_code TEXT,
            rate_set_name TEXT,
            client_id TEXT,
            matter_id TEXT,
            effective_date DATE,
            end_date DATE,
            is_active VARCHAR(1) DEFAULT 'Y',
            FOREIGN KEY (client_id) REFERENCES client(client_id),
            FOREIGN KEY (matter_id) REFERENCES matter(matter_id)
        );

        -- Individual rates
        CREATE TABLE IF NOT EXISTS rate (
            rate_id TEXT PRIMARY KEY,
            rate_set_id TEXT,
            timekeeper_id TEXT NOT NULL,
            rate_amount DECIMAL(10,2) NOT NULL,
            effective_date DATE NOT NULL,
            end_date DATE,
            is_active VARCHAR(1) DEFAULT 'Y',
            FOREIGN KEY (rate_set_id) REFERENCES rate_set(rate_set_id),
            FOREIGN KEY (timekeeper_id) REFERENCES timekeeper(timekeeper_id)
        );

        -- Invoice management
        CREATE TABLE IF NOT EXISTS invoice (
            invoice_id TEXT PRIMARY KEY,
            matter_id TEXT NOT NULL,
            client_id TEXT NOT NULL,
            invoice_number TEXT,
            amount DECIMAL(10,2) NOT NULL,
            date DATE NOT NULL,
            due_date DATE,
            status TEXT DEFAULT 'PENDING',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (matter_id) REFERENCES matter(matter_id),
            FOREIGN KEY (client_id) REFERENCES client(client_id)
        );

        -- Expense tracking
        CREATE TABLE IF NOT EXISTS expense (
            expense_id TEXT PRIMARY KEY,
            matter_id TEXT NOT NULL,
            timekeeper_id TEXT,
            amount DECIMAL(10,2) NOT NULL,
            date DATE NOT NULL,
            description TEXT,
            category TEXT,
            is_billable VARCHAR(1) DEFAULT 'Y',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (matter_id) REFERENCES matter(matter_id),
            FOREIGN KEY (timekeeper_id) REFERENCES timekeeper(timekeeper_id)
        );

        -- Task management
        CREATE TABLE IF NOT EXISTS task (
            task_id TEXT PRIMARY KEY,
            matter_id TEXT NOT NULL,
            timekeeper_id TEXT,
            description TEXT NOT NULL,
            due_date DATE,
            status TEXT DEFAULT 'PENDING',
            priority TEXT DEFAULT 'MEDIUM',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP,
            FOREIGN KEY (matter_id) REFERENCES matter(matter_id),
            FOREIGN KEY (timekeeper_id) REFERENCES timekeeper(timekeeper_id)
        );

        -- Document management
        CREATE TABLE IF NOT EXISTS document (
            document_id TEXT PRIMARY KEY,
            matter_id TEXT NOT NULL,
            timekeeper_id TEXT,
            filename TEXT NOT NULL,
            file_path TEXT,
            file_size INTEGER,
            mime_type TEXT,
            upload_date DATE DEFAULT CURRENT_DATE,
            description TEXT,
            tags TEXT,
            FOREIGN KEY (matter_id) REFERENCES matter(matter_id),
            FOREIGN KEY (timekeeper_id) REFERENCES timekeeper(timekeeper_id)
        );

        -- Contact management
        CREATE TABLE IF NOT EXISTS contact (
            contact_id TEXT PRIMARY KEY,
            client_id TEXT NOT NULL,
            name TEXT NOT NULL,
            email TEXT,
            phone TEXT,
            role TEXT,
            is_primary VARCHAR(1) DEFAULT 'N',
            address TEXT,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (client_id) REFERENCES client(client_id)
        );

        -- Create indexes for better performance
        CREATE INDEX IF NOT EXISTS idx_matter_client_id ON matter(client_id);
        CREATE INDEX IF NOT EXISTS idx_timecard_timekeeper_id ON timecard(timekeeper_id);
        CREATE INDEX IF NOT EXISTS idx_timecard_matter_id ON timecard(matter_id);
        CREATE INDEX IF NOT EXISTS idx_timecard_date ON timecard(date);
        CREATE INDEX IF NOT EXISTS idx_matter_timekeeper_summary_matter_id ON matter_timekeeper_summary(matter_id);
        CREATE INDEX IF NOT EXISTS idx_matter_timekeeper_summary_date ON matter_timekeeper_summary(date);
        CREATE INDEX IF NOT EXISTS idx_invoice_matter_id ON invoice(matter_id);
        CREATE INDEX IF NOT EXISTS idx_expense_matter_id ON expense(matter_id);
        CREATE INDEX IF NOT EXISTS idx_task_matter_id ON task(matter_id);
        CREATE INDEX IF NOT EXISTS idx_document_matter_id ON document(matter_id);
        CREATE INDEX IF NOT EXISTS idx_contact_client_id ON contact(client_id);
    """)

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

    print("Underbilling-CLI ready. Type your question (or ‘quit’ to exit).")
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




# python -m http.server 8000 --directory charts------oka terminal lo idi run chey and inko terminal lo python app.py
# then neeku charts link open avuthundi