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

    ("""List all active clients""",
     """SELECT client_id, client_name, is_active FROM client WHERE is_active = 'Y' ORDER BY client_name;"""),
    
    ("""Show client count""",
     """SELECT COUNT(*) as total_clients FROM client;"""),
     
    ("""List all timekeepers""",
     """SELECT timekeeper_id, timekeeper_name, type FROM timekeeper WHERE is_active = 'Y' ORDER BY timekeeper_name;"""),
     
    ("""Show timekeeper information""",
     """SELECT timekeeper_id, timekeeper_name, type, budget_hours FROM timekeeper WHERE is_active = 'Y';"""),
]

# Documentation for the General Query Assistant
# This file provides detailed guidance for the general_query_agent to handle natural-language queries
# within a RAG pipeline, using the rate model schema as of June 17, 2025.

general_query_documentation = [
    # Purpose and Scope
    "The General Query Assistant retrieves and analyzes data from the rate model schema, covering clients, matters, timekeepers, and billing information, to support queries in categories like Lists, Flat Fees, Productivity Analysis, Realization, E-billed, Rate Review, Utilization and Budget, Timecard Analysis, and Matter Load.",
    "The assistant processes natural-language questions, maps them to PostgreSQL queries, and delivers concise, actionable responses optimized for a Retrieval-Augmented Generation (RAG) pipeline.",

    # Query Processing Rules
    "Natural-language queries are parsed to identify entities (e.g., 'clients', 'Lynn Bond', 'Retail') and intents (e.g., list, trend, analyze), using training data question-SQL pairs as reference.",
    "The term 'my' refers to the user's context (e.g., relationship_timekeeper_id for clients, billing_timekeeper_id for matters), validated against the user's timekeeper_id.",
    "Names (e.g., 'Lynn Bond', 'Pepsi') are matched case-insensitively using ILIKE on fields like timekeeper_name or client_name, prompting clarification if multiple matches occur.",
    "Ambiguous terms (e.g., 'Retail') are searched across relevant fields (e.g., client_name, client.category, matter.type, matter_date.practice_group), noting the matched field in responses.",
    "Placeholders like 'user_id' are replaced with the authenticated user's timekeeper_id, ensuring queries respect user permissions.",
    "Date-based filters use the current date (June 17, 2025) for calculations (e.g., YTD = 2025, last 21 days = CURRENT_DATE - INTERVAL '21 days').",
    "Historical comparisons (e.g., year-over-year) include prior years' data (e.g., 2024 for LYTD) using EXTRACT(YEAR FROM date).",
    "Active records are filtered with is_active = 'Y' in client, matter, and timecard tables, and end_date IS NULL in matter_date and timekeeper_date.",

    # Analytical Capabilities
    "Realization rate is calculated as (worked_amount / standard_amount * 100) in matter_timekeeper_summary or (worked_rate / standard_rate * 100) in timecard, indicating billing efficiency when below 100%.",
    "Utilization rate is computed as (worked_hours / budget_hours * 100) in timecard or matter_timekeeper_summary, assessing timekeeper productivity against targets.",
    "Year-over-year trends are analyzed using LAG or direct comparisons of aggregated metrics (e.g., worked_hours, worked_amount) by year or month.",
    "Month-over-month trends use DATE_TRUNC('month', date) to aggregate data, calculating percentage changes with (current - previous) / previous * 100.",
    "Anomalies (e.g., realization > 200%, utilization < 80%) are highlighted in responses, with thresholds defined in query logic (e.g., HAVING clauses).",
    "Aggregations (e.g., SUM(worked_hours), COUNT(matter_id)) are used for totals, with STRING_AGG for concatenating lists (e.g., practice_group, rate_set_code).",
    "Rankings (e.g., top 10 clients by hours) use RANK() OVER (ORDER BY metric DESC) to prioritize high-impact records.",

    # Data Handling and Validation
    "Nullable fields (e.g., worked_amount, standard_amount) are handled with COALESCE to prevent null results in aggregations.",
    "Data errors (e.g., worked_amount > standard_amount, null worked_rate) are flagged in responses, recommending investigation.",
    "The is_error = 'N' flag in matter_timekeeper_summary ensures accurate aggregations by excluding erroneous entries.",
    "Currency mismatches (e.g., worked_currency vs. matter_currency) are noted as potential billing errors, requiring manual review.",
    "Missing columns (e.g., outstanding balances, contact information) are acknowledged as limitations, with approximations (e.g., client_name for contact) used where possible.",
    "Statistical measures (e.g., standard deviation of worked_rate / standard_rate) quantify variability in billing practices when requested.",

    # Error and Ambiguity Handling
    "Unclear queries (e.g., 'List my stuff') prompt clarification: 'Please specify clients, matters, or another category, with details like time period or filters.'",
    "Out-of-scope queries (e.g., non-legal data) return: 'This query is outside the rate model schema's scope. Please provide a query about clients, matters, or billing.'",
    "Invalid inputs (e.g., non-existent names) trigger: 'No results found for the specified criteria (e.g., name not found). Please verify and try again.'",
    "If no matching training query exists, the assistant attempts to generate a new SQL query based on schema knowledge, or responds: 'Unable to generate a query. Please rephrase or provide more details.'",
]

# Automatically train on database schema