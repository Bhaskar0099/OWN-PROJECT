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
# Training data for the General Query Assistant
# This dataset includes SQL queries paired with natural-language questions, covering various categories.
# All column names match the YAML configuration exactly.

general_query_sql_training = [
    # --- Core Analytics: Hours and Trends ---
    (
        'SELECT DATE_TRUNC(\'month\', tc.date)::DATE AS work_month, SUM(tc.worked_hours) AS total_hours\n'
        'FROM public.timecard tc\n'
        'JOIN public.timekeeper_date tkd ON tc.timekeeper_id = tkd.timekeeper_id\n'
        'WHERE tkd.office = \'New York\' AND tkd.title = \'Paralegal\' AND tkd.end_date IS NULL\n'
        'GROUP BY work_month\n'
        'ORDER BY work_month;',
        "What are the hours trends for New York paralegals?"
    ),
    (
        '-- Note: Assumes "my team" are timekeepers on matters the current user supervises.\n'
        'SELECT DATE_TRUNC(\'month\', tc.date)::DATE AS work_month, SUM(tc.worked_hours) AS total_team_hours\n'
        'FROM public.timecard tc\n'
        'JOIN public.matter m ON tc.matter_id = m.matter_id\n'
        'WHERE m.supervising_timekeeper_id = \'your_timekeeper_id\' -- Placeholder for the current user\n'
        'GROUP BY work_month\n'
        'ORDER BY work_month;',
        "What are my team’s total hours over time?"
    ),

    # --- Core Analytics: Profitability and Margins ---
    (
        '-- Note: "Low profit margin" is defined as below 50%.\n'
        'WITH ProfitabilityData AS (\n'
        '    SELECT tkd.office, SUM(tc.worked_amount) AS total_revenue, SUM(tc.worked_hours * tk.cost_rate) AS total_cost\n'
        '    FROM public.timecard tc\n'
        '    JOIN public.timekeeper tk ON tc.timekeeper_id = tk.timekeeper_id\n'
        '    JOIN public.timekeeper_date tkd ON tk.timekeeper_id = tkd.timekeeper_id\n'
        '    WHERE tkd.end_date IS NULL AND tc.is_nonbillable = \'N\' AND tk.cost_rate IS NOT NULL AND tc.worked_amount > 0\n'
        '    GROUP BY tkd.office\n'
        ')\n'
        'SELECT office, (total_revenue - total_cost) / total_revenue AS profit_margin\n'
        'FROM ProfitabilityData\n'
        'WHERE (total_revenue - total_cost) / NULLIF(total_revenue, 0) < 0.5\n'
        'ORDER BY profit_margin;',
        "Which offices have profit margins below 50%?"
    ),
    (
        '-- Note: "Unusually low" is defined as a profit margin below 40%.\n'
        'SELECT t.timekeeper_name, (SUM(mts.worked_amount) - SUM(mts.worked_hours * t.cost_rate)) / NULLIF(SUM(mts.worked_amount), 0) AS profit_margin\n'
        'FROM public.matter_timekeeper_summary mts\n'
        'JOIN public.timekeeper t ON mts.timekeeper_id = t.timekeeper_id\n'
        'JOIN public.timekeeper_date tkd ON t.timekeeper_id = tkd.timekeeper_id\n'
        'WHERE tkd.title = \'Partner\' AND tkd.end_date IS NULL AND t.cost_rate > 0\n'
        'GROUP BY t.timekeeper_id, t.timekeeper_name\n'
        'HAVING (SUM(mts.worked_amount) - SUM(mts.worked_hours * t.cost_rate)) / NULLIF(SUM(mts.worked_amount), 0) < 0.4\n'
        'ORDER BY profit_margin;',
        "Which partners have unusually low profit margins?"
    ),

    # --- Core Analytics: Rates and Exceptions ---
    (
        '-- Note: "Exception rate" is interpreted as a record in `rate_detail` with `deviation_percent` > 10.\n'
        'SELECT DISTINCT t.timekeeper_name\n'
        'FROM public.timekeeper t\n'
        'JOIN public.rate_detail rd ON t.timekeeper_id = rd.timekeeper_id\n'
        'WHERE rd.deviation_percent > 10;',
        "Which timekeepers have exception rates greater than 10%?"
    ),
    (
        'SELECT t.timekeeper_name, MAX(rd.rate_amount) as max_rate\n'
        'FROM public.rate_detail rd\n'
        'JOIN public.timekeeper t ON rd.timekeeper_id = t.timekeeper_id\n'
        'JOIN public.timekeeper_date tkd ON t.timekeeper_id = tkd.timekeeper_id\n'
        'WHERE tkd.title = \'Partner\' AND tkd.end_date IS NULL AND rd.end_date IS NULL\n'
        'GROUP BY t.timekeeper_name\n'
        'ORDER BY max_rate DESC LIMIT 10;',
        "Which partners currently have the highest billing rates?"
    ),

    # --- Core Analytics: Utilization and Budget ---
    (
        '-- Note: "Low utilization" is defined as < 80% of annual budget year-to-date.\n'
        'WITH TimekeeperHours AS (\n'
        '    SELECT timekeeper_id, SUM(worked_hours) as ytd_hours\n'
        '    FROM public.timecard\n'
        '    WHERE EXTRACT(YEAR FROM date) = EXTRACT(YEAR FROM CURRENT_DATE) AND is_nonbillable = \'N\'\n'
        '    GROUP BY timekeeper_id\n'
        ')\n'
        'SELECT t.timekeeper_name, (th.ytd_hours / NULLIF(t.budget_hours, 0)) * 100 as utilization_pct\n'
        'FROM public.timekeeper t\n'
        'JOIN TimekeeperHours th ON t.timekeeper_id = th.timekeeper_id\n'
        'WHERE (th.ytd_hours / NULLIF(t.budget_hours, 0) * 100) < 80 AND t.is_active = \'Y\' AND t.budget_hours > 0\n'
        'ORDER BY utilization_pct;',
        "Which timekeepers are showing low utilization?"
    ),
    (
        'SELECT t.timekeeper_name, t.budget_hours AS annual_budget_hours, SUM(tc.worked_hours) AS ytd_worked_hours,\n'
        '  (SUM(tc.worked_hours) / NULLIF(t.budget_hours, 0)) * 100 AS budget_utilization_pct\n'
        'FROM public.timekeeper t\n'
        'LEFT JOIN public.timecard tc ON t.timekeeper_id = tc.timekeeper_id AND EXTRACT(YEAR FROM tc.date) = EXTRACT(YEAR FROM CURRENT_DATE) AND tc.is_nonbillable = \'N\'\n'
        'WHERE t.timekeeper_name = \'Mike\'\n'
        'GROUP BY t.timekeeper_id, t.timekeeper_name, t.budget_hours;',
        "How is Mike pacing against his budgeted hours and fees?"
    ),

    # --- Core Analytics: Realization ---
    (
        'SELECT (SUM(mts.worked_amount) / NULLIF(SUM(mts.standard_amount), 0)) * 100 AS realization_rate\n'
        'FROM public.matter_timekeeper_summary mts\n'
        'JOIN public.timekeeper_date tkd ON mts.timekeeper_id = tkd.timekeeper_id\n'
        'WHERE tkd.office = \'Chicago\' AND tkd.title = \'Associate\' AND tkd.end_date IS NULL AND mts.standard_amount > 0;',
        "What is the realization rate for Chicago associates?"
    ),
    (
        'SELECT m.matter_name, c.client_name, (SUM(mts.worked_amount) / NULLIF(SUM(mts.standard_amount), 0)) * 100 AS realization_rate\n'
        'FROM public.matter_timekeeper_summary mts\n'
        'JOIN public.matter m ON mts.matter_id = m.matter_id\n'
        'JOIN public.client c ON m.client_id = c.client_id\n'
        'WHERE m.is_active = \'Y\'\n'
        'GROUP BY m.matter_id, m.matter_name, c.client_name\n'
        'HAVING SUM(mts.standard_amount) > 0\n'
        'ORDER BY realization_rate ASC LIMIT 10;',
        "Which matters have the worst realization percentages?"
    ),
    (
        '-- Note: The number of matters [N] is set to 5 in this example.\n'
        'WITH LowRealizationMatters AS (\n'
        '    SELECT m.matter_id FROM public.matter_timekeeper_summary mts JOIN public.matter m ON mts.matter_id = m.matter_id\n'
        '    WHERE m.is_active = \'Y\'\n'
        '    GROUP BY m.matter_id\n'
        '    HAVING SUM(worked_amount) / NULLIF(SUM(standard_amount), 0) < 0.8\n'
        ')\n'
        'SELECT t.timekeeper_name, COUNT(DISTINCT m.matter_id) as low_realization_matter_count\n'
        'FROM public.matter m\n'
        'JOIN public.timekeeper t ON m.billing_timekeeper_id = t.timekeeper_id\n'
        'WHERE m.matter_id IN (SELECT matter_id FROM LowRealizationMatters)\n'
        'GROUP BY t.timekeeper_name\n'
        'HAVING COUNT(DISTINCT m.matter_id) > 5\n'
        'ORDER BY low_realization_matter_count DESC;',
        "Which billing attorneys have more than [N] matters with realization under 80%?"
    ),

    # --- Core Analytics: Matter and Client Load ---
    (
        'SELECT t.timekeeper_name, tkd.office, COUNT(DISTINCT tc.matter_id) as active_matter_count\n'
        'FROM public.timecard tc\n'
        'JOIN public.timekeeper t ON tc.timekeeper_id = t.timekeeper_id\n'
        'JOIN public.timekeeper_date tkd ON t.timekeeper_id = tkd.timekeeper_id\n'
        'JOIN public.matter m ON tc.matter_id = m.matter_id\n'
        'WHERE tkd.title = \'Paralegal\' AND tkd.end_date IS NULL AND m.is_active = \'Y\'\n'
        'GROUP BY t.timekeeper_name, tkd.office\n'
        'ORDER BY active_matter_count DESC;',
        "How many matters is each paralegal handling (matter load by paralegal)?"
    ),
    (
        '-- Note: "Handling" is interpreted as being a designated timekeeper (billing, responsible, or supervising).\n'
        'SELECT m.matter_number, m.matter_name, c.client_name FROM public.matter m\n'
        'JOIN public.client c ON m.client_id = c.client_id\n'
        'WHERE m.is_active = \'Y\' AND (\n'
        '    m.billing_timekeeper_id = (SELECT timekeeper_id FROM public.timekeeper WHERE timekeeper_name = \'Batty Lynn Bond\') OR\n'
        '    m.responsible_timekeeper_id = (SELECT timekeeper_id FROM public.timekeeper WHERE timekeeper_name = \'Batty Lynn Bond\') OR\n'
        '    m.supervising_timekeeper_id = (SELECT timekeeper_id FROM public.timekeeper WHERE timekeeper_name = \'Batty Lynn Bond\')\n'
        ');',
        "Which matters is Batty Lynn Bond currently handling?"
    ),
    (
        'SELECT md.department, COUNT(DISTINCT m.matter_id) as open_matter_count\n'
        'FROM public.matter m\n'
        'JOIN public.matter_date md ON m.matter_id = md.matter_id\n'
        'WHERE m.is_active = \'Y\' AND md.end_date IS NULL AND md.department IS NOT NULL\n'
        'GROUP BY md.department\n'
        'ORDER BY open_matter_count DESC;',
        "How many open matters exist by department?"
    ),
    (
        'SELECT matter_name, client_id, rate_review_date FROM public.matter\n'
        'WHERE (rate_review_date IS NULL OR rate_review_date < CURRENT_DATE - INTERVAL \'1 year\') AND is_active = \'Y\';',
        "Which matters have not been reviewed in the last year?"
    ),
    (
        'SELECT c2.client_name FROM public.client c1\n'
        'JOIN public.client c2 ON c1.related_client_id = c2.related_client_id AND c1.client_id != c2.client_id\n'
        'WHERE c1.client_name = \'Pepsi\' AND c2.is_active = \'Y\';',
        "Which clients are related to Pepsi?"
    ),

    # --- Core Analytics: Rate Sets and Configuration ---
    (
        'SELECT\n'
        '  rs.rate_set_code, rs.rate_set_name,\n'
        '  rc.rate_type, rc.source_name, rc.start_date, rc.end_date,\n'
        '  rd.ordinal, rd.rate_amount, rd.rate_currency, rd.title, rd.officet\n'
        'FROM public.rate_set rs\n'
        'JOIN public.rate_set_link rsl ON rs.rate_set_id = rsl.rate_set_id\n'
        'JOIN public.rate_component rc ON rsl.rate_component_id = rc.rate_component_id\n'
        'JOIN public.rate_detail rd ON rc.rate_component_id = rd.rate_component_id\n'
        'WHERE rs.rate_set_code = \'CORP2024\'; -- Example rate_set_code',
        "What are the full rate-definition details for rate code GR001‑MG00365‑CX00083?"
    ),
    (
        'SELECT COUNT(DISTINCT m.rate_set_id) as rate_set_count\n'
        'FROM public.matter m\n'
        'JOIN public.client c ON m.client_id = c.client_id\n'
        'WHERE c.client_name = \'Apple\' AND m.is_active = \'Y\';',
        "How many rate sets does Apple have?"
    )
]
# Documentation for the General Query Assistant
# This file provides detailed guidance for the general_query_agent to handle natural-language queries
# within a RAG pipeline, using the rate model schema as of June 17, 2025.

general_query_documentation = [
    # Purpose and Scope
    "The General Query Assistant retrieves and analyzes data from the rate model schema, covering clients, matters, timekeepers, and billing information, to support queries in categories like Lists, Flat Fees, Productivity Analysis, Realization, E-billed, Rate Review, Utilization and Budget, Timecard Analysis, and Matter Load.",
    "The assistant processes natural-language questions, maps them to PostgreSQL queries, and delivers concise, actionable responses optimized for a Retrieval-Augmented Generation (RAG) pipeline.",

    # Query Processing Rules
    "Natural-language queries are parsed to identify entities (e.g., ‘clients’, ‘Lynn Bond’, ‘Retail’) and intents (e.g., list, trend, analyze), using training data question-SQL pairs as reference.",
    "The term ‘my’ refers to the user’s context (e.g., relationship_timekeeper_id for clients, billing_timekeeper_id for matters), validated against the user’s timekeeper_id.",
    "Names (e.g., ‘Lynn Bond’, ‘Pepsi’) are matched case-insensitively using ILIKE on fields like timekeeper_name or client_name, prompting clarification if multiple matches occur.",
    "Ambiguous terms (e.g., ‘Retail’) are searched across relevant fields (e.g., client_name, client.category, matter.type, matter_date.practice_group), noting the matched field in responses.",
    "Placeholders like ‘user_id’ are replaced with the authenticated user’s timekeeper_id, ensuring queries respect user permissions.",
    "Date-based filters use the current date (June 17, 2025) for calculations (e.g., YTD = 2025, last 21 days = CURRENT_DATE - INTERVAL ‘21 days’).",
    "Historical comparisons (e.g., year-over-year) include prior years’ data (e.g., 2024 for LYTD) using EXTRACT(YEAR FROM date).",
    "Active records are filtered with is_active = ‘Y’ in client, matter, and timecard tables, and end_date IS NULL in matter_date and timekeeper_date.",

    # Analytical Capabilities
    "Realization rate is calculated as (worked_amount / standard_amount * 100) in matter_timekeeper_summary or (worked_rate / standard_rate * 100) in timecard, indicating billing efficiency when below 100%.",
    "Utilization rate is computed as (worked_hours / budget_hours * 100) in timecard or matter_timekeeper_summary, assessing timekeeper productivity against targets.",
    "Year-over-year trends are analyzed using LAG or direct comparisons of aggregated metrics (e.g., worked_hours, worked_amount) by year or month.",
    "Month-over-month trends use DATE_TRUNC(‘month’, date) to aggregate data, calculating percentage changes with (current - previous) / previous * 100.",
    "Anomalies (e.g., realization > 200%, utilization < 80%) are highlighted in responses, with thresholds defined in query logic (e.g., HAVING clauses).",
    "Aggregations (e.g., SUM(worked_hours), COUNT(matter_id)) are used for totals, with STRING_AGG for concatenating lists (e.g., practice_group, rate_set_code).",
    "Rankings (e.g., top 10 clients by hours) use RANK() OVER (ORDER BY metric DESC) to prioritize high-impact records.",

    # Data Handling and Validation
    "Nullable fields (e.g., worked_amount, standard_amount) are handled with COALESCE to prevent null results in aggregations.",
    "Data errors (e.g., worked_amount > standard_amount, null worked_rate) are flagged in responses, recommending investigation.",
    "The is_error = ‘N’ flag in matter_timekeeper_summary ensures accurate aggregations by excluding erroneous entries.",
    "Currency mismatches (e.g., worked_currency vs. matter_currency) are noted as potential billing errors, requiring manual review.",
    "Missing columns (e.g., outstanding balances, contact information) are acknowledged as limitations, with approximations (e.g., client_name for contact) used where possible.",
    "Statistical measures (e.g., standard deviation of worked_rate / standard_rate) quantify variability in billing practices when requested.",

    # Error and Ambiguity Handling
    "Unclear queries (e.g., ‘List my stuff’) prompt clarification: ‘Please specify clients, matters, or another category, with details like time period or filters.’",
    "Out-of-scope queries (e.g., non-legal data) return: ‘This query is outside the rate model schema’s scope. Please provide a query about clients, matters, or billing.’",
    "Invalid inputs (e.g., non-existent names) trigger: ‘No results found for the specified criteria (e.g., name not found). Please verify and try again.’",
    "If no matching training query exists, the assistant attempts to generate a new SQL query based on schema knowledge, or responds: ‘Unable to generate a query. Please rephrase or provide more details.’",
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

    # 3) Connect to PostgreSQL
    vn.connect_to_postgres(
        host=os.getenv('POSTGRES_HOST'),
        dbname=os.getenv('POSTGRES_DB'),
        user=os.getenv('UNDERBILLING_USER'),
        password=os.getenv('UNDERBILLING_PASSWORD'),
        port=int(os.getenv('POSTGRES_PORT', 5432))
    )

    # 4) Automatically train on database schema
    print("Auto-discovering database schema...")
    try:
        # Get database schema information from INFORMATION_SCHEMA
        df_information_schema = vn.run_sql("SELECT * FROM INFORMATION_SCHEMA.COLUMNS")
        
        # Generate training plan from the schema
        plan = vn.get_training_plan_generic(df_information_schema)
        
        # Train Vanna with the auto-generated plan
        vn.train(plan=plan)
        
        print(f"Successfully trained on database schema from the connected database.")
    except Exception as e:
        print(f"Warning: Could not auto-train on database schema: {e}")
        print("Continuing with manual training data only...")

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



