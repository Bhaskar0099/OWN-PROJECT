#!/usr/bin/env python3

import os
import openai
from dotenv import load_dotenv

# === IMPORT AGENTS ===
from underbilling_agent.cli import (
    MyVanna as UnderVanna,
    underbilling_sql_training,
    underbilling_documentation,
    ddl_statements as underbilling_ddl,
)
from profitability_agent.cli import (
    MyVanna as ProfitVanna,
    profitability_risks_sql_training,
    profitability_risks_documentation,
    profitability_risks_ddl_statements,
)
from cross_matter_rate_consistency_agent.cli import (
    MyVanna as CrossVanna,
    cross_matter_rate_consistency_sql_training,
    cross_matter_rate_consistency_documentation,
    cross_matter_rate_consistency_ddl_statements,
)

# === LOAD ENV VARIABLES ===
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_KEY
MODEL = os.getenv("MODEL_NAME", "gpt-4o-mini")

shared_config = {
    "api_key": OPENAI_KEY,
    "model": MODEL,
    "embedding_model": "text-embedding-3-small",
    "allow_llm_to_see_data": True,
}

# === SETUP UNDERBILLING AGENT ===
under_vanna = UnderVanna(config=shared_config)
for sql, q in underbilling_sql_training:
    under_vanna.train(sql=sql, question=q)
for doc in underbilling_documentation:
    under_vanna.train(documentation=doc)
for ddl in underbilling_ddl:
    under_vanna.train(ddl=ddl)
under_vanna.connect_to_postgres(
    host=os.getenv("POSTGRES_HOST"),
    dbname=os.getenv("POSTGRES_DB"),
    user=os.getenv("UNDERBILLING_USER"),
    password=os.getenv("UNDERBILLING_PASSWORD"),
    port=int(os.getenv("POSTGRES_PORT", 5432))
)

# === SETUP PROFITABILITY AGENT ===
profit_vanna = ProfitVanna(config=shared_config)
for sql, q in profitability_risks_sql_training:
    profit_vanna.train(sql=sql, question=q)
for doc in profitability_risks_documentation:
    profit_vanna.train(documentation=doc)
for ddl in profitability_risks_ddl_statements:
    profit_vanna.train(ddl=ddl)
profit_vanna.connect_to_postgres(
    host=os.getenv("POSTGRES_HOST"),
    dbname=os.getenv("POSTGRES_DB"),
    user=os.getenv("PROFITABILITY_USER"),
    password=os.getenv("PROFITABILITY_PASSWORD"),
    port=int(os.getenv("POSTGRES_PORT", 5432))
)

# === SETUP CROSS-MATTER AGENT ===
cross_vanna = CrossVanna(config=shared_config)
for sql, q in cross_matter_rate_consistency_sql_training:
    cross_vanna.train(sql=sql, question=q)
for doc in cross_matter_rate_consistency_documentation:
    cross_vanna.train(documentation=doc)
for ddl in cross_matter_rate_consistency_ddl_statements:
    cross_vanna.train(ddl=ddl)
cross_vanna.connect_to_postgres(
    host=os.getenv("POSTGRES_HOST"),
    dbname=os.getenv("POSTGRES_DB"),
    user=os.getenv("CROSS_MATTER_AGENT_USER"),
    password=os.getenv("CROSS_MATTER_AGENT_PASSWORD"),
    port=int(os.getenv("POSTGRES_PORT", 5432))
)

# === SUPERVISOR PROMPT ===
SUPERVISOR_PROMPT = """
You are the Legal Billing Supervisor Agent. Your task is to classify a user's question and route it to one of three specialized agents, each focused on a distinct type of analysis.

You must return ONLY one of the following tokens:
- underbilling
- profitability
- cross_matter

No other text. Just the token.

---

Agent: **underbilling**

Focuses on identifying cases where actual billed rates are lower than standard rates, or where non-billable, discounted, or zero-charge timecards suggest revenue leakage.

Detection logic:
- If actual rate < 60% of standard rate for at least two consecutive months, flag it.
- Metrics: worked_rate, standard_rate, worked_amount, standard_amount, discount %
- Also monitors: timekeeper roles (e.g. senior partners billing low), repeated underbilling, or billing mismatches.

Example questions:
- "Which timecards are below 60% of standard rate?"
- "Who billed below standard for three months?"
- "Are there any timekeepers discounting heavily?"

---

Agent: **profitability**

Evaluates profitability risks using realization rate, billing efficiency, collections, and discounts by practice area.

Detection logic:
- Group by practice area
- If discount > 25% OR realization < 80%, flag as risk
- Analyze timecard-based and matter-based billing efficiency

Example questions:
- "Which practice groups have low realization?"
- "Identify profitability risks by area"
- "Where is the revenue vs cost mismatch?"

---

Agent: **cross_matter**

Detects inconsistencies in rate usage across matters or clients.

Detection logic:
- Compare rates across multiple matters under the same client
- If a matter deviates from client-average discount by more than 10%, flag it
- Highlight matters using inconsistent currencies or rate sets

Example questions:
- "Which matters have inconsistent rates for the same client?"
- "Are there currency mismatches in standard rates?"
- "Find matters using deviated rate details"

---

Classify based on intent and logic match. Return ONLY one of:
- underbilling
- profitability
- cross_matter
"""

def choose_agent(question: str) -> str:
    resp = openai.chat.completions.create(
        model=MODEL,
        temperature=0.0,
        messages=[
            {"role": "system", "content": SUPERVISOR_PROMPT},
            {"role": "user", "content": question},
        ]
    )
    agent_type_raw = resp.choices[0].message.content
    cleaned = agent_type_raw.strip().lower()
    print(f"[DEBUG] LLM raw response: {repr(agent_type_raw)}")
    print(f"[DEBUG] Normalized agent type: '{cleaned}'")
    return cleaned

def print_result(result):
    df = result.get("df")
    chart_url = result.get("chart_url")
    summary = result.get("summary")

    print("\n📊 TOP 10 RESULTS:")
    if df is not None and not df.empty:
        print(df.head(10).to_string(index=False))
    else:
        print("No data returned.")

    if chart_url:
        print(f"\n📈 CHART: {chart_url}")

    if summary:
        print(f"\n🧠 SUMMARY:\n{summary}")

# === MAIN LOOP ===
if __name__ == "__main__":
    print("🔍 Supervisor Agent ready. (type 'quit' to exit')")
    while True:
        q = input("\n> ").strip()
        if not q or q.lower() in ("quit", "exit"):
            break

        agent_type = choose_agent(q)

        result = None
        if agent_type == "underbilling":
            print("📌 Routing to Underbilling Agent...")
            result = under_vanna.ask(q)
        elif agent_type == "profitability":
            print("📌 Routing to Profitability Agent...")
            result = profit_vanna.ask(q)
        elif agent_type in ("cross_matter", "cross-matter"):
            print("📌 Routing to Cross-Matter Consistency Agent...")
            result = cross_vanna.ask(q)
        else:
            print(f"❓ Unrecognized agent: '{agent_type}'")
            continue

        if result:
            print_result(result)
        else:
            print("⚠️ No response from selected agent.")

    print("\n👋 Goodbye!")
