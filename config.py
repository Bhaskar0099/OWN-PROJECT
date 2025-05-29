import psycopg2
import os
import yaml
from psycopg2 import sql
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load YAML config
with open("agent_access_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# PostgreSQL connection
conn = psycopg2.connect(
    dbname=os.getenv("POSTGRES_DB"),
    user=os.getenv("POSTGRES_USER"),
    password=os.getenv("POSTGRES_PASSWORD"),
    host=os.getenv("POSTGRES_HOST"),
    port=os.getenv("POSTGRES_PORT")
)
conn.autocommit = True
cursor = conn.cursor()

for agent_name, agent_data in config["agents"].items():
    role_env_var = agent_data["role_env"]
    role = os.getenv(role_env_var)
    password_env_var = agent_data["password_env"]
    password = os.getenv(password_env_var)

    if not role:
        print(f"[ERROR] Role not found for {agent_name}. Set {role_env_var} in .env")
        continue
    if not password:
        print(f"[ERROR] Password not found for {agent_name}. Set {password_env_var} in .env")
        continue

    print(f"🔧 Setting up role: {role}")

    try:
        # Create role if not exists
        cursor.execute(sql.SQL("DO $$ BEGIN IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = %s) THEN CREATE ROLE {} LOGIN PASSWORD %s; END IF; END $$;").format(sql.Identifier(role)), [role, password])
    except Exception as e:
        print(f"[ERROR] Failed to create role {role}: {e}")
        continue

    try:
        # Grant connect and usage
        cursor.execute(sql.SQL("GRANT CONNECT ON DATABASE {} TO {};").format(
            sql.Identifier(os.getenv("POSTGRES_DB")),
            sql.Identifier(role)
        ))
        cursor.execute(sql.SQL("GRANT USAGE ON SCHEMA public TO {};").format(sql.Identifier(role)))
    except Exception as e:
        print(f"[ERROR] Failed to grant DB/schema access to {role}: {e}")
        continue

    # Grant table/column access
    for table_name, columns in agent_data["tables"].items():
        try:
            grant_stmt = sql.SQL("GRANT SELECT ({}) ON public.{} TO {};").format(
                sql.SQL(', ').join(map(sql.Identifier, columns)),
                sql.Identifier(table_name),
                sql.Identifier(role)
            )
            cursor.execute(grant_stmt)
            print(f"✅ Granted access on {table_name}: {columns}")
        except Exception as e:
            print(f"[ERROR] Failed to grant access on {table_name} to {role}: {e}")

cursor.close()
conn.close()
