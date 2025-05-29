import pandas as pd
from sqlalchemy import create_engine

# Load the CSV file
file_path = '/home/bhaskar/Downloads/U_A/Underbilling-Agent/data/data.csv'
df = pd.read_csv(file_path)

# Database connection configuration
db_user = 'postgres'
db_password = '123456'
db_host = 'localhost'  # or your remote host
db_port = '5432'
db_name = 'WyzeAssist'

# Create a connection string
connection_string = f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'

# Create SQLAlchemy engine
engine = create_engine(connection_string)

# Push the DataFrame to PostgreSQL
df.to_sql('timekeeper_details', engine, if_exists='replace', index=False)

print("Data pushed to PostgreSQL table 'legal_billing' successfully!")
