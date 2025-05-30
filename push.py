import pandas as pd
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values
import re

# --- Configuration ---
EXCEL_FILE_PATH = r'/home/bhaskar/Downloads/MAIN/OWN-PROJECT/RateModel.xlsx'  # Replace with your Excel file path

DB_HOST = 'localhost'
DB_PORT = '5432'  # Default PostgreSQL port
DB_NAME = 'WyzeRates'
DB_USER = 'postgres' 
DB_PASSWORD = '123456'

# --- Helper Functions ---

def sanitize_name(name):
    """Converts a string to a suitable SQL identifier (lowercase, underscores)."""
    name = str(name).strip()
    name = re.sub(r'\s+', '_', name)      # Replace spaces with underscores
    name = re.sub(r'[^a-zA-Z0-9_]', '', name) # Remove special characters except underscore
    name = name.lower()
    if not name: # Handle case where name becomes empty after sanitization
        raise ValueError("Sanitized name is empty. Original was: " + str(name))
    if name[0].isdigit(): # SQL identifiers cannot start with a digit
        name = "_" + name
    return name

def get_pg_type(pandas_dtype):
    """Maps pandas dtype to PostgreSQL data type."""
    if pd.api.types.is_integer_dtype(pandas_dtype):
        return 'BIGINT'  # Or INTEGER if your numbers aren't huge
    elif pd.api.types.is_float_dtype(pandas_dtype):
        return 'DOUBLE PRECISION' # Or NUMERIC for exact precision
    elif pd.api.types.is_bool_dtype(pandas_dtype):
        return 'BOOLEAN'
    elif pd.api.types.is_datetime64_any_dtype(pandas_dtype):
        return 'TIMESTAMP WITHOUT TIME ZONE'
    elif pd.api.types.is_timedelta64_dtype(pandas_dtype):
        return 'INTERVAL'
    # Default to TEXT for object types or any other unhandled types
    return 'TEXT'

def create_table_from_dataframe(cursor, table_name, df):
    """Creates a PostgreSQL table based on DataFrame columns and inferred types."""
    if df.empty:
        print(f"DataFrame for table '{table_name}' is empty. Skipping table creation.")
        return False

    columns_defs = []
    for col_name, dtype in df.dtypes.items():
        pg_type = get_pg_type(dtype)
        sanitized_col_name = sanitize_name(col_name)
        if not sanitized_col_name:
            print(f"Warning: Column name '{col_name}' became empty after sanitization. Skipping this column for table '{table_name}'.")
            continue
        columns_defs.append(sql.SQL("{} {}").format(
            sql.Identifier(sanitized_col_name),
            sql.SQL(pg_type)
        ))

    if not columns_defs:
        print(f"No valid columns found for table '{table_name}' after sanitization. Skipping table creation.")
        return False

    create_table_query = sql.SQL("CREATE TABLE IF NOT EXISTS {} ({})").format(
        sql.Identifier(table_name),
        sql.SQL(', ').join(columns_defs)
    )
    try:
        print(f"Executing: {create_table_query.as_string(cursor.connection)}")
        cursor.execute(create_table_query)
        print(f"Table '{table_name}' created successfully or already exists.")
        return True
    except psycopg2.Error as e:
        print(f"Error creating table '{table_name}': {e}")
        return False

def insert_data_from_dataframe(cursor, table_name, df):
    """Inserts data from DataFrame into the specified PostgreSQL table."""
    if df.empty:
        print(f"DataFrame for table '{table_name}' is empty. No data to insert.")
        return

    # Sanitize column names in the DataFrame to match table column names
    # This is important if the original Excel headers had spaces/special chars
    df_columns_sanitized = [sanitize_name(col) for col in df.columns]
    
    # Filter out columns that might have been skipped during table creation (e.g. empty sanitized name)
    valid_df_columns = [col for col in df_columns_sanitized if col]
    
    if not valid_df_columns:
        print(f"No valid columns to insert for table '{table_name}'. Skipping insertion.")
        return

    # Ensure the DataFrame columns used for insertion match the sanitized ones
    # Create a temporary DataFrame with sanitized column names for insertion
    temp_df = df.copy()
    temp_df.columns = df_columns_sanitized
    
    # Select only columns that were valid (non-empty after sanitization)
    temp_df_for_insert = temp_df[[col for col in df_columns_sanitized if col]]


    cols_sql = sql.SQL(', ').join(map(sql.Identifier, temp_df_for_insert.columns))
    insert_query = sql.SQL("INSERT INTO {} ({}) VALUES %s").format(
        sql.Identifier(table_name),
        cols_sql
    )

    # Convert DataFrame to list of tuples for execute_values
    # Handle NaT (Not a Time for datetime) and NaN (Not a Number for float) by converting to None
    data_tuples = []
    for row_tuple in temp_df_for_insert.itertuples(index=False, name=None):
        processed_row = []
        for val in row_tuple:
            if pd.isna(val):
                processed_row.append(None)
            else:
                processed_row.append(val)
        data_tuples.append(tuple(processed_row))


    if not data_tuples:
        print(f"No data rows to insert into '{table_name}'.")
        return

    try:
        print(f"Inserting {len(data_tuples)} rows into '{table_name}'...")
        execute_values(cursor, insert_query, data_tuples)
        print(f"Data inserted successfully into '{table_name}'.")
    except psycopg2.Error as e:
        print(f"Error inserting data into table '{table_name}': {e}")
        print(f"Failed query (values not shown for brevity): {insert_query.as_string(cursor.connection)}")
        # For more detailed debugging, you might want to log the first few data_tuples
        # print(f"First few data tuples: {data_tuples[:3]}")


# --- Main Logic ---
def main():
    conn = None
    try:
        # Connect to PostgreSQL
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        conn.autocommit = False # Use transactions
        cursor = conn.cursor()
        print("Successfully connected to PostgreSQL.")

        # Load the Excel file
        xls = pd.ExcelFile(EXCEL_FILE_PATH)
        sheet_names = xls.sheet_names
        print(f"Found sheets: {sheet_names}")

        for sheet_name in sheet_names:
            print(f"\n--- Processing sheet: '{sheet_name}' ---")
            try:
                df = pd.read_excel(xls, sheet_name=sheet_name)
            except Exception as e:
                print(f"Error reading sheet '{sheet_name}': {e}")
                continue

            if df.empty and df.columns.empty:
                print(f"Sheet '{sheet_name}' is completely empty. Skipping.")
                continue
            
            # Clean column names in DataFrame (e.g. remove leading/trailing spaces)
            df.columns = [str(col).strip() for col in df.columns]

            # Determine table name
            table_name = sanitize_name(sheet_name)
            if not table_name:
                print(f"Sheet name '{sheet_name}' resulted in an empty table name after sanitization. Skipping.")
                continue
            
            print(f"Target PostgreSQL table name: '{table_name}'")

            # Create table if it doesn't exist
            table_created_or_exists = create_table_from_dataframe(cursor, table_name, df)

            if table_created_or_exists:
                # Insert data
                # For simplicity, we are appending. If you want to clear the table first:
                # cursor.execute(sql.SQL("DELETE FROM {}").format(sql.Identifier(table_name)))
                # print(f"Cleared existing data from '{table_name}'.")
                insert_data_from_dataframe(cursor, table_name, df)
            else:
                print(f"Skipping data insertion for '{table_name}' due to table creation issues.")
            
            conn.commit() # Commit after each sheet/table is processed

        print("\n--- All sheets processed. ---")

    except FileNotFoundError:
        print(f"Error: Excel file not found at '{EXCEL_FILE_PATH}'")
    except psycopg2.Error as e:
        print(f"PostgreSQL Error: {e}")
        if conn:
            conn.rollback() # Rollback any pending transaction on global error
    except ValueError as e:
        print(f"Configuration or Naming Error: {e}")
        if conn:
            conn.rollback()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            cursor.close()
            conn.close()
            print("PostgreSQL connection closed.")

if __name__ == "__main__":
    main()