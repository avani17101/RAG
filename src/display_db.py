import sqlite3
# Connect to the SQLite database
conn = sqlite3.connect("/home/avani/roshn/RAG/data/vector_db/urlavani17101.github.io_/chroma.sqlite3")


# Create a cursor object
cur = conn.cursor()

# Retrieve all tables in the database
cur.execute("SELECT name FROM sqlite_master WHERE type='table';")

tables = cur.fetchall()

# Loop through the tables and print their names and schemas
for table in tables:
    table_name = table[0]
    print(f"Table Name: {table_name}")
    
    # Retrieve the schema of the current table
    cur.execute(f"PRAGMA table_info({table_name});")
    columns = cur.fetchall()
    
    # Print the schema of the table
    print("Columns:")
    for column in columns:
        print(f"  {column[1]} ({column[2]})")  # Column name and type
    print()  # Add a newline for better readability between tables

# Close the connection to the database
conn.close()