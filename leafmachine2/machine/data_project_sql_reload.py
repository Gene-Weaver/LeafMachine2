import os, time
import sqlite3
from dataclasses import dataclass, field
from sqlite3 import Error

@dataclass
class LoadProjectDB:
    db_path: str
    dir_images: str = field(default=None)  # Add the dir_images field
    conn: object = field(init=False)

    def __post_init__(self):
        """Initialize the database connection and ensure db_path is valid."""
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database not found at: {self.db_path}")
        self.conn = self.create_connection()
        
        # If dir_images is not provided, try to fetch it from the database or set it explicitly later
        if self.dir_images is None:
            self.dir_images = self.get_image_dir_from_db()

    def create_connection(self, retries=5, delay=0.1):
        """Establish connection to the SQLite database."""
        conn = None
        self.db_path = os.path.abspath(self.db_path)  # Ensure the path is absolute

        for attempt in range(retries):
            try:
                conn = sqlite3.connect(self.db_path)
                print(f"Connected to database at {self.db_path}")
                return conn
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e):
                    print(f"Database is locked, retrying in {delay} seconds... (Attempt {attempt + 1}/{retries})")
                    time.sleep(delay)
                else:
                    print(f"An error occurred: {e}")
                    break
        return conn

    def close_connection(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            print(f"Closed connection to database at {self.db_path}")
    
    def fetch_table(self, table_name: str):
        """Fetch all records from a table."""
        if not self.table_exists(table_name):
            raise ValueError(f"Table '{table_name}' does not exist.")
        
        cur = self.conn.cursor()
        cur.execute(f"SELECT * FROM {table_name}")
        rows = cur.fetchall()
        return rows
    
    def table_exists(self, table_name):
        """Check if a table exists in the database."""
        cur = self.conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        return cur.fetchone() is not None

    def execute_query(self, query: str, params: tuple = ()):
        """Execute a custom SQL query."""
        cur = self.conn.cursor()
        cur.execute(query, params)
        self.conn.commit()

    def get_image_metadata(self):
        """Fetch image metadata from the 'images' table."""
        return self.fetch_table('images')

    def get_image_dir_from_db(self):
        """Retrieve the image directory from the database, if stored."""
        # If the image directory is stored in the database, fetch it
        # This is just an example. Adjust based on how/where the directory is stored
        query = "SELECT path FROM images LIMIT 1"
        cur = self.conn.cursor()
        cur.execute(query)
        result = cur.fetchone()
        if result:
            return os.path.dirname(result[0])  # Extract directory from image path
        return None

    def get_database_path(self):
        """Return the absolute path to the database."""
        return self.db_path
    
    @property
    def database(self):
        """Alias for db_path to maintain compatibility with existing code."""
        return self.db_path


# Example usage:

# Assuming Dirs.database points to the database file path
# db_loader = LoadProjectDB(Dirs.database)

# # Now you can use db_loader with methods like `check_num_workers` and `test_sql`:
# cfg = check_num_workers(cfg, db_loader.dir_images)
# test_sql(db_loader.get_database_path())

# db_loader.close_connection()
