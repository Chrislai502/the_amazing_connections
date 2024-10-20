import sqlite3
import datetime

class EvaluationDatabase:
    def __init__(self, db_filepath="evaluation_metrics.db"):
        """
        Initializes the database connection and creates the metrics table if it doesn't exist.
        """
        self.db_filepath = db_filepath
        self.conn = sqlite3.connect(self.db_filepath)
        self.create_table_if_not_exists()

    def create_table_if_not_exists(self):
        """
        Creates a table for storing evaluation metrics if it doesn't already exist.
        """
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    hallucination_rate REAL,
                    num_failed_guesses INTEGER,
                    solve_rate REAL,
                    solve_order TEXT,
                    num_tokens_generated INTEGER,
                    num_tokens_ingested INTEGER
                )
            """)
        print("Database table created or verified.")

    def get_hallucination_rate(self):
        # Placeholder for logic to get hallucination rate
        return None

    def get_num_failed_guesses(self):
        # Placeholder for logic to get the number of failed guesses
        return None

    def get_solve_rate(self):
        # Placeholder for logic to get the solve rate
        return None

    def get_solve_order(self):
        # Placeholder for logic to get the solve order (should return a list or string representation of it)
        return None

    def get_num_tokens_generated(self):
        # Placeholder for logic to get the number of tokens generated
        return None

    def get_num_tokens_ingested(self):
        # Placeholder for logic to get the number of tokens ingested
        return None

    def add_evaluation_entry(self):
        """
        Collects metrics and adds a row entry to the database.
        """
        # Get the current date and time as a timestamp
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Collect all metric values (currently placeholders)
        hallucination_rate = self.get_hallucination_rate()
        num_failed_guesses = self.get_num_failed_guesses()
        solve_rate = self.get_solve_rate()
        solve_order = str(self.get_solve_order())  # Assuming a list, convert to string for storage
        num_tokens_generated = self.get_num_tokens_generated()
        num_tokens_ingested = self.get_num_tokens_ingested()

        # Insert the collected data into the database
        with self.conn:
            self.conn.execute("""
                INSERT INTO evaluations (
                    timestamp, hallucination_rate, num_failed_guesses, solve_rate, 
                    solve_order, num_tokens_generated, num_tokens_ingested
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp, hallucination_rate, num_failed_guesses, solve_rate,
                solve_order, num_tokens_generated, num_tokens_ingested
            ))
        print(f"Entry added at {timestamp}.")

    def close(self):
        """
        Closes the database connection.
        """
        self.conn.close()


# Example Usage:
if __name__ == "__main__":
    # Initialize the database
    db = EvaluationDatabase()

    # Add a new evaluation entry (this will collect data and insert a row)
    db.add_evaluation_entry()

    # Close the database connection when done
    db.close()
