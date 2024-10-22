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

    def add_evaluation_entry(self, hallucination_rate: float, num_failed_guesses: int, solve_rate: float, solve_order: list[int], num_tokens_generated: int, num_tokens_ingested: int):
        """
        Collects metrics and adds a row entry to the database.

        :param hallucination_rate: Number of hallucinated words.
        :param num_failed_guesses: Number of failed guesses.
        :param solve_rate: Solve rate percentage.
        :param solve_order: The order in which levels were solved.
        :param num_tokens_generated: Total tokens generated during the game.
        :param num_tokens_ingested: Total tokens ingested during the game.
        """
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        solve_order_str = str(solve_order)

        with self.conn:
            self.conn.execute("""
                INSERT INTO evaluations (
                    timestamp, hallucination_rate, num_failed_guesses, solve_rate, 
                    solve_order, num_tokens_generated, num_tokens_ingested
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp, hallucination_rate, num_failed_guesses, solve_rate,
                solve_order_str, num_tokens_generated, num_tokens_ingested
            ))
        print(f"Entry added at {timestamp}.")

    def close(self):
        """
        Closes the database connection.
        """
        self.conn.close()
