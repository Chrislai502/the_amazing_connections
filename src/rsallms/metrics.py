from dataclasses import dataclass, field
from typing import List
import sqlite3
import datetime
from sentence_transformers import SentenceTransformer
import numpy as np


@dataclass
class Metrics:
    total_levels: int = 4
    points_per_correct: int = 5
    penalty_per_failed_guess: int = 1
    solves: List[bool] = field(default_factory=lambda: [False]*4)
    failed_guesses: int = 0
    solve_order: List[int] = field(default_factory=list)
    points: int = 0
    tokens_used: dict[str, dict[str, int]] = field(default_factory=dict)
    hallucinated_words: int = 0
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    category_similarity: float = 0.0

    def increment_failed_guesses(self):
        """Increment the count of failed guesses."""
        self.failed_guesses += 1

    def add_solve(self, level: int):
        """Record a successful solve at the given level."""
        if not self.solves[level]:
            self.solves[level] = True
            self.solve_order.append(level)
            self.points += self.points_per_correct

    def add_tokens(self, model_name: str, prompt_tokens: int, completion_tokens: int):
        if model_name in self.tokens_used:
            self.tokens_used[model_name] = {
                "prompt_tokens": self.tokens_used[model_name]["prompt_tokens"] + prompt_tokens,
                "completion_tokens": self.tokens_used[model_name]["completion_tokens"] + completion_tokens
            }
        else:
            self.tokens_used[model_name] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens
            }

    @property
    def solve_rate(self) -> float:
        """Calculate the solve rate as a percentage."""
        return (sum(self.solves) / self.total_levels) * 100

    @property
    def final_points(self) -> float:
        """Adjust the total points based on penalties."""
        f_points = self.points - self.failed_guesses * self.penalty_per_failed_guess
        return max(f_points, 0)  # Ensure points are not negative

    def to_dict(self) -> dict:
        """Convert the metrics data to a dictionary."""
        return {
            'solves': self.solves,
            'failed_guesses': self.failed_guesses,
            'solve_order': self.solve_order,
            'points': self.points,
            'solve_rate': self.solve_rate,
            'tokens_used': self.tokens_used,
            'num_hallucinated_words': self.hallucinated_words,
            'cosine_similarity': self.category_similarity
        }
    
    def hallucination_words(self, guess_word_lst: list[str], all_board_words: list[str]) -> float:
        """Get the number of words that are guessed, but not on the board"""
        board_word_set = set(all_board_words)

        hallucinated_words = sum(1 for word in guess_word_lst if word not in board_word_set)
        
        self.hallucinated_words += hallucinated_words
        
        return hallucinated_words
    
    def cosine_similarity_category(self, guessed_cat: str, correct_cat: str) -> float:
        """Given correct guess of words, return cosine similarity of guessed cat with the ground truth connections category"""
        embeddings = self.model.encode([guessed_cat, correct_cat])
        embedding1, embedding2 = embeddings[0], embeddings[1]
        similarity = np.dot(embedding1, embedding2)
        normalized_similarity = (similarity + 1) / 2
        print(f"Normalized similarity: {normalized_similarity}")
        print(f"category guess: {guessed_cat}")
        print(f"correct category: {correct_cat}")
        self.category_similarity = (((len(self.solve_order) - 1) * self.category_similarity) + normalized_similarity) / len(self.solve_order)
        return normalized_similarity

    def commit(self, to_db="evaluations.db"):
        conn = sqlite3.connect(to_db)

        # Make sure the table exists
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    hallucination_rate REAL,
                    num_failed_guesses INTEGER,
                    solve_rate REAL,
                    solve_order TEXT,
                    num_tokens_generated INTEGER,
                    num_tokens_ingested INTEGER,
                    cosine_similarity REAL
                )
            """)

        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Insert row
        with conn:
            conn.execute("""
                INSERT INTO evaluations (
                    timestamp, hallucination_rate, num_failed_guesses, solve_rate, 
                    solve_order, num_tokens_generated, num_tokens_ingested, cosine_similarity
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp,
                self.hallucinated_words,
                self.failed_guesses,
                self.solve_rate,
                str(self.solve_order),
                sum(t['completion_tokens'] for t in self.tokens_used.values()),
                sum(t['prompt_tokens'] for t in self.tokens_used.values()),
                self.category_similarity
            ))

        conn.close()
