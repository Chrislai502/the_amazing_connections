import sqlite3
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_evaluations_pandadataframe(db_name="evals.db"):
    conn = sqlite3.connect(db_name)
    
    df = pd.read_sql_query("""
    SELECT id, timestamp, hallucination_rate, num_failed_guesses, solve_rate, solve_order, num_tokens_generated, num_tokens_ingested
    FROM evaluations
    ORDER BY timestamp DESC
    """, conn)
    
    df.columns = [
        'Evaluation ID', 
        'Timestamp', 
        'Hallucination Rate', 
        'Failed Guesses', 
        'Solve Rate', 
        'Solve Order', 
        'Tokens Generated (Completion)', 
        'Tokens Ingested (Prompt)'
    ]
    
    conn.close()
    return df

def analyze_evaluations(df):
    """Provide a comprehensive analysis of evaluation metrics"""
    analysis = {
        'Total Evaluations': len(df),
        'Performance Metrics': {
            'Solve Rate': {
                'Mean': df['Solve Rate'].mean(),
                'Median': df['Solve Rate'].median(),
                'Success Rate (100%)': (df['Solve Rate'] == 100.0).sum() / len(df) * 100
            },
            'Failed Guesses': {
                'Mean': df['Failed Guesses'].mean(),
                'Median': df['Failed Guesses'].median(),
                'Max': df['Failed Guesses'].max()
            },
            'Hallucination Rate': {
                'Mean': df['Hallucination Rate'].mean(),
                'Median': df['Hallucination Rate'].median(),
                'Evaluations with Hallucinations': (df['Hallucination Rate'] > 0).sum()
            },
            'Tokens': {
                'Mean Generated': df['Tokens Generated (Completion)'].mean(),
                'Mean Ingested': df['Tokens Ingested (Prompt)'].mean(),
                'Token Ratio (Generated/Ingested)': df['Tokens Generated (Completion)'].sum() / df['Tokens Ingested (Prompt)'].sum()
            }
        }
    }
    return analysis

def create_metrics_visualization(df):
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    sns.histplot(df['Solve Rate'], kde=True)
    plt.title('Solve Rate Distribution')
    plt.xlabel('Solve Rate')
    
    plt.subplot(2, 2, 2)
    sns.histplot(df['Failed Guesses'], kde=True)
    plt.title('Failed Guesses Distribution')
    plt.xlabel('Number of Failed Guesses')
    
    plt.subplot(2, 2, 3)
    sns.histplot(df['Hallucination Rate'], kde=True)
    plt.title('Hallucination Rate Distribution')
    plt.xlabel('Hallucination Rate')
    
    plt.subplot(2, 2, 4)
    plt.scatter(df['Tokens Ingested (Prompt)'], df['Tokens Generated (Completion)'])
    plt.title('Tokens Ingested vs Generated')
    plt.xlabel('Tokens Ingested')
    plt.ylabel('Tokens Generated')
    
    plt.tight_layout()
    plt.savefig('evaluation_metrics.png')
    plt.close()

def json_numpy_serializer(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return int(obj) if isinstance(obj, np.integer) else float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def main():
    # Get the DataFrame
    df = get_evaluations_pandadataframe()

    # Display the DataFrame
    print(df)
    print("\nBasic Statistics:")
    print(df.describe())

    # Perform analysis
    analysis = analyze_evaluations(df)

    # Create visualization
    create_metrics_visualization(df)

    # Print analysis
    print("Comprehensive Evaluation Analysis:")
    try:
        print(json.dumps(analysis, indent=2, default=json_numpy_serializer))
    except Exception as e:
        print(f"Serialization error: {e}")

if __name__ == "__main__":
    main()