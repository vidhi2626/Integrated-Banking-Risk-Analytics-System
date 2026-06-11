import pandas as pd

def merge_datasets(loan_path, trans_path):
    loan_df = pd.read_csv(loan_path)
    trans_df = pd.read_csv(trans_path)

    # Aggregate transaction data
    trans_summary = trans_df.groupby('customer_id').agg({
        'transaction_amount': ['mean', 'sum', 'count'],
        'fraud_flag': 'sum'
    })

    trans_summary.columns = [
        'avg_transaction',
        'total_transaction',
        'transaction_count',
        'fraud_transactions'
    ]

    trans_summary.reset_index(inplace=True) # reset_index() : Convert index → column

    # Merge datasets
    df = pd.merge(loan_df, trans_summary, on='customer_id', how='left')

    return df