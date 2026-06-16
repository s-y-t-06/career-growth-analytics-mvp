"""Chronological train/validation/test split by signup timestamp."""

import pandas as pd


def chronological_split(
    df: pd.DataFrame,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    signup_col: str = "signup_timestamp",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split users chronologically by signup timestamp.

    The split is deterministic: users are sorted by ``signup_col`` and partitioned
    into train/validation/test sets using the provided fractions. No user appears
    in more than one split.

    Parameters
    ----------
    df: DataFrame containing a ``signup_col`` column.
    train_frac: Fraction of users assigned to the training set.
    val_frac: Fraction of users assigned to the validation set.
    signup_col: Column name for the signup timestamp.

    Returns
    -------
    train, validation, test DataFrames.
    """
    if train_frac <= 0 or val_frac <= 0 or train_frac + val_frac >= 1.0:
        raise ValueError("train_frac and val_frac must be positive and sum to < 1.0")

    df = df.sort_values(by=signup_col).reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    return train_df, val_df, test_df


def split_users_and_labels(
    users: pd.DataFrame,
    labels: pd.DataFrame,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split users and labels together by signup timestamp.

    Returns
    -------
    train_users, val_users, test_users, train_labels, val_labels, test_labels.
    """
    users = users.copy()
    users["signup_timestamp"] = pd.to_datetime(users["signup_timestamp"])

    sorted_users = users.sort_values(by="signup_timestamp").reset_index(drop=True)
    n = len(sorted_users)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)

    train_users = sorted_users.iloc[:train_end]
    val_users = sorted_users.iloc[train_end:val_end]
    test_users = sorted_users.iloc[val_end:]

    train_labels = labels[labels["user_id"].isin(train_users["user_id"])].copy()
    val_labels = labels[labels["user_id"].isin(val_users["user_id"])].copy()
    test_labels = labels[labels["user_id"].isin(test_users["user_id"])].copy()

    return train_users, val_users, test_users, train_labels, val_labels, test_labels
