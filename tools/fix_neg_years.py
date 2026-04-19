import polars as pl

users = pl.read_csv("./data/users.csv")
interactions = pl.read_parquet("./data/interactions/part_*.parquet")

# Recover decisive_ts from experience_years
# Then replace negative experience_years with time since user's first submission
SECS_PER_YEAR = 365.25 * 24 * 3600

interactions = (
    interactions
    .join(
        users.select(["user_index", "registration_time_seconds"]),
        on="user_index",
        how="left",
    )
    # Recover the absolute timestamp of the decisive submission
    .with_columns(
        (pl.col("experience_years") * SECS_PER_YEAR + pl.col("registration_time_seconds"))
        .alias("decisive_ts")
    )
    # For each user, find their earliest decisive_ts as a proxy for first submission
    .with_columns(
        pl.col("decisive_ts")
        .min()
        .over("user_index")
        .alias("first_submission_ts")
    )
    # Fix negative rows: experience = decisive_ts - first_submission_ts, min 1 day
    .with_columns(
        pl.when(pl.col("experience_years") < 0)
        .then(
            ((pl.col("decisive_ts") - pl.col("first_submission_ts")) / SECS_PER_YEAR)
            .clip(lower_bound=24 * 3600 / SECS_PER_YEAR)
        )
        .otherwise(pl.col("experience_years"))
        .alias("experience_years")
    )
    .drop(["registration_time_seconds", "decisive_ts", "first_submission_ts"])
)
interactions.write_parquet("./data/interactions/fixed.parquet")