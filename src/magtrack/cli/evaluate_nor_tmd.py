import zipfile
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import typer
from loguru import logger
from sklearn.metrics import (accuracy_score, classification_report,
                             f1_score, matthews_corrcoef, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from xgboost import XGBClassifier

from magtrack.utils.utils import resolve_seed

app = typer.Typer(help="NOR-TMD evaluation pipeline (parse → preprocess → pivot → train)")

# ── Default constants ────────────────────────────────────────────────────────

DEFAULT_CSV_FILE = "nor_tmd.csv"
DEFAULT_DB_DIR = "data/nor_tmd/"
DEFAULT_DB_NAME = "nor_tmd_complete"
DEFAULT_CHUNK_SIZE = 10 ** 6
DEFAULT_SEG_DF_NAME = "segmented_df"
DEFAULT_TRAIN_DF_NAME = "data_android_centered"
DEFAULT_SEED = "magtrack"


# ── Helpers (parse) ──────────────────────────────────────────────────────────


def _init_db(db_path: Path, db_name: str, overwrite: bool = False) -> duckdb.DuckDBPyConnection:
    if db_path.exists() and db_path.stat().st_size > 0:
        if overwrite:
            db_path.unlink()
            logger.info(f"Existing database '{db_path}' deleted.")
        else:
            logger.info(f"Adding to existing database '{db_path}'.")
    else:
        logger.info(f"Creating new database '{db_path}'.")

    conn = duckdb.connect(str(db_path))
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {db_name} (
            installationId STRING,
            journeyNumber INTEGER,
            tripNumber INTEGER,
            manufacturer STRING,
            model STRING,
            timestamp TIMESTAMP,
            transportType STRING,
            typeInteger INTEGER,
            typeString STRING,
            deviceLocation STRING,
            value_0 FLOAT,
            value_1 FLOAT,
            value_2 FLOAT,
            value_3 FLOAT,
            value_4 FLOAT,
            value_5 FLOAT,
            OS STRING,
        )
    """)
    return conn


def _import_csv(conn: duckdb.DuckDBPyConnection, db_name: str,
                zip_path: str, csv_file: str, chunk_size: int):
    with zipfile.ZipFile(zip_path) as z:
        with z.open(csv_file) as f:
            with pd.read_csv(
                    f, chunksize=chunk_size, dtype={"activity_recognition": "string"}
            ) as reader:
                for chunk_num, chunk in enumerate(reader, start=1):
                    selected_columns = chunk.drop(columns=["activity_recognition"])
                    logger.info(f"Chunk {chunk_num}: {len(selected_columns)} rows inserted")
                    if len(chunk) > 0:
                        conn.execute(
                            f"INSERT INTO {db_name} BY NAME SELECT * FROM selected_columns"
                        )
                        conn.commit()
    conn.commit()


def _log_db_stats(conn: duckdb.DuckDBPyConnection, db_name: str):
    stats = conn.execute(f"""
        WITH by_trip AS (
            SELECT installationId, journeyNumber, tripNumber, COUNT(*) AS row_count
            FROM {db_name}
            GROUP BY installationId, journeyNumber, tripNumber
        )
        SELECT
            SUM(row_count),
            COUNT(DISTINCT installationId),
            COUNT(DISTINCT (installationId, journeyNumber)),
            COUNT(*)
        FROM by_trip
    """).fetchone()

    labels = ["rows", "installations", "journeys", "trips"]
    for label, value in zip(labels, stats):
        logger.info(f"Total {label}: {value}")


# ── Helpers (preprocess) ─────────────────────────────────────────────────────


def _get_journey_pairs(db_path: str, db_name: str) -> pd.DataFrame:
    """Get all installationId/journeyNumber pairs (for installations that provide
    pressure sensor data) from the database."""
    con = duckdb.connect(db_path)
    query = f"""
        WITH installations_with_pressure AS (
            SELECT DISTINCT installationId
            FROM '{db_name}'
            WHERE typeInteger IN (6, 1002)
        )
        SELECT installationId, journeyNumber,
               COUNT(DISTINCT tripNumber) AS trips, COUNT(*) AS count
        FROM '{db_name}'
        WHERE installationId IN (SELECT installationId FROM installations_with_pressure)
        GROUP BY installationId, journeyNumber
        ORDER BY installationId, journeyNumber
    """
    return con.execute(query).df()


def _load_journey_df(db_path: str, db_name: str,
                     installation_id: str, journey_number: int) -> pd.DataFrame:
    """Load data for a specific installationId/journeyNumber pair.
    Excludes ActivityRecognition (sensor ids 5577 / 1006) datapoints."""
    con = duckdb.connect(db_path)
    query = f"""
        SELECT *
        FROM '{db_name}'
        WHERE installationId = '{installation_id}'
        AND journeyNumber = {journey_number}
        AND typeInteger NOT IN (5577, 1006)
    """
    return con.execute(query).df()


def _calculate_magnitude(df: pd.DataFrame) -> pd.DataFrame:
    single_value_mask = df["typeInteger"].isin([6, 1002])
    three_value_mask = ~single_value_mask

    df = df.copy()
    df["magnitude"] = np.nan
    df.loc[three_value_mask, "magnitude"] = np.sqrt(
        df.loc[three_value_mask, "value_0"] ** 2
        + df.loc[three_value_mask, "value_1"] ** 2
        + df.loc[three_value_mask, "value_2"] ** 2
    )
    df.loc[single_value_mask, "magnitude"] = df.loc[single_value_mask, "value_0"]
    return df


def _segment_and_aggregate(
        df: pd.DataFrame, window_len_sec: int = 10, overlap_sec: int = 5, centered: bool = False
) -> pd.DataFrame:
    group_columns = ["installationId", "journeyNumber", "tripNumber", "typeInteger"]
    other_columns = [
        col for col in df.columns.tolist()
        if col not in group_columns + ["magnitude", "timestamp"]
    ]
    closed = "left" if centered else "right"

    df_sorted = df.sort_values("timestamp").set_index("timestamp").groupby(group_columns)
    roller = df_sorted["magnitude"].rolling(f"{window_len_sec}s", closed=closed, center=centered)

    df_rolled = pd.DataFrame({
        "min": roller.min(),
        "max": roller.max(),
        "mean": roller.mean(),
        "var": roller.var(),
        "std": roller.std(),
        "kurt": roller.kurt(),
        "q1": roller.quantile(0.25),
        "q2": roller.quantile(0.50),
        "q3": roller.quantile(0.75),
        "count": roller.count(),
    })

    df_rolled["range"] = df_rolled["max"] - df_rolled["min"]
    df_rolled["iqr"] = df_rolled["q3"] - df_rolled["q1"]

    df_windowed = (
        df_rolled.reset_index()
        .set_index("timestamp")
        .groupby(group_columns)
        .resample(f"{overlap_sec}s", include_groups=False)
        .first()
    )

    metadata = df_sorted[other_columns].first()
    df_windowed = df_windowed.join(metadata, on=group_columns)
    return df_windowed


# ── Helpers (pivot) ──────────────────────────────────────────────────────────


def _relabel(df: pd.DataFrame) -> pd.DataFrame:
    other_mask = df["transportType"].isin(["INSIDE", "OUTSIDE", "ESCOOTER"])
    df = df.copy()
    df.loc[other_mask, "transportType"] = "OTHER"
    return df


def _pivot_sensors_to_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    res_dict = {}

    for os_name in ("ANDROID", "iOS"):
        df_os = df[df["OS"] == os_name]

        id_columns = ["installationId", "journeyNumber", "tripNumber", "timestamp"]
        metadata_columns = ["manufacturer", "model", "transportType", "deviceLocation", "OS"]
        agg_columns = ["min", "max", "mean", "var", "std", "kurt",
                       "q1", "q2", "q3", "count", "range", "iqr"]

        metadata = df_os.groupby(id_columns)[metadata_columns].first()

        df_pivot = df_os.pivot_table(
            index=id_columns,
            columns="typeInteger",
            values=agg_columns,
            aggfunc="first",
        )
        df_pivot.columns = [f"{col[1]}_{col[0]}" for col in df_pivot.columns]

        df_result = df_pivot.join(metadata, on=id_columns).reset_index()
        res_dict[os_name] = df_result

    return res_dict["ANDROID"], res_dict["iOS"]


def _filter_trip_durations(
        df: pd.DataFrame, min_seconds: int = 60, max_seconds: int = 3600
) -> pd.DataFrame:
    trip_durations = df.groupby(
        ["installationId", "journeyNumber", "tripNumber"]
    )["timestamp"].agg(start_time="min", end_time="max")
    trip_durations["duration_seconds"] = (
            trip_durations["end_time"] - trip_durations["start_time"]
    ).dt.total_seconds()

    valid_trips = trip_durations[
        (trip_durations["duration_seconds"] >= min_seconds)
        & (trip_durations["duration_seconds"] <= max_seconds)
        ].index

    mask = df.set_index(
        ["installationId", "journeyNumber", "tripNumber"]
    ).index.isin(valid_trips)

    trips_before = df.groupby(["installationId", "journeyNumber", "tripNumber"]).ngroups
    logger.info(f"Trips before filtering: {trips_before}")
    logger.info(f"Trips after filtering:  {len(valid_trips)}")
    logger.info(f"Rows before: {len(df)}, rows after: {mask.sum()}")

    return df[mask]


# ── Helpers (train) ──────────────────────────────────────────────────────────


def _prepare_features_and_labels(df: pd.DataFrame):
    sensors = tuple(id_ + '_' for id_ in map(str, [
        1, 2, 4, 9, 11, 15, 10, 14, 16, 6, 20
    ]))
    feature_cols = [col for col in df.columns
                    if col.startswith(sensors) and "count" not in col]
    X = df[feature_cols]
    y = df["transportType"]

    logger.info(f"Features: {len(feature_cols)} ({', '.join(feature_cols)})")
    logger.info(f"Classes: {y.astype(str).nunique()} ({', '.join(y.astype(str).unique())})")

    class_counts = y.astype(str).value_counts()
    logger.info("Samples per class:")
    for class_name, count in class_counts.items():
        logger.info(f"  {class_name}: {count}")

    if y.astype(str).nunique() > 2:
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        class_names = label_encoder.classes_
    else:
        y_encoded = y.astype(int)
        class_names = ["Not TRAIN", "TRAIN"]

    logger.info(f"Labels encoded: {dict(enumerate(class_names))}")
    return X, y_encoded, class_names


# ── Commands ─────────────────────────────────────────────────────────────────


@app.command("parse")
def parse(
        zip_path: str = typer.Option(..., "--zip-path", "-z",
                                     help="Path to the NOR-TMD zip archive."),
        csv_file: str = typer.Option(DEFAULT_CSV_FILE, "--csv-file", "-c",
                                     help="Name of the CSV file inside the zip."),
        db_dir: str = typer.Option(DEFAULT_DB_DIR, "--db-dir", "-d",
                                   help="Directory for the DuckDB database."),
        db_name: str = typer.Option(DEFAULT_DB_NAME, "--db-name",
                                    help="Name of the DuckDB table/database."),
        chunk_size: int = typer.Option(DEFAULT_CHUNK_SIZE, "--chunk-size",
                                       help="Number of CSV rows per chunk."),
        overwrite: bool = typer.Option(False, "--overwrite", "-o",
                                       help="Overwrite existing database if it exists."),
):
    """Stage 1/4: Parse the NOR-TMD CSV from a zip archive into a DuckDB database."""
    logger.info("=== Stage 1/4: Parse ===")

    db_path = Path(db_dir) / f"{db_name}.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = _init_db(db_path, db_name, overwrite=overwrite)
    _import_csv(conn, db_name, zip_path, csv_file, chunk_size)
    _log_db_stats(conn, db_name)
    conn.close()

    logger.info("Parse complete.")


@app.command("preprocess")
def preprocess(
        db_dir: str = typer.Option(DEFAULT_DB_DIR, "--db-dir", "-d",
                                   help="Directory containing the DuckDB database."),
        db_name: str = typer.Option(DEFAULT_DB_NAME, "--db-name",
                                    help="Name of the DuckDB table/database."),
        output_name: str = typer.Option(DEFAULT_SEG_DF_NAME, "--output-name",
                                        help="Output parquet file name (without extension)."),
        window_len_sec: int = typer.Option(10, "--window-len", "-w",
                                           help="Rolling window length in seconds."),
        overlap_sec: int = typer.Option(5, "--overlap", help="Resample step in seconds."),
        centered: bool = typer.Option(True, "--centered/--no-centered",
                                      help="Whether to center rolling windows."),
):
    """Stage 2/4: Preprocess – segment and aggregate sensor data from the database."""
    logger.info("=== Stage 2/4: Preprocess ===")

    db_path = f"{db_dir}/{db_name}.db"
    output_path = f"{db_dir}/{output_name}.parquet"

    logger.info(f"Fetching journey pairs from '{db_name}'...")
    journey_pairs = _get_journey_pairs(db_path, db_name)
    logger.info(f"Found {len(journey_pairs)} journey pairs.")

    batches = []
    for idx, row in journey_pairs.iterrows():
        installation_id = row["installationId"]
        journey_number = row["journeyNumber"]
        logger.info(
            f"[{idx + 1}/{len(journey_pairs)}] installationId={installation_id}, "
            f"journeyNumber={journey_number}, trips={row['trips']}, events={row['count']}"
        )

        df = _load_journey_df(db_path, db_name, installation_id, journey_number)
        if df.empty:
            logger.warning("Skipped (no data after filtering)")
            continue

        df = _calculate_magnitude(df)
        df.drop(columns=[f"value_{i}" for i in range(6)], inplace=True)

        segmented_df = _segment_and_aggregate(
            df, window_len_sec=window_len_sec, overlap_sec=overlap_sec, centered=centered
        )
        batches.append(segmented_df)
        logger.info(f"  → {len(segmented_df)} windows")

    pd.concat(batches).to_parquet(output_path, engine="pyarrow")
    logger.info(f"Output saved to '{output_path}'")
    logger.info("Preprocess complete.")


@app.command("pivot")
def pivot(
        data_dir: str = typer.Option(DEFAULT_DB_DIR, "--data-dir", "-d",
                                     help="Directory containing the segmented parquet."),
        input_name: str = typer.Option(DEFAULT_SEG_DF_NAME, "--input-name",
                                       help="Segmented parquet file name (without extension)."),
        min_trip_sec: int = typer.Option(60, "--min-trip-sec",
                                         help="Minimum trip duration in seconds."),
        max_trip_sec: int = typer.Option(3600, "--max-trip-sec",
                                         help="Maximum trip duration in seconds."),
):
    """Stage 3/4: Pivot sensors to columns, split by OS, and filter trip durations."""
    logger.info("=== Stage 3/4: Pivot ===")

    input_path = f"{data_dir}/{input_name}.parquet"
    logger.info(f"Loading '{input_path}' and relabeling classes...")
    df = pd.read_parquet(input_path)
    df_relabeled = _relabel(df)
    logger.info(f"Loaded {len(df_relabeled)} rows. Minority classes merged into OTHER.")

    logger.info("Pivoting sensors to columns (split by OS)...")
    df_android, df_ios = _pivot_sensors_to_columns(df_relabeled)
    logger.info(f"Android: {len(df_android)} rows, {len(df_android.columns)} columns")
    logger.info(f"iOS:     {len(df_ios)} rows, {len(df_ios.columns)} columns")

    logger.info("Filtering trip durations (Android)...")
    df_filtered_android = _filter_trip_durations(df_android, min_trip_sec, max_trip_sec)
    logger.info("Filtering trip durations (iOS)...")
    df_filtered_ios = _filter_trip_durations(df_ios, min_trip_sec, max_trip_sec)

    android_path = f"{data_dir}/data_android_centered.parquet"
    ios_path = f"{data_dir}/data_ios_centered.parquet"
    df_filtered_android.to_parquet(android_path, engine="pyarrow")
    df_filtered_ios.to_parquet(ios_path, engine="pyarrow")
    logger.info(f"Saved Android data to '{android_path}'")
    logger.info(f"Saved iOS data to '{ios_path}'")
    logger.info("Pivot complete.")


@app.command("train")
def train(
        data_dir: str = typer.Option(DEFAULT_DB_DIR, "--data-dir", "-d",
                                     help="Directory containing the pivoted parquet."),
        input_name: str = typer.Option(DEFAULT_TRAIN_DF_NAME, "--input-name",
                                       help="Pivoted parquet file name (without extension)."),
        test_size: float = typer.Option(0.33, "--test-size",
                                        help="Fraction of data to use for testing."),
        seed: str = typer.Option(DEFAULT_SEED, "--seed", "-s",
                                 help="Random seed for reproducibility."),
        runs: int = typer.Option(None, "--runs", help="Number of training runs with different seeds.")
):
    """Stage 4/4: Train an XGBoost classifier and evaluate it."""
    logger.info("=== Stage 4/4: Train ===")

    input_path = f"{data_dir}/{input_name}.parquet"
    logger.info(f"Loading data from '{input_path}'...")
    df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    X, y_encoded, class_names = _prepare_features_and_labels(df)

    if runs is not None and runs > 1:
        all_results = []
        for run_idx in range(1, runs + 1):
            run_seed = f"{seed}{run_idx}"
            logger.info(f"\n{'=' * 50}")
            logger.info(f"Run {run_idx}/{runs} (seed='{run_seed}')")
            logger.info(f"{'=' * 50}")
            result = _train_single_run(X, y_encoded, class_names, test_size, run_seed)
            all_results.append(result)

        logger.info(f"\n{'=' * 80}")
        logger.info(f"Summary over {runs} runs (seed base='{seed}')")
        logger.info(f"{'=' * 80}")

        def _fmt(key):
            values = np.array([r[key] for r in all_results])
            return f"{values.mean():.4f}±{values.std():.4f}"

        col_w = 16
        header = f"{'':>20s} {'precision':>{col_w}s} {'recall':>{col_w}s} {'f1-score':>{col_w}s} {'mcc':>{col_w}s}"
        summary_lines = [header, ""]
        for name in class_names:
            summary_lines.append(
                f"{name:>20s} {_fmt(f'precision_{name}'):>{col_w}s} {_fmt(f'recall_{name}'):>{col_w}s} "
                f"{_fmt(f'f1_{name}'):>{col_w}s} {_fmt(f'mcc_{name}'):>{col_w}s}"
            )
        summary_lines.append("")
        for avg_key, avg_label in [("macro_avg", "macro avg"), ("weighted_avg", "weighted avg")]:
            summary_lines.append(
                f"{avg_label:>20s} {_fmt(f'precision_{avg_key}'):>{col_w}s} {_fmt(f'recall_{avg_key}'):>{col_w}s} "
                f"{_fmt(f'f1_{avg_key}'):>{col_w}s} {_fmt('mcc'):>{col_w}s}"
            )
        summary_lines.append(
            f"\n{'accuracy':>20s} {'':>{col_w}s} {'':>{col_w}s} {_fmt('accuracy'):>{col_w}s} {_fmt('mcc'):>{col_w}s}"
        )

        logger.info("\n" + "\n".join(summary_lines))
    else:
        _train_single_run(X, y_encoded, class_names, test_size, seed)

    logger.info("Train complete.")


def _train_single_run(X, y_encoded, class_names, test_size: float, seed: str) -> dict:
    seed_int = resolve_seed(seed)

    logger.info(f"Splitting train/test ({int((1 - test_size) * 100)}%/{int(test_size * 100)}%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=seed_int
    )
    logger.info(f"Train set: {len(X_train)} samples")
    logger.info(f"Test set:  {len(X_test)} samples")

    logger.info("Scaling features with MinMaxScaler...")
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logger.info("Training XGBoost model...")
    model = XGBClassifier(random_state=seed_int, verbosity=1)
    model.fit(X_train_scaled, y_train, verbose=True)

    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall_val = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    mcc = matthews_corrcoef(y_test, y_pred)

    logger.info(f"Accuracy:  {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall:    {recall_val:.4f}")
    logger.info(f"F1 Score:  {f1:.4f}")
    logger.info(f"MCC:       {mcc:.4f}")

    # per-class MCC computed one-vs-rest
    report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0,
                                   output_dict=True)
    per_class_mcc = {}
    for i, name in enumerate(class_names):
        y_test_bin = (y_test == i).astype(int)
        y_pred_bin = (y_pred == i).astype(int)
        per_class_mcc[name] = matthews_corrcoef(y_test_bin, y_pred_bin)

    header = f"{'':>20s} {'precision':>10s} {'recall':>10s} {'f1-score':>10s} {'mcc':>10s} {'support':>10s}"
    lines = [header, ""]
    for name in class_names:
        r = report[name]
        m = per_class_mcc[name]
        lines.append(
            f"{name:>20s} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1-score']:>10.4f} {m:>10.4f} {int(r['support']):>10d}"
        )
    lines.append("")
    for avg_key, avg_label in [("macro avg", "macro avg"), ("weighted avg", "weighted avg")]:
        r = report[avg_key]
        lines.append(
            f"{avg_label:>20s} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1-score']:>10.4f} {mcc:>10.4f} {int(r['support']):>10d}"
        )
    lines.append(
        f"\n{'accuracy':>20s} {'':>10s} {'':>10s} {report['accuracy']:>10.4f} {mcc:>10.4f} {int(report['weighted avg']['support']):>10d}")

    logger.info("\n" + "\n".join(lines))

    result = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall_val,
        "f1": f1,
        "mcc": mcc,
    }
    for name in class_names:
        result[f"mcc_{name}"] = per_class_mcc[name]
        result[f"precision_{name}"] = report[name]["precision"]
        result[f"recall_{name}"] = report[name]["recall"]
        result[f"f1_{name}"] = report[name]["f1-score"]
    for avg_key in ("macro avg", "weighted avg"):
        safe_key = avg_key.replace(" ", "_")
        result[f"precision_{safe_key}"] = report[avg_key]["precision"]
        result[f"recall_{safe_key}"] = report[avg_key]["recall"]
        result[f"f1_{safe_key}"] = report[avg_key]["f1-score"]
    return result


@app.command("all")
def all_stages(
        zip_path: str = typer.Option(..., "--zip-path", "-z",
                                     help="Path to the NOR-TMD zip archive."),
        csv_file: str = typer.Option(DEFAULT_CSV_FILE, "--csv-file", "-c",
                                     help="Name of the CSV file inside the zip."),
        db_dir: str = typer.Option(DEFAULT_DB_DIR, "--db-dir", "-d",
                                   help="Directory for the DuckDB database and output files."),
        db_name: str = typer.Option(DEFAULT_DB_NAME, "--db-name",
                                    help="Name of the DuckDB table/database."),
        chunk_size: int = typer.Option(DEFAULT_CHUNK_SIZE, "--chunk-size",
                                       help="Number of CSV rows per chunk."),
        overwrite: bool = typer.Option(False, "--overwrite", "-o",
                                       help="Overwrite existing database if it exists."),
        window_len_sec: int = typer.Option(10, "--window-len", "-w",
                                           help="Rolling window length in seconds."),
        overlap_sec: int = typer.Option(5, "--overlap",
                                        help="Resample step in seconds."),
        centered: bool = typer.Option(True, "--centered/--no-centered",
                                      help="Whether to center rolling windows."),
        min_trip_sec: int = typer.Option(60, "--min-trip-sec",
                                         help="Minimum trip duration in seconds."),
        max_trip_sec: int = typer.Option(3600, "--max-trip-sec",
                                         help="Maximum trip duration in seconds."),
        test_size: float = typer.Option(0.33, "--test-size",
                                        help="Fraction of data to use for testing."),
        seed: str = typer.Option(DEFAULT_SEED, "--seed", "-s",
                                 help="Random seed for reproducibility."),
        runs: int = typer.Option(None, "--runs", help="Number of training runs with different seeds."),
):
    """Run the full pipeline: parse → preprocess → pivot → train."""
    parse(
        zip_path=zip_path, csv_file=csv_file, db_dir=db_dir,
        db_name=db_name, chunk_size=chunk_size, overwrite=overwrite,
    )
    seg_name = DEFAULT_SEG_DF_NAME
    preprocess(
        db_dir=db_dir, db_name=db_name, output_name=seg_name,
        window_len_sec=window_len_sec, overlap_sec=overlap_sec, centered=centered,
    )
    pivot(
        data_dir=db_dir, input_name=seg_name,
        min_trip_sec=min_trip_sec, max_trip_sec=max_trip_sec,
    )
    train(
        data_dir=db_dir, input_name=DEFAULT_TRAIN_DF_NAME,
        test_size=test_size, seed=seed, runs=runs,
    )


if __name__ == "__main__":
    app()
