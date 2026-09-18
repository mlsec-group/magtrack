import hashlib
import os
import random
from pathlib import Path

import pandas as pd
import typer
from tqdm import tqdm

from magtrack.utils.coloc_builder import build_rows, canonicalize_dtype_identity, warm_trace_cache

app = typer.Typer(help="Create several colocation datasets in a single pass")

ALLOWED_TRAIN_TYPES = {"long_distance", "regional"}


def _parse_dataset_spec(spec: str) -> tuple[str, list[str], str]:
    """Parse ``name:type[,type...][:output_dir]`` into ``(name, filter_train_type, output_dir)``."""
    parts = spec.split(':', 2)
    name = parts[0].strip()
    if not name:
        raise typer.BadParameter(f"Invalid dataset spec {spec!r}: missing dataset name")
    types = parts[1].strip() if len(parts) > 1 else ''
    train_types = [t.strip() for t in types.split(',') if t.strip()] if types else \
        ['long_distance', 'regional']
    for train_type in train_types:
        if train_type not in ALLOWED_TRAIN_TYPES:
            raise typer.BadParameter(
                f"filter_train_type values must be one of {sorted(ALLOWED_TRAIN_TYPES)}")
    output_dir = parts[2].strip() if len(parts) > 2 and parts[2].strip() else ''
    return name, train_types, output_dir


@app.command("create")
def create(
        dataset_path: Path = typer.Option(..., help="Path to traintrack dataset (yaml)"),
        output_path: Path = typer.Option(..., help="Directory where the datasets will be stored"),
        duration: list[int] = typer.Option(..., help="Durations in seconds for each extracted window"),
        sampling_rate: list[float] = typer.Option(..., help="Expected sampling rates in Hz"),
        rolling_window: list[int] = typer.Option(..., help="Rolling window sizes"),
        trainride_start_seconds: list[int] = typer.Option(...,
                                                          help="Seconds to use from the start of trainride (0 = all)"),
        dataset_spec: list[str] = typer.Option(
            ["all:long_distance,regional", "long_distance:long_distance", "regional:regional"],
            help="Dataset to emit as 'name:train_type[,train_type][:output_dir]'. Repeatable."),
        extraction_function_name: str = typer.Option("filter_notch_highpass",
                                                     help="Name of extraction function in utils.filters"),
        sensor_type: str = typer.Option("Magnetometer", help="Sensor type to extract from zipfiles"),
        start_delay: int = typer.Option(2, help="Seconds to delay from recording start"),
        end_buffer: int = typer.Option(2, help="Seconds to trim from the end of each recording"),
        min_sample_frequency: float = typer.Option(90.0, help="Minimum sample frequency in Hz for samples"),
        normalize: str = typer.Option("trace",
                                      help="Normalization mode: 'none', 'chunk' (normalize each chunk), or 'trace' (normalize whole trace before chunking)"),
        seed=typer.Option('magtrack', help="Random seed"),
        overwrite: bool = typer.Option(False, help="Overwrite existing output files"),
        compression: str = typer.Option("zstd", help="Compression codec for pickle IO."),
        trace_cache_dir: Path = typer.Option(None,
                                             help="Directory for the decoded/filtered trace cache shared between runs"),
        skip_short_windows: bool = typer.Option(True,
                                                help="Skip combinations where duration * sampling_rate <= rolling_window"),
        warm_trace_cache_only: bool = typer.Option(False,
                                                   help="Only fill --trace-cache-dir with the decoded traces and exit"),
):
    """Create every dataset of the requested parameter grid in one process.

    This produces exactly the same files as running ``create-colocation-dataset``
    once per parameter combination, but decodes each recording only once and
    shares chunking and sampling-rate checks between the combinations.
    """
    random.seed(seed)

    try:
        import magtrack.utils.filters as _filters
        if extraction_function_name and hasattr(_filters, extraction_function_name):
            EXTRACTION_FUNCTION = getattr(_filters, extraction_function_name)
        else:
            raise AttributeError("Extraction function not found")
    except Exception:
        typer.echo(f"Failed to import extraction function {extraction_function_name}.", err=True)
        raise typer.Exit(1)

    allowed_norms = {"none", "chunk", "trace"}
    if normalize not in allowed_norms:
        raise typer.BadParameter(f"normalize must be one of {sorted(allowed_norms)}")

    specs = [_parse_dataset_spec(spec) for spec in dataset_spec]
    train_types = sorted({t for _, types, _ in specs for t in types})

    if warm_trace_cache_only:
        if trace_cache_dir is None:
            raise typer.BadParameter("--warm-trace-cache-only requires --trace-cache-dir")
        cached, total = warm_trace_cache(dataset_path, train_types, EXTRACTION_FUNCTION,
                                         extraction_function_name, sensor_type, trace_cache_dir)
        typer.echo(f"Cached {cached}/{total} traces in {trace_cache_dir}")
        return

    durations = list(dict.fromkeys(duration))
    sampling_rates = list(dict.fromkeys(sampling_rate))
    rolling_windows = list(dict.fromkeys(rolling_window))
    starts = list(dict.fromkeys(trainride_start_seconds))

    rows_by_combination, types_by_combination = build_rows(
        dataset_path=dataset_path,
        durations=durations,
        sampling_rates=sampling_rates,
        rolling_windows=rolling_windows,
        trainride_start_seconds=starts,
        train_types=train_types,
        extraction_function=EXTRACTION_FUNCTION,
        extraction_function_name=extraction_function_name,
        sensor_type=sensor_type,
        start_delay=start_delay,
        end_buffer=end_buffer,
        min_sample_frequency=min_sample_frequency,
        normalize=normalize,
        trace_cache_dir=trace_cache_dir,
        skip_short_windows=skip_short_windows,
    )

    spec_dirs = {}
    for dataset_name, _, output_dir in specs:
        target_dir = Path(output_dir) if output_dir else Path(output_path)
        target_dir.mkdir(parents=True, exist_ok=True)
        spec_dirs[dataset_name] = target_dir

    written = 0
    for combination in sorted(rows_by_combination):
        rolling_window_value, duration_value, start_seconds, sampling_rate_value = combination
        rows = rows_by_combination.pop(combination)
        row_types = types_by_combination.pop(combination)
        for dataset_name, filter_train_type, _ in specs:
            written += _write_dataset(
                rows=rows,
                row_types=row_types,
                dataset_name=dataset_name,
                filter_train_type=filter_train_type,
                output_path=spec_dirs[dataset_name],
                duration=duration_value,
                trainride_start_seconds=start_seconds,
                sampling_rate=sampling_rate_value,
                rolling_window=rolling_window_value,
                extraction_function_name=extraction_function_name,
                sensor_type=sensor_type,
                start_delay=start_delay,
                end_buffer=end_buffer,
                min_sample_frequency=min_sample_frequency,
                normalize=normalize,
                seed=seed,
                overwrite=overwrite,
                compression=compression,
            )
        del rows, row_types

    typer.echo(f"Wrote {written} dataset(s)")


def _write_dataset(rows, row_types, dataset_name, filter_train_type, output_path, duration,
                   trainride_start_seconds, sampling_rate, rolling_window, extraction_function_name,
                   sensor_type, start_delay, end_buffer, min_sample_frequency, normalize, seed,
                   overwrite, compression) -> int:
    out_filename = (f"{dataset_name}_coloc_first{trainride_start_seconds}_{duration}s_"
                    f"window{rolling_window}_{int(sampling_rate)}Hz.pkl")
    out_file = os.path.join(output_path, out_filename)
    if os.path.exists(out_file) and not overwrite:
        print(f"Output file {out_file} already exists. Use --overwrite to overwrite.")
        return 0

    wanted = set(filter_train_type)
    selected = [row for row, train_type in zip(rows, row_types) if train_type in wanted]
    if not selected:
        typer.echo(f"No valid colocated samples found for {out_filename}. Skipping.")
        return 0

    coloc_df = pd.DataFrame(selected)
    old_length = len(coloc_df)

    target_chunk_samples = int(duration * sampling_rate)
    valid_length_mask = coloc_df['data'].apply(lambda df: len(df) == target_chunk_samples)
    if not valid_length_mask.all():
        invalid_count = (~valid_length_mask).sum()
        tqdm.write(f"Dropping {invalid_count} sample(s) due to invalid chunk length "
                   f"(expected {target_chunk_samples} samples per chunk).")
        coloc_df = coloc_df[valid_length_mask].reset_index(drop=True)

    if trainride_start_seconds > 0:
        target_recording_length = trainride_start_seconds / duration
        segment_length_mask = coloc_df.groupby(['source_file', 'segment_id'])['data'].transform(
            lambda group: len(group) == target_recording_length)
        if not segment_length_mask.all():
            invalid_count = (~segment_length_mask).sum()
            tqdm.write(f"Dropping {invalid_count} sample(s) due to invalid segment length "
                       f"(expected {target_recording_length} samples per segment for "
                       f"trainride_start_seconds > 0).")
            coloc_df = coloc_df[segment_length_mask].reset_index(drop=True)
    else:
        id_counts = coloc_df['id'].value_counts()
        ids_to_keep = id_counts[id_counts >= 2].index
        coloc_df = coloc_df[coloc_df['id'].isin(ids_to_keep)].reset_index(drop=True)
        dropped = old_length - len(coloc_df)
        if dropped > 0:
            tqdm.write(f"Dropped {dropped} item(s) because their id appeared < 2 times.")

    if len(coloc_df) == 0:
        typer.echo(f"No valid colocated samples for {out_filename}. Skipping.")
        return 0

    canonicalize_dtype_identity(coloc_df)
    coloc_df.to_pickle(out_file, compression=compression)
    meta_df = pd.DataFrame({
        'dataset_name': [dataset_name],
        'dataset_md5': [hashlib.md5(open(out_file, 'rb').read()).hexdigest()],
        'extraction_function_name': [extraction_function_name],
        'duration': [duration],
        'trainride_start_seconds': [trainride_start_seconds],
        'sensor_type': [sensor_type],
        'filter_train_type': [filter_train_type],
        'start_delay': [start_delay],
        'end_buffer': [end_buffer],
        'min_sample_frequency': [min_sample_frequency],
        'rolling_window': [rolling_window],
        'normalize': [normalize],
        'sampling_rate': [sampling_rate],
        'seed': [seed],
        'num_samples': [len(coloc_df)],
    })
    meta_df.to_csv(os.path.join(output_path, f"{os.path.splitext(out_filename)[0]}.meta.csv"), index=False)

    typer.echo(f"Saved coloc dataset with {len(coloc_df)} samples to {out_file}")
    return 1


if __name__ == "__main__":
    app()
