from pathlib import Path

import pandas as pd
import typer

from magtrack.utils.loader import read_pickle, get_metadata_from_dataset


def _stitch_chunks_by_recording(data: pd.DataFrame) -> pd.DataFrame:
    """Merge chunk rows into recording rows by (segment_id, source_file)."""
    required = {"id", "segment_id", "source_file", "data"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    rows: list[dict] = []
    for (seg, src), grp in data.groupby(["segment_id", "source_file"], sort=False, dropna=False):
        grp = grp.sort_values("id")
        traces = []
        for t in grp["data"].tolist():
            if not isinstance(t, pd.DataFrame):
                raise ValueError(f"Expected DataFrame in 'data' column, got {type(t)!r}")
            traces.append(t)
        rows.append({
            "id": f"{seg}::{src}",
            "segment_id": seg,
            "source_file": src,
            "data": pd.concat(traces, ignore_index=True),
            "n_chunks": len(grp),
        })
    return pd.DataFrame(rows)


def load_and_stitch(pkl_path: Path) -> tuple[pd.DataFrame, dict]:
    """Load a .pkl dataset and stitch chunks into recording-level rows."""
    typer.echo(f"Loading {pkl_path.name} …")
    data = read_pickle(pkl_path)
    required = {"id", "segment_id", "source_file", "data"}
    if not required.issubset(data.columns):
        typer.echo(f"Error: missing columns {required - set(data.columns)}", err=True)
        raise typer.Exit(code=1)
    if len(data) < 2:
        typer.echo("Error: dataset has fewer than 2 rows.", err=True)
        raise typer.Exit(code=1)

    data = _stitch_chunks_by_recording(data)

    valid_n_chunks_mask = data['n_chunks'].apply(lambda x: x == data['n_chunks'].iloc[0])
    if not valid_n_chunks_mask.all():
        typer.echo(f"Error: Not all rows in {pkl_path.name} have the same n_chunks value.", err=True)
        raise typer.Exit(code=1)
    n_chunks = data['n_chunks'].iloc[0]

    meta = get_metadata_from_dataset(pkl_path).iloc[0].to_dict()
    if meta:
        dataset_duration = int(meta.get("duration", 0))
        sr_int = int(float(meta.get("sampling_rate", 0)))
        if dataset_duration > 0 and sr_int > 0:
            target_chunk_samples = int(dataset_duration * sr_int * n_chunks)
            valid_length_mask = data['data'].apply(lambda df: len(df) == target_chunk_samples)
            if not valid_length_mask.all():
                typer.echo(
                    f"Error: Not all data fields in {pkl_path.name} have the expected length "
                    f"of {target_chunk_samples} samples.", err=True)
                raise typer.Exit(code=1)

    n_recordings = len(data)
    n_segment_ids = data['segment_id'].nunique()
    n_ids = data['id'].nunique()
    typer.echo(f"  {n_recordings} recordings after stitching "
               f"({n_segment_ids} segment_ids, {n_ids} unique ids, {n_chunks} chunks each).")
    return data, meta
