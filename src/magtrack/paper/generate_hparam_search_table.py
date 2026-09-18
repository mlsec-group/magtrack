"""LaTeX table of the best hyperparameter-search trials.

One column per trial, one row per hyperparameter.
"""
from pathlib import Path

import optuna
import typer

app = typer.Typer(help="Generate the Optuna hyperparameter search LaTeX table.")

# Row label column width, so the '&' separators line up in the .tex source.
_LABEL_WIDTH = 14
_DASH = "--"


def _row(label: str, cells: list[str], width: int = 0) -> str:
    """One tabular row: padded label, cells, LaTeX row terminator."""
    body = " & ".join(c.ljust(width) for c in cells)
    return f"{label.ljust(_LABEL_WIDTH)} & {body} \\\\"


@app.command()
def main(
        db_path: Path = typer.Option(
            Path("results/coloc_ml/hyperparam_search.db"), "--db",
            help="Optuna SQLite database written by the hyperparameter search.",
        ),
        study_name: str = typer.Option(
            "colocoation_cnn_param_search", "--study-name",
            help="Optuna study name inside the database.",
        ),
        n_trials: int = typer.Option(
            5, "--n-trials", help="How many of the Pareto-front trials to list.",
        ),
        max_epochs: int = typer.Option(
            200, "--max-epochs", help="Epoch count named in the caption.",
        ),
) -> None:
    """Print the table to stdout, for redirection into a .tex file."""
    if not db_path.is_file():
        typer.echo(f"Optuna database not found: {db_path}", err=True)
        raise typer.Exit(code=1)

    study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{db_path}")
    trials = study.best_trials[-n_trials:]
    if not trials:
        typer.echo("The study has no trials on the Pareto front.", err=True)
        raise typer.Exit(code=1)

    params = [t.params for t in trials]
    # As many FC width rows as the widest network needs; shorter ones get a dash.
    max_fc_layers = max((p.get("fc_layers", 0) for p in params), default=0)

    def cells(key, fmt="{}"):
        return [fmt.format(p[key]) if key in p else _DASH for p in params]

    def fc_cells(i):
        out = []
        for p in params:
            dim = p.get(f"fc_dim_{i}")
            out.append(str(dim) if dim is not None and i < p.get("fc_layers", 0) else _DASH)
        return out

    # The FC width rows share one width so their dashes line up as a block.
    fc_rows = [fc_cells(i) for i in range(max_fc_layers)]
    fc_width = max((len(c) for row in fc_rows for c in row), default=0)

    print("\\begin{table}[htbp]")
    print("\\centering")
    print(f"\\caption{{Best hyperparameters from the Optuna search after {max_epochs} epochs. "
          "A dash marks layers that are not present.}")
    print("\\label{tab:optuna_hparams}")
    print("\\small")
    print("\\setlength{\\tabcolsep}{5pt}")
    print(f"\\begin{{tabular}}{{l{'c' * len(trials)}}}")
    print("\\toprule")
    print("Trial & " + " & ".join(str(t.number) for t in trials) + " \\\\")
    print("\\midrule")

    print(_row("MCC", [f"{t.values[0]:.3f}" if t.values else _DASH for t in trials]))
    print("\\addlinespace")

    print(_row("Conv.\\ layers", cells("num_conv_layers")))
    print(_row("Base channels", cells("base_channels")))
    print(_row("Kernel size", cells("kernel_size")))
    print(_row("Stride", cells("stride")))
    print(_row("Pool kernel", cells("pool_kernel")))
    print("\\addlinespace")

    print(_row("FC layers", cells("fc_layers")))
    for i, row in enumerate(fc_rows, start=1):
        print(_row(f"FC width {i}", row, fc_width))
    print("\\addlinespace")

    print(_row("Dropout", cells("fc_dropout", "{:.3f}")))
    print(_row("Learning rate", cells("lr", "{:.3f}")))
    print(_row("Batch size", cells("batch_size")))

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


if __name__ == "__main__":
    app()
