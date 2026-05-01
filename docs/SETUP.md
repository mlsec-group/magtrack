## Setup

Our pipeline is using python.
As of May 2026 strongly recommend to use Python 3.13 to match all dependencies.


### Python Setup

For we recomment using `uv` to manage environments and dependencies.

From the repository root in editable mode:

```bash
uv venv
uv pip install -e .
```

### Apptainer Setup
Assuming Apptainer is already installed, build the container image from the repository root.

Run the build command:

```bash
apptainer build python.sif apptainer.def
```

This creates `python.sif` in the repository root.

You can run the commands inside the container with:

```bash
apptainer exec python.sif <command>
```
### Commands

The following commands are available:

- `traintrack-dataset`
- `evaluate-nor-tmd`
- `create-tmd-dataset`
- `evaluate-tmd`
- `create-colocation-dataset`
- `evaluate-coloc-distance`
- `hyperparameter-search`
- `train-ml-model`
- `ml-inference`
- `evaluate-majority-ml`