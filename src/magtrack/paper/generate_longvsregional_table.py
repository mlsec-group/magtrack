import os
import re

import pandas as pd


def load_mcc_data(directory, name):
    """Load MCC scores, calculate overall stats, and group them by 'first' duration."""
    csv_path = os.path.join(directory, "evaluation_results.results_ml.csv")
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found")
        return None, {}

    df = pd.read_csv(csv_path)

    df = df[df['seed'] != 'seed']
    df['mcc'] = pd.to_numeric(df['mcc'], errors='coerce')
    df = df.dropna(subset=['mcc'])

    # --- 1. Calculate Overall Statistics ---
    best_idx = df['mcc'].idxmax()
    best_row = df.loc[best_idx]

    overall_stats = {
        'name': name,
        'mcc_scores': df['mcc'].values,
        'mean': df['mcc'].mean(),
        'std': df['mcc'].std(),
        'best': df['mcc'].max(),
        'best_config': {
            'seed': best_row['seed'],
            'dataset': best_row['dataset_file'],
            'mcc': best_row['mcc']
        }
    }

    # --- 2. Calculate Grouped Statistics by 'first' parameter ---
    def extract_first(filename):
        match = re.search(r'first(\d+)', str(filename))
        if match:
            return int(match.group(1))
        return None

    df['first_duration'] = df['dataset_file'].apply(extract_first)
    df = df.dropna(subset=['first_duration'])
    df['first_duration'] = df['first_duration'].astype(int)

    grouped_stats = {}
    for duration, group in df.groupby('first_duration'):
        config_stats = group.groupby('dataset_file')['mcc'].agg(['mean', 'std', 'count']).reset_index()

        best_config = config_stats.loc[config_stats['mean'].idxmax()]

        std_val = best_config['std']
        if pd.isna(std_val):
            std_val = 0.0

        grouped_stats[duration] = {
            'mean': best_config['mean'],
            'std': std_val,
            'count': best_config['count'],
            'best_dataset': best_config['dataset_file']  # Optional: tracking what the best config was
        }

    return overall_stats, grouped_stats


def main():
    base_dir = "."

    # Load data
    overall_all, grouped_all = load_mcc_data(os.path.join(base_dir, "results/coloc_ml"), "results")
    overall_long, grouped_long = load_mcc_data(os.path.join(base_dir, "results/coloc_ml/long_distance"),
                                               "results_long_distance")
    overall_reg, grouped_reg = load_mcc_data(os.path.join(base_dir, "results/coloc_ml/regional"), "results_regional")

    # ==========================================
    # Print Overall Summaries
    # ==========================================
    print("=" * 80)
    print("MCC Score Comparison (Overall ML Approach)")
    print("=" * 80)

    for data in [overall_all, overall_long, overall_reg]:
        if not data:
            continue

        print(f"\n{data['name']}:")
        print(f"  Mean \u00B1 Std: {data['mean']:.2f} \u00B1 {data['std']:.2f}")
        print(f"  Best MCC:   {data['best']:.2f}")
        print(f"  N samples:  {len(data['mcc_scores'])}")

        best_cfg = data['best_config']
        dataset = best_cfg['dataset']
        match = re.search(r'first(\d+)_([0-9]+)s_window([0-9]+)_([0-9]+)Hz', dataset)

        if match:
            chunk_size = match.group(1)
            duration = match.group(2)
            window_size = match.group(3)
            hz = match.group(4)
            print(f"  Best config: chunk={chunk_size}, duration={duration}s, window={window_size}, Hz={hz}")
        else:
            print(f"  Best config dataset file: {dataset}")

        print(f"  Seed: {best_cfg['seed']}")

    # ==========================================
    # Generate LaTeX Table
    # ==========================================
    durations = [60, 300, 600, 900]
    modes = [
        ('All', grouped_all),
        ('Long distance', grouped_long),
        ('Regional', grouped_reg)
    ]

    # 1. Generate Distance-based rows
    words = {60: 'Sixty', 300: 'ThreeHundred', 600: 'SixHundred', 900: 'NineHundred'}
    dist_macros = {
        'All': r'\ColocDistanceDTWROneBestMCCFirst',
        'Long distance': r'\ColocDistanceLongDistanceDTWROneBestMCCFirst',
        'Regional': r'\ColocDistanceRegionalDTWROneBestMCCFirst'
    }

    dist_rows = []
    for i, d in enumerate(durations):
        prefix = r"\textbf{Distance-based}" if i == 0 else ""
        row_str = f"    {prefix:<22} & {d:<3} "

        for mode_name, _ in modes:
            # Regional is missing for 600 and 900 based on the original table
            if mode_name == 'Regional' and d in [600, 900]:
                row_str += "& - "
            else:
                row_str += f"& {dist_macros[mode_name]}{words[d]} "

        row_str += r"\\"
        dist_rows.append(row_str)

    # 2. Generate ML-based rows dynamically
    ml_rows = []
    for i, d in enumerate(durations):
        prefix = r"\textbf{Machine Learning}" if i == 0 else ""
        row_str = f"    {prefix:<22} & {d:<3} "

        best_mean = -1
        for _, data in modes:
            if data and d in data and data[d]['mean'] > best_mean:
                best_mean = data[d]['mean']

        for mode_name, data in modes:
            if data and d in data:
                mean = data[d]['mean']
                std = data[d]['std']

                # Highlight bold if it's the best performing train type for this duration
                if abs(mean - best_mean) < 1e-5:
                    row_str += f"& \\textbf{{{mean:.2f}}} $\\pm$ \\textbf{{{std:.2f}}} "
                else:
                    row_str += f"& {mean:.2f} $\\pm$ {std:.2f} "
            else:
                row_str += "& - "

        row_str += r"\\"
        ml_rows.append(row_str)

    # 3. Assemble the single-column LaTeX table
    latex_table = f"""\\begin{{table}}[htbp]
  \\centering
  \\caption{{Colocation inference performance (MCC score) of the distance-based and machine learning approaches across different train types and initial time windows. Best ML performances per duration are highlighted in bold.}}
  \\label{{tab:coloc_traintype_combined}}
  \\resizebox{{\\columnwidth}}{{!}}{{
  \\begin{{tabular}}{{ll ccc}}
    \\toprule
    \\multirow{{2}}{{*}}{{\\textbf{{Approach}}}} & \\multirow{{2}}{{*}}{{\\textbf{{First (s)}}}} & \\multicolumn{{3}}{{c}}{{\\textbf{{Train Type}}}} \\\\
    \\cmidrule(lr){{3-5}}
    & & \\textbf{{All}} & \\textbf{{Long distance}} & \\textbf{{Regional}} \\\\
    \\midrule
{chr(10).join(dist_rows)}
    \\midrule
{chr(10).join(ml_rows)}
    \\bottomrule
  \\end{{tabular}}
  }}
\\end{{table}}"""

    print("\n\n" + "=" * 80)
    print("Generated Single-Column LaTeX Table:")
    print("=" * 80 + "\n")
    print(latex_table)

    # Save to file
    output_path = os.path.join("results/paper/tables", "mcc_train_types.tex")
    with open(output_path, 'w') as f:
        f.write(latex_table)
    print(f"\n% Table saved to: {output_path}")


if __name__ == "__main__":
    main()
