import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
import glob

def get_latest_summary(experiment_dir):
    """Get the most recent summary file from an experiment directory."""
    summary_files = glob.glob(os.path.join(experiment_dir, 'summary_*.csv'))
    if not summary_files:
        return None
    # Sort by timestamp in filename
    latest_file = max(summary_files, key=lambda x: x.split('summary_')[1])
    return latest_file

def load_experiment_data(results_dir):
    """Load experiment data from results directory."""
    print(f"Loading data from directory: {results_dir}")
    data = {
        'Experiment': [],
        'RetrievalPrecision_mean': [],
        'RetrievalPrecision_std': [],
        'AnswerSimilarity_mean': [],
        'AnswerSimilarity_std': []
    }
    
    # Map from directory names to display names
    name_mapping = {
        'no_rerank': 'No Rerank',
        'cohere_rerank': 'Cohere Rerank',
        'llm_rerank': 'LLM Rerank',
        'one-turn_rebel_rerank': 'One-Turn REBEL Rerank',
        'two-turn_relevance-only_rebel_rerank': 'Two-Turn Relevance-Only REBEL Rerank',
        'two-turn_rebel_rerank': 'Two-Turn REBEL Rerank',
        'hyde': 'HyDE',
    }
    
    # Iterate through experiment directories
    exp_dirs = glob.glob(os.path.join(results_dir, '*'))
    print(f"Found experiment directories: {exp_dirs}")
    
    for exp_dir in exp_dirs:
        if not os.path.isdir(exp_dir):
            print(f"Skipping non-directory: {exp_dir}")
            continue
            
        exp_name = os.path.basename(exp_dir)
        print(f"\nProcessing experiment: {exp_name}")
        
        if exp_name not in name_mapping:
            print(f"Warning: Unknown experiment directory {exp_name}")
            continue
            
        summary_file = get_latest_summary(exp_dir)
        if not summary_file:
            print(f"Warning: No summary file found for {exp_name}")
            continue
            
        print(f"Loading summary file: {summary_file}")
        try:
            df = pd.read_csv(summary_file)
            print(f"DataFrame columns: {df.columns}")
            print(f"DataFrame content:\n{df}")
            
            # Get the actual values from row 2 (index 2)
            rp_mean = float(df.iloc[2]['RetrievalPrecision'])
            rp_std = float(df.iloc[2]['RetrievalPrecision.1'])
            as_mean = float(df.iloc[2]['AnswerSimilarity'])
            as_std = float(df.iloc[2]['AnswerSimilarity.1'])
            
            data['Experiment'].append(name_mapping[exp_name])
            data['RetrievalPrecision_mean'].append(rp_mean)
            data['RetrievalPrecision_std'].append(rp_std)
            data['AnswerSimilarity_mean'].append(as_mean)
            data['AnswerSimilarity_std'].append(as_std)
            print(f"Successfully processed {exp_name}")
            
        except Exception as e:
            print(f"Error loading data from {summary_file}: {str(e)}")
            continue
    
    print("\nFinal loaded data:")
    for key, value in data.items():
        print(f"{key}: {value}")

    return data

def figure_1(output_path, results_dir):
    # Load data from experiment results
    data = load_experiment_data(results_dir)
    if not data['Experiment']:
        raise ValueError("No experiment data found in results directory")

    plt.rcParams.update({'font.size': 20})  # Default is usually 10

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))

    # Separate multi-criteria, relevance-only, and no-rerank methods
    multi_criteria_indices = [i for i, exp in enumerate(data['Experiment']) 
                            if exp in ['One-Turn REBEL Rerank', 'Two-Turn REBEL Rerank']]
    no_rerank_idx = data['Experiment'].index('No Rerank')
    relevance_only_indices = [i for i in range(len(data['Experiment'])) 
                            if i not in multi_criteria_indices and i != no_rerank_idx]

    # Plot relevance-only methods
    relevance_scatter = ax.errorbar([data['RetrievalPrecision_mean'][i] for i in relevance_only_indices],
              [data['AnswerSimilarity_mean'][i] for i in relevance_only_indices],
              xerr=[data['RetrievalPrecision_std'][i] for i in relevance_only_indices],
              yerr=[data['AnswerSimilarity_std'][i] for i in relevance_only_indices],
              fmt='o', markersize=8, capsize=5, color='blue', label='Relevance-Only Methods')

    # Plot multi-criteria methods
    multi_scatter = ax.errorbar([data['RetrievalPrecision_mean'][i] for i in multi_criteria_indices],
              [data['AnswerSimilarity_mean'][i] for i in multi_criteria_indices],
              xerr=[data['RetrievalPrecision_std'][i] for i in multi_criteria_indices],
              yerr=[data['AnswerSimilarity_std'][i] for i in multi_criteria_indices],
              fmt='o', markersize=8, capsize=5, color='red', label='Our Multi-Criteria Methods')
              
    # Plot No Rerank in black
    no_rerank_scatter = ax.errorbar([data['RetrievalPrecision_mean'][no_rerank_idx]],
              [data['AnswerSimilarity_mean'][no_rerank_idx]],
              xerr=[data['RetrievalPrecision_std'][no_rerank_idx]],
              yerr=[data['AnswerSimilarity_std'][no_rerank_idx]],
              fmt='o', markersize=8, capsize=5, color='black', label='No Rerank')

    # Add labels for each point
    for i, txt in enumerate(data['Experiment']):
        if txt == 'No Rerank':
            ax.annotate(txt, (data['RetrievalPrecision_mean'][i], data['AnswerSimilarity_mean'][i]),
                      xytext=(10, -8), textcoords='offset points')
        else:
            ax.annotate(txt, (data['RetrievalPrecision_mean'][i], data['AnswerSimilarity_mean'][i]),
                      xytext=(10, 10), textcoords='offset points')

    # Get indices for the multi-criteria line
    multi_indices = [no_rerank_idx] + [
        data['Experiment'].index(exp) for exp in ['One-Turn REBEL Rerank', 'Two-Turn REBEL Rerank']
    ]

    # Plot multi-criteria line of best fit
    x_multi = np.array([data['RetrievalPrecision_mean'][i] for i in multi_indices])
    y_multi = np.array([data['AnswerSimilarity_mean'][i] for i in multi_indices])
    
    # Calculate line of best fit
    z_multi = np.polyfit(x_multi, y_multi, 1)
    p_multi = np.poly1d(z_multi)
    
    # Create smooth line
    x_multi_line = np.linspace(min(x_multi), max(x_multi), 100)
    y_multi_line = p_multi(x_multi_line)
    
    line_multi = ax.plot(x_multi_line, y_multi_line, '--', color='red', alpha=0.5, 
                        label='Multi-Criteria Surpassing\nInformation Bottleneck')

    # Get indices for the relevance-only line
    relevance_indices = [no_rerank_idx] + [
        data['Experiment'].index(exp) for exp in [
            'HyDE', 'Two-Turn Relevance-Only REBEL Rerank', 'Cohere Rerank', 'LLM Rerank'
        ]
    ]

    # Plot relevance-only line of best fit
    x_relevance = np.array([data['RetrievalPrecision_mean'][i] for i in relevance_indices])
    y_relevance = np.array([data['AnswerSimilarity_mean'][i] for i in relevance_indices])
    
    # Calculate line of best fit
    z_relevance = np.polyfit(x_relevance, y_relevance, 1)
    p_relevance = np.poly1d(z_relevance)
    
    # Create smooth line
    x_relevance_line = np.linspace(min(x_relevance), max(x_relevance), 100)
    y_relevance_line = p_relevance(x_relevance_line)
    
    line_relevance = ax.plot(x_relevance_line, y_relevance_line, '--', color='blue', alpha=0.5,
                           label='Relevance-Only\nInformation Bottleneck')

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Customize the plot
    ax.set_xlabel('Retrieval Precision')
    ax.set_ylabel('Answer Similarity')

    # Add legend with custom order
    handles = [relevance_scatter, multi_scatter, line_relevance[0], line_multi[0]]
    labels = ['Relevance-Only Methods', 'Our Multi-Criteria Methods', 
              'Relevance-Only\nInformation Bottleneck', 'Multi-Criteria Surpassing\nInformation Bottleneck']
    ax.legend(handles, labels, loc='lower left', bbox_to_anchor=(0.02, 0.02))

    # Add some padding to the axes
    plt.margins(0.1)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'figure_1.png'))

    # Save the data used to generate the plot
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(output_path, 'figure_1_data.csv'), index=False)

def figure_2(inference_data, output_path, results_dir):
    # Load data from experiment results
    data = load_experiment_data(results_dir)
    if not data['Experiment']:
        raise ValueError("No experiment data found in results directory")

    # Create mapping between different naming conventions
    name_mapping = {
        k:k for k in inference_data["Method"]
    }
    name_mapping['Two-Turn Relevance-Only REBEL Rerank'] = 'Two-Turn Relevance-Only\nREBEL Rerank'
    name_mapping['LLM Rerank'] = 'LLM\nRerank'
    name_mapping['Cohere Rerank'] = 'Cohere\nRerank'

    # Calculate combined metric and prepare data for plotting
    metrics = []
    metrics_std = []
    speeds = []
    labels = []
    method_types = []  # To track if method is relevance-only or multi-criteria

    for method, time, length in zip(inference_data['Method'], inference_data['Total Time (s)'], inference_data['Response Length']):
        
        idx = data['Experiment'].index(method)
        standard_name = name_mapping[method]

        combined_metric = data['RetrievalPrecision_mean'][idx] * data['AnswerSimilarity_mean'][idx]
        def calculate_metric_std(mx, my, sx, sy):
            return np.sqrt(sx**2 * sy**2 + mx**2 * sy**2 + my**2 * sx**2)
        combined_metric_std = calculate_metric_std(data['RetrievalPrecision_mean'][idx], data['AnswerSimilarity_mean'][idx], data['RetrievalPrecision_std'][idx], data['AnswerSimilarity_std'][idx])
        speed = length/time

        metrics.append(combined_metric)
        metrics_std.append(combined_metric_std)
        speeds.append(speed)
        labels.append(standard_name)

        # Determine if method is relevance-only or multi-criteria
        if standard_name in ['Two-Turn Relevance-Only\nREBEL Rerank', 'LLM\nRerank', 'Cohere\nRerank', 'HyDE', 'No Rerank']:
            method_types.append('relevance')
        else:
            method_types.append('multi')

    plt.rcParams.update({'font.size': 20})

    fig, ax = plt.subplots(figsize=(12, 8))

    print(metrics)
    metrics_std = [float(s) for s in metrics_std]
    print(metrics_std)
    # Plot points with different colors based on method type
    for speed, metric, metric_std, method_type, label in zip(speeds, metrics, metrics_std, method_types, labels):
        if label == 'No Rerank':
            color = 'black'
        else:
            color = 'red' if method_type == 'multi' else 'blue'
        ax.errorbar(speed, metric, yerr=metric_std, c=color, fmt='o', markersize=10, capsize=5)

    # Add labels for points with adjusted positions for HyDE and LLM Rerank
    for i, txt in enumerate(labels):
        if txt == 'HyDE':
            ax.annotate(txt, (speeds[i], metrics[i]), xytext=(10, -15), textcoords='offset points')
        elif txt == 'LLM Rerank':
            ax.annotate(txt, (speeds[i], metrics[i]), xytext=(10, -15), textcoords='offset points')
        else:
            ax.annotate(txt, (speeds[i], metrics[i]), xytext=(10, 10), textcoords='offset points')

    # Get indices for multi-criteria line
    multi_indices = [labels.index(label) for label in ['No Rerank', 'One-Turn REBEL Rerank', 'Two-Turn REBEL Rerank']]
    multi_indices = sorted(multi_indices, key=lambda i: speeds[i])
    
    # Create points for the line in order of speed
    x_line = [speeds[i] for i in multi_indices]
    y_line = [metrics[i] for i in multi_indices]
    y_std = [metrics_std[i] for i in multi_indices]
    
    # Create interpolation points for smooth shading
    x_interp = np.linspace(min(x_line), max(x_line), 100)
    y_interp = np.interp(x_interp, x_line, y_line)
    std_interp = np.interp(x_interp, x_line, y_std)
    
    # Plot line and shading
    ax.plot(x_interp, y_interp, '--', color='red', alpha=0.5,
            label='Multi-Criteria Surpassing\nInformation Bottleneck')
    ax.fill_between(x_interp,
                    y_interp - std_interp,
                    y_interp + std_interp,
                    color='red', alpha=0.1)

    # Customize the plot
    ax.set_xlabel('Generated Output Characters Per Second')
    ax.set_ylabel('System Quality\n(Answer Similarity × Retrieval Precision)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Create custom legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Relevance-Only Methods'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Our Multi-Criteria Methods'),
        Line2D([0], [0], linestyle='--', color='red', alpha=0.5, label='Multi-Criteria\nSystem Quality/Speed\nScaling'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.margins(0.1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'figure_2.png'))

    # Save the data used to generate the plot
    inference_data.to_csv(os.path.join(output_path, 'figure_2_data.csv'), index=False)

def parse_args():
    parser = argparse.ArgumentParser(description='Generate figures from experiment results')
    parser.add_argument('--figure', type=int, choices=[1, 2], help='Which figure to generate (1 or 2)')
    parser.add_argument('--output-path', type=str, default='figures', help='Path to save the output figures and data')
    parser.add_argument('--results-dir', type=str, required=True, help='Directory containing experiment results')
    parser.add_argument('--inference-times', type=str, help='Path to inference times CSV file (required for figure 2)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)
    
    if args.figure == 1:
        figure_1(args.output_path, args.results_dir)
        print(f"Figure 1 and its data have been saved to {args.output_path}")
    
    elif args.figure == 2:
        if not args.inference_times:
            raise ValueError("--inference-times argument is required for figure 2")
        if not os.path.exists(args.inference_times):
            raise FileNotFoundError(f"Inference times file not found: {args.inference_times}")
            
        inference_times = pd.read_csv(args.inference_times)
        figure_2(inference_times, args.output_path, args.results_dir)
        print(f"Figure 2 and its data have been saved to {args.output_path}")
    
    else:
        print("Please specify which figure to generate using --figure argument")

if __name__ == "__main__":
    main()