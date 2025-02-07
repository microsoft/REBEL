import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os

def figure_1(output_path):
    # Data
    data = {
        'Experiment': ['Cohere Rerank', 'Two-Turn REBEL\nRerank', 'LLM Rerank', 'One-Turn REBEL Rerank', 'No Rerank', 'Two-Turn Relevance-Only REBEL Rerank', 'HyDE'],
        'RetrievalPrecision_mean': [0.8846, 0.9557, 0.9278, 0.8860, 0.8598, 0.9011, 0.8735],
        'RetrievalPrecision_std': [0.0003, 0.0055, 0.0072, 0.0063, 0.0000, 0.0057, 0.0055],
        'AnswerSimilarity_mean': [4.0907, 4.3439, 3.8916, 4.2981, 4.2243, 4.1505, 4.2523],
        'AnswerSimilarity_std': [0.0279, 0.0424, 0.0776, 0.0450, 0.0216, 0.0277, 0.0406],
    }

    plt.rcParams.update({'font.size': 20})  # Default is usually 10

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))

    # Separate multi-criteria and relevance-only methods
    multi_criteria_indices = [data['Experiment'].index(x) for x in ['One-Turn REBEL Rerank', 'Two-Turn REBEL\nRerank']]
    relevance_only_indices = [i for i in range(len(data['Experiment'])) if i not in multi_criteria_indices]

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

    # Add labels for each point
    for i, txt in enumerate(data['Experiment']):
        if txt == 'No Rerank':
            ax.annotate(txt, (data['RetrievalPrecision_mean'][i], data['AnswerSimilarity_mean'][i]),
                      xytext=(10, -8), textcoords='offset points')
        else:
            ax.annotate(txt, (data['RetrievalPrecision_mean'][i], data['AnswerSimilarity_mean'][i]),
                      xytext=(10, 10), textcoords='offset points')

    # Plot first line: No Rerank to Cohere Rerank
    no_rerank_idx = data['Experiment'].index('No Rerank')
    cohere_idx = data['Experiment'].index('Cohere Rerank')
    llm_idx = data['Experiment'].index('LLM Rerank')

    # Second line: Cohere Rerank to LLM Rerank
    x_line2 = [data['RetrievalPrecision_mean'][cohere_idx], data['RetrievalPrecision_mean'][llm_idx]]
    y_line2 = [data['AnswerSimilarity_mean'][cohere_idx], data['AnswerSimilarity_mean'][llm_idx]]
    line2 = ax.plot(x_line2, y_line2, '--', color='#1f77b4', alpha=0.5, label='One-Turn Relevance-Only\nTradeoff Curve')

    # Add new line between HyDE and Dynamic Relevance-Only REBEL Rerank
    hyde_idx = data['Experiment'].index('HyDE')
    dynamic_idx = data['Experiment'].index('Two-Turn Relevance-Only REBEL Rerank')

    x_line3 = [data['RetrievalPrecision_mean'][hyde_idx], data['RetrievalPrecision_mean'][dynamic_idx]]
    y_line3 = [data['AnswerSimilarity_mean'][hyde_idx], data['AnswerSimilarity_mean'][dynamic_idx]]
    line3 = ax.plot(x_line3, y_line3, '--', color='#9ecae1', alpha=0.5,
                    label='Two-Turn Relevance-Only\nTradeoff Curve')

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Customize the plot
    ax.set_xlabel('Retrieval Precision')
    ax.set_ylabel('Answer Similarity')

    # Add legend with custom order
    handles = [relevance_scatter, multi_scatter, line2[0], line3[0]]
    labels = ['Relevance-Only Methods', 'Our Multi-Criteria Methods',
              'One-Turn Relevance-Only\nTradeoff Curve', 'Two-Turn Relevance-Only\nTradeoff Curve']
    ax.legend(handles, labels, loc='lower left', bbox_to_anchor=(0.02, 0.02))

    # Add some padding to the axes
    plt.margins(0.1)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'figure_1.png'))

    # Save the data used to generate the plot
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(output_path, 'figure_1_data.csv'), index=False)

def figure_2(inference_data, output_path):
    data = {
        'Experiment': ['Cohere Rerank', 'Two-Turn REBEL Rerank', 'LLM Rerank', 'One-Turn REBEL Rerank', 'No Rerank', 'Two-Turn\nRelevance-Only\nREBEL Rerank', 'HyDE'],
        'RetrievalPrecision_mean': [0.8846, 0.9557, 0.9278, 0.8860, 0.8598, 0.9011, 0.8735],
        'RetrievalPrecision_std': [0.0003, 0.0055, 0.0072, 0.0063, 0.0000, 0.0057, 0.0055],
        'AnswerSimilarity_mean': [4.0907, 4.3439, 3.8916, 4.2981, 4.2243, 4.1505, 4.2523],
        'AnswerSimilarity_std': [0.0279, 0.0424, 0.0776, 0.0450, 0.0216, 0.0277, 0.0406],
    }

    # inference_data = {
    #     'Method': ['Cohere rerank', 'Two-Turn REBEL Rerank', 'LLM Rerank', '1-Turn REBEL Rerank', 'No Rerank', '2-Turn Relevance-Only REBEL Rerank', 'HyDE'],
    #     'Time': [1.9880, 17.0301, 2.8455, 4.2466, 1.9734, 8.5820, 7.8250],
    #     'ResponseLength': [135, 213, 213, 250, 226, 206, 213]
    # }

    # Create mapping between different naming conventions
    name_mapping = {
        k:k for k in inference_data["Method"]
    }
    name_mapping['Two-Turn Relevance-Only REBEL Rerank'] = 'Two-Turn\nRelevance-Only\nREBEL Rerank'

    # Calculate combined metric and prepare data for plotting
    metrics = []
    metrics_std = []
    speeds = []
    labels = []
    method_types = []  # To track if method is relevance-only or multi-criteria

    for method, time, length in zip(inference_data['Method'], inference_data['Total Time (s)'], inference_data['Response Length']):
        standard_name = name_mapping[method]
        idx = data['Experiment'].index(standard_name)

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
        if standard_name in ['Two-Turn\nRelevance-Only\nREBEL Rerank', 'LLM Rerank', 'Cohere Rerank', 'HyDE', 'No Rerank']:
            method_types.append('relevance')
        else:
            method_types.append('multi')

    plt.rcParams.update({'font.size': 20})

    fig, ax = plt.subplots(figsize=(12, 8))

    # # Add vertical lines - now purple and solid
    # ax.axvline(x=50, color='purple', linestyle='-', alpha=0.3)
    # ax.axvline(x=100, color='purple', linestyle='-', alpha=0.3)

    # # Add region labels
    # ax.text(25, 4.3, 'Two-Turn Reranking\nInference Band',
    #         horizontalalignment='center', verticalalignment='bottom', color='purple')
    # ax.text(75, 4.3, 'One-Turn Reranking\nInference Band',
    #         horizontalalignment='center', verticalalignment='bottom', color='purple')
    # ax.text(125, 4.3, 'No Reranking\nInference Band',
    #         horizontalalignment='center', verticalalignment='bottom', color='purple')

    print(metrics)
    metrics_std = [float(s) for s in metrics_std]
    print(metrics_std)
    # Plot points with different colors based on method type
    for speed, metric, metric_std, method_type in zip(speeds, metrics, metrics_std, method_types):
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

    # Add tradeoff curve
    one_turn_idx = labels.index('One-Turn REBEL Rerank')
    two_turn_idx = labels.index('Two-Turn REBEL Rerank')
    no_rerank_idx = labels.index('No Rerank')

    # Add line from Two-Turn to One-Turn with shading
    x_line1 = [speeds[two_turn_idx], speeds[one_turn_idx]]
    y_line1 = [metrics[two_turn_idx], metrics[one_turn_idx]]
    y_std1 = [metrics_std[two_turn_idx], metrics_std[one_turn_idx]]
    
    # Create interpolation points for smooth shading
    x_interp1 = np.linspace(x_line1[0], x_line1[1], 100)
    y_interp1 = np.interp(x_interp1, x_line1, y_line1)
    std_interp1 = np.interp(x_interp1, x_line1, y_std1)
    
    # Plot line and shading for first segment
    ax.plot(x_interp1, y_interp1, '--', color='red', alpha=0.5)
    ax.fill_between(x_interp1, 
                    y_interp1 - std_interp1,
                    y_interp1 + std_interp1,
                    color='red', alpha=0.1)

    # Add line from One-Turn to No Rerank with shading
    x_line2 = [speeds[one_turn_idx], speeds[no_rerank_idx]]
    y_line2 = [metrics[one_turn_idx], metrics[no_rerank_idx]]
    y_std2 = [metrics_std[one_turn_idx], metrics_std[no_rerank_idx]]
    
    # Create interpolation points for smooth shading
    x_interp2 = np.linspace(x_line2[0], x_line2[1], 100)
    y_interp2 = np.interp(x_interp2, x_line2, y_line2)
    std_interp2 = np.interp(x_interp2, x_line2, y_std2)
    
    # Plot line and shading for second segment
    line = ax.plot(x_interp2, y_interp2, '--', color='red', alpha=0.5, 
                  label='Our System Quality/Speed\nTradeoff Curve')[0]
    ax.fill_between(x_interp2,
                    y_interp2 - std_interp2,
                    y_interp2 + std_interp2,
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
        Line2D([0], [0], linestyle='--', color='red', alpha=0.5, label='Our System Quality/Speed\nTradeoff Curve'),
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
    parser.add_argument('--inference-times', type=str, help='Path to inference times CSV file (required for figure 2)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)
    
    if args.figure == 1:
        figure_1(args.output_path)
        print(f"Figure 1 and its data have been saved to {args.output_path}")
    
    elif args.figure == 2:
        if not args.inference_times:
            raise ValueError("--inference-times argument is required for figure 2")
        if not os.path.exists(args.inference_times):
            raise FileNotFoundError(f"Inference times file not found: {args.inference_times}")
            
        inference_times = pd.read_csv(args.inference_times)
        figure_2(inference_times, args.output_path)
        print(f"Figure 2 and its data have been saved to {args.output_path}")
    
    else:
        print("Please specify which figure to generate using --figure argument")

if __name__ == "__main__":
    main()