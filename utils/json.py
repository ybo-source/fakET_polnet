import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import tkinter as tk
from tkinter import filedialog


def analyze_json_files(input_dir):
    # Initialize data structures
    all_data = []
    style_usage = defaultdict(int)
    content_layers_stats = defaultdict(list)
    loss_values = []
    iteration_counts = []
    gpu_ram_usage = []
    
    # Process each JSON file in directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(input_dir, filename)
            with open(filepath, 'r') as f:
                try:
                    data = json.load(f)
                    
                    # Extract basic info
                    content_path = data['args']['content']
                    styles = data['args']['styles']
                    iterations = len(data['iterates'])
                    
                    # Record style usage
                    for style in styles:
                        style_name = os.path.basename(style)
                        style_usage[style_name] += 1
                    
                    # Record content layers
                    if 'content_layers' in data['args']:
                        for layer in data['args']['content_layers']:
                            content_layers_stats[layer].append(content_path)
                    
                    # Record performance metrics
                    if data['iterates']:
                        loss_values.append(data['iterates'][-1]['loss'])
                        iteration_counts.append(iterations)
                        gpu_ram_usage.append(data['iterates'][0]['gpu_ram'] / (1024**3))  # Convert to GB
                    
                    # Add to complete dataset
                    all_data.append({
                        'file': filename,
                        'content': content_path,
                        'styles': [os.path.basename(s) for s in styles],
                        'iterations': iterations,
                        'final_loss': data['iterates'][-1]['loss'] if data['iterates'] else None,
                        'gpu_ram_gb': data['iterates'][0]['gpu_ram'] / (1024**3) if data['iterates'] else None,
                        'content_weight': data['args']['content_weight'],
                        'style_weights': data['args']['style_weights'],
                        'content_layers': data['args'].get('content_layers', []),
                        'random_seed': data['args']['random_seed']
                    })
                    
                except json.JSONDecodeError as e:
                    print(f"Error reading {filename}: {e}")
                except KeyError as e:
                    print(f"Missing key in {filename}: {e}")
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(all_data)
    
    # Generate statistics
    stats = {
        'total_files': len(all_data),
        'avg_loss': sum(loss_values) / len(loss_values) if loss_values else 0,
        'avg_iterations': sum(iteration_counts) / len(iteration_counts) if iteration_counts else 0,
        'avg_gpu_ram_gb': sum(gpu_ram_usage) / len(gpu_ram_usage) if gpu_ram_usage else 0,
        'style_usage': dict(sorted(style_usage.items(), key=lambda x: x[1], reverse=True)),
        'content_layers_distribution': {k: len(v) for k, v in content_layers_stats.items()}
    }
    
    return df, stats

def print_style_stats(style_usage):
    print("\nStyle Image Usage Statistics:")
    print("----------------------------")
    print(f"{'Style Image':<40} {'Count':<10}")
    print("-" * 50)
    for style, count in style_usage.items():
        print(f"{style:<40} {count:<10}")
    print("\n")

def visualize_results(df, stats, output_dir):
    # Create plots directory
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot loss distribution
    plt.figure(figsize=(10, 6))
    df['final_loss'].plot(kind='hist', bins=20)
    plt.title('Final Loss Distribution')
    plt.xlabel('Loss Value')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(plots_dir, 'loss_distribution.png'))
    plt.close()
    
    # Plot GPU RAM usage
    plt.figure(figsize=(10, 6))
    df['gpu_ram_gb'].plot(kind='box')
    plt.title('GPU RAM Usage Distribution')
    plt.ylabel('GB')
    plt.savefig(os.path.join(plots_dir, 'gpu_ram_usage.png'))
    plt.close()
    
    # Plot style usage
    style_df = pd.DataFrame.from_dict(stats['style_usage'], orient='index', columns=['count'])
    style_df.sort_values('count', ascending=False).plot(kind='bar', figsize=(12, 6))
    plt.title('Style Image Usage Frequency')
    plt.xlabel('Style Image')
    plt.ylabel('Usage Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'style_usage.png'))
    plt.close()
    
    # Save statistics to file
    with open(os.path.join(output_dir, 'summary_stats.txt'), 'w') as f:
        f.write("SUMMARY STATISTICS\n")
        f.write("=================\n\n")
        for key, value in stats.items():
            if key == 'style_usage':
                f.write("\nStyle Usage Counts:\n")
                f.write("------------------\n")
                for style, count in value.items():
                    f.write(f"{style}: {count}\n")
            elif key == 'content_layers_distribution':
                f.write("\nContent Layers Distribution:\n")
                f.write("--------------------------\n")
                for layer, count in value.items():
                    f.write(f"Layer {layer}: {count} uses\n")
            else:
                f.write(f"{key}: {value}\n")
    
    # Save full dataframe to CSV
    df.to_csv(os.path.join(output_dir, 'full_analysis.csv'), index=False)

