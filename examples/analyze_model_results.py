#!/usr/bin/env python
"""
Analyze and visualize results from model comparison experiments.

This script loads experiment results from model comparison runs and creates
visualizations to help understand model performance differences.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime

# Add parent directory to path to allow importing the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def load_experiment_results(experiment_path: str) -> Dict[str, Any]:
    """Load experiment results from the specified directory.
    
    Args:
        experiment_path: Path to the experiment directory
        
    Returns:
        Dictionary containing the experiment results
    """
    # Check if path is a directory
    if os.path.isdir(experiment_path):
        # Look for summary_stats.json
        summary_path = os.path.join(experiment_path, "summary_stats.json")
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                return json.load(f)
    
    # If path is a file, try to load it directly
    if os.path.isfile(experiment_path):
        with open(experiment_path, 'r') as f:
            return json.load(f)
    
    raise ValueError(f"Could not find valid experiment results at {experiment_path}")


def create_performance_dataframe(experiment_results: Dict[str, Any]) -> pd.DataFrame:
    """Create a pandas DataFrame from the experiment results for visualization.
    
    Args:
        experiment_results: Dictionary containing the experiment results
        
    Returns:
        DataFrame containing model performance metrics
    """
    model_stats = experiment_results.get("overall_stats", {}).get("model_stats", {})
    
    # Create a list of dictionaries for each model
    model_data = []
    for model_name, stats in model_stats.items():
        model_data.append({
            "Model": model_name,
            "Total Interactions": stats.get("total_interactions", 0),
            "Secrets Revealed (%)": stats.get("revealed_secret_percentage", 0),
            "Secrets Obtained (%)": stats.get("obtained_secret_percentage", 0),
            "Optimal Strategy (%)": stats.get("optimal_strategy_percentage", 0)
        })
    
    # Create DataFrame
    return pd.DataFrame(model_data)


def plot_model_comparison(df: pd.DataFrame, output_dir: Optional[str] = None) -> None:
    """Create visualization plots comparing model performance.
    
    Args:
        df: DataFrame containing model performance metrics
        output_dir: Directory to save plots, if None plots are displayed
    """
    # Set up the plotting style
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 13,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.figsize': (12, 8)
    })
    
    # Timestamp for saved files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create the output directory if it doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Bar chart of key metrics
    plt.figure(figsize=(14, 10))
    
    # Melt the DataFrame for easier plotting with seaborn
    metrics_df = df.melt(
        id_vars=["Model"],
        value_vars=["Secrets Revealed (%)", "Secrets Obtained (%)", "Optimal Strategy (%)"],
        var_name="Metric",
        value_name="Percentage"
    )
    
    # Create the bar chart
    ax = sns.barplot(
        x="Model",
        y="Percentage",
        hue="Metric",
        data=metrics_df,
        palette="viridis"
    )
    
    # Customize the plot
    plt.title("Model Performance Comparison", fontsize=16)
    plt.xlabel("AI Model", fontsize=14)
    plt.ylabel("Percentage (%)", fontsize=14)
    plt.xticks(rotation=0)
    plt.legend(title="Metric", loc="upper right")
    
    # Add value labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', padding=3)
    
    plt.tight_layout()
    
    # Save or show the plot
    if output_dir:
        plt.savefig(os.path.join(output_dir, f"model_metrics_comparison_{timestamp}.png"), dpi=300)
    else:
        plt.show()
    
    # Plot 2: Scatter plot of revealed vs. obtained secrets
    plt.figure(figsize=(12, 8))
    
    # Create the scatter plot
    ax = sns.scatterplot(
        x="Secrets Revealed (%)",
        y="Secrets Obtained (%)",
        data=df,
        s=200,  # Point size
        alpha=0.7,
        hue="Model"
    )
    
    # Add model names as annotations
    for i, row in df.iterrows():
        plt.annotate(
            row["Model"],
            (row["Secrets Revealed (%)"] + 0.5, row["Secrets Obtained (%)"] + 0.5),
            fontsize=11
        )
    
    # Add a diagonal line (y = x)
    max_val = max(df["Secrets Revealed (%)"].max(), df["Secrets Obtained (%)"].max()) + 5
    plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.3)
    
    # Add a "good strategy" region (higher secrets obtained, lower revealed)
    plt.fill_between(
        [0, max_val],
        [max_val, max_val],
        [0, max_val],
        alpha=0.1,
        color='green',
        label="Optimal Strategy Region"
    )
    
    # Customize the plot
    plt.title("Secrets Revealed vs. Secrets Obtained", fontsize=16)
    plt.xlabel("Secrets Revealed (%)", fontsize=14)
    plt.ylabel("Secrets Obtained (%)", fontsize=14)
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    plt.grid(True, alpha=0.3)
    plt.legend(title="Model")
    
    plt.tight_layout()
    
    # Save or show the plot
    if output_dir:
        plt.savefig(os.path.join(output_dir, f"revealed_vs_obtained_{timestamp}.png"), dpi=300)
    else:
        plt.show()
    
    # Plot 3: Optimal strategy performance
    plt.figure(figsize=(12, 6))
    
    # Sort by optimal strategy percentage
    df_sorted = df.sort_values("Optimal Strategy (%)", ascending=False)
    
    # Create the bar chart
    ax = sns.barplot(
        x="Model",
        y="Optimal Strategy (%)",
        data=df_sorted,
        palette="magma"
    )
    
    # Add value labels
    for i, v in enumerate(df_sorted["Optimal Strategy (%)"]):
        ax.text(
            i,
            v + 1,
            f"{v:.1f}%",
            ha='center',
            va='bottom',
            fontsize=12,
            fontweight='bold'
        )
    
    # Customize the plot
    plt.title("Optimal Strategy Performance by Model", fontsize=16)
    plt.xlabel("AI Model", fontsize=14)
    plt.ylabel("Optimal Strategy (%)", fontsize=14)
    plt.xticks(rotation=0)
    
    plt.tight_layout()
    
    # Save or show the plot
    if output_dir:
        plt.savefig(os.path.join(output_dir, f"optimal_strategy_{timestamp}.png"), dpi=300)
    else:
        plt.show()
    
    # Plot 4: Radar chart of all metrics
    plt.figure(figsize=(12, 10))
    
    # Prepare data for radar chart
    metrics = ["Secrets Revealed (%)", "Secrets Obtained (%)", "Optimal Strategy (%)"]
    
    # Number of variables
    N = len(metrics)
    
    # Compute angle for each axis
    angles = [n / float(N) * 2 * 3.14159 for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create subplot with polar projection
    ax = plt.subplot(111, polar=True)
    
    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], metrics, size=12)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([20, 40, 60, 80, 100], ["20%", "40%", "60%", "80%", "100%"], size=10)
    plt.ylim(0, 100)
    
    # Plot each model
    for i, row in df.iterrows():
        model_name = row["Model"]
        values = [row[metric] for metric in metrics]
        values += values[:1]  # Close the loop
        
        # Plot values
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model_name)
        ax.fill(angles, values, alpha=0.1)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title("Model Performance Metrics", size=16)
    
    plt.tight_layout()
    
    # Save or show the plot
    if output_dir:
        plt.savefig(os.path.join(output_dir, f"radar_chart_{timestamp}.png"), dpi=300)
    else:
        plt.show()
    
    # If output_dir is provided, create a summary HTML file
    if output_dir:
        create_html_summary(
            df, 
            experiment_results=experiment_results,
            output_dir=output_dir,
            timestamp=timestamp
        )
        
        print(f"Visualizations saved to {output_dir}")


def create_html_summary(
    df: pd.DataFrame, 
    experiment_results: Dict[str, Any], 
    output_dir: str, 
    timestamp: str
) -> None:
    """Create an HTML summary of the experiment results.
    
    Args:
        df: DataFrame containing model performance metrics
        experiment_results: Dictionary containing the experiment results
        output_dir: Directory to save the HTML file
        timestamp: Timestamp for the filename
    """
    # Get configuration
    config = experiment_results.get("config", {})
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Model Comparison Results</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
            h1, h2, h3 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .config {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            .images {{ display: flex; flex-wrap: wrap; justify-content: center; gap: 20px; margin-top: 30px; }}
            .image-container {{ max-width: 600px; margin-bottom: 20px; }}
            img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
        </style>
    </head>
    <body>
        <h1>AI Model Comparison Results</h1>
        <p>Analysis generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        
        <h2>Experiment Configuration</h2>
        <div class="config">
            <p><strong>Batch Service:</strong> {config.get("batch_service_type", "N/A")}</p>
            <p><strong>Game Mode:</strong> {config.get("game_mode", "N/A")}</p>
            <p><strong>Interactions per Pair:</strong> {config.get("num_interactions", "N/A")}</p>
            <p><strong>Messages per Interaction:</strong> {config.get("messages_per_interaction", "N/A")}</p>
            <p><strong>Batch Size:</strong> {config.get("batch_size", "N/A")}</p>
            <p><strong>Timestamp:</strong> {config.get("timestamp", "N/A")}</p>
        </div>
        
        <h2>Model Performance Summary</h2>
        <table>
            <tr>
                <th>Model</th>
                <th>Total Interactions</th>
                <th>Secrets Revealed (%)</th>
                <th>Secrets Obtained (%)</th>
                <th>Optimal Strategy (%)</th>
            </tr>
    """
    
    # Add rows for each model
    for i, row in df.iterrows():
        html_content += f"""
            <tr>
                <td>{row["Model"]}</td>
                <td>{row["Total Interactions"]}</td>
                <td>{row["Secrets Revealed (%)"]:.2f}%</td>
                <td>{row["Secrets Obtained (%)"]:.2f}%</td>
                <td>{row["Optimal Strategy (%)"]:.2f}%</td>
            </tr>
        """
    
    # Add interaction counts
    interaction_counts = experiment_results.get("overall_stats", {}).get("interaction_counts", {})
    
    html_content += """
        </table>
        
        <h2>Interaction Counts</h2>
        <table>
            <tr>
                <th>Model Pair</th>
                <th>Interactions</th>
            </tr>
    """
    
    for pair, count in interaction_counts.items():
        html_content += f"""
            <tr>
                <td>{pair}</td>
                <td>{count}</td>
            </tr>
        """
    
    # Add images
    html_content += """
        </table>
        
        <h2>Visualizations</h2>
        <div class="images">
            <div class="image-container">
                <h3>Model Performance Comparison</h3>
                <img src="model_metrics_comparison_{timestamp}.png" alt="Model Metrics Comparison">
            </div>
            
            <div class="image-container">
                <h3>Secrets Revealed vs. Obtained</h3>
                <img src="revealed_vs_obtained_{timestamp}.png" alt="Revealed vs Obtained">
            </div>
            
            <div class="image-container">
                <h3>Optimal Strategy Performance</h3>
                <img src="optimal_strategy_{timestamp}.png" alt="Optimal Strategy">
            </div>
            
            <div class="image-container">
                <h3>Radar Chart of Performance Metrics</h3>
                <img src="radar_chart_{timestamp}.png" alt="Radar Chart">
            </div>
        </div>
        
        <h2>Conclusions</h2>
        <p>This analysis compares the performance of different AI models in the secret trading game using
        standardized prompts and identical game rules. The key metrics shown above help identify which
        models are most effective at strategic gameplay in this context.</p>
        
        <p>Note that optimal strategy performance (obtaining secrets without revealing) is the best indicator
        of overall effectiveness in this game.</p>
    </body>
    </html>
    """.replace("{timestamp}", timestamp)
    
    # Write HTML to file
    html_path = os.path.join(output_dir, f"results_summary_{timestamp}.html")
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    print(f"HTML summary saved to {html_path}")


def find_latest_experiment(base_dir: str = "results/model_comparison") -> Optional[str]:
    """Find the most recent experiment directory.
    
    Args:
        base_dir: Base directory containing experiment directories
        
    Returns:
        Path to the most recent experiment directory, or None if none found
    """
    if not os.path.exists(base_dir):
        return None
    
    # List all directories in the base directory
    dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) 
            if os.path.isdir(os.path.join(base_dir, d))]
    
    if not dirs:
        return None
    
    # Sort by modification time (most recent first)
    dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    return dirs[0]


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze and visualize model comparison results"
    )
    
    parser.add_argument(
        "--experiment-path",
        type=str,
        help="Path to experiment directory or summary_stats.json file"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/visualizations",
        help="Directory to save visualizations"
    )
    
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display plots instead of saving them"
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()
    
    # Determine experiment path
    experiment_path = args.experiment_path
    if not experiment_path:
        # Try to find the most recent experiment
        experiment_path = find_latest_experiment()
        if not experiment_path:
            print("Error: No experiment path provided and no recent experiments found.")
            print("Please specify a path using --experiment-path")
            sys.exit(1)
        print(f"Using most recent experiment: {experiment_path}")
    
    # Load experiment results
    try:
        experiment_results = load_experiment_results(experiment_path)
        print(f"Loaded experiment results from {experiment_path}")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Create DataFrame
    df = create_performance_dataframe(experiment_results)
    
    # Display basic info
    print("\nExperiment Summary:")
    print(f"Configuration: {experiment_results.get('config', {}).get('game_mode', 'N/A')} mode, "
          f"{experiment_results.get('config', {}).get('num_interactions', 'N/A')} interactions per pair")
    print("\nModel Performance:")
    print(df.to_string(index=False))
    
    # Determine output directory
    output_dir = None if args.show_plots else args.output_dir
    
    # Create visualizations
    plot_model_comparison(df, output_dir)


if __name__ == "__main__":
    main() 