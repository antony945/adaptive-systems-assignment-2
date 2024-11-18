import pandas as pd
import matplotlib.pyplot as plt

DPI = 300

def draw_1(file="output_q1.csv"):
    # Read CSV data into a pandas DataFrame
    df = pd.read_csv(file)

    # Step 2: Group by Sparsity and find the Neighbors that minimize the MAE for each Sparsity
    min_mae_neighbors = df.loc[df.groupby("Sparsity")["MAE"].idxmin()]

    # Step 3: Create separate plots for each sparsity value
    for sparsity in df["Sparsity"].unique():
        # Subset the data for the current sparsity level
        subset = df[df["Sparsity"] == sparsity]
        
        # Plot MAE vs Neighbors for the current sparsity using pandas' plot method
        ax = subset.plot(x="Neighbors", y="MAE", kind="line", linestyle='-', legend=False)

        # Highlight the Neighbor values that minimize MAE and show corresponding MAE value
        for _, row in min_mae_neighbors[min_mae_neighbors["Sparsity"] == sparsity].iterrows():
            ax.scatter(row["Neighbors"], row["MAE"], color='red', zorder=5)
            # Display both the number of neighbors and the MAE value
            ax.text(row["Neighbors"], row["MAE"], f"({int(row['Neighbors'])}, {row['MAE']:.2f})", 
                    color='red', fontsize=12, ha='left', va='bottom')

        # Labels and title
        ax.set_xlabel("Number of Neighbors")
        ax.set_ylabel("MAE")
        # ax.set_title(f"MAE vs. Number of Neighbors (Sparsity = {sparsity})")
        ax.grid(True)
        # ax.legend(loc='best')

        # Step 4: Show the plot
        plt.tight_layout()
        plt.savefig(f"figures_clo/1_{sparsity}.png", dpi=DPI)
        plt.close()  # Close the plot to avoid overlapping when creating multiple plots

def draw_2a(file="output_q2.csv"):
    # Read CSV data into a pandas DataFrame
    df = pd.read_csv(file)
    
    # Step 1: Filter the DataFrame to keep only Sparsity values of 0.25 and 0.75
    df = df[df["Sparsity"].isin([0.25, 0.75])]

    # Step 2: Pivot the DataFrame to make it easier to plot
    pivot_df = df.pivot(index="Sparsity", columns="Algo", values="MAE")

    # Step 3: Create the bar plot
    ax = pivot_df.plot(kind='bar', width=0.8, edgecolor='black')
        # Step 4: Add the labels to the bars using ax.bar_label()
    for container in ax.containers:
        ax.bar_label(container, label_type='edge', fontsize=10, padding=3, labels=[f'{v:.2f}' for v in container.datavalues])

    # Step 5: Customize the plot
    plt.xlabel("Sparsity")
    plt.ylabel("MAE")
    # plt.title("Comparison of KNN and SVD MAE for Different Sparsity Levels")
    plt.xticks(rotation=0)  # Keep the Sparsity labels horizontal
    plt.legend(title="Algorithm", loc='best')

    y_max = pivot_df.max().max()  # Find the maximum MAE value
    plt.ylim(0, y_max * 1.1)  # Increase the y-axis upper limit by 10%

    # Step 6: Show the plot
    plt.tight_layout()
    plt.savefig(f"figures_clo/2_a.png", dpi=DPI)
    plt.close()  # Close the plot to avoid overlapping when creating multiple plots

def draw_2b(file="output_q2.csv"):
    # Read CSV data into a pandas DataFrame
    df = pd.read_csv(file)

    # Step 1: Pivot the DataFrame to make it easier to plot
    pivot_df = df.pivot(index="Sparsity", columns="Algo", values="MAE")

    # Step 2: Create the plot using the plot function on pivot_df
    # plt.figure(figsize=(8, 6))

    # Plot the data with markers
    pivot_df.plot(marker='o', markersize=6, linestyle='-', linewidth=2)

    # Step 3: Customize the plot
    plt.xlabel("Sparsity")
    plt.ylabel("MAE")
    # plt.title("Comparison of KNN and SVD MAE for Different Sparsity Levels")
    plt.xticks(rotation=0)  # Keep the Sparsity labels horizontal
    plt.grid(True)
    plt.legend(title="Algorithm", loc='best')

    # Step 4: Show the plot
    plt.tight_layout()
    plt.savefig(f"figures_clo/2_b.png", dpi=DPI)
    plt.close()  # Close the plot to avoid overlapping when creating multiple plots

def draw_3(file="output_q3.csv"):
    # Read CSV data into a pandas DataFrame
    df = pd.read_csv(file)

    # Step 1: Filter the DataFrame to keep only Sparsity values of 0.25 and 0.75
    df = df[df["Sparsity"].isin([0.25, 0.75])]
    
    # Define the default colors (first three from the default color cycle)
    default_colors = plt.cm.tab10.colors[:3]  # Get the first 3 colors from tab10 colormap
    
    # Define markers for each algorithm (separate from metrics)
    algorithm_markers = {
        "KNN": 'o',    # Circle for KNN
        "SVD": '^',    # Triangle for SVD
        "Algorithm3": 's',  # Square for another algorithm (replace with actual names)
        # Add more algorithms if necessary
    }

    # Step 2: Create a plot for each sparsity level
    for sparsity in df["Sparsity"].unique():
        # Create a new figure for each sparsity level
        # plt.figure(figsize=(8, 6))
        plt.figure()

        # Subset the data for the current sparsity level
        subset = df[df["Sparsity"] == sparsity]

        # Define metrics to plot
        metrics = ["Precision", "Recall", "F1"]
        
        # Plot Precision, Recall, F1 vs TopN for all algorithms at the specified sparsity
        for i, metric in enumerate(metrics):
            for j, algorithm in enumerate(subset["Algo"].unique()):
                # Subset data for the current algorithm and metric
                algo_data = subset[subset["Algo"] == algorithm]
                
                # Plot the metric with a fixed color for the metric and a unique marker for each algorithm
                plt.plot(algo_data["TopN"], algo_data[metric], 
                         label=f"{algorithm} - {metric}", 
                         color=default_colors[i],  # Use default colors (blue, orange, green)
                         marker=algorithm_markers[algorithm],  # Set marker for the algorithm
                         linestyle='-', 
                         markersize=6)

        # Labels and title
        plt.xlabel("Number of Neighbors (TopN)")
        plt.ylabel("Metrics")
        # plt.title(f"Metrics vs. TopN (Sparsity = {sparsity})")
        
        # Set the y-axis limit between 0 and 1
        plt.ylim(0, 1.05)
        
        # Ensure that all lines are labeled in the legend
        plt.legend(loc="best")
        plt.grid(True)
        plt.savefig(f"figures_clo/3_{sparsity}.png", dpi=DPI)
        plt.close()  # Close the plot to avoid overlapping when creating multiple plots

if __name__ == '__main__':
    draw_1()
    draw_2a()
    draw_2b()
    draw_3()