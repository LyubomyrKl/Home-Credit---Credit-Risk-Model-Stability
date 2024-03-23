import statistics
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_distribution(
        column,

        dataframe,
        bins='auto',
        histplot=True,
        kdeplot=False,
        scatterplot=False,
        scatter_plot_y=False,

        xlim=False,
        ylim=False,

        is_categorical=False,
        countplot=True,
        pieplot=False,
        groupby=False,
        numeric_column_to_compare=False,
        categorical_column_to_compare=False,

        title=None,
        xlabel=None,
        ylabel=None,

        show_mean=False,
        show_median=False,
        show_mode=False,

        total_num=True,
        total_nan=True,
        count_zeros=True,
        total_non_nan_values=True,
        percentage_zeros=True,
        drop_invalid=True,
        describe=True
):
    """
    Plot the distribution of a column in a DataFrame.

    Parameters:
    - dataframe: pandas.DataFrame
    - column: str, the column name to plot
    - bins: int or 'auto', the number of bins for continuous data or 'auto' for automatic binning
    """

    # Check if the column exists in the DataFrame
    if column not in dataframe.columns:
        print(f"Error: Column '{column}' not found in the DataFrame.")
        return

    # Drop NaNs or empties if specified
    if drop_invalid:
        dataframe = dataframe.dropna(subset=[column])

    plt.figure(figsize=(10, 6))

    if is_categorical:
        if countplot:
            sns.countplot(x=dataframe[column], data=dataframe)
            plt.title(f'Distribution of {column} - Categorical')
            plt.ylabel('Frequency')
            plt.xlabel(column)
            plt.show()
        if pieplot:
            dataframe[column].value_counts().plot.pie(autopct='%1.1f%%')
            plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            plt.show()

        if groupby:
            if numeric_column_to_compare:
                dataframe.groupby(column)[numeric_column_to_compare].mean().plot(kind='bar')
                plt.show()
            elif categorical_column_to_compare:
                # Your existing code
                sns.countplot(x=categorical_column_to_compare, hue=column, data=dataframe)
                # Adding x and y labels
                plt.xlabel('Credit Account Actual Balance')  # Replace with the actual label for the x-axis
                plt.ylabel('Count')  # Replace with the actual label for the y-axis
                plt.show()

    else:
        mean = dataframe[column].mean()
        median = dataframe[column].median()
        mode_value = statistics.mode(dataframe[column])

        if histplot:
            sns.histplot(dataframe[column], bins=bins, kde=kdeplot)
            plt.title(title or f'Distribution of {column}')
            plt.xlabel(xlabel or f'{column} values')
            plt.ylabel(ylabel or 'Frequency')
            plt.ylim(ylim)
            plt.xlim(xlim)

            if show_mean: plt.axvline(mean, color='red', linestyle='dashed', label='Mean')
            if show_median: plt.axvline(median, color='green', linestyle='dashed', label='Median')
            if show_mode: plt.axvline(mode_value, color='blue', linestyle='dashed', label='Mode')
            plt.legend()
            plt.show()

        if kdeplot:
            sns.kdeplot(dataframe[column])
            plt.xlabel(xlabel or f'{column} values')
            plt.ylabel(ylabel or 'Frequency')
            plt.ylim(ylim)
            plt.xlim(xlim)
            if show_mean: plt.axvline(mean, color='red', linestyle='dashed', label='Mean')
            if show_median: plt.axvline(median, color='green', linestyle='dashed', label='Median')
            if show_mode: plt.axvline(mode_value, color='blue', linestyle='dashed', label='Mode')
            plt.legend()
            plt.show()

        if scatterplot:
            sns.set_palette("husl")
            sns.set_theme(style="whitegrid")
            scatter_plot = sns.scatterplot(x=column, y=scatter_plot_y, data=dataframe.iloc[:1000], palette="viridis",
                                           alpha=0.7)

            scatter_plot.set(
                xlabel=xlabel or f'{column} values',
                ylabel=ylabel or 'Frequency',
                title=title or f'Distribution of {column}'
            )

            scatter_plot.legend()
            scatter_plot.grid(True, linestyle='--', alpha=0.7)
            sns.regplot(x=column, y=scatter_plot_y, data=dataframe.iloc[:1000], scatter=False, ax=scatter_plot)
            plt.show()

    nzeros = (dataframe[column] == 0).sum()
    ntotalnonnan = dataframe[column].notna().sum()
    if total_num and not describe: print('Total: ', dataframe[column].shape[0])
    if total_non_nan_values: print('Total non zero value: ', ntotalnonnan)
    if total_nan: print('Missed values num. ', dataframe[column].isnull().sum())
    if count_zeros and not is_categorical: print('Count zeros: ', nzeros)
    if percentage_zeros: print("Zero %: ", np.round(nzeros / ntotalnonnan * 100, 2))
    if show_mean and not is_categorical and not describe: print("Mean: ", dataframe[column].mean())
    if show_median and not is_categorical and not describe: print('Median: ', dataframe[column].median())
    if show_mode: print('Mode: ', dataframe[column].mode()[0])
    if describe: dataframe[column].describe()
