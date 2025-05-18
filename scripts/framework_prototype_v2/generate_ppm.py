import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_ppm(x_vals, y_vals, grid_size=32):
    # grid_size = [10, 32]
    # x then y (x=10, y=32)
    # if one value, then square grid.
    # cartesian plotting and numpy 2d histogram has 2 different orientation.
    # 1st argument in histogram2d is x-axis (CD69), but becomes rows in output.
    # 2nd argument in histogram2d is y-axis (CD99), but becomes columns in output.
    ppm, xedges, yedges = np.histogram2d(
        x=x_vals, 
        y=y_vals, 
        bins=[grid_size, grid_size]
    )

    # note the ppm output here, x_vals (CD69) will be rows
    # y_vals (CD99) is columns.
    # it's just the way numpy intrepret x and y differently from us.

    # Normalize to get probabilities
    total_cells = len(x_vals)
    ppm = ppm / total_cells

    xcenters = 0.5 * (xedges[:-1] + xedges[1:])
    ycenters = 0.5 * (yedges[:-1] + yedges[1:])

    # get bin ids and probabilities
    # get position and index of bins
    bin_coords = []
    bin_id = 0
    for x_idx in range(len(xcenters)):
        for y_idx in range(len(ycenters)):
            bin_coords.append([
                f"bin_{x_idx}_{y_idx}",
                xedges[x_idx],
                xedges[x_idx+1],
                xcenters[x_idx],
                yedges[y_idx],
                yedges[y_idx+1],
                ycenters[y_idx],
                ppm[x_idx, y_idx]
            ])
            bin_id += 1
    bin_coords = pd.DataFrame(bin_coords, columns=["bin_id", "x_min", "x_max", "x_center", "y_min", "y_max", "y_center", "prob"])

    # Find bin index for each point
    # Default behavior is (right==False) indicating that the interval does not include the right edge.
    # Subtract 1 to get 0-based bin index
    x_bin = np.digitize(x_vals, xedges) - 1  
    # digitize operate off the bin edges.
    # any value that is >= last bin edge will be given the len(xedges) id.
    # which does not exist as we only have len(xedges)-1 bins.
    # hence we need to clip the bins so it maxed out at number of bins - 1
    x_bin = np.clip(x_bin, 0, grid_size - 1)

    y_bin = np.digitize(y_vals, yedges) - 1
    y_bin = np.clip(y_bin, 0, grid_size - 1)

    bin_ids = [f"bin_{x_idx}_{y_idx}" for x_idx, y_idx in zip(x_bin, y_bin)]

    return bin_coords, bin_ids

def compare_ppms(
    df_ref: pd.DataFrame,
    ppm_ref_bin_df: pd.DataFrame,
    df_new: pd.DataFrame,
    ppm_new_bin_df: pd.DataFrame,
    x_axis: str,
    y_axis: str
) -> None:
    """
    Visualizes and compares reference and new population data alongside their corresponding PPM bin data.

    Produces a 2x2 grid of scatterplots:
      - Top left: Reference data colored by population.
      - Top right: Reference PPM bin data colored by probability.
      - Bottom left: New data colored by population.
      - Bottom right: New PPM bin data colored by probability.

    Parameters
    ----------
    df_ref : pandas.DataFrame
        Reference dataset containing at least columns for `x_axis`, `y_axis`, and 'Population'.
    ppm_ref_bin_df : pandas.DataFrame
        Reference PPM bin dataset containing columns 'x_center', 'y_center', and 'prob'.
    df_new : pandas.DataFrame
        New dataset containing at least columns for `x_axis`, `y_axis`.
    ppm_new_bin_df : pandas.DataFrame
        New PPM bin dataset containing columns 'x_center', 'y_center', and 'prob'.
    x_axis : str
        Column name in `df_ref` and `df_new` to use for the x-axis.
    y_axis : str
        Column name in `df_ref` and `df_new` to use for the y-axis.

    Returns
    -------
    None
        This function creates a matplotlib figure and displays it.

    Example
    -------
    >>> compare_ppms(df_ref, ppm_ref_bin_df, df_new, ppm_new_bin_df, 'CD69', 'CD99')
    """
    fig, axs = plt.subplots(2, 2, figsize=(10, 10), sharey=True, sharex=True)

    # Reference data
    sns.scatterplot(
        data=df_ref,
        x=x_axis,
        y=y_axis,
        hue='Population',
        ax=axs[0, 0]
    ).set_title("Reference data")

    # Reference PPM bin data
    sns.kdeplot(
        x=ppm_ref_bin_df["x_center"],
        y=ppm_ref_bin_df["y_center"],
        weights=ppm_ref_bin_df["prob"],
        fill=True,      # Optional: fill the contours
        cmap="viridis",  # Optional: color map
        ax=axs[0, 1]
    ).set_title("PPM Reference data")

    # New data
    sns.scatterplot(
        data=df_new,
        x=x_axis,
        y=y_axis,
        ax=axs[1, 0]
    ).set_title("New data")

    # New PPM bin data
    sns.kdeplot(
        x=ppm_new_bin_df["x_center"],
        y=ppm_new_bin_df["y_center"],
        weights=ppm_new_bin_df["prob"],
        fill=True,      # Optional: fill the contours
        cmap="viridis",  # Optional: color map
        ax=axs[1, 1]
    ).set_title("PPM New data")

    plt.tight_layout()
    plt.show()
