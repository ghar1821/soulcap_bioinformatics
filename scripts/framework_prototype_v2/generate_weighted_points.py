import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_weighted_point_set(
    ppm_bins_df,
    probability_threshold=0.0005,
    adaptive_sampling=True,
    max_points_per_bin=10
):
    """
    Generate a weighted point set from a Probability Point Map (PPM).

    Args:
        ppm_bins_df (pd.DataFrame): DataFrame with columns ['bin_id', 'x_center', 'y_center', 'prob'].
        probability_threshold (float, optional): Minimum probability for a bin to be included. Defaults to 0.0005.
        adaptive_sampling (bool, optional): If True, sample more points in higher-probability bins. If False, one point per bin. Defaults to True.
        max_points_per_bin (int, optional): Maximum points sampled per bin. Defaults to 10.

    Returns:
        points (np.ndarray): Array of (x, y) coordinates.
        weights (np.ndarray): Array of normalized weights for each point.
    """

    # Find the maximum probability for scaling
    max_probability = ppm_bins_df['prob'].max()

    # Keep only bins above the probability threshold
    bins_above_thresh = ppm_bins_df[ppm_bins_df['prob'] > probability_threshold]

    bin_probs = bins_above_thresh['prob']
    bin_x_centers = bins_above_thresh['x_center'].to_numpy()
    bin_y_centers = bins_above_thresh['y_center'].to_numpy()
    bin_ids = bins_above_thresh['bin_id'].to_numpy()

    if adaptive_sampling:
        # Scale number of points per bin by probability, capped at max_points_per_bin.
        # We scale the probability again so the bin with max_probability end up with max_points_per_bin points.
        # E.g., if we have probabilities of 0.2, 0.5, 0.6, 
        # dividing by max probability will yield 0.3, 0.83, 1.
        # Multiply this by max_points_per_bin will yield 3, 8, 10 points.
        scaled_probs = bin_probs / max_probability
        points_per_bin = np.maximum(
            1,
            np.ceil(scaled_probs * max_points_per_bin).astype(int)
        )

        # Generate random jitter for all points
        total_points = points_per_bin.sum()
        jitter = np.random.uniform(0.1, 0.9, (total_points, 2))

        # Repeat bin centers according to how many points each bin gets
        sampled_x = np.repeat(bin_x_centers, points_per_bin)
        sampled_y = np.repeat(bin_y_centers, points_per_bin)

        # Get the bin ids
        sampled_bin_ids = np.repeat(bin_ids, points_per_bin)

        # Add jitter to bin centers
        points = np.column_stack([
            sampled_x + jitter[:, 0],
            sampled_y + jitter[:, 1]
        ])
        point_weights = np.repeat(bin_probs, points_per_bin)
    else:
        # One point per bin
        points = np.column_stack([bin_x_centers, bin_y_centers])
        point_weights = bin_probs
        sampled_bin_ids = bin_ids

    # Normalize weights to sum to 1
    point_weights /= point_weights.sum()

    return points, point_weights, sampled_bin_ids


def compare_weighted_points(
    ref_set_points: np.ndarray,
    new_dat_set_points: np.ndarray,
    df: pd.DataFrame,
    x_axis: str,
    y_axis: str
) -> None:
    """
    Compare actual and generated weighted point datasets using scatter plots.

    Creates a side-by-side visualization:
      - The first plot displays the actual data colored by batch.
      - The second plot shows the generated weighted points, colored by their origin ('ref' or 'new_data').

    Parameters
    ----------
    ref_set_points : np.ndarray
        Array of shape (n_ref, 2) containing reference set points for [x_axis, y_axis].
    new_dat_set_points : np.ndarray
        Array of shape (n_new, 2) containing new data set points for [x_axis, y_axis].
    df : pandas.DataFrame
        DataFrame containing the actual data, with columns for `x_axis`, `y_axis`, and 'batch'.
        Batch denote whether the row comes from reference data or new data.
    x_axis : str
        Name of the column to use for the x-axis in both plots.
    y_axis : str
        Name of the column to use for the y-axis in both plots.

    Returns
    -------
    None
        Displays matplotlib figures; does not return anything.

    Example
    -------
    >>> compare_weighted_points(ref_points, new_points, df, 'CD69', 'CD99')
    """
    # Concatenate points and create batch labels
    all_points = np.vstack([ref_set_points, new_dat_set_points])
    batch_labels = np.concatenate([
        np.repeat('ref', len(ref_set_points)),
        np.repeat('new_data', len(new_dat_set_points))
    ])
    set_points = pd.DataFrame(all_points, columns=[x_axis, y_axis])
    set_points['batch'] = batch_labels

    _, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True, sharex=True)

    # Plot actual data
    sns.scatterplot(
        data=df,
        x=x_axis,
        y=y_axis,
        hue='batch',
        ax=axs[0]
    ).set_title("Actual data")

    # Plot generated weighted points
    sns.scatterplot(
        data=set_points,
        x=x_axis,
        y=y_axis,
        hue='batch',
        ax=axs[1]
    ).set_title("Generated weighted points")

    plt.tight_layout()
    plt.show()



