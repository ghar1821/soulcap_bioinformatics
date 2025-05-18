import pandas as pd

from sklearn.neighbors import KNeighborsClassifier

def label_new_data(
    ref_set_points_df, 
    new_dat_set_points_df,
    ppm_dat_bin_coords,
    df_new_with_bins,
    markers
):
    
    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(
        ref_set_points_df[markers].to_numpy(), 
        ref_set_points_df['Population'].to_numpy()
    )

    pred_labels = neigh.predict(new_dat_set_points_df[markers].to_numpy())
    new_dat_set_points_df.loc[:, 'Population'] = pred_labels

    # find the label for each bin based on majority
    new_dat_bin_labels = new_dat_set_points_df.groupby('bin_id')['Population'].agg(lambda x: x.value_counts().idxmax())
    ppm_dat_bin_coords.loc[:, "Population"] = ppm_dat_bin_coords['bin_id'].map(new_dat_bin_labels)

    # For bin with no label because no weighted set points were generated, train knn on the labelled bins and label those nan ones.
    ppm_dat_bin_coords_nolabs = ppm_dat_bin_coords[ppm_dat_bin_coords['Population'].isna()]
    ppm_dat_bin_coords_labs = ppm_dat_bin_coords[~ppm_dat_bin_coords['Population'].isna()]

    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(
        ppm_dat_bin_coords_labs[['x_center', 'y_center']].to_numpy(), 
        ppm_dat_bin_coords_labs['Population'].to_numpy()
    )

    pred_labels = neigh.predict(ppm_dat_bin_coords_nolabs[['x_center', 'y_center']].to_numpy())
    ppm_dat_bin_coords_nolabs.loc[:, 'Population'] = pred_labels

    # merge
    ppm_dat_bin_coords_labelled = pd.concat([ppm_dat_bin_coords_nolabs, ppm_dat_bin_coords_labs])
    
    # label the new data from bin
    ppm_dat_labs = ppm_dat_bin_coords_labelled['Population']
    ppm_dat_labs.index = ppm_dat_bin_coords_labelled['bin_id']

    new_labels = df_new_with_bins['bin_id'].map(ppm_dat_labs)
    
    return(new_labels)


