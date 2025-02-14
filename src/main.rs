mod read_data;
mod knn;
mod kmeans;
mod dbscan;
use read_data::{read_features, read_genres, align_genres};
use linfa::prelude::*;
use linfa::dataset::Labels;
use kmeans::kmeans;
use dbscan::dbscan;
use ndarray::s;
use ndarray::{Array1, Array2};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (headers, track_nums, features) = read_features()?;
    let genre_map: std::collections::HashMap<u32, String> = read_genres()?;
    let true_labels: Vec<String> = align_genres(track_nums.clone(), &genre_map);
    let num_rows_to_keep = 100000;

    let new_features = features.slice(s![..num_rows_to_keep, ..]).to_owned();
    let new_labels = (&true_labels[0..num_rows_to_keep]).to_owned();

    let dataset = Dataset::new(new_features.clone(), new_labels.into());
    kmeans::kmeans(dataset, 3);

    let dbscan_res = dbscan::dbscan(track_nums, new_features.clone(), 5, 0.9);
    if let Err(e) = dbscan_res {
        eprintln!("Error during DBSCAN: {}", e);
    }
    Ok(())
}
