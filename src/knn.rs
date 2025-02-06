use linfa::prelude::*;
use linfa_nn::{distance::*, CommonNearestNeighbour, NearestNeighbour};
use ndarray::{Array1, Array2};
use std::io::Error;


fn knn(track_nums: Array1<f64>, features: Array2<f64>) -> Result<(), Error> {

    // Build a K-D tree using Euclidean distance
    let nn = CommonNearestNeighbour::KdTree
        .from_batch(&features, L2Dist)
        .expect("Failed to build K-D tree");

    // Choose a sample query point (first track's features)
    let pt = features.row(0).to_owned();

    // Find the 10 nearest neighbors
    let nearest = nn.k_nearest(pt.view(), 10).unwrap();
    println!("10 Nearest Neighbors: {:?}", nearest);

    // Find all points within a range of 100 units
    let range = nn.within_range(pt.view(), 100.0).unwrap();
    println!("Points within 100.0 units: {:?}", range);

    Ok(())
}
