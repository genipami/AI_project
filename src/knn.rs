use linfa::prelude::*;
use linfa_nn::{distance::*, CommonNearestNeighbour, NearestNeighbour};
use ndarray::{Array1, Array2};
use std::io::Error;


pub fn knn(track_nums: Array1<f64>, features: Array2<f64>) -> Result<(), Error> {

    let nn = CommonNearestNeighbour::KdTree
        .from_batch(&features, L2Dist)
        .expect("Failed to build K-D tree");

    let pt = features.row(0).to_owned();

    let nearest = nn.k_nearest(pt.view(), 10).unwrap();
    println!("10 Nearest Neighbors: {:?}", nearest);

    let range = nn.within_range(pt.view(), 100.0).unwrap();
    println!("Points within 100.0 units: {:?}", range);

    Ok(())
}
