use linfa::prelude::*;
use linfa_nn::{distance::*, CommonNearestNeighbour, NearestNeighbour};
use ndarray::{Array1, Array2};
use std::io::Error;
mod read_data;

// fn knn_classification(data: &Array2<f64>, labels: &Array1<usize>, test_data: &Array2<f64>) -> Vec<usize> {
//     // Create and train the KNN model
//     let model = NearestNeighbour::new(3)  // Using k=3
//         .fit(data, labels)
//         .expect("Failed to fit KNN model");

//     // Predict labels for test data
//     let predictions = model.predict(test_data);

//     predictions.to_vec()
// }

// fn knn(data: &ArrayBase<DT, Dim<[usize; 2]>>, point: Array1<f64>) -> (ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 2]>>, ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 2]>>) 
// {
//     let nn: Box<dyn NearestNeighbourIndex<f64> + Send + Sync> = CommonNearestNeighbour::KdTree.from_batch(&data, L2Dist).unwrap();
//     let nearest: ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 1]>> = nn.k_nearest(point.view(), 5).unwrap();
//     let range: ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 2]>> = nn.within_range(point.view(), 100.0).unwrap();

//     (nearest, range)
// }


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
