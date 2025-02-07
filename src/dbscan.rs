use linfa::traits::Transformer;
use linfa_clustering::Dbscan;
use ndarray::{Array1, Array2};
use std::error::Error;

pub fn dbscan(track_nums: Array1<f64>, features:Array2<f64>, min_points: usize, tolerance: f64) -> Result<(), Box<dyn Error>> {

    let clusters = Dbscan::params(min_points)
        .tolerance(tolerance)
        .transform(&features)?;

    for (i, cluster) in clusters.iter().enumerate() {
        match cluster {
            Some(cluster_id) => println!("Track {} -> Cluster {}", track_nums[i], cluster_id),
            None => println!("Track {} -> Noise", track_nums[i]),
        }
    }

    Ok(())
}
