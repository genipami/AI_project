mod read_data;
mod knn;
mod kmeans;
mod dbscan;
mod evaluation;
use linfa_clustering::{KMeans, Dbscan};
use read_data::{read_features, read_genres, align_genres};
use evaluation::evaluate_clusters;
use linfa::prelude::*;
use linfa::dataset::Labels;
use kmeans::kmeans;
use dbscan::dbscan;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (headers, track_nums, features) = read_features()?;
    let genre_map = read_genres()?;
    let true_labels = align_genres(track_nums.clone(), &genre_map);

    let dataset = Dataset::new(features.clone(), true_labels.into());
    let best_k = (5..15)
        .map(|k| {
            let model = KMeans::params(k).fit(&dataset).unwrap();
            let clusters = model.predict(&features);
            let score = model.inertia(); 
            (k, score)
        })
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap()
        .0;

    println!("🔍 Best K for KMeans: {}", best_k);
    let dataset = DatasetBase::new(features.clone(), track_nums.clone());

    kmeans(dataset, best_k);

    let best_dbscan = (1..6)
        .flat_map(|min_pts| [0.01, 0.05, 0.1].iter().map(move |&tol| (min_pts, tol)))
        .map(|(min_pts, tol)| {
            let model = Dbscan::params(min_pts).tolerance(tol).transform(&features).unwrap();
            let clusters: Vec<Option<usize>> = model.labels().iter().map(|l| l.map(|v| v as usize)).collect();
            let valid_clusters: Vec<usize> = clusters.iter().flatten().cloned().collect();
            let score = valid_clusters.len();
            (min_pts, tol, score)
        })
        .max_by(|a, b| a.2.cmp(&b.2))
        .unwrap();

    println!("Best DBSCAN Params: min_points={} tolerance={}", best_dbscan.0, best_dbscan.1);
    
    dbscan(track_nums.clone(), features.clone(), best_dbscan.0, best_dbscan.1)?; 
    Ok(())
}
