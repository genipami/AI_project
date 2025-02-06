mod read_data;
mod knn;
mod kmeans;
mod dbscan;
mod evaluation;
use linfa_clustering::{KMeans, Dbscan};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (headers, track_nums, features) = read_features()?;
    let genre_map = read_genres()?;
    let true_labels = align_genres(track_nums, &genre_map);

    // 🔹 Try different cluster numbers for KMeans
    let best_k = (5..15)
        .map(|k| {
            let model = KMeans::params(k).fit(&features).unwrap();
            let clusters = model.predict(&features);
            let score = model.inertia();  // Lower is better
            (k, score)
        })
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap()
        .0;

    println!("🔍 Best K for KMeans: {}", best_k);

    let kmeans = KMeans::params(best_k).fit(&features)?;
    let kmeans_clusters = kmeans.predict(&features);

    // 🔹 Tune DBSCAN min_points and tolerance
    let best_dbscan = (1..6)
        .flat_map(|min_pts| [0.01, 0.05, 0.1].iter().map(move |&tol| (min_pts, tol)))
        .map(|(min_pts, tol)| {
            let model = Dbscan::params(min_pts).tolerance(tol).transform(&features).unwrap();
            let clusters: Vec<Option<usize>> = model.labels().iter().map(|l| l.map(|v| v as usize)).collect();
            let valid_clusters: Vec<usize> = clusters.iter().flatten().cloned().collect();
            let score = valid_clusters.len();  // More clusters mean fewer noise points
            (min_pts, tol, score)
        })
        .max_by(|a, b| a.2.cmp(&b.2))
        .unwrap();

    println!("🔍 Best DBSCAN Params: min_points={} tolerance={}", best_dbscan.0, best_dbscan.1);

    let dbscan = Dbscan::params(best_dbscan.0).tolerance(best_dbscan.1).transform(&features)?;
    let dbscan_clusters: Vec<Option<usize>> = dbscan.labels().iter().map(|l| l.map(|v| v as usize)).collect();

    // Evaluate KMeans
    println!("\n📌 Evaluating KMeans...");
    evaluate_clusters(kmeans_clusters, true_labels.clone(), &features);

    // Evaluate DBSCAN
    println!("\n📌 Evaluating DBSCAN...");
    evaluate_clusters(dbscan_clusters, true_labels, &features);

    Ok(())
}
