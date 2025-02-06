use linfa::prelude::*;
use linfa::metrics::{RandIndex, SilhouetteScore};

use linfa::DatasetBase;
use ndarray::{Array1, Array2};
use std::collections::HashMap;

fn encode_labels(labels: &[Option<String>]) -> (Vec<usize>, HashMap<String, usize>) {
    let mut label_map = HashMap::new();
    let mut encoded_labels = Vec::new();
    let mut next_id = 0;

    for label in labels.iter().flatten() {
        if !label_map.contains_key(label) {
            label_map.insert(label.clone(), next_id);
            next_id += 1;
        }
        encoded_labels.push(label_map[label]);
    }
    
    (encoded_labels, label_map)
}

fn evaluate_clusters(predicted_labels: Vec<Option<usize>>, true_labels: Vec<Option<String>>, features: &Array2<f64>) {
    let valid_indices: Vec<usize> = true_labels
        .iter()
        .enumerate()
        .filter_map(|(i, label)| if label.is_some() { Some(i) } else { None })
        .collect();

    let filtered_preds: Vec<usize> = valid_indices.iter().map(|&i| predicted_labels[i].unwrap()).collect();
    let (filtered_true, label_map) = encode_labels(&true_labels);

    // Adjusted Rand Index
    let ari = RandIndex::adjusted_rand_index(&filtered_true, &filtered_preds);
    
    // Normalized Mutual Information (NMI)
    let nmi = RandIndex::normalized_mutual_info_score(&filtered_true, &filtered_preds);

    // Silhouette Score
    let dataset = DatasetBase::from((features.clone(), Array1::from_vec(filtered_preds.clone())));
    let silhouette_score = dataset.silhouette_score().unwrap_or(0.0);

    println!("Evaluation Metrics:");
    println!("Adjusted Rand Index: {:.4}", ari);
    println!("Normalized Mutual Information: {:.4}", nmi);
    println!("Silhouette Score: {:.4}", silhouette_score);
}

