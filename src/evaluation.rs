use linfa::prelude::*;
use linfa::metrics:: SilhouetteScore;
use std::collections::HashMap;
use linfa::DatasetBase;
use ndarray::{Array1, Array2};


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

pub fn adjusted_rand_index(true_labels: &Array1<usize>, predicted_labels: &Array1<Option<usize>>) -> f64 {
    let mut contingency: HashMap<(usize, usize), usize> = HashMap::new();
    let mut true_cluster_sizes: HashMap<usize, usize> = HashMap::new();
    let mut pred_cluster_sizes: HashMap<usize, usize> = HashMap::new();
    let mut n = 0;

    for (true_label, pred_label) in true_labels.iter().zip(predicted_labels.iter()) {
        if let Some(pred) = pred_label {
            *contingency.entry((*true_label, *pred)).or_insert(0) += 1;
            *true_cluster_sizes.entry(*true_label).or_insert(0) += 1;
            *pred_cluster_sizes.entry(*pred).or_insert(0) += 1;
            n += 1;
        }
    }

    let sum_comb = |counts: &HashMap<usize, usize>| -> usize {
        counts.values().map(|&size| size * (size - 1) / 2).sum()
    };

    let sum_contingency: usize = contingency.values().map(|&size| size * (size - 1) / 2).sum();
    let sum_true = sum_comb(&true_cluster_sizes);
    let sum_pred = sum_comb(&pred_cluster_sizes);

    let expected_index = (sum_true as f64 * sum_pred as f64) / (n * (n - 1) / 2) as f64;
    let max_index = (sum_true + sum_pred) as f64 / 2.0;

    (sum_contingency as f64 - expected_index) / (max_index - expected_index)
}

pub fn evaluate_clusters(predicted_labels: Vec<Option<usize>>, true_labels: Vec<Option<String>>, features: &Array2<f64>) {
    let valid_indices: Vec<usize> = true_labels
        .iter()
        .enumerate()
        .filter_map(|(i, label)| if label.is_some() { Some(i) } else { None })
        .collect();

    let filtered_preds: Vec<Option<usize>> = valid_indices.iter().map(|&i| predicted_labels[i]).collect();
    let (filtered_true, label_map) = encode_labels(&true_labels);
    let filtered_preds_arr = Array1::from(filtered_preds.clone());
    let filtered_true_arr = Array1::from(filtered_true);
    let ari = adjusted_rand_index(&filtered_true_arr, &filtered_preds_arr);
    
    let dataset = DatasetBase::from((features.clone(), Array1::from_vec(filtered_preds.clone())));
    let silhouette_score = dataset.silhouette_score().unwrap_or(0.0);

    println!("Evaluation Metrics:");
    println!("Adjusted Rand Index: {:.4}", ari);
    println!("Silhouette Score: {:.4}", silhouette_score);
}

