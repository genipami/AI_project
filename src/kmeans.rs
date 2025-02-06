use linfa::DatasetBase;
use linfa_clustering::KMeans;
use linfa_nn::distance::LInfDist;
use ndarray::{Array1, Array2};

fn kmeans(data: DatasetBase<Array2<f64>, Array1<f64>>, n: usize) -> ()
{
    
    let model = KMeans::params_with(n, data, LInfDist)
        .max_n_iterations(200)
        .tolerance(1e-5)
        .fit(&data)
        .expect("KMeans fitted");


    let dataset = model.predict(data);
    let DatasetBase {
        records, targets, ..
    } = dataset;

    write_npy("clustered_dataset.npy", &records).expect("Failed to write .npy file");
    write_npy("clustered_memberships.npy", &targets.map(|&x| x as u64))
        .expect("Failed to write .npy file");
}