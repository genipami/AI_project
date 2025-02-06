use linfa::prelude::*;
use linfa::DatasetBase;
use linfa_clustering::KMeans;
use linfa_nn::distance::LInfDist;
use ndarray::{Array1, Array2};
use ndarray_npy::write_npy;
use ndarray_rand::rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;

const SEED: u64 = 42;

fn kmeans(data: DatasetBase<Array2<f64>, Array1<f64>>, n: usize) -> ()
{
    
    let rng: Xoshiro256Plus = Xoshiro256Plus::seed_from_u64(SEED);
    let model = KMeans::params_with(n, rng.clone(), LInfDist)
        .max_n_iterations(200)
        .tolerance(1e-5)
        .fit(&data)
        .expect("KMeans not fitted");


    let dataset = model.predict(data);
    let DatasetBase {
        records, targets, ..
    } = dataset;

    let records_f64: Array2<f64> = records.mapv(|x| x as f64);
    write_npy("clustered_dataset.npy", &records_f64).expect("Failed to write .npy file");

    let converted_targets: Array1<f64> = targets.mapv(|x| x as f64);
    write_npy("clustered_memberships.npy", &converted_targets).expect("Failed to write .npy file");
}