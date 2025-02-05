extern crate csv;
extern crate ndarray_csv;
use std::io::{BufReader, BufRead};
use csv::ReaderBuilder;
use ndarray::{array, Array1, Array2};
use std::fs::File;
use linfa_nn::{distance::*, CommonNearestNeighbour, NearestNeighbour, NearestNeighbourIndex};
use linfa_clustering::KMeans;
use linfa::traits::Fit;
use linfa::traits::Predict;
use linfa::DatasetBase;


fn read_features() -> Result<Array2<f64>, std::io::Error> 
{
    let file = File::open("fma/fma/fma_metadata/features.csv")?;
    let mut buf_reader = BufReader::new(&file);

    let mut header_rows: Vec<Vec<String>> = Vec::new();
    for _ in 0..4 {
        let mut line = String::new();
        buf_reader.read_line(&mut line)?;
        let row: Vec<String> = line.trim().split(',').map(|s| s.to_string()).collect();
        header_rows.push(row);
    }

    let num_columns = header_rows[0].len();
    let mut combined_headers = Vec::new();
    for col in 0..num_columns {
        let combined = header_rows
            .iter()
            .map(|row| &row[col])
            .collect::<Vec<&String>>();
        combined_headers.push(combined);
    }

    let mut reader = ReaderBuilder::new().has_headers(false).flexible(true).from_reader(file);
    let mut valid_rows: Vec<Vec<f64>> = Vec::new();
    for result in reader.records() {
        let record = result?;
        if record.len() == num_columns {
            let row: Vec<f64> = record.iter()
                .map(|s| s.parse::<f64>().unwrap_or(0.0))
                .collect();
            valid_rows.push(row);
        }
    }

    let data:Array2<f64> = Array2::from_shape_vec((valid_rows.len(), num_columns), valid_rows.concat()).unwrap();

    println!("Array shape: {:?}", data.dim());
    return Ok(data);
}

fn knn(data: Array2<f64>, point: Array1<f64>) -> (Array1<f64>, Array2<f64>) 
{
    let nn: Box<dyn NearestNeighbourIndex<f64> + Send + Sync> = CommonNearestNeighbour::KdTree.from_batch(&data, L2Dist).unwrap();
    let nearest: ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 1]>> = nn.k_nearest(point.view(), 5).unwrap();
    let range: ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 2]>> = nn.within_range(point.view(), 100.0).unwrap();

    (nearest, range)
}

fn kmeans(data: Array2<f64>, n: usize) -> ()
{
    
    let model = KMeans::params_with(n, data, LInfDist)
        .max_n_iterations(200)
        .tolerance(1e-5)
        .fit(&dataset)
        .expect("KMeans fitted");


    let dataset = model.predict(data);
    let DatasetBase {
        records, targets, ..
    } = dataset;

    write_npy("clustered_dataset.npy", &records).expect("Failed to write .npy file");
    write_npy("clustered_memberships.npy", &targets.map(|&x| x as u64))
        .expect("Failed to write .npy file");
}
fn main(){
    
}
