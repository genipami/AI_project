extern crate csv;
extern crate ndarray_csv;
use std::io::{BufReader, BufRead};
use csv::ReaderBuilder;
use ndarray::{array, Array1, Array2, ArrayBase};
use std::fs::File;
use ndarray::prelude::*;
fn read_features() -> Result<(Vec<String>, Array1<f64>, Array2<f64>), std::io::Error> 
{
    let file: File = File::open("fma/fma/fma_metadata/features.csv")?;
    let mut buf_reader: BufReader<&File> = BufReader::new(&file);

    let mut header_rows: Vec<Vec<String>> = Vec::new();
    for _ in 0..4 {
        let mut line: String = String::new();
        buf_reader.read_line(&mut line)?;
        let row: Vec<String> = line.trim().split(',').map(|s| s.to_string()).collect();
        header_rows.push(row);
    }

    let num_columns: usize = header_rows[0].len();
    let mut combined_headers: Vec<String> = Vec::new();
    for col in 0..num_columns {
        let combined:String = header_rows
            .iter()
            .map(|row| row[col].clone())
            .collect::<Vec<String>>().join("_");
        combined_headers.push(combined);
    }

    let mut reader: csv::Reader<File> = ReaderBuilder::new().has_headers(false).flexible(true).from_reader(file);
    let mut valid_rows: Vec<Vec<f64>> = Vec::new();
    for result in reader.records() {
        let record: csv::StringRecord = result?;
        if record.len() == num_columns {
            let row: Vec<f64> = record.iter()
                .map(|s| s.parse::<f64>().unwrap_or(0.0))
                .collect();
            valid_rows.push(row);
        }
    }

    let data:Array2<f64> = Array2::from_shape_vec((valid_rows.len(), num_columns), valid_rows.concat()).unwrap();

    let track_nums = data.column(0).to_owned();
    let features = data.slice(s![.., 1..num_columns]).to_owned();
    println!("Array shape: {:?}", data.dim());
    
    return Ok((combined_headers, track_nums, features));
}