extern crate csv;
extern crate ndarray;
extern crate ndarray_csv;
use std::io::{BufReader, BufRead};
use csv::{ReaderBuilder, WriterBuilder};
use ndarray::{array, Array2};
use ndarray_csv::{Array2Reader, Array2Writer};
use std::error::Error;
use std::fs::File;
fn main() -> Result<(), Box<dyn Error>> {
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

    println!("Combined Headers: {:?}", combined_headers);
    let mut reader = ReaderBuilder::new().has_headers(false).from_reader(file);
    let array_read: Array2<u64> = reader.deserialize_array2((2, 3))?;
    return Ok(());
}
