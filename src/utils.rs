use bytemuck::checked::cast_slice;
use bytemuck::Pod;
use std::io::{self, Read};

pub fn read_file_to_struct<T: Pod>(file: &mut std::fs::File) -> io::Result<T> {
    let mut buffer = vec![0u8; std::mem::size_of::<T>()];
    file.read_exact(&mut buffer)?;

    let ret = bytemuck::try_from_bytes::<T>(&buffer).expect("FAIL");
    Ok(*ret)
}

pub fn read_variable_length_string(file: &mut std::fs::File, size: usize) -> io::Result<String> {
    let mut buffer = vec![0u8; size];
    file.read_exact(&mut buffer)?;

    let ret = String::from_utf8_lossy(&buffer).to_string();
    Ok(ret)
}

pub fn read_variable_length_data<T: Pod>(
    file: &mut std::fs::File,
    size: usize,
) -> io::Result<Box<[T]>> {
    let mut buffer = vec![0u8; size * std::mem::size_of::<T>()];
    file.read_exact(&mut buffer)?;

    let ret = cast_slice::<u8, T>(&buffer).to_vec().into_boxed_slice();
    Ok(ret)
}
