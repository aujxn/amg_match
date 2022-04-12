pub mod partitioner;
pub mod preconditioner;
pub mod solver;

#[macro_use]
extern crate log;

use sprs::CsMat;

/// Saves the sparsity graph of the matrix as a png
pub fn mat_to_image<T>(mat: &CsMat<T>, filename: &str) {
    let image_as_2darray = sprs::visu::nnz_image(mat.view());
    let (h, w) = image_as_2darray.dim();
    let raw_image = image_as_2darray.into_raw_vec();
    let png = image::GrayImage::from_raw(w as u32, h as u32, raw_image).unwrap();
    png.save(filename).unwrap();
}
