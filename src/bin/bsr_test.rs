use nalgebra::DMatrix;
use nalgebra::DVector;
use nalgebra::Matrix3;
use nalgebra::Matrix3x1;
use nalgebra::Vector3;
use nalgebra_sparse::CooMatrix;
use nalgebra_sparse::CsrMatrix;

fn main() {
    /*
        let mut block_coo: CooMatrix<Matrix3<f64>> = CooMatrix::new(5, 5);
        let data: Matrix3<f64> = Matrix3::from_iterator((0..9).map(|i| i as f64));
        block_coo.push(0, 0, data);
        block_coo.push(1, 2, data);
        block_coo.push(2, 2, data);
        block_coo.push(2, 1, data);
        block_coo.push(3, 3, data);
        block_coo.push(4, 3, data);
        block_coo.push(4, 4, data);

        let bsr: CsrMatrix<Matrix3<f64>> = CsrMatrix::from(&block_coo);
        let bvector: DVector<Matrix3x1<f64>> = DVector::from_element(5, Vector3::new(1.0, 2.0, 3.0));

        let test = &bsr * &bvector;
    */
}
