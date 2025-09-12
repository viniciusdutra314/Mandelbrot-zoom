use ndarray::Array2;
use ndarray_npy::NpzWriter;
use rug::ops::CompleteRound;

fn pixel_color(c:rug::Complex,max_iterations:usize,precision:u32)->u8{
    let mut z=rug::Complex::with_val(precision,(0,0));
    let two=rug::Float::with_val(precision,2);
    
    let mut iterations=0;
    while z.clone().norm().real()<&two && iterations<max_iterations {
        z.square_mut();
        z+=&c;
        iterations+=1;
    }
    return ((255.0*iterations as f64)/(max_iterations as f64)) as u8;
}

fn mandelbrot_set(
    resolution: [usize; 2],
    limit_points: [rug::Complex;2],
    max_iterations: usize,
    precision: u32,
) -> Array2<u8> {
    let (width, height) = (resolution[0], resolution[1]);
    let mut result = Array2::<u8>::zeros((height, width));

    let (x0, y0) = (limit_points[0].real().clone(), limit_points[0].imag().clone());
    let (x1, y1) = (limit_points[1].real().clone(), limit_points[1].imag().clone());
    
    let dx = (&x1 - &x0).complete(precision) / rug::Float::with_val(precision, width as f64);
    let dy = (&y1 - &y0).complete(precision) / rug::Float::with_val(precision, width as f64);

    use rayon::prelude::*;

    result.axis_iter_mut(ndarray::Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(j, mut row)| {
            let y = &y0 + &dy * rug::Float::with_val(precision, j as f64);
            for (i, pixel) in row.iter_mut().enumerate() {
                let x = &x0 + &dx * rug::Float::with_val(precision, i as f64);
                let c = rug::Complex::with_val(precision, (x, y.clone()));
                *pixel = pixel_color(c, max_iterations, precision);
            }
        });
    return result;
}  




fn main() {
    let resolution_x=800;
    let resolution_y=800;
    let lower_left=rug::Complex::with_val(256,(-1.0,-1.0));
    let upper_right=rug::Complex::with_val(256,(1.0,1.0));
    let image=mandelbrot_set([resolution_x,resolution_y], [lower_left,upper_right], 200, 16);
    let mut npz=NpzWriter::new(std::fs::File::create("output.npz").unwrap());
    npz.add_array("1",&image).unwrap();
    npz.finish().unwrap();
}
