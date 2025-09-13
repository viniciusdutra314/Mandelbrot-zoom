use core::num;
use rayon::prelude::*;
use ndarray::Array2;
use ndarray_npy::NpzWriter;
use rug::ops::{CompleteRound, Pow};


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
    limit_points: [&rug::Complex;2],
    max_iterations: usize,
    precision: u32,
) -> Array2<u8> {
    let (width, height) = (resolution[0], resolution[1]);
    let mut result = Array2::<u8>::zeros((height, width));

    let (x0, y0) = (limit_points[0].real().clone(), limit_points[0].imag().clone());
    let (x1, y1) = (limit_points[1].real().clone(), limit_points[1].imag().clone());
    
    let dx = (&x1 - &x0).complete(precision) / rug::Float::with_val(precision, width as f64);
    let dy = (&y1 - &y0).complete(precision) / rug::Float::with_val(precision, height as f64);

    let x_vals: Vec<rug::Float> = (0..width)
    .map(|i| &x0 + &dx * rug::Float::with_val(precision, i as f64))
    .collect();

    let y_vals: Vec<rug::Float> = (0..height)
    .map(|j| &y0 + &dy * rug::Float::with_val(precision, j as f64))
    .collect();

    result.axis_iter_mut(ndarray::Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(j, mut row)| {
            let y = y_vals[j].clone();
            for (i, pixel) in row.iter_mut().enumerate() {
                let x=x_vals[i].clone();
                let c = rug::Complex::with_val(precision, (x, y.clone()));
                *pixel = pixel_color(c, max_iterations, precision);
            }
        });
    return result;
}  




fn main() {
    let resolution_x=400;
    let resolution_y=400;
    let num_frames=300;
    let max_iteration=200;
    let zoom_factor:f64=1.15;
    let initial_lower_left=rug::Complex::with_val(256,(-1.5,-1.5));
    let initial_upper_right=rug::Complex::with_val(256,(1.5,1.5));
    let focus_point=rug::Complex::with_val(256,(0.001643721971153,-0.822467633298876));
    let shift_lower_left_vec=(&focus_point-&initial_lower_left).complete((256,256));
    let shift_upper_right_vec=(&focus_point-&initial_upper_right).complete((256,256));
    let mut npz=NpzWriter::new(std::fs::File::create("output.npz").unwrap());


    for frame in 0..num_frames {
        let precision=32+((frame as f64)*(zoom_factor.log2())) as u32;
        println!("Generating frame {frame}/{num_frames} (precision={precision})");
        let reduction_factor=1.0-1/rug::Float::with_val(precision, zoom_factor).pow(frame);
        let current_shift_lower_left_vec = shift_lower_left_vec.clone()*&reduction_factor;
        let current_shift_upper_right_vec = shift_upper_right_vec.clone()*&reduction_factor;
        let new_lower_left = &initial_lower_left+current_shift_lower_left_vec;
        let new_upper_right = &initial_upper_right+current_shift_upper_right_vec;
        let image = mandelbrot_set([resolution_x, resolution_y], 
            [&new_lower_left, &new_upper_right], max_iteration, 
            precision);
        npz.add_array(frame.to_string(), &image).unwrap();


    }
    npz.finish().unwrap();
}
