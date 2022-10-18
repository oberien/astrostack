use image::{DynamicImage, Rgb, Rgb64FImage};
use crate::{helpers, Postprocessing};

pub fn process(buf: &mut Rgb64FImage, num_files: usize, postprocessing: &[Postprocessing]) {
    for postprocess in postprocessing {
        match postprocess {
            Postprocessing::Average => average(buf, num_files),
            Postprocessing::Maxscale => maxscale(buf),
            Postprocessing::Sqrt => sqrt(buf),
            Postprocessing::Asinh => asinh(buf),
            &Postprocessing::Akaze(threshold) => akaze_draw(buf, threshold),
            &Postprocessing::Sobel(blur) => sobel(buf, blur),
            &Postprocessing::Blur(sigma) => gaussian_blur(buf, sigma),
            &Postprocessing::BGone(threshold) => background_extract(buf, threshold),
            &Postprocessing::BlackWhite(threshold) => black_while(buf, threshold),
            &Postprocessing::SingleObjectDetection(threshold) => single_object_detection(buf, threshold),
        }
    }
}

pub fn average(buf: &mut Rgb64FImage, num_files: usize) {
    for pixel in buf.pixels_mut() {
        pixel.0[0] /= num_files as f64;
        pixel.0[1] /= num_files as f64;
        pixel.0[2] /= num_files as f64;
    }
}

pub fn maxscale(buf: &mut Rgb64FImage) {
    let maxcol = buf.pixels()
        .fold(0_f64, |maxcol, px| {
            let maxcol = maxcol.max(px.0[0]);
            let maxcol = maxcol.max(px.0[1]);
            let maxcol = maxcol.max(px.0[2]);
            maxcol
        });
    for pixel in buf.pixels_mut() {
        pixel.0[0] /= maxcol;
        pixel.0[1] /= maxcol;
        pixel.0[2] /= maxcol;
    }
}

pub fn sqrt(buf: &mut Rgb64FImage) {
    for pixel in buf.pixels_mut() {
        pixel.0[0] = pixel.0[0].sqrt();
        pixel.0[1] = pixel.0[1].sqrt();
        pixel.0[2] = pixel.0[2].sqrt();
    }
}

pub fn asinh(buf: &mut Rgb64FImage) {
    for pixel in buf.pixels_mut() {
        pixel.0[0] = pixel.0[0].asinh();
        pixel.0[1] = pixel.0[1].asinh();
        pixel.0[2] = pixel.0[2].asinh();
    }
}

pub fn sobel(buf: &mut Rgb64FImage, blur: i32) {
    let sobeled = crate::helpers::edgy_sobel(&DynamicImage::ImageRgb64F(buf.clone()), blur);
    *buf = DynamicImage::ImageLuma8(sobeled).into_rgb64f();
}

pub fn gaussian_blur(buf: &mut Rgb64FImage, sigma: f32) {
    *buf = image::imageops::blur(buf, sigma)
}

pub fn background_extract(buf: &mut Rgb64FImage, threshold: f64) {
    for pixel in buf.pixels_mut() {
        let value = pixel.0.into_iter().sum::<f64>() / 3.;
        if value < threshold {
            *pixel = Rgb([0., 0., 0.]);
        }
    }
}

pub fn black_while(buf: &mut Rgb64FImage, threshold: f64) {
    for pixel in buf.pixels_mut() {
        let value = pixel.0.into_iter().sum::<f64>() / 3.;
        if value < threshold {
            *pixel = Rgb([0., 0., 0.]);
        } else {
            *pixel = Rgb([1., 1., 1.]);
        }
    }
}

pub fn single_object_detection(buf: &mut Rgb64FImage, threshold: f64) {
    let object = helpers::single_object_detection(buf, threshold);
    helpers::draw_object(buf, object);
}

pub fn akaze_draw(buf: &mut Rgb64FImage, threshold: f64) {
    let (keypoints, _descriptors) = helpers::akaze(buf, threshold);
    for keypoint in keypoints {
        helpers::akaze_draw_kp(buf, keypoint);
    }
}
