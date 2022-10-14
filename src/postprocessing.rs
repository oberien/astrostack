use cv::feature::akaze::KeyPoint;
use image::{DynamicImage, Rgb, Rgb64FImage};
use crate::Postprocessing;

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

pub fn akaze_draw(buf: &mut Rgb64FImage, threshold: f64) {
    let (keypoints, _descriptors) = crate::helpers::akaze(buf, threshold);
    for keypoint in keypoints {
        akaze_draw_kp(buf, keypoint);
    }
}

pub fn akaze_draw_kp(buf: &mut Rgb64FImage, keypoint: KeyPoint) {
    let KeyPoint { point, size, angle, .. } = keypoint;
    let color = Rgb([1.,0.,0.]);
    imageproc::drawing::draw_hollow_circle_mut(buf, (point.0 as i32, point.1 as i32), size as i32, color);
    let (endx, endy) = (point.0 + size * angle.cos(), point.1 + size * angle.sin());
    imageproc::drawing::draw_line_segment_mut(buf, point, (endx, endy), color);
}
