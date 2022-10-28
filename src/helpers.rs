use std::fs::File;
use std::path::{Path, PathBuf};
use cv::feature::akaze::KeyPoint;
use image::{DynamicImage, GenericImageView, ImageBuffer, Luma, Rgb, Rgb64FImage};
use image::io::Reader;
use crate::Colorspace;
use crate::register::{Registration, SodRegistration};

pub fn load_image<P: AsRef<Path>>(path: P, colorspace: Colorspace) -> Rgb64FImage {
    let mut img = Reader::open(path).unwrap().decode().unwrap().into_rgb64f();
    for px in img.pixels_mut() {
        *px = colorspace.convert_into(*px);
    }
    img
}
pub fn save_image<P: AsRef<Path>>(mut img: Rgb64FImage, path: P, colorspace: Colorspace) {
    for pixel in img.pixels_mut() {
        *pixel = colorspace.convert_back(*pixel);
    }
    DynamicImage::ImageRgb64F(img).into_rgb16().save(path).unwrap();
}

pub fn load_registration<P: AsRef<Path>>(path: P) -> Registration {
    serde_json::from_reader(File::open(path).unwrap()).unwrap()
}
pub fn save_registration<P: AsRef<Path>>(path: P, registration: &Registration) {
    serde_json::to_writer(File::create(path).unwrap(), registration).unwrap();
}

pub fn path_with_suffix<P: AsRef<Path>>(prefix: P, suffix: &str) -> PathBuf {
    let file_name = format!("{}_{}", prefix.as_ref().file_name().unwrap().to_str().unwrap(), suffix);
    prefix.as_ref().with_file_name(file_name)
}

// from https://github.com/dangreco/edgy/blob/master/src/main.rs
pub fn edgy_sobel(image: &DynamicImage, blur_modifier: i32) -> ImageBuffer<Luma<u8>, Vec<u8>> {
    let (width, height) = image.dimensions();
    let sigma = (((width * height) as f32) / 3630000.0) * blur_modifier as f32;
    let gaussed = image.blur(sigma);
    let gray = gaussed.into_luma8();

    let width: u32 = gray.width() - 2;
    let height: u32 = gray.height() - 2;
    let mut buff: ImageBuffer<Luma<u8>, Vec<u8>> = ImageBuffer::new(width, height);

    for i in 0..width {
        for j in 0..height {
            /* Unwrap those loops! */
            let val0 = gray.get_pixel(i, j).0[0] as i32;
            let val1 = gray.get_pixel(i + 1, j).0[0] as i32;
            let val2 = gray.get_pixel(i + 2, j).0[0] as i32;
            let val3 = gray.get_pixel(i, j + 1).0[0] as i32;
            let val5 = gray.get_pixel(i + 2, j + 1).0[0] as i32;
            let val6 = gray.get_pixel(i, j + 2).0[0] as i32;
            let val7 = gray.get_pixel(i + 1, j + 2).0[0] as i32;
            let val8 = gray.get_pixel(i + 2, j + 2).0[0] as i32;
            /* Apply Sobel kernels */
            let gx = (-1 * val0) + (-2 * val3) + (-1 * val6) + val2 + (2 * val5) + val8;
            let gy = (-1 * val0) + (-2 * val1) + (-1 * val2) + val6 + (2 * val7) + val8;
            let mut mag = ((gx as f64).powi(2) + (gy as f64).powi(2)).sqrt();

            if mag > 255.0 {
                mag = 255.0;
            }

            buff.put_pixel(i, j, Luma([mag as u8]));
        }
    }
    buff
}

pub fn draw_object(buf: &mut Rgb64FImage, sod: SodRegistration) {
    let left = (sod.left as f32, sod.middle().1 as f32);
    let right = (sod.right as f32, sod.middle().1 as f32);
    let top = (sod.middle().0 as f32, sod.top as f32);
    let bottom = (sod.middle().0 as f32, sod.bottom as f32);
    imageproc::drawing::draw_line_segment_mut(buf, left, right, Rgb([1., 0., 0.]));
    imageproc::drawing::draw_line_segment_mut(buf, top, bottom, Rgb([1., 0., 0.]));
}

pub fn draw_cross(buf: &mut Rgb64FImage, center: (f32, f32)) {
    let size = 20.;
    let left = (center.0 - size, center.1);
    let right = (center.0 + size, center.1);
    let top = (center.0, center.1 - size);
    let bottom = (center.0, center.1 + size);
    imageproc::drawing::draw_line_segment_mut(buf, left, right, Rgb([1., 0., 0.]));
    imageproc::drawing::draw_line_segment_mut(buf, top, bottom, Rgb([1., 0., 0.]));
}

pub fn akaze_draw_kp(buf: &mut Rgb64FImage, keypoint: KeyPoint) {
    let KeyPoint { point, size, angle, .. } = keypoint;
    let color = Rgb([1.,0.,0.]);
    imageproc::drawing::draw_hollow_circle_mut(buf, (point.0 as i32, point.1 as i32), size as i32, color);
    let (endx, endy) = (point.0 + size * angle.cos(), point.1 + size * angle.sin());
    imageproc::drawing::draw_line_segment_mut(buf, point, (endx, endy), color);
}
