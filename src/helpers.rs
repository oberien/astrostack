use cv::bitarray::BitArray;
use cv::feature::akaze::{Akaze, KeyPoint};
use image::{DynamicImage, GenericImageView, ImageBuffer, Luma, Rgb, Rgb64FImage};
use image::buffer::ConvertBuffer;

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

pub fn akaze(buf: &Rgb64FImage, threshold: f64) -> (Vec<KeyPoint>, Vec<BitArray<64>>) {
    let detector = Akaze::new(threshold);
    let luma16: ImageBuffer<Luma<u16>, Vec<u16>> = buf.convert();
    detector.extract(&DynamicImage::ImageLuma16(luma16))
}

#[derive(Debug, Copy, Clone)]
pub struct Object {
    pub left: u32,
    pub right: u32,
    pub top: u32,
    pub bottom: u32,
    pub middle: (u32, u32),
    pub width: u32,
    pub height: u32,
}

pub fn single_object_detection(buf: &Rgb64FImage, threshold: f64) -> Object {
    let mut left = u32::MAX;
    let mut right = 0;
    let mut top = u32::MAX;
    let mut bottom = 0;
    for (x, y, pixel) in buf.enumerate_pixels() {
        let value = pixel.0.into_iter().sum::<f64>() / 3.;
        if value >= threshold {
            left = x.min(left);
            right = x.max(right);
            top = y.min(top);
            bottom = y.max(bottom);
        }
    }
    let middlex = left + (right - left) / 2;
    let middley = top + (bottom - top) / 2;
    Object {
        left, right, top, bottom,
        middle: (middlex, middley),
        width: right - left,
        height: bottom - top,
    }
}

pub fn draw_object(buf: &mut Rgb64FImage, object: Object) {
    let left = (object.left as f32, object.middle.1 as f32);
    let right = (object.right as f32, object.middle.1 as f32);
    let top = (object.middle.0 as f32, object.top as f32);
    let bottom = (object.middle.0 as f32, object.bottom as f32);
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
