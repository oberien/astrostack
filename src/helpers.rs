use cv::bitarray::BitArray;
use cv::feature::akaze::{Akaze, KeyPoint};
use image::{DynamicImage, GenericImageView, ImageBuffer, Luma, Rgb64FImage};
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
