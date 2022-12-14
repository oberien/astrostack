use image::{DynamicImage, imageops, Rgb, Rgb64FImage};
use ordered_float::NotNan;
use crate::{helpers, Processing, register};

pub fn process(buf: &mut Rgb64FImage, num_files: usize, processing: &[Processing]) {
    for postprocess in processing {
        match postprocess {
            Processing::Average => average(buf, num_files),
            Processing::Maxscale => maxscale(buf),
            &Processing::MaxscaleFixed(maxcol) => maxscale_fixed(buf, maxcol),
            Processing::Sqrt => sqrt(buf),
            Processing::Asinh => asinh(buf),
            Processing::Sharpen => sharpen(buf),
            &Processing::Sobel(blur) => sobel(buf, blur),
            &Processing::Blur(sigma) => gaussian_blur(buf, sigma),
            &Processing::Median(radius) => median(buf, radius),
            &Processing::BGone(threshold) => background_extract(buf, threshold),
            &Processing::BlackWhite(threshold) => black_while(buf, threshold),
            &Processing::Akaze(threshold) => akaze_draw(buf, threshold),
            &Processing::SingleObjectDetection(threshold) => single_object_detection(buf, threshold),
            &Processing::AverageBrightnessAlignment(threshold) => average_brightness_alignment(buf, threshold),
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

pub fn maxcol(buf: &Rgb64FImage) -> f64 {
    buf.pixels()
        .fold(0_f64, |maxcol, px| {
            let maxcol = maxcol.max(px.0[0]);
            let maxcol = maxcol.max(px.0[1]);
            let maxcol = maxcol.max(px.0[2]);
            maxcol
        })
}
pub fn maxscale(buf: &mut Rgb64FImage) {
    let maxcol = maxcol(buf);
    for pixel in buf.pixels_mut() {
        pixel.0[0] /= maxcol;
        pixel.0[1] /= maxcol;
        pixel.0[2] /= maxcol;
    }
}

pub fn maxscale_fixed(buf: &mut Rgb64FImage, maxcol: f64) {
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

pub fn sharpen(buf: &mut Rgb64FImage) {
    *buf = imageops::filter3x3(buf, &[
         0., -1.,  0.,
        -1.,  5., -1.,
         0., -1.,  0.,
    ]);
}

pub fn median(buf: &mut Rgb64FImage, radius: u32) {
    // prepare bitmap
    let radius: i32 = radius.try_into().unwrap();
    let side_len = radius as usize * 2 + 1;
    let mut bitmap = bitvec::bitvec![0; side_len * side_len];
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let in_range = dy.abs() + dx.abs() <= radius;
            let index = (dy + radius) as usize * side_len + (dx + radius) as usize;
            bitmap.set(index, in_range);
        }
    }

    let orig = buf.clone();
    let mut reds = Vec::with_capacity(side_len * side_len);
    let mut greens = Vec::with_capacity(side_len * side_len);
    let mut blues = Vec::with_capacity(side_len * side_len);
    for (x, y, pixel) in buf.enumerate_pixels_mut() {
        reds.clear();
        greens.clear();
        blues.clear();
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let index = (dy + radius) as usize * side_len + (dx + radius) as usize;
                if !bitmap[index] {
                    continue;
                }
                let x = (x as i32 + dx).max(0).min(orig.width() as i32 - 1) as u32;
                let y = (y as i32 + dy).max(0).min(orig.height() as i32 - 1) as u32;
                reds.push(NotNan::new(orig[(x, y)].0[0]).unwrap());
                greens.push(NotNan::new(orig[(x, y)].0[1]).unwrap());
                blues.push(NotNan::new(orig[(x, y)].0[2]).unwrap());
            }
        }
        reds.sort();
        greens.sort();
        blues.sort();
        let red = reds[reds.len() / 2].into_inner();
        let green = greens[greens.len() / 2].into_inner();
        let blue = blues[blues.len() / 2].into_inner();
        *pixel = Rgb([red, green, blue]);
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
    let sod = register::single_object_detection(buf, threshold);
    helpers::draw_object(buf, sod);
}
pub fn average_brightness_alignment(buf: &mut Rgb64FImage, threshold: f64) {
    let aba = register::average_brightness(buf, threshold);
    helpers::draw_cross(buf, (aba.middlex, aba.middley));
}

pub fn akaze_draw(buf: &mut Rgb64FImage, threshold: f64) {
    let akaze = register::akaze(buf, threshold);
    for keypoint in akaze.keypoints {
        helpers::akaze_draw_kp(buf, keypoint);
    }
}
