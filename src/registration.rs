use std::path::PathBuf;
use cv::feature::akaze::KeyPoint;
use image::{DynamicImage, GenericImage, ImageBuffer, Rgb};
use image::io::Reader;
use crate::Postprocessing;

pub fn register(files: &[PathBuf], width: u32, height: u32, num_files: usize, mut processing: &[Postprocessing]) {
    let akaze_threshold = match processing.last() {
        None => return,
        Some(&Postprocessing::Akaze(threshold)) => threshold,
        _ => unreachable!(),
    };
    processing = &processing[..processing.len()-1];

    let mut first = Reader::open(&files[0]).unwrap().decode().unwrap().into_rgb64f();
    let mut second = Reader::open(&files[1]).unwrap().decode().unwrap().into_rgb64f();
    crate::postprocessing::process(&mut first, num_files, &processing);
    crate::postprocessing::process(&mut second, num_files, &processing);

    let (kps1, desc1) = crate::helpers::akaze(&first, akaze_threshold);
    let (kps2, desc2) = crate::helpers::akaze(&second, akaze_threshold);
    const THRESH: f32 = 0.8;

    let mut bests = Vec::new();
    for desc1 in desc1 {
        let mut vice = u32::MAX;
        let mut best = (u32::MAX, 0);
        for (i, desc2) in desc2.iter().enumerate() {
            let dist = desc1.distance(desc2);
            if dist < best.0 {
                vice = best.0;
                best = (dist, i);
            }
            if (best.0 as f32) < vice as f32 * THRESH {
                bests.push(Some(best.1));
            } else {
                bests.push(None);
            }
        }
    }

    let mut res = ImageBuffer::new(width * 2, height);
    res.copy_from(&first, 0, 0).unwrap();
    res.copy_from(&second, width, 0).unwrap();

    for &kp in &kps1 {
        crate::postprocessing::akaze_draw_kp(&mut res, kp);
    }
    for &kp2 in &kps2 {
        let kp2 = KeyPoint { point: (kp2.point.0 + width as f32, kp2.point.1), ..kp2 };
        crate::postprocessing::akaze_draw_kp(&mut res, kp2);
    }
    for (kp, best) in kps1.into_iter().zip(bests) {
        let best = match best {
            Some(best) => best,
            None => continue,
        };
        let kp2 = kps2[best];
        let kp2 = KeyPoint { point: (kp2.point.0 + width as f32, kp2.point.1), ..kp2 };
        imageproc::drawing::draw_line_segment_mut(&mut res, kp.point, kp2.point, Rgb([1.,0.,0.]));
    }
    DynamicImage::ImageRgb64F(res).into_rgb16().save("registration.png").unwrap();
}