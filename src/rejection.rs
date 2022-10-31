use crate::register::{ImageRegistration, Registration};
use crate::{Colorspace, helpers, Rejection};

pub fn reject(registration: &Registration, mut images: Vec<ImageRegistration>, rejections: &[Rejection]) -> Vec<ImageRegistration> {
    let reference = &registration.images[registration.reference_image];
    let reference_image = helpers::load_image(&reference.image, Colorspace::Srgb);
    let width = reference_image.width();
    let height = reference_image.height();
    for rejection in rejections {
        images = match rejection {
            &Rejection::AverageSod(threshold) => average(&images, threshold, width, height, |r| { let (a,b) = r.sod.middle(); (a as f32, b as f32) }),
            &Rejection::AverageAba(threshold) => average(&images, threshold, width, height, |r| (r.aba.middlex, r.aba.middley)),
            &Rejection::RegressionAkaze(threshold) => regression(&images, threshold, width, height, |r| r.akaze.unwrap().offset()),
            &Rejection::RegressionSod(threshold) => regression(&images, threshold, width, height, |r| r.sod.offset(&reference.sod)),
            &Rejection::RegressionAba(threshold) => regression(&images, threshold, width, height, |r| r.aba.offset(&reference.aba)),
            &Rejection::WidthHeight(threshold) => width_height(&images, threshold, &reference),
        }
    }
    images
}

fn average(images: &[ImageRegistration], threshold: f32, width: u32, height: u32, middle_fn: impl Fn(&ImageRegistration) -> (f32, f32)) -> Vec<ImageRegistration> {
    images.windows(3).filter_map(|window| {
        let (a,b,c) = match window {
            [a, b, c] => (a,b,c),
            _ => unreachable!(),
        };
        let (ax, ay) = middle_fn(a);
        let (bx, by) = middle_fn(b);
        let (cx, cy) = middle_fn(c);
        let avgx = (ax + bx + cx) / 3.;
        let avgy = (ay + by + cy) / 3.;
        let dx = (bx - avgx).abs();
        let dy = (by - avgy).abs();
        let distance = (dx.powi(2) + dy.powi(2)).sqrt();
        let wh = ((width*width + height*height) as f32).sqrt();
        if distance / wh >= threshold {
            None
        } else {
            Some(b.clone())
        }
    }).collect()
}

fn regression(images: &[ImageRegistration], threshold: f32, width: u32, height: u32, offset_fn: impl Fn(&ImageRegistration) -> (i32, i32)) -> Vec<ImageRegistration> {
    images.windows(9)
        .filter_map(|window| {
            let points: Vec<_> = window.iter()
                .map(|p| { let (a,b) = offset_fn(p); (a as f32, b as f32) })
                .collect();
            let p = points[4];
            let (m1, n1): (f32, f32) = linreg::linear_regression_of(&points).unwrap();

            // calculate intersecting line
            // perpendicular => m2 = -1 / m1
            // y = m1 * x + n1
            // y = m2 * x + n2
            // => m1 * x + n1 = m2 * x + n2
            // => n1 - n2 = x * (m2 - m1)
            // => x = (n1 / n2) / (m2 - m1)
            let m2 = -1. / m1;
            let n2 = p.1 - m2 * p.0;
            let x1 = (n1 - n2) / (m2 - m1);
            let y1 = m1 * x1 + n1;
            let dx = (x1 - p.0).abs();
            let dy = (y1 - p.1).abs();
            let percentage = (dx + dy) / (width + height) as f32;
            if percentage >= threshold {
                None
            } else {
                Some(window[4].clone())
            }
        }).collect()
}

fn width_height(images: &[ImageRegistration], threshold: f32, reference: &ImageRegistration) -> Vec<ImageRegistration> {
    images.into_iter().cloned().filter(|i| {
        let dwidth = (i.sod.width() as f32 / reference.sod.width() as f32 - 1.).abs();
        let dheight = (i.sod.height() as f32 / reference.sod.height() as f32 - 1.).abs();
        dwidth >= threshold || dheight >= threshold
    }).collect()
}
