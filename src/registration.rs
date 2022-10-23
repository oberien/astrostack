use std::f32::consts::PI;
use cv::feature::akaze::KeyPoint;
use image::{DynamicImage, GenericImage, ImageBuffer, Rgb, Rgb64FImage};
use itertools::Itertools;
use plotters::backend::BitMapBackend;
use plotters::chart::ChartBuilder;
use plotters::drawing::IntoDrawingArea;
use plotters::series::Histogram;
use plotters::style::{Color, RED, WHITE};
use crate::helpers::Object;
use crate::{helpers, Postprocessing};

#[derive(Debug, Copy, Clone)]
struct Match {
    left: (f32, f32),
    right: (f32, f32),
    dx: f32,
    dy: f32,
    arc: f32,
    arcdeg: i32,
}

impl Match {
    pub fn new(left: (f32, f32), right: (f32, f32)) -> Match {
        let dx = left.0 - right.0;
        let dy = left.1 - right.1;
        let arc = (dy / dx).atan();
        Match {
            left, right, arc, dx, dy,
            arcdeg: (arc / PI * 180.).round() as i32,
        }
    }
}

pub struct ReferenceImage(Rgb64FImage);

pub fn prepare_reference_image(mut reference_image: Rgb64FImage, num_files: usize, processing: &[Postprocessing]) -> ReferenceImage {
    if processing.is_empty() {
        return ReferenceImage(reference_image);
    }
    crate::postprocessing::process(&mut reference_image, num_files, &processing[..processing.len() - 1]);
    ReferenceImage(reference_image)
}

pub fn register(reference: &ReferenceImage, mut img: Rgb64FImage, num_files: usize, mut processing: &[Postprocessing], debug: bool) -> Option<(i32, i32)> {
    let (registration_function, threshold): (fn(_, _, _, _) -> Vec<Match>, _) = match processing.last() {
        None => return Some((0, 0)),
        Some(&Postprocessing::Akaze(threshold)) => (akaze, threshold),
        Some(&Postprocessing::SingleObjectDetection(threshold)) => (single_object_detection, threshold),
        _ => unreachable!(),
    };
    processing = &processing[..processing.len()-1];

    crate::postprocessing::process(&mut img, num_files, &processing);

    assert_eq!(reference.0.width(), img.width());
    assert_eq!(reference.0.height(), img.height());
    let mut res = ImageBuffer::new(reference.0.width() * 2, reference.0.height());
    res.copy_from(&reference.0, 0, 0).unwrap();
    res.copy_from(&img, reference.0.width(), 0).unwrap();

    let mut matches = registration_function(reference, &img, threshold, &mut res);
    matches.sort_by(|m1, m2| m1.arc.total_cmp(&m2.arc));

    if matches.is_empty() {
        return None;
    }

    // reject everything deviating >5° from the median
    let median_arcdeg = matches[matches.len() / 2].arcdeg;
    matches.retain(|m| (median_arcdeg - m.arcdeg).abs() <= 5);

    if debug {
        DynamicImage::ImageRgb64F(res).into_rgb16().save("registration.png").unwrap();

        let max_frequency = matches.iter().dedup_by_with_count(|m1, m2| m1.arcdeg == m2.arcdeg)
            .map(|(count, _arc)| count)
            .max().unwrap();

        let root = BitMapBackend::new("arc-histogram.png", (1920, 1080)).into_drawing_area();
        root.fill(&WHITE).unwrap();
        let mut chart = ChartBuilder::on(&root)
            .x_label_area_size(100)
            .y_label_area_size(100)
            .margin(5)
            .caption("Arc Histogram", ("sans-serif", 100.0))
            .build_cartesian_2d(-180i32..180i32, 0u32..max_frequency as u32 + 1).unwrap();

        chart.configure_mesh()
            .disable_x_mesh()
            .bold_line_style(&WHITE.mix(0.3))
            .y_desc("Count")
            .x_desc("Arc in deg")
            .axis_desc_style(("sans-serif", 50))
            .label_style(("sans-serif", 50))
            .draw().unwrap();

        chart.draw_series(
            Histogram::vertical(&chart)
                .style(RED.filled())
                .data(matches.iter().map(|m| (m.arcdeg, 1)))
        ).unwrap();
        root.present().unwrap();
    }

    // average all resulting offsets
    let (dx, dy) = matches.iter().fold((0., 0.), |(dx, dy), m| (dx+m.dx, dy+m.dy));
    Some((
        (dx / matches.len() as f32).round() as i32,
        (dy / matches.len() as f32).round() as i32,
    ))
}

fn akaze(reference: &ReferenceImage, img: &Rgb64FImage, threshold: f64, res: &mut Rgb64FImage) -> Vec<Match> {
    let width = reference.0.width();
    let height = reference.0.height();

    let (kps1, desc1) = crate::helpers::akaze(&reference.0, threshold);
    let (kps2, desc2) = crate::helpers::akaze(img, threshold);
    const THRESH: f32 = 0.8;

    let mut matches = Vec::new();
    for (i1, desc1) in desc1.iter().enumerate() {
        let mut vice = u32::MAX;
        let mut best = (u32::MAX, 0);
        for (i2, desc2) in desc2.iter().enumerate() {
            let dist = desc1.distance(desc2);
            if dist < best.0 {
                vice = best.0;
                best = (dist, i2);
            }
        }
        // only keep matches which are 20% apart from the next-best match
        if (best.0 as f32) < vice as f32 * THRESH {
            matches.push((kps1[i1], kps2[best.1]));
        }
    }

    // reject all matches that are further apart than 5% of the image size
    matches.retain(|(kp1, kp2)| {
        let dx = kp1.point.0 - kp2.point.0;
        let dy = kp1.point.1 - kp2.point.1;
        let dist = (dx*dx + dy*dy).sqrt();
        dist < (width + height) as f32 / 2. / 20.
    });

    for &kp in &kps1 {
        crate::helpers::akaze_draw_kp(res, kp);
    }
    for &kp2 in &kps2 {
        let kp2 = KeyPoint { point: (kp2.point.0 + width as f32, kp2.point.1), ..kp2 };
        crate::helpers::akaze_draw_kp(res, kp2);
    }
    for &(kp1, kp2) in &matches {
        let kp2 = KeyPoint { point: (kp2.point.0 + width as f32, kp2.point.1), ..kp2 };
        imageproc::drawing::draw_line_segment_mut(res, kp1.point, kp2.point, Rgb([1.,0.,0.]));
    }

    matches.into_iter().map(|(kp1, kp2)| Match::new(kp1.point, kp2.point)).collect()
}

fn single_object_detection(reference: &ReferenceImage, img: &Rgb64FImage, threshold: f64, res: &mut Rgb64FImage) -> Vec<Match> {
    let o1 = helpers::single_object_detection(&reference.0, threshold);
    let o2 = helpers::single_object_detection(img, threshold);
    // reject everything with more than 2% diff from the reference image
    let dwidth = (o1.width as f32 / o2.width as f32 - 1.).abs();
    let dheight = (o1.height as f32 / o2.height as f32 - 1.).abs();
    if dwidth > 0.02 || dheight > 0.02 {
        return Vec::new();
    }

    let o2right = Object {
        left: o2.left + reference.0.width(),
        right: o2.right + reference.0.width(),
        top: o2.top,
        bottom: o2.bottom,
        middle: (o2.middle.0 + reference.0.width(), o2.middle.1),
        width: o2.width,
        height: o2.height,
    };
    helpers::draw_object(res, o1);
    helpers::draw_object(res, o2right);

    vec![Match::new(
        (o1.middle.0 as f32, o1.middle.1 as f32),
        (o2.middle.0 as f32, o2.middle.1 as f32),
    )]
}
