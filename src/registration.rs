use std::f32::consts::PI;
use std::path::PathBuf;
use cv::feature::akaze::KeyPoint;
use image::{DynamicImage, GenericImage, ImageBuffer, Rgb};
use image::io::Reader;
use itertools::Itertools;
use plotters::backend::BitMapBackend;
use plotters::chart::ChartBuilder;
use plotters::drawing::IntoDrawingArea;
use plotters::prelude::IntoSegmentedCoord;
use plotters::series::Histogram;
use plotters::style::{Color, RED, WHITE};
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
    for &(kp1, kp2) in &matches {
        let kp2 = KeyPoint { point: (kp2.point.0 + width as f32, kp2.point.1), ..kp2 };
        imageproc::drawing::draw_line_segment_mut(&mut res, kp1.point, kp2.point, Rgb([1.,0.,0.]));
    }
    DynamicImage::ImageRgb64F(res).into_rgb16().save("registration.png").unwrap();


    let mut arcs: Vec<_> = matches.iter().copied()
        .map(|(kp1, kp2)| {
            let dx = kp1.point.0 - kp2.point.0;
            let dy = kp1.point.1 - kp2.point.1;
            let arc = (dy / dx).atan() / PI * 180.;
            arc as i32
        }).collect();
    arcs.sort();
    let max_frequency = arcs.iter().dedup_with_count().map(|(count, _arc)| count).max().unwrap();

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
            .data(arcs.iter().copied().map(|arc| (arc, 1)))
    ).unwrap();
    root.present().unwrap();
}