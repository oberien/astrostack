use std::path::PathBuf;
use cv::feature::akaze::KeyPoint;
use image::{GenericImage, Rgb, Rgb64FImage};
use itertools::Itertools;
use plotters::backend::BitMapBackend;
use plotters::chart::ChartBuilder;
use plotters::drawing::IntoDrawingArea;
use plotters::series::Histogram;
use plotters::style::{Color, RED, WHITE};
use crate::{CommonArgs, Compare, helpers, processing, register};
use crate::register::{AkazeData, Match, SodRegistration};

pub fn compare(common: CommonArgs, compare: Compare) {
    let CommonArgs { colorspace, num_files, skip_files: _ } = common;
    let Compare { first, second, preprocessing_akaze, preprocessing_rest, akaze, single_object_detection, average_brightness_alignment, outfile_prefix } = compare;

    let first = helpers::load_image(first, colorspace);
    let second = helpers::load_image(second, colorspace);
    assert_eq!(first.width(), second.width());
    assert_eq!(first.height(), second.height());

    let algs: &[(_, for<'a, 'b, 'c, 'd> fn (&'a _, &'b _, _, &'c mut _, &'d _) -> _, _, &[_])] = &[
        ("akaze", self::akaze, akaze, &preprocessing_akaze),
        ("sod", self::single_object_detection, single_object_detection, &preprocessing_rest),
        ("aba", self::average_brightness_alignment, average_brightness_alignment, &preprocessing_rest),
        ("orig", noop, 0.0, &[]),
    ];

    for &(name, register, threshold, preprocessing) in algs {
        let mut first = first.clone();
        processing::process(&mut first, num_files, preprocessing);
        let mut second = second.clone();
        processing::process(&mut second, num_files, preprocessing);

        let mut res = Rgb64FImage::new(first.width() * 2, first.height());
        res.copy_from(&first, 0, 0).unwrap();
        res.copy_from(&second, first.width(), 0).unwrap();

        register(&first, &second, threshold, &mut res, &outfile_prefix);
        let outfile = helpers::path_with_suffix(&outfile_prefix, &format!("{name}.png"));
        helpers::save_image(res, outfile, colorspace);
    }
}

fn akaze(left: &Rgb64FImage, right: &Rgb64FImage, threshold: f64, res: &mut Rgb64FImage, outfile_prefix: &PathBuf) {
    let width = left.width();
    let height = left.height();
    let data1 = register::akaze(left, threshold);
    let data2 = register::akaze(right, threshold);
    let matches_unrejected = data1.matches_unrejected(&data2);

    // arc statistics

    let outfile = helpers::path_with_suffix(outfile_prefix, "akaze_arcs.png");
    let max_frequency = matches_unrejected.iter().dedup_by_with_count(|m1, m2| m1.arcdeg() == m2.arcdeg())
        .map(|(count, _arc)| count)
        .max().unwrap();
    dbg!(max_frequency);

    let root = BitMapBackend::new(&outfile, (1920, 1080)).into_drawing_area();
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
            .data(matches_unrejected.iter().map(|m| (m.arcdeg(), 1)))
    ).unwrap();
    root.present().unwrap();

    // render into result image

    for &kp in &data1.keypoints {
        helpers::akaze_draw_kp(res, kp);
    }
    for &kp2 in &data2.keypoints {
        let kp2 = KeyPoint { point: (kp2.point.0 + width as f32, kp2.point.1), ..kp2 };
        helpers::akaze_draw_kp(res, kp2);
    }
    for &Match { left, right, .. } in &matches_unrejected {
        let right = (right.0 + width as f32, right.1);
        imageproc::drawing::draw_line_segment_mut(res, left, right, Rgb([1.,0.,0.]));
    }

    let mut matches = matches_unrejected;
    AkazeData::reject_matches(&mut matches, width, height);

    for &Match { left, right, .. } in &matches {
        let right = (right.0 + width as f32, right.1);
        imageproc::drawing::draw_line_segment_mut(res, left, right, Rgb([0.,1.,0.]));
    }

}

fn single_object_detection(left: &Rgb64FImage, right: &Rgb64FImage, threshold: f64, res: &mut Rgb64FImage, _outfile_prefix: &PathBuf) {
    let o1 = register::single_object_detection(left, threshold);
    let o2 = register::single_object_detection(right, threshold);

    let o2right = SodRegistration {
        left: o2.left + left.width(),
        right: o2.right + left.width(),
        top: o2.top,
        bottom: o2.bottom,
    };
    helpers::draw_object(res, o1);
    helpers::draw_object(res, o2right);
    imageproc::drawing::draw_line_segment_mut(res, (o1.middle().0 as f32, o1.middle().1 as f32), (o2right.middle().0 as f32, o2right.middle().1 as f32), Rgb([1., 0., 0.]));
}

fn average_brightness_alignment(left: &Rgb64FImage, right: &Rgb64FImage, threshold: f64, res: &mut Rgb64FImage, _outfile_prefix: &PathBuf) {
    let width = left.width() as f32;
    let left = register::average_brightness(left, threshold);
    let right = register::average_brightness(right, threshold);

    helpers::draw_cross(res, (left.middlex, left.middley));
    helpers::draw_cross(res, ((right.middlex + width), right.middley));
    imageproc::drawing::draw_line_segment_mut(res, (left.middlex, left.middley), (right.middlex + width, right.middley), Rgb([1., 0., 0.]));
}

fn noop(_left: &Rgb64FImage, _right: &Rgb64FImage, _threshold: f64, _res: &mut Rgb64FImage, _outfile_prefix: &PathBuf) {}
