use std::f32::consts::PI;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use cv::bitarray::BitArray;
use cv::feature::akaze::{Akaze, KeyPoint};
use either::Either;
use image::{DynamicImage, ImageBuffer, Luma, Rgb64FImage};
use image::buffer::ConvertBuffer;
use plotters::backend::BitMapBackend;
use plotters::chart::ChartBuilder;
use plotters::drawing::IntoDrawingArea;
use plotters::element::Circle;
use plotters::style::{BLUE, Color, GREEN, RED, WHITE};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Serialize, Deserialize};
use crate::{CommonArgs, helpers, processing, Register};

pub fn register(common: CommonArgs, register: Register) {
    let CommonArgs { colorspace, num_files, skip_files } = common;
    let Register { imagepaths, reference_image, preprocessing_akaze, preprocessing_rest, outfile, akaze, single_object_detection, average_brightness_alignment } = register;

    let mut files: Vec<_> = imagepaths.into_iter()
        .flat_map(|path| {
            if path.is_dir() {
                Either::Left(path.read_dir().unwrap().map(|entry| entry.unwrap().path()))
            } else if path.is_file() {
                Either::Right([path].into_iter())
            } else {
                panic!("input path {} is neither directory nor file", path.display())
            }
        }).collect();
    files.sort_by_key(|path| path.file_name().unwrap().to_owned());
    let files: Vec<_> = files.into_iter()
        .skip(skip_files)
        .take(num_files)
        .collect();

    // akaze reference image
    let mut reference_image_akaze = helpers::load_image(&files[reference_image], colorspace);
    processing::process(&mut reference_image_akaze, num_files, &preprocessing_akaze);
    let reference_akaze_data = akaze.map(|akaze| (akaze, self::akaze(&reference_image_akaze, akaze)));

    let counter = AtomicU32::new(0);
    let image_registrations: Vec<_> = files.into_par_iter()
        .map(|path| (helpers::load_image(&path, colorspace), path))
        .map(|(image, path)| {
            let count = counter.fetch_add(1, Ordering::Relaxed);
            if count % 50 == 0 {
                println!("{count}");
            }
            let akaze = reference_akaze_data.as_ref().map(|(akaze, reference_akaze_data)| {
                let mut preprocessed = image.clone();
                processing::process(&mut preprocessed, num_files, &preprocessing_akaze);
                let akaze_data = self::akaze(&preprocessed, *akaze);
                akaze_data.akaze_registration(reference_akaze_data, reference_image_akaze.width(), reference_image_akaze.height())
            });
            let mut preprocessed = image;
            processing::process(&mut preprocessed, num_files, &preprocessing_rest);
            let (sod, aba) = sod_aba(&preprocessed, single_object_detection, average_brightness_alignment);
            ImageRegistration {
                image: path,
                akaze,
                sod,
                aba,
            }
        }).collect();

    let reg = Registration {
        reference_image,
        images: image_registrations,
    };
    helpers::save_registration(outfile, &reg);
    statistics(&reg);
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Registration {
    pub reference_image: usize,
    pub images: Vec<ImageRegistration>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageRegistration {
    pub image: PathBuf,
    pub akaze: Option<AkazeRegistration>,
    pub sod: SodRegistration,
    pub aba: AbaRegistration,
}
impl ImageRegistration {
    pub fn offsets(&self, reference: &ImageRegistration) -> ((i32, i32), (i32, i32), (i32, i32)) {
        (
            self.akaze.map(|a| a.offset()).unwrap_or_default(),
            self.sod.offset(&reference.sod),
            self.aba.offset(&reference.aba),
        )
    }
}
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub enum AkazeRegistration {
    Offset(f32, f32),
    Rejected,
}
impl AkazeRegistration {
    pub fn offset(&self) -> (i32, i32) {
        match self {
            AkazeRegistration::Rejected => (0, 0),
            AkazeRegistration::Offset(dx, dy) => (dx.round() as i32, dy.round() as i32)
        }
    }
}
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct SodRegistration {
    pub left: u32,
    pub right: u32,
    pub top: u32,
    pub bottom: u32,
}
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct AbaRegistration {
    pub middlex: f32,
    pub middley: f32,
}
impl SodRegistration {
    pub fn middle(&self) -> (u32, u32) {
        let middlex = self.left + (self.right - self.left) / 2;
        let middley = self.top + (self.bottom - self.top) / 2;
        (middlex, middley)
    }
    pub fn width(&self) -> u32 {
        self.right - self.left
    }
    pub fn height(&self) -> u32 {
        self.bottom - self.top
    }
    pub fn offset(&self, reference: &SodRegistration) -> (i32, i32) {
        let (x1, y1) = reference.middle();
        let (x2, y2) = self.middle();
        (x1 as i32 - x2 as i32, y1 as i32 - y2 as i32)
    }
}
impl AbaRegistration {
    pub fn offset(&self, reference: &AbaRegistration) -> (i32, i32) {
        let AbaRegistration { middlex: x1, middley: y1 } = reference;
        let AbaRegistration { middlex: x2, middley: y2 } = self;
        (x1.round() as i32 - x2.round() as i32, y1.round() as i32 - y2.round() as i32)
    }
}

fn statistics(reg: &Registration) {
    let reference = &reg.images[reg.reference_image];
    let (maxabsx, maxabsy) = reg.images.iter()
        .fold((i32::MIN, i32::MIN), |(maxabsx, maxabsy), reg| {
            let ((dx1, dy1), (dx2, dy2), (dx3, dy3)) = reg.offsets(reference);
            (
                maxabsx.max(dx1.abs()).max(dx2.abs()).max(dx3.abs()),
                maxabsy.max(dy1.abs()).max(dy2.abs()).max(dy3.abs()),
            )
        });

    let root = BitMapBackend::new("registration-scatter.png", (1920, 1080)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let mut scatter_ctx = ChartBuilder::on(&root)
        .x_label_area_size(60)
        .y_label_area_size(60)
        .build_cartesian_2d(-maxabsx as f32 -5.0..maxabsx as f32+5.0, maxabsy as f32+5.0..-maxabsy as f32-5.0).unwrap();
    scatter_ctx
        .configure_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .draw().unwrap();
    scatter_ctx.draw_series(
        reg.images.iter()
            .flat_map(|reg| {
                let (a, b, c) = reg.offsets(reference);
                [
                    ((a.0 as f32, a.1 as f32), RED),
                    ((b.0 as f32 + 0.3, b.1 as f32), GREEN),
                    ((c.0 as f32, c.1 as f32 + 0.3), BLUE),
                ]
            }).map(|((x, y), col)| Circle::new((x, y), 2, col.filled())),
    ).unwrap();

    root.present().unwrap();
}

#[derive(Debug, Clone)]
pub struct AkazeData {
    pub keypoints: Vec<KeyPoint>,
    pub descriptions: Vec<BitArray<64>>,
}
#[derive(Debug, Copy, Clone)]
pub struct Match {
    pub left: (f32, f32),
    pub right: (f32, f32),
}
impl Match {
    pub fn dx(&self) -> f32 {
        self.left.0 - self.right.0
    }
    pub fn dy(&self) -> f32 {
        self.left.1 - self.right.1
    }
    pub fn arc(&self) -> f32 {
        (self.dy() / self.dx()).atan()
    }
    pub fn arcdeg(&self) -> i32 {
        (self.arc() / PI * 180.).round() as i32
    }
}

impl AkazeData {
    pub fn matches(&self, other: &AkazeData, width: u32, height: u32) -> Vec<Match> {
        let mut matches = self.matches_unrejected(other);
        Self::reject_matches(&mut matches, width, height);
        matches
    }
    pub fn matches_unrejected(&self, other: &AkazeData) -> Vec<Match> {
        const THRESH: f32 = 0.8;

        let mut matches = Vec::new();
        for (i1, desc1) in self.descriptions.iter().enumerate() {
            let mut vice = u32::MAX;
            let mut best = (u32::MAX, 0);
            for (i2, desc2) in other.descriptions.iter().enumerate() {
                let dist = desc1.distance(desc2);
                if dist < best.0 {
                    vice = best.0;
                    best = (dist, i2);
                }
            }
            // only keep matches which are 20% apart from the next-best match
            if (best.0 as f32) < vice as f32 * THRESH {
                let kp1 = self.keypoints[i1];
                let kp2 = other.keypoints[best.1];
                matches.push(Match {
                    left: kp1.point,
                    right: kp2.point,
                });
            }
        }
        matches
    }
    pub fn reject_matches(matches: &mut Vec<Match>, _width: u32, _height: u32) {
        // // reject all matches that are further apart than 5% of the image size
        // matches.retain(|Match { left, right }| {
        //     let dx = left.0 - right.0;
        //     let dy = left.1 - right.1;
        //     let dist = (dx*dx + dy*dy).sqrt();
        //     dist < (width + height) as f32 / 2. / 20.
        // });

        // reject everything deviating >5° from the median
        matches.sort_by(|m1, m2| m1.arc().total_cmp(&m2.arc()));
        let median_arcdeg = matches[matches.len() / 2].arcdeg();
        matches.retain(|m| (median_arcdeg - m.arcdeg()).abs() <= 5);
    }

    pub fn akaze_registration(&self, reference: &AkazeData, width: u32, height: u32) -> AkazeRegistration {
        let matches = reference.matches(self, width, height);
        if matches.is_empty() {
            return AkazeRegistration::Rejected;
        }

        // average all resulting offsets
        let matches_len = matches.len();
        let (dx, dy) = matches.into_iter().fold((0., 0.), |(dx, dy), m| (dx+m.dx(), dy+m.dy()));
        AkazeRegistration::Offset(dx / matches_len as f32, dy / matches_len as f32)
    }
}

pub fn akaze(buf: &Rgb64FImage, threshold: f64) -> AkazeData {
    let detector = Akaze::new(threshold);
    let luma16: ImageBuffer<Luma<u16>, Vec<u16>> = buf.convert();
    let (key_points, descriptions) = detector.extract(&DynamicImage::ImageLuma16(luma16));
    AkazeData { keypoints: key_points, descriptions }
}

pub fn sod_aba(buf: &Rgb64FImage, threshold_sod: f64, threshold_aba: f64) -> (SodRegistration, AbaRegistration) {
    let mut left = u32::MAX;
    let mut right = 0;
    let mut top = u32::MAX;
    let mut bottom = 0;
    let mut sum = 0.0;
    let mut weighted_sum_rows = 0.0;
    let mut weighted_sum_columns = 0.0;

    for (x, y, pixel) in buf.enumerate_pixels() {
        let value = pixel.0.into_iter().sum::<f64>() / 3.;
        // single object detection
        if value >= threshold_sod {
            left = x.min(left);
            right = x.max(right);
            top = y.min(top);
            bottom = y.max(bottom);
        }
        // average brightness alignment
        if value >= threshold_aba {
            sum += value;
            weighted_sum_rows += y as f64 * value;
            weighted_sum_columns += x as f64 * value;
        }
    }
    let sod = SodRegistration { left, right, top, bottom };

    let middlex = (weighted_sum_columns / sum) as f32;
    let middley = (weighted_sum_rows / sum) as f32;
    let aba = AbaRegistration { middlex, middley };

    (sod, aba)
}

pub fn single_object_detection(buf: &Rgb64FImage, threshold: f64) -> SodRegistration {
    sod_aba(buf, threshold, 1.0).0
}

pub fn average_brightness(buf: &Rgb64FImage, threshold: f64) -> AbaRegistration {
    sod_aba(buf, 1.0, threshold).1
}
