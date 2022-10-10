use std::path::Path;
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb, RgbImage};
use image::io::Reader;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use clap::{Parser, ValueEnum};
use image::buffer::ConvertBuffer;
// use cv::feature::akaze::Akaze;

#[derive(Parser)]
struct Args {
    #[arg(short = 'c', long, value_enum)]
    colorspace: Colorspace,
    #[arg(short = 'p', long, value_enum, value_delimiter=',')]
    postprocessing: Vec<Postprocessing>,
    #[arg(short = 'n', default_value_t = 100)]
    num_files: usize,
}
#[derive(Clone, ValueEnum)]
enum Colorspace {
    Srgb,
    Linear,
    Quadratic,
    Sqrt,
}
#[derive(Clone, ValueEnum)]
enum Postprocessing {
    Average,
    Maxscale,
    Sqrt,
    Asinh,
}

type Rgb64FImage = ImageBuffer<Rgb<f64>, Vec<f64>>;

fn linear(srgb: f64) -> f64 {
    if srgb <= 0.04045 {
        srgb / 12.92
    } else {
        ((srgb + 0.055) / 1.055).powf(2.4)
    }
}
fn srgb(linear: f64) -> f64 {
    if linear <= 0.0031308 {
        linear * 12.92
    } else {
        1.055 * linear.powf(1.0/2.4) - 0.055
    }
}
impl Colorspace {
    fn convert_into(&self, px: Rgb<f64>) -> Rgb<f64> {
        match self {
            Colorspace::Srgb => px,
            Colorspace::Linear => {
                let [r, g, b] = px.0;
                Rgb::from([linear(r), linear(g), linear(b)])
            }
            Colorspace::Quadratic => {
                let [r, g, b] = px.0;
                Rgb::from([r.powi(2), g.powi(2), b.powi(2)])
            }
            Colorspace::Sqrt => {
                let [r, g, b] = px.0;
                Rgb::from([r.sqrt(), g.sqrt(), b.sqrt()])
            }
        }
    }
    fn convert_back(&self, px: Rgb<f64>) -> Rgb<f64> {
        match self {
            Colorspace::Srgb => px,
            Colorspace::Linear => {
                let [r, g, b] = px.0;
                Rgb::from([srgb(r), srgb(g), srgb(b)])
            }
            Colorspace::Quadratic => {
                let [r, g, b] = px.0;
                Rgb::from([r.sqrt(), g.sqrt(), b.sqrt()])
            }
            Colorspace::Sqrt => {
                let [r, g, b] = px.0;
                Rgb::from([r.powi(2), g.powi(2), b.powi(2)])
            }
        }
    }
}

fn main() {
    let args: Args = Args::parse();

    let paths = [
        "/Astrophotos/2022-10-03/jupiterpipp/pipp_20221004_173153/MVI_6825",
        "/Astrophotos/2022-10-03/jupiterpipp/pipp_20221004_173153/MVI_6826",
    ];

    let files: Vec<_> = paths.into_iter()
        .flat_map(|path| Path::new(path).read_dir().unwrap())
        .map(|entry| entry.unwrap())
        .map(|entry| entry.path())
        .take(args.num_files)
        .collect();
    let (width, height) = Reader::open(&files[0]).unwrap().decode().unwrap().dimensions();

    // let detector = Akaze::sparse();
    // let (keypoints1, descriptors1) = detector.extract_path(&files[0]).unwrap();
    // let (keypoints2, descriptors2) = detector.extract_path(&files[0]).unwrap();
    //
    // panic!("mi done");

    let num_files = files.len();

    let mut res = files.into_par_iter()
        .map(|path| {
            let rgb8 = Reader::open(path).unwrap().decode().unwrap().into_rgb8();
            let mut rgb64f = Rgb64FImage::new(rgb8.width(), rgb8.height());
            for (to, from) in rgb64f.pixels_mut().zip(rgb8.pixels()) {
                to.0[0] = from.0[0] as f64 / 256.0;
                to.0[1] = from.0[1] as f64 / 256.0;
                to.0[2] = from.0[2] as f64 / 256.0;
            }
            rgb64f
        })
        .reduce(|| ImageBuffer::new(width, height), |mut buf: ImageBuffer<Rgb<f64>, _>, image| {
            for (x, y, pixel) in buf.enumerate_pixels_mut() {
                let px = image.get_pixel(x, y).clone();
                let px = args.colorspace.convert_into(px);
                pixel.0[0] += px.0[0];
                pixel.0[1] += px.0[1];
                pixel.0[2] += px.0[2];
            }
            buf
        });

    for pixel in res.pixels_mut() {
        *pixel = args.colorspace.convert_back(*pixel);
    }

    for postprocess in args.postprocessing {
        match postprocess {
            Postprocessing::Average => average(&mut res, num_files),
            Postprocessing::Maxscale => maxscale(&mut res),
            Postprocessing::Sqrt => sqrt(&mut res),
            Postprocessing::Asinh => asinh(&mut res),
        }
    }

    let mut rgb8 = RgbImage::new(res.width(), res.height());
    for (to, from) in rgb8.pixels_mut().zip(res.pixels()) {
        to.0[0] = (from.0[0] * 256.0) as u8;
        to.0[1] = (from.0[1] * 256.0) as u8;
        to.0[2] = (from.0[2] * 256.0) as u8;
    }
    rgb8.save("stacked.png").unwrap();

    // let mut f = File::open("/Astrophotos/2022-10-03/jupiterpipp/lights.fit").unwrap();
    // // let br = BufReader::with_capacity(1*1024*1024, f);
    // let mut buf = vec![0; 1*1024*1024];
    // let time = Instant::now();
    // let mut read_total = 0;
    // while let Ok(read) = f.read(&mut buf) {
    //     if read == 0 {
    //         break;
    //     }
    //     println!("read {}", read);
    //     read_total += read;
    //     if read_total > 10*1024*1024*1024 {
    //         break;
    //     }
    // }
    // let time = time.elapsed();
    // let mbps = read_total as f64 / 1024. / 1024. / (time.as_millis() as f64 / 1000.);
    // println!("{mbps:.1} MB/s");
}

fn average(buf: &mut Rgb64FImage, num_files: usize) {
    for pixel in buf.pixels_mut() {
        pixel.0[0] /= num_files as f64;
        pixel.0[1] /= num_files as f64;
        pixel.0[2] /= num_files as f64;
    }
}

fn maxscale(buf: &mut Rgb64FImage) {
    let maxcol = buf.pixels()
        .fold(0_f64, |maxcol, px| {
            let maxcol = maxcol.max(px.0[0]);
            let maxcol = maxcol.max(px.0[1]);
            let maxcol = maxcol.max(px.0[2]);
            maxcol
        });
    for pixel in buf.pixels_mut() {
        pixel.0[0] /= maxcol;
        pixel.0[1] /= maxcol;
        pixel.0[2] /= maxcol;
    }
}

fn sqrt(buf: &mut Rgb64FImage) {
    for pixel in buf.pixels_mut() {
        pixel.0[0] = pixel.0[0].sqrt();
        pixel.0[1] = pixel.0[1].sqrt();
        pixel.0[2] = pixel.0[2].sqrt();
    }
}
fn asinh(buf: &mut Rgb64FImage) {
    for pixel in buf.pixels_mut() {
        pixel.0[0] = pixel.0[0].asinh();
        pixel.0[1] = pixel.0[1].asinh();
        pixel.0[2] = pixel.0[2].asinh();
    }
}
