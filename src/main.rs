use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use image::{DynamicImage, Rgb64FImage};
use image::io::Reader;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use clap::{builder::ValueParser, Parser, ValueEnum, error::ErrorKind, CommandFactory};
use either::Either;

mod helpers;
mod registration;
mod postprocessing;
mod colorspace;
mod stack;

#[derive(Debug, Parser)]
struct Args {
    #[arg(short = 'i', long)]
    imagepaths: Vec<PathBuf>,
    #[arg(short = 'c', long, value_enum)]
    colorspace: Colorspace,
    /// Processing chain for alignment registration; must end with `akaze` or `sod`
    #[arg(short = 'r', long, value_parser=ValueParser::new(parse_postprocessing), value_delimiter=',')]
    registration: Vec<Postprocessing>,
    #[arg(short = 'p', long, value_parser=ValueParser::new(parse_postprocessing), value_delimiter=',')]
    postprocessing: Vec<Postprocessing>,
    #[arg(short = 'n', default_value_t = 100)]
    num_files: usize,
    #[arg(short = 's', default_value_t = 0)]
    skip_files: usize,
    #[arg(short = 'o', long, default_value = "stacked.png")]
    outfile: PathBuf,
    #[arg(short = 'd', long)]
    debug: bool,
}
#[derive(Debug, Copy, Clone, ValueEnum)]
pub enum Colorspace {
    Srgb,
    Linear,
    Quadratic,
    Sqrt,
}
#[derive(Debug, Copy, Clone)]
pub enum Postprocessing {
    Average,
    Maxscale,
    Sqrt,
    Asinh,
    /// akaze feature detection with passed threshold, 0.0008 by default
    Akaze(f64),
    /// sobel edgeg detection with passed blur, 1 by default
    Sobel(i32),
    /// gaussian blur with passed sigma, 1.0 by default
    Blur(f32),
    /// bg extraction using the given threshold to make pixels black (0.2)
    BGone(f64),
    /// convert the image to a black-white image using the given threshold (0.5)
    BlackWhite(f64),
    /// single object detection with the given threshold (0.2)
    SingleObjectDetection(f64),
}

fn main() {
    let args: Args = Args::parse();
    // validate arguments
    match args.registration.last() {
        Some(Postprocessing::Akaze(_)) => (),
        Some(Postprocessing::SingleObjectDetection(_)) => (),
        None => (),
        _ => Args::command().error(
            ErrorKind::InvalidValue,
            "last registration processing chain must be `akaze`",
        ).exit(),
    }
    dbg!(&args);

    let mut files: Vec<_> = args.imagepaths.into_iter()
        .flat_map(|path| {
            if path.is_dir() {
                Either::Left(path.read_dir().unwrap().map(|entry| entry.unwrap().path()))
            } else if path.is_file() {
                Either::Right([path].into_iter())
            } else {
                panic!("input path {} is neither directory nor file", path.display())
            }
        })
        .skip(args.skip_files)
        .take(args.num_files)
        .collect();
    files.sort();
    let first = Reader::open(&files[0]).unwrap().decode().unwrap().into_rgb64f();
    let (width, height) = first.dimensions();
    let num_files = files.len();
    let reference_image = registration::prepare_reference_image(first, num_files, &args.registration);

    let counter = AtomicU32::new(0);
    let mut res = files.into_par_iter()
        .map(|path| Reader::open(path).unwrap().decode().unwrap().into_rgb64f())
        .fold(|| Rgb64FImage::new(width*3, height*3), |mut buf, image| {
            let count = counter.fetch_add(1, Ordering::Relaxed);
            if count % 50 == 0 {
                println!("{count}");
            }
            let (dx, dy) = registration::register(
                &reference_image,
                image.clone(),
                num_files,
                &args.registration,
                args.debug,
            );
            if args.debug {
                println!("{dx}:{dy}");
            }
            stack::stack_into(&mut buf, &image, dx + width as i32, dy + height as i32, args.colorspace);
            buf
        }).reduce(|| Rgb64FImage::new(width*3, height*3), |mut buf, image| {
            stack::stack_into(&mut buf, &image, 0, 0, Colorspace::Srgb);
            buf
        });

    postprocessing::process(&mut res, num_files, &args.postprocessing);

    for pixel in res.pixels_mut() {
        *pixel = args.colorspace.convert_back(*pixel);
    }

    DynamicImage::ImageRgb64F(res).into_rgb16().save(args.outfile).unwrap();
}

fn parse_postprocessing(p: &str) -> Result<Postprocessing, String> {
    let mut parts = p.split("=");
    let typ = parts.next().unwrap();
    let value = parts.next();
    macro_rules! value {
        ($value:expr, $default:expr) => {
            $value.map(|s| s.parse()).unwrap_or(Ok($default)).map_err(|e| format!("{e}"))?
        }
    }
    match typ {
        "average" => Ok(Postprocessing::Average),
        "maxscale" => Ok(Postprocessing::Maxscale),
        "sqrt" => Ok(Postprocessing::Sqrt),
        "asinh" => Ok(Postprocessing::Asinh),
        "akaze" => Ok(Postprocessing::Akaze(value!(value, 0.0008))),
        "sobel" => Ok(Postprocessing::Sobel(value!(value, 1))),
        "blur" => Ok(Postprocessing::Blur(value!(value, 1.0))),
        "bgone" => Ok(Postprocessing::BGone(value!(value, 0.2))),
        "bw" => Ok(Postprocessing::BlackWhite(value!(value, 0.5))),
        "sod" => Ok(Postprocessing::SingleObjectDetection(value!(value, 0.2))),
        _ => Err(format!("unknown postprocessing `{typ}`, allowed values are `average`, `maxscale`, `sqrt`, `asinh`, `akaze=0.008`, `sobel=1`, `blur=1.0."))
    }
}
