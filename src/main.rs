use std::path::PathBuf;
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb};
use image::io::Reader;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use clap::{builder::ValueParser, Parser, ValueEnum};

mod postprocessing;
mod colorspace;

#[derive(Parser)]
struct Args {
    #[arg(short = 'i', long)]
    imagepaths: Vec<PathBuf>,
    #[arg(short = 'c', long, value_enum)]
    colorspace: Colorspace,
    #[arg(short = 'p', long, value_parser=ValueParser::new(parse_postprocessing), value_delimiter=',')]
    postprocessing: Vec<Postprocessing>,
    #[arg(short = 'n', default_value_t = 100)]
    num_files: usize,
    #[arg(short = 's', default_value_t = 0)]
    skip_files: usize,
    #[arg(short = 'o', long, default_value = "stacked.png")]
    outfile: PathBuf,
}
#[derive(Clone, ValueEnum)]
enum Colorspace {
    Srgb,
    Linear,
    Quadratic,
    Sqrt,
}
#[derive(Clone)]
enum Postprocessing {
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
}

fn main() {
    let args: Args = Args::parse();

    let files: Vec<_> = args.imagepaths.into_iter()
        .flat_map(|path| path.read_dir().unwrap())
        .skip(args.skip_files)
        .take(args.num_files)
        .map(|entry| entry.unwrap())
        .map(|entry| entry.path())
        .collect();
    let (width, height) = Reader::open(&files[0]).unwrap().decode().unwrap().dimensions();

    let num_files = files.len();

    let mut res = files.into_par_iter()
        .map(|path| Reader::open(path).unwrap().decode().unwrap().into_rgb64f())
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

    for postprocess in args.postprocessing {
        match postprocess {
            Postprocessing::Average => postprocessing::average(&mut res, num_files),
            Postprocessing::Maxscale => postprocessing::maxscale(&mut res),
            Postprocessing::Sqrt => postprocessing::sqrt(&mut res),
            Postprocessing::Asinh => postprocessing::asinh(&mut res),
            Postprocessing::Akaze(threshold) => postprocessing::akaze_draw(&mut res, threshold),
            Postprocessing::Sobel(blur) => postprocessing::sobel(&mut res, blur),
            Postprocessing::Blur(sigma) => postprocessing::gaussian_blur(&mut res, sigma),
        }
    }

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
        ($default:expr) => {
            value.map(|s| s.parse()).unwrap_or(Ok($default)).map_err(|e| format!("{e}"))?
        }
    }
    match typ {
        "average" => Ok(Postprocessing::Average),
        "maxscale" => Ok(Postprocessing::Maxscale),
        "sqrt" => Ok(Postprocessing::Sqrt),
        "asinh" => Ok(Postprocessing::Asinh),
        "akaze" => Ok(Postprocessing::Akaze(value!(0.0008))),
        "sobel" => Ok(Postprocessing::Sobel(value!(1))),
        "blur" => Ok(Postprocessing::Blur(value!(1.0))),
        _ => Err(format!("unknown postprocessing `{typ}`, allowed values are `average`, `maxscale`, `sqrt`, `asinh`, `akaze=0.008`, `sobel=1`, `blur=1.0."))
    }
}
