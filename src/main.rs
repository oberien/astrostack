use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use image::{DynamicImage, Rgb64FImage};
use image::io::Reader;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use clap::{builder::ValueParser, Parser, ValueEnum, error::ErrorKind, CommandFactory};
use either::Either;
use plotters::backend::BitMapBackend;
use plotters::chart::ChartBuilder;
use plotters::drawing::IntoDrawingArea;
use plotters::element::Circle;
use plotters::style::{BLUE, Color, GREEN, RED, WHITE};
use rayon::slice::ParallelSlice;

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
    /// Processing chain for alignment registration; must end with `akaze`, `sod` or `aba`
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
    /// detect the pixel with the average brightness given threshold (0.2)
    AverageBrightnessAlignment(f64),
}

fn main() {
    let args: Args = Args::parse();
    // validate arguments
    match args.registration.last() {
        Some(Postprocessing::Akaze(_)) => (),
        Some(Postprocessing::SingleObjectDetection(_)) => (),
        Some(Postprocessing::AverageBrightnessAlignment(_)) => (),
        None => (),
        _ => Args::command().error(
            ErrorKind::InvalidValue,
            "last registration processing chain must be `akaze`, `sod` or `aba`",
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
        .collect();
    files.sort_by_key(|path| path.file_name().unwrap().to_owned());
    let files: Vec<_> = files.into_iter()
        .skip(args.skip_files)
        .take(args.num_files)
        .collect();

    let res = files.par_windows(2)
        .map(|paths| (
            Reader::open(&paths[0]).unwrap().decode().unwrap().into_rgb64f(),
            Reader::open(&paths[1]).unwrap().decode().unwrap().into_rgb64f()
        )).fold(|| Vec::new(), |mut diffs, (left, right)| {
            let processing_akaze = &[Postprocessing::Maxscale, Postprocessing::Blur(20.), Postprocessing::Sobel(0), Postprocessing::Maxscale, Postprocessing::Akaze(0.001)];
            let processing_sod = &[Postprocessing::Maxscale, Postprocessing::Blur(20.), Postprocessing::Maxscale, Postprocessing::SingleObjectDetection(0.2)];
            let processing_aba = &[Postprocessing::Maxscale, Postprocessing::Blur(20.), Postprocessing::Maxscale, Postprocessing::AverageBrightnessAlignment(0.2)];
            let ref_akaze = registration::prepare_reference_image(left.clone(), files.len(), processing_akaze);
            let ref_sod = registration::prepare_reference_image(left.clone(), files.len(), processing_sod);
            let ref_aba = registration::prepare_reference_image(left.clone(), files.len(), processing_aba);
            let akaze = registration::register(&ref_akaze, right.clone(), files.len(), processing_akaze, false).unwrap_or((0, 0));
            let sod = registration::register(&ref_sod, right.clone(), files.len(), processing_sod, false).unwrap_or((0, 0));
            let aba = registration::register(&ref_aba, right.clone(), files.len(), processing_aba, false).unwrap_or((0, 0));
            diffs.push((akaze, sod, aba));
            diffs
        }).reduce(|| Vec::new(), |mut a, b| {
            a.extend(b);
            a
        });

    let (maxabsx, maxabsy) = res.iter().copied()
        .fold((i32::MIN, i32::MIN), |(maxabsx, maxabsy), diff| {
            let ((dx1, dy1), (dx2, dy2), (dx3, dy3)) = diff;
            (
                maxabsx.max(dx1.abs()).max(dx2.abs()).max(dx3.abs()),
                maxabsy.max(dy1.abs()).max(dy2.abs()).max(dy3.abs()),
            )
        });

    let root = BitMapBackend::new("registration-scatter.png", (1920, 1080)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let mut scatter_ctx = ChartBuilder::on(&root)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(-maxabsx as f32 -1.0..maxabsx as f32+1.0, -maxabsy as f32-1.0..maxabsy as f32+1.0).unwrap();
    scatter_ctx
        .configure_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .draw().unwrap();
    scatter_ctx.draw_series(
        res.iter().copied()
            .flat_map(|(a, b, c)| [
                ((a.0 as f32, a.1 as f32), RED),
                ((b.0 as f32 + 0.3, b.1 as f32), GREEN),
                ((c.0 as f32, c.1 as f32 + 0.3), BLUE),
            ]).map(|((x, y), col)| Circle::new((x, y), 2, col.filled())),
    ).unwrap();

    root.present().unwrap();

    panic!("done");

    let first = Reader::open(&files[0]).unwrap().decode().unwrap().into_rgb64f();
    let (width, height) = first.dimensions();
    let num_files = files.len();
    let reference_image = registration::prepare_reference_image(first, num_files, &args.registration);

    println!("Starting stacking.");
    let counter = AtomicU32::new(0);
    let mut res = files.into_par_iter()
        .map(|path| Reader::open(path).unwrap().decode().unwrap().into_rgb64f())
        .fold(|| Rgb64FImage::new(width*3, height*3), |mut buf, image| {
            let count = counter.fetch_add(1, Ordering::Relaxed);
            if count % 50 == 0 {
                println!("{count}");
            }
            let delta = registration::register(
                &reference_image,
                image.clone(),
                num_files,
                &args.registration,
                args.debug,
            );
            let (dx, dy) = match delta {
                Some((dx, dy)) => (dx, dy),
                None => {
                    if args.debug {
                        println!("rejected");
                    }
                    return buf
                },
            };
            if args.debug {
                println!("{dx}:{dy}");
            }
            stack::stack_into(&mut buf, &image, dx + width as i32, dy + height as i32, args.colorspace);
            buf
        }).reduce(|| Rgb64FImage::new(width*3, height*3), |mut buf, image| {
            stack::stack_into(&mut buf, &image, 0, 0, Colorspace::Srgb);
            buf
        });

    println!("Stacking completed");
    println!("Starting postprocessing");
    postprocessing::process(&mut res, num_files, &args.postprocessing);

    for pixel in res.pixels_mut() {
        *pixel = args.colorspace.convert_back(*pixel);
    }

    println!("Postprocessing completed");
    println!("Saving Image");
    DynamicImage::ImageRgb64F(res).into_rgb16().save(args.outfile).unwrap();
    println!("Saving completed");
    println!("Done");
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
        "aba" => Ok(Postprocessing::AverageBrightnessAlignment(value!(value, 0.2))),
        _ => Err(format!("unknown postprocessing `{typ}`, allowed values are `average`, `maxscale`, `sqrt`, `asinh`, `akaze=0.008`, `sobel=1`, `blur=1.0."))
    }
}
