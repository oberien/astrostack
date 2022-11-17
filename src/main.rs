use std::path::PathBuf;
use clap::{builder::ValueParser, Parser, ValueEnum, Args, Subcommand};

mod helpers;
mod processing;
mod colorspace;
mod process;
mod compare;
mod register;
mod video;
mod stack;
mod rejection;

fn main() {
    let args: Cli = Cli::parse();
    dbg!(&args);

    match args.command {
        Command::Process(proc) => process::process(args.common, proc),
        Command::Register(reg) => register::register(args.common, reg),
        Command::Compare(cmp) => compare::compare(args.common, cmp),
        Command::Video(video) => video::video(args.common, video),
        Command::Stack(stack) => stack::stack(args.common, stack),
    }
}

#[derive(Debug, Parser)]
struct Cli {
    #[command(subcommand)]
    command: Command,

    #[command(flatten)]
    common: CommonArgs,
}

#[derive(Debug, Args)]
pub struct CommonArgs {
    #[arg(global = true, short = 'c', long, value_enum, default_value = "srgb")]
    colorspace: Colorspace,
    #[arg(global = true, short = 'n', default_value_t = 100)]
    num_files: usize,
    #[arg(global = true, short = 's', default_value_t = 0)]
    skip_files: usize,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Apply a process-chain to a single image
    Process(Process),
    /// Output comparison of two images as will be done during registration
    Compare(Compare),
    /// Register images
    Register(Register),
    /// Create a video from the aligned registered images
    Video(Video),
    /// Stack registered images
    Stack(Stack),
}

#[derive(Debug, Args)]
pub struct Process {
    image: PathBuf,
    #[arg(short = 'p', long, value_parser=ValueParser::new(parse_postprocessing), value_delimiter=',')]
    processing: Vec<Processing>,
    #[arg(short = 'o', long, default_value = "processed.png")]
    outfile: PathBuf,
}

#[derive(Debug, Args)]
pub struct Compare {
    first: PathBuf,
    second: PathBuf,
    #[arg(
        long = "pa", long, value_parser=ValueParser::new(parse_postprocessing), value_delimiter=',',
        default_value = "maxscale,blur=20,sobel=0,maxscale",
    )]
    preprocessing_akaze: Vec<Processing>,
    #[arg(
        long = "pr", long, value_parser=ValueParser::new(parse_postprocessing), value_delimiter=',',
        default_value = "maxscale,blur=20,maxscale",
    )]
    preprocessing_rest: Vec<Processing>,
    #[arg(long, default_value_t = 0.001)]
    akaze: f64,
    #[arg(long, long = "sod", default_value_t = 0.2)]
    single_object_detection: f64,
    #[arg(long, long = "aba", default_value_t = 0.2)]
    average_brightness_alignment: f64,
    #[arg(short = 'o', long, default_value = "compared")]
    outfile_prefix: PathBuf,
}

#[derive(Debug, Args)]
pub struct Register {
    #[arg(short = 'i', long)]
    imagepaths: Vec<PathBuf>,
    #[arg(short = 'r', long, default_value_t = 0)]
    reference_image: usize,
    #[arg(
        long = "pa", long, value_parser=ValueParser::new(parse_postprocessing), value_delimiter=',',
        default_value = "maxscale,blur=20,sobel=0,maxscale",
    )]
    preprocessing_akaze: Vec<Processing>,
    #[arg(
        long = "pr", long, value_parser=ValueParser::new(parse_postprocessing), value_delimiter=',',
        default_value = "maxscale,blur=20,maxscale",
    )]
    preprocessing_rest: Vec<Processing>,
    #[arg(short = 'o', long, default_value = "registration_data.json")]
    outfile: PathBuf,
    #[arg(long)]
    akaze: Option<f64>,
    #[arg(long, long = "sod", default_value_t = 0.2)]
    single_object_detection: f64,
    #[arg(long, long = "aba", default_value_t = 0.2)]
    average_brightness_alignment: f64,
}

#[derive(Debug, Args)]
pub struct Video {
    #[arg(short = 'i', long, default_value = "registration_data.json")]
    registration_input: PathBuf,
    #[arg(short = 'r', long, value_parser=ValueParser::new(parse_rejection), value_delimiter=',')]
    rejection: Vec<Rejection>,
    #[arg(
        short = 'p', long, value_parser=ValueParser::new(parse_postprocessing), value_delimiter=',',
        default_value = "maxscale",
    )]
    processing: Vec<Processing>,
    #[arg(short = 'o', long, default_value = "video_aligned")]
    outfile_prefix: PathBuf,
}

#[derive(Debug, Args)]
pub struct Stack {
    #[arg(short = 'i', long, default_value = "registration_data.json")]
    registration_input: PathBuf,
    #[arg(
        short = 'r', long, value_parser=ValueParser::new(parse_rejection), value_delimiter=',',
        default_value = "regressionaba,widthheight",
    )]
    rejection: Vec<Rejection>,
    #[arg(
        long = "pre", value_parser=ValueParser::new(parse_postprocessing), value_delimiter=',',
        default_value = "bgone=0.025",
    )]
    preprocessing: Vec<Processing>,
    #[arg(
        short = 'p', long = "post", value_parser=ValueParser::new(parse_postprocessing), value_delimiter=',',
        default_value = "maxscale",
    )]
    postprocessing: Vec<Processing>,
    #[arg(short = 'o', long, default_value = "stacked")]
    outfile_prefix: PathBuf,
}

#[derive(Debug, Copy, Clone, ValueEnum)]
pub enum Colorspace {
    Srgb,
    Linear,
    Quadratic,
    Sqrt,
}

#[derive(Debug, Copy, Clone)]
pub enum Processing {
    Average,
    Maxscale,
    MaxscaleFixed(f64),
    Sqrt,
    Asinh,
    Sharpen,
    /// sobel edgeg detection with passed blur, 1 by default
    Sobel(i32),
    /// gaussian blur with passed sigma, 1.0 by default
    Blur(f32),
    /// Median of a single image of pixels in the given radius
    Median(u32),
    /// bg extraction using the given threshold to make pixels black (0.2)
    BGone(f64),
    /// convert the image to a black-white image using the given threshold (0.5)
    BlackWhite(f64),
    /// akaze feature detection with passed threshold, 0.0008 by default
    Akaze(f64),
    /// single object detection with the given threshold (0.2)
    SingleObjectDetection(f64),
    /// detect the pixel with the average brightness given threshold (0.2)
    AverageBrightnessAlignment(f64),
}
fn parse_postprocessing(p: &str) -> Result<Processing, String> {
    let mut parts = p.split("=");
    let typ = parts.next().unwrap();
    let value = parts.next();
    macro_rules! value {
        ($value:expr, $default:expr) => {
            $value.map(|s| s.parse()).unwrap_or(Ok($default)).map_err(|e| format!("{e}"))?
        }
    }
    let no_value = |p| {
        dbg!(value);
        if value.is_none() {
            Ok(p)
        } else {
            Err("no value allowed".to_string())
        }
    };
    match typ {
        "average" => no_value(Processing::Average),
        "maxscale" => no_value(Processing::Maxscale),
        "sqrt" => no_value(Processing::Sqrt),
        "asinh" => no_value(Processing::Asinh),
        "sharpen" => no_value(Processing::Sharpen),
        "akaze" => Ok(Processing::Akaze(value!(value, 0.0008))),
        "sobel" => Ok(Processing::Sobel(value!(value, 0))),
        "blur" => Ok(Processing::Blur(value!(value, 1.0))),
        "median" => Ok(Processing::Median(value!(value, 2))),
        "bgone" => Ok(Processing::BGone(value!(value, 0.2))),
        "bw" => Ok(Processing::BlackWhite(value!(value, 0.2))),
        "sod" => Ok(Processing::SingleObjectDetection(value!(value, 0.2))),
        "aba" => Ok(Processing::AverageBrightnessAlignment(value!(value, 0.2))),
        _ => Err(format!(
            "unknown processing `{typ}`, allowed values are `average`, `maxscale`,\
            `sqrt`, `asinh`, `akaze=0.0008`, `sobel=0`, `blur=1.0`, `bgone=0.2`,\
            `bw=0.2`, `sod=0.2`, `aba=0.2`."
        ))
    }
}

#[derive(Debug, Copy, Clone)]
pub enum Rejection {
    AverageSod(f32),
    AverageAba(f32),
    RegressionAkaze(f32),
    RegressionSod(f32),
    RegressionAba(f32),
    WidthHeight(f32),
}
fn parse_rejection(p: &str) -> Result<Rejection, String> {
    let mut parts = p.split("=");
    let typ = parts.next().unwrap();
    let value = parts.next();
    macro_rules! value {
        ($value:expr, $default:expr) => {
            $value.map(|s| s.parse()).unwrap_or(Ok($default)).map_err(|e| format!("{e}"))?
        }
    }
    match typ {
        "averagesod" => Ok(Rejection::AverageSod(value!(value, 0.01))),
        "averageaba" => Ok(Rejection::AverageAba(value!(value, 0.01))),
        "regressionakaze" => Ok(Rejection::RegressionAkaze(value!(value, 0.001))),
        "regressionsod" => Ok(Rejection::RegressionSod(value!(value, 0.001))),
        "regressionaba" => Ok(Rejection::RegressionAba(value!(value, 0.001))),
        "widthheight" => Ok(Rejection::WidthHeight(value!(value, 0.02))),
        _ => Err(format!(
            "unknown rejection `{typ}`, allowed values are `averagesod=0.01`, `averageaba=0.01`,\
            `regressionakaze=0.001`, `regressionsod=0.001`, `regressionaba=0.001`."
        ))
    }
}
