use std::path::PathBuf;
use clap::{builder::ValueParser, Parser, ValueEnum, Args, Subcommand};

mod helpers;
mod postprocessing;
mod colorspace;
mod register;
mod compare;
mod video;
mod stack;

#[derive(Debug, Parser)]
struct Cli {
    #[command(subcommand)]
    command: Command,

    #[command(flatten)]
    common: CommonArgs,
}

#[derive(Debug, Args)]
pub struct CommonArgs {
    #[arg(short = 'c', long, value_enum, default_value = "srgb")]
    colorspace: Colorspace,
    #[arg(short = 'n', default_value_t = 100)]
    num_files: usize,
    #[arg(short = 's', default_value_t = 0)]
    skip_files: usize,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    Register(Register),
    Compare(Compare),
    Video(Video),
    Stack(Stack),
}

#[derive(Debug, Args)]
pub struct Register {
    #[arg(short = 'i', long)]
    imagepaths: Vec<PathBuf>,
    #[arg(
        long = "pa", long, value_parser=ValueParser::new(parse_postprocessing), value_delimiter=',',
        default_value = "maxscale,blur=20,maxscale",
    )]
    preprocessing_akaze: Vec<Processing>,
    #[arg(long = "pr", long, value_parser=ValueParser::new(parse_postprocessing), value_delimiter=',')]
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
pub struct Compare {
    #[arg(short = 'i', long)]
    registration_input: PathBuf,
    #[arg(short = 'o', long, default_value = "compare")]
    outfile_prefix: PathBuf,
}

#[derive(Debug, Args)]
pub struct Video {
    #[arg(short = 'i', long)]
    registration_input: PathBuf,
    #[arg(short = 'o', long, default_value = "video_aligned")]
    outfile_prefix: PathBuf,
}

#[derive(Debug, Args)]
pub struct Stack {
    #[arg(short = 'i', long)]
    registration_input: PathBuf,
    #[arg(short = 'p', long, value_parser=ValueParser::new(parse_postprocessing), value_delimiter=',')]
    postprocessing: Vec<Processing>,
    #[arg(short = 'o', long, default_value = "stacked.png")]
    outfile: PathBuf,
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
    Sqrt,
    Asinh,
    /// sobel edgeg detection with passed blur, 1 by default
    Sobel(i32),
    /// gaussian blur with passed sigma, 1.0 by default
    Blur(f32),
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

fn main() {
    let args: Cli = Cli::parse();
    dbg!(&args);

    match args.command {
        Command::Register(reg) => register::register(args.common, reg),
        Command::Compare(cmp) => compare::compare(args.common, cmp),
        Command::Video(video) => video::video(args.common, video),
        Command::Stack(stack) => stack::stack(args.common, stack),
    }



    panic!("done");

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
    match typ {
        "average" => Ok(Processing::Average),
        "maxscale" => Ok(Processing::Maxscale),
        "sqrt" => Ok(Processing::Sqrt),
        "asinh" => Ok(Processing::Asinh),
        "akaze" => Ok(Processing::Akaze(value!(value, 0.0008))),
        "sobel" => Ok(Processing::Sobel(value!(value, 1))),
        "blur" => Ok(Processing::Blur(value!(value, 1.0))),
        "bgone" => Ok(Processing::BGone(value!(value, 0.2))),
        "bw" => Ok(Processing::BlackWhite(value!(value, 0.5))),
        "sod" => Ok(Processing::SingleObjectDetection(value!(value, 0.2))),
        "aba" => Ok(Processing::AverageBrightnessAlignment(value!(value, 0.2))),
        _ => Err(format!("unknown postprocessing `{typ}`, allowed values are `average`, `maxscale`, `sqrt`, `asinh`, `akaze=0.008`, `sobel=1`, `blur=1.0."))
    }
}
