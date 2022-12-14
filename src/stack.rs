use std::sync::atomic::{AtomicU32, Ordering};
use image::Rgb64FImage;
use rayon::iter::{ParallelIterator, IntoParallelRefIterator};
use crate::{Colorspace, CommonArgs, helpers, processing, rejection, Stack};
use crate::register::AkazeRegistration;

pub fn stack(common: CommonArgs, stack: Stack) {
    let CommonArgs { colorspace, num_files, skip_files } = common;
    let Stack { registration_input, rejection, preprocessing, postprocessing, outfile_prefix } = stack;

    let registration = helpers::load_registration(registration_input);
    let reference_image = &registration.images[registration.reference_image];
    let ref_img = helpers::load_image(&reference_image.image, Colorspace::Srgb);
    let width = ref_img.width();
    let height = ref_img.height();

    println!("Starting rejection");
    let images = helpers::clamp_slice(&registration.images, skip_files, num_files);
    let images = rejection::reject(&registration, images.to_owned(), &rejection);
    println!("Rejection finished");

    let creation_fn = || {
        let img = Rgb64FImage::new(width, height);
        (img.clone(), img.clone(), img.clone())
    };

    println!("Starting Stacking");
    let counter = AtomicU32::new(0);
    let (mut akaze, mut sod, mut aba) = images.par_iter()
        .map(|reg| (helpers::load_image(&reg.image, colorspace), reg))
        .fold(creation_fn, |(mut akaze, mut sod, mut aba), (mut image, reg)| {
            let count = counter.fetch_add(1, Ordering::Relaxed);
            if count % 50 == 0 {
                println!("{count}");
            }

            processing::process(&mut image, num_files, &preprocessing);

            // akaze
            match reg.akaze {
                Some(AkazeRegistration::Offset(dx, dy)) => {
                    stack_into(&mut akaze, &image, dx as i32, dy as i32);
                }
                Some(AkazeRegistration::Rejected) => println!("rejected akaze {count:05}"),
                None => (),
            }

            // sod
            let (dx, dy) = reg.sod.offset(&reference_image.sod);
            stack_into(&mut sod, &image, dx, dy);

            // aba
            let (dx, dy) = reg.aba.offset(&reference_image.aba);
            stack_into(&mut aba, &image, dx, dy);
            (akaze, sod, aba)
        }).reduce(creation_fn, |(mut akaze1, mut sod1, mut aba1), (akaze2, sod2, aba2)| {
            stack_into(&mut akaze1, &akaze2, 0, 0);
            stack_into(&mut sod1, &sod2, 0, 0);
            stack_into(&mut aba1, &aba2, 0, 0);
            (akaze1, sod1, aba1)
        });

    println!("Stacking completed");
    println!("Starting postprocessing");
    processing::process(&mut akaze, num_files, &postprocessing);
    processing::process(&mut sod, num_files, &postprocessing);
    processing::process(&mut aba, num_files, &postprocessing);

    println!("Processing completed");
    println!("Saving Image");
    helpers::save_image(akaze, helpers::path_with_suffix(&outfile_prefix, "akaze.png"), colorspace);
    helpers::save_image(sod, helpers::path_with_suffix(&outfile_prefix, "sod.png"), colorspace);
    helpers::save_image(aba, helpers::path_with_suffix(&outfile_prefix, "aba.png"), colorspace);
    println!("Saving completed");
    println!("Done");
}

pub fn stack_into(buf: &mut Rgb64FImage, img: &Rgb64FImage, dx: i32, dy: i32) {
    for (x, y, &pixel) in img.enumerate_pixels() {
        let bufx = (x as i32 + dx) as u32;
        let bufy = (y as i32 + dy) as u32;
        let bufpx = match buf.get_pixel_mut_checked(bufx, bufy) {
            Some(px) => px,
            None => continue,
        };
        bufpx.0[0] += pixel.0[0];
        bufpx.0[1] += pixel.0[1];
        bufpx.0[2] += pixel.0[2];
    }
}