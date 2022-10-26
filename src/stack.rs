use image::Rgb64FImage;
use crate::{Colorspace, CommonArgs, Stack};

pub fn stack(common: CommonArgs, stack: Stack) {
    let CommonArgs { colorspace, num_files, skip_files } = common;
    let Stack { registration_input, postprocessing, outfile } = stack;
    todo!()

    // let reference_image = registration::prepare_reference_image(first, num_files, &registration);
    //
    // println!("Starting stacking.");
    // let counter = AtomicU32::new(0);
    // let mut res = files.into_par_iter()
    //     .map(|path| Reader::open(path).unwrap().decode().unwrap().into_rgb64f())
    //     .fold(|| Rgb64FImage::new(width*3, height*3), |mut buf, image| {
    //         let count = counter.fetch_add(1, Ordering::Relaxed);
    //         if count % 50 == 0 {
    //             println!("{count}");
    //         }
    //         let delta = register::register(
    //             &reference_image,
    //             image.clone(),
    //             num_files,
    //             &args.registration,
    //             args.debug,
    //         );
    //         let (dx, dy) = match delta {
    //             Some((dx, dy)) => (dx, dy),
    //             None => {
    //                 if args.debug {
    //                     println!("rejected");
    //                 }
    //                 return buf
    //             },
    //         };
    //         if args.debug {
    //             println!("{dx}:{dy}");
    //         }
    //         stack::stack_into(&mut buf, &image, dx + width as i32, dy + height as i32, args.colorspace);
    //         buf
    //     }).reduce(|| Rgb64FImage::new(width*3, height*3), |mut buf, image| {
    //     stack::stack_into(&mut buf, &image, 0, 0, Colorspace::Srgb);
    //     buf
    // });
    //
    // println!("Stacking completed");
    // println!("Starting postprocessing");
    // postprocessing::process(&mut res, num_files, &args.postprocessing);
    //
    // for pixel in res.pixels_mut() {
    //     *pixel = args.colorspace.convert_back(*pixel);
    // }
    //
    // println!("Processing completed");
    // println!("Saving Image");
    // DynamicImage::ImageRgb64F(res).into_rgb16().save(args.outfile).unwrap();
    // println!("Saving completed");
    // println!("Done");
}

pub fn stack_into(buf: &mut Rgb64FImage, img: &Rgb64FImage, dx: i32, dy: i32, colorspace: Colorspace) {
    for (x, y, &pixel) in img.enumerate_pixels() {
        let bufx = (x as i32 + dx) as u32;
        let bufy = (y as i32 + dy) as u32;
        let bufpx = buf.get_pixel_mut(bufx, bufy);
        let pixel = colorspace.convert_into(pixel);
        bufpx.0[0] += pixel.0[0];
        bufpx.0[1] += pixel.0[1];
        bufpx.0[2] += pixel.0[2];
    }
}