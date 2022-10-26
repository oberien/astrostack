use image::{GenericImage, Rgb64FImage};
use crate::{CommonArgs, Compare, helpers, processing, register};

pub fn compare(common: CommonArgs, compare: Compare) {
    let CommonArgs { colorspace, num_files, skip_files } = common;
    let Compare { first, second, preprocessing_akaze, preprocessing_rest, akaze, single_object_detection, average_brightness_alignment, outfile_prefix } = compare;

    let first = helpers::load_image(first, colorspace);
    let second = helpers::load_image(second, colorspace);
    assert_eq!(first.width(), second.width());
    assert_eq!(first.height(), second.height());

    let algs: &[(_, for<'a, 'b, 'c> fn (&'a _, &'b _, _, Option<&'c mut _>) -> _, _, &[_])] = &[
        ("akaze", register::akaze, akaze, &preprocessing_akaze),
        ("sod", register::single_object_detection, single_object_detection, &preprocessing_rest),
        ("aba", register::average_brightness_alignment, average_brightness_alignment, &preprocessing_rest),
        ("orig", register::noop, 0.0, &[]),
    ];

    for &(name, register, threshold, preprocessing) in algs {
        let reference_image = register::prepare_reference_image(first.clone(), num_files, preprocessing);
        let mut second = second.clone();
        processing::process(&mut second, num_files, preprocessing);

        let mut res = Rgb64FImage::new(first.width() * 2, first.height());
        res.copy_from(&reference_image.0, 0, 0).unwrap();
        res.copy_from(&second, first.width(), 0).unwrap();

        register(&reference_image, &second, threshold, Some(&mut res));
        let outfile_name = format!("{}_{}.png", outfile_prefix.file_name().unwrap().to_str().unwrap(), name);
        let outfile = outfile_prefix.with_file_name(outfile_name);
        helpers::save_image(res, outfile, colorspace);
    }
}