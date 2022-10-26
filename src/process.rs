use crate::{CommonArgs, helpers, Process, processing};

pub fn process(common: CommonArgs, process: Process) {
    let CommonArgs { colorspace, num_files, skip_files: _ } = common;
    let Process { image, processing, outfile } = process;
    let mut img = helpers::load_image(image, colorspace);
    processing::process(&mut img, num_files, &processing);
    helpers::save_image(img, outfile, colorspace);
}
