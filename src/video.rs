use std::fs::File;
use image::{EncodableLayout, RgbImage};
use image::buffer::ConvertBuffer;
use minimp4::Mp4Muxer;
use openh264::encoder::{Encoder, EncoderConfig};
use openh264::formats::RBGYUVConverter;
use crate::{CommonArgs, helpers, processing, Processing, rejection, Video};

pub fn video(common: CommonArgs, video: Video) {
    let CommonArgs { colorspace, num_files, skip_files } = common;
    let Video { registration_input, rejection, processing, outfile_prefix } = video;

    let registration = helpers::load_registration(registration_input);
    let reference = &registration.images[registration.reference_image];
    let reference_image = helpers::load_image(&reference.image, colorspace);
    let width = reference_image.width();
    let height = reference_image.height();

    let images = helpers::clamp_slice(&registration.images, skip_files, num_files);
    let images = rejection::reject(&registration, images.to_owned(), &rejection);

    // replace maxscale with maxscale_fixed based on first image
    let processing: Vec<_> = processing.into_iter()
        .map(|processing| match processing {
            Processing::Maxscale => Processing::MaxscaleFixed(processing::maxcol(&reference_image)),
            p => p,
        }).collect();

    let config = EncoderConfig::new(width, height);
    let mut encoder_orig = Encoder::with_config(config).unwrap();
    let mut encoder_akaze = Encoder::with_config(config).unwrap();
    let mut encoder_sod = Encoder::with_config(config).unwrap();
    let mut encoder_aba = Encoder::with_config(config).unwrap();

    let mut buf_orig = Vec::new();
    let mut buf_akaze = Vec::new();
    let mut buf_sod = Vec::new();
    let mut buf_aba = Vec::new();

    let encode_into = |encoder: &mut Encoder, buf: &mut Vec<u8>, frame: &RgbImage| {
        let mut yuv = RBGYUVConverter::new(width as usize, height as usize);
        yuv.convert(frame.as_bytes());
        // Encode YUV into H.264.
        let bitstream = encoder.encode(&yuv).unwrap();
        bitstream.write_vec(buf);
    };

    for (i, reg) in images.iter().enumerate() {
        if i % 50 == 0 {
            println!("{i}");
        }
        let mut image = helpers::load_image(&reg.image, colorspace);
        processing::process(&mut image, num_files, &processing);
        let image: RgbImage = image.convert();
        let frame_sod = helpers::offset_image(&image, reg.sod.offset(&reference.sod));
        let frame_aba = helpers::offset_image(&image, reg.aba.offset(&reference.aba));
        encode_into(&mut encoder_orig, &mut buf_orig, &image);
        if let Some(akaze) = &reg.akaze {
            let frame = helpers::offset_image(&image, akaze.offset());
            encode_into(&mut encoder_akaze, &mut buf_akaze, &frame);

        }
        encode_into(&mut encoder_sod, &mut buf_sod, &frame_sod);
        encode_into(&mut encoder_aba, &mut buf_aba, &frame_aba);
    }

    let save_buf = |name: &str, data: &[u8]| {
        let file_name = helpers::path_with_suffix(&outfile_prefix, &format!("{}.mp4", name));
        let file = File::create(file_name).unwrap();
        let mut mp4muxer = Mp4Muxer::new(file);
        mp4muxer.init_video(width as i32, height as i32, false, name);
        mp4muxer.write_video_with_fps(&data, 25);
        mp4muxer.close();
    };

    save_buf("orig", &buf_orig);
    save_buf("akaze", &buf_akaze);
    save_buf("sod", &buf_sod);
    save_buf("aba", &buf_aba);
}