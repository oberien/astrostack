use std::fs::File;
use image::{EncodableLayout, RgbImage};
use minimp4::Mp4Muxer;
use openh264::encoder::{Encoder, EncoderConfig};
use openh264::formats::RBGYUVConverter;
use crate::{CommonArgs, helpers, Video};

pub fn video(common: CommonArgs, video: Video) {
    let CommonArgs { colorspace, num_files, skip_files } = common;
    let Video { registration_input, outfile_prefix } = video;

    let registration = helpers::load_registration(registration_input);
    let reference = &registration.images[registration.reference_image];
    let reference_image = helpers::load_image(&reference.image, colorspace);
    let width = reference_image.width();
    let height = reference_image.height();

    let images = helpers::clamp_slice(&registration.images, skip_files, num_files);

    // temporary movement smoothing
    let images: Vec<_> = images.windows(3)
        .filter_map(|window| {
            let (a,b,c) = match window {
                [a, b, c] => (a,b,c),
                _ => unreachable!(),
            };
            let avgx = (a.aba.middlex + b.aba.middlex + c.aba.middlex) / 3.;
            let avgy = (a.aba.middley + b.aba.middley + c.aba.middley) / 3.;
            let dx = (b.aba.middlex - avgx).abs();
            let dy = (b.aba.middley - avgy).abs();
            if (dx + dy) / (width + height) as f32  > 0.02 {
                None
            } else {
                Some(b)
            }
        }).collect();

    let config = EncoderConfig::new(width, height);
    let mut encoder_orig = Encoder::with_config(config).unwrap();
    let mut encoder_akaze = Encoder::with_config(config).unwrap();
    let mut encoder_sod = Encoder::with_config(config).unwrap();
    let mut encoder_sod_reject = Encoder::with_config(config).unwrap();
    let mut encoder_aba = Encoder::with_config(config).unwrap();
    let mut encoder_aba_reject = Encoder::with_config(config).unwrap();

    let mut buf_orig = Vec::new();
    let mut buf_akaze = Vec::new();
    let mut buf_sod = Vec::new();
    let mut buf_sod_reject = Vec::new();
    let mut buf_aba = Vec::new();
    let mut buf_aba_reject = Vec::new();

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
        let image = helpers::load_image_rgb8(&reg.image);
        let frame_sod = helpers::offset_image(&image, reg.sod.offset(&reference.sod));
        let frame_aba = helpers::offset_image(&image, reg.aba.offset(&reference.aba));
        encode_into(&mut encoder_orig, &mut buf_orig, &image);
        if let Some(akaze) = &reg.akaze {
            let frame = helpers::offset_image(&image, akaze.offset());
            encode_into(&mut encoder_akaze, &mut buf_akaze, &frame);

        }
        encode_into(&mut encoder_sod, &mut buf_sod, &frame_sod);
        encode_into(&mut encoder_aba, &mut buf_aba, &frame_aba);
        if !reg.sod.should_reject(&reference.sod) {
            encode_into(&mut encoder_sod_reject, &mut buf_sod_reject, &frame_sod);
            encode_into(&mut encoder_aba_reject, &mut buf_aba_reject, &frame_aba);
        }
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
    save_buf("sod_reject", &buf_sod_reject);
    save_buf("aba", &buf_aba);
    save_buf("aba_reject", &buf_aba_reject);
}