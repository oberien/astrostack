use image::Rgb64FImage;
use crate::Colorspace;

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