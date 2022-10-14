use image::Rgb;
use crate::Colorspace;

fn linear(srgb: f64) -> f64 {
    if srgb <= 0.04045 {
        srgb / 12.92
    } else {
        ((srgb + 0.055) / 1.055).powf(2.4)
    }
}
fn srgb(linear: f64) -> f64 {
    if linear <= 0.0031308 {
        linear * 12.92
    } else {
        1.055 * linear.powf(1.0/2.4) - 0.055
    }
}
impl Colorspace {
    pub fn convert_into(&self, px: Rgb<f64>) -> Rgb<f64> {
        match self {
            Colorspace::Srgb => px,
            Colorspace::Linear => {
                let [r, g, b] = px.0;
                Rgb::from([linear(r), linear(g), linear(b)])
            }
            Colorspace::Quadratic => {
                let [r, g, b] = px.0;
                Rgb::from([r.powi(2), g.powi(2), b.powi(2)])
            }
            Colorspace::Sqrt => {
                let [r, g, b] = px.0;
                Rgb::from([r.sqrt(), g.sqrt(), b.sqrt()])
            }
        }
    }
    pub fn convert_back(&self, px: Rgb<f64>) -> Rgb<f64> {
        match self {
            Colorspace::Srgb => px,
            Colorspace::Linear => {
                let [r, g, b] = px.0;
                Rgb::from([srgb(r), srgb(g), srgb(b)])
            }
            Colorspace::Quadratic => {
                let [r, g, b] = px.0;
                Rgb::from([r.sqrt(), g.sqrt(), b.sqrt()])
            }
            Colorspace::Sqrt => {
                let [r, g, b] = px.0;
                Rgb::from([r.powi(2), g.powi(2), b.powi(2)])
            }
        }
    }
}