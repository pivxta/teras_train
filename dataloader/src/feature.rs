use dama::{Color, Piece, Square};

pub const MAX_ACTIVE_FEATURES: usize = 64;

#[inline]
pub fn feature(perspective: Color, color: Color, piece: Piece, square: Square) -> u32 {
    let square = match perspective {
        Color::White => square,
        Color::Black => square.flip_vertical()
    };
    let index = if perspective == color {
        0
    } else {
        1 
    }; 
    let index = index * Piece::COUNT as u32 + piece as u32;
    let index = index * Square::COUNT as u32 + square as u32;
    index
}
