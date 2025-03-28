use dama::{
    position, ByColor, Color, InvalidPositionError, Outcome, Piece, Position, Rank, Square,
    SquareSet,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sample {
    pub position: Position,
    pub outcome: Outcome,
    pub eval: Option<i16>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct PackedSample {
    pieces: PackedPieces,
    occupied: u64,
    eval: i16,
    fullmove_number: u16,
    halfmove_clock: u8,
    en_passant: u8,
    side_to_move: u8,
    game_outcome: u8,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Error)]
pub enum PackError {
    #[error("no position can have more than 32 pieces.")]
    TooManyPieces,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Error)]
pub enum UnpackError {
    #[error("invalid position: {0}")]
    InvalidPosition(InvalidPositionError),
    #[error("invalid \"outcome\" field for packed position.")]
    InvalidOutcome,
    #[error("invalid \"side to move\" field for packed position.")]
    InvalidSideToMove,
    #[error("invalid \"en passant\" field for packed position.")]
    InvalidEnPassant,
    #[error("too many pieces in packed position.")]
    TooManyPieces,
}

impl Sample {
    #[inline]
    pub fn pack(&self) -> Result<PackedSample, PackError> {
        PackedSample::new(self)
    }
}

impl PackedSample {
    #[inline]
    fn new(sample: &Sample) -> Result<Self, PackError> {
        let position = &sample.position;
        let mut pieces = PackedPieces::default();

        if position.occupied().count() > 32 {
            return Err(PackError::TooManyPieces);
        }

        for (n, square) in position.occupied().iter().enumerate() {
            if let Some(piece) = position.piece_at(square) {
                let color = position.color_at(square).expect("piece has no color.");
                let castling = position.castling(color);
                let backrank = Rank::back_rank(color);

                if piece == Piece::Rook
                    && square.rank() == backrank
                    && castling.contains(square.file())
                {
                    pieces.set(n, encode_color(color) | CASTLING_ROOK);
                } else {
                    pieces.set(n, encode_color(color) | encode_piece(piece));
                }
            }
        }

        Ok(PackedSample {
            pieces,
            occupied: position.occupied().into(),
            en_passant: position.en_passant().map(|sq| sq as u8).unwrap_or(0),
            halfmove_clock: position.halfmove_clock().min(255) as u8,
            fullmove_number: position.fullmove_number() as u16,
            side_to_move: position.side_to_move() as u8,
            eval: sample.eval.unwrap_or(NO_EVAL),
            game_outcome: match sample.outcome {
                Outcome::Draw => 0b11,
                Outcome::Winner(Color::White) => 0b10,
                Outcome::Winner(Color::Black) => 0b01,
            },
        })
    }

    pub fn unpack(&self) -> Result<Sample, UnpackError> {
        let mut setup = position::Setup::new_empty();

        if self.en_passant != 0 {
            let en_passant = Square::try_from_index(self.en_passant as usize)
                .ok_or(UnpackError::InvalidEnPassant)?;
            setup.set_en_passant(Some(en_passant));
        }

        let side_to_move = *Color::ALL
            .get(self.side_to_move as usize)
            .ok_or(UnpackError::InvalidSideToMove)?;
        setup.set_side_to_move(side_to_move);

        setup
            .set_fullmove_number(self.fullmove_number as u32)
            .set_halfmove_clock(self.halfmove_clock as u32);

        let occupied = SquareSet::from(self.occupied);
        if occupied.count() > 32 {
            return Err(UnpackError::TooManyPieces);
        }

        let mut saw_king = ByColor::default();
        for (n, square) in occupied.iter().enumerate() {
            let (color, piece, is_castling_rook) = self.pieces.get(n);
            if piece == Piece::King {
                saw_king[color] = true;
            }
            if is_castling_rook {
                if saw_king[color] {
                    setup.castling[color].king_side = Some(square.file());
                } else {
                    setup.castling[color].queen_side = Some(square.file());
                }
            }
            setup.put_piece(square, color, piece);
        }

        let position = setup
            .into_position()
            .map_err(|err| UnpackError::InvalidPosition(err))?;
        
        let outcome = match self.game_outcome {
            0b11 => Outcome::Draw,
            0b10 => Outcome::Winner(Color::White),
            0b01 => Outcome::Winner(Color::Black),
            _ => return Err(UnpackError::InvalidOutcome),
        };

        let eval = match self.eval {
            NO_EVAL => None,
            eval => Some(eval),
        };

        Ok(Sample {
            position,
            outcome,
            eval,
        })
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
struct PackedPieces([u8; 16]);

impl PackedPieces {
    #[inline]
    fn set(&mut self, index: usize, bits: u8) {
        self.0[index / 2] |= match index % 2 {
            0 => bits,
            1 => bits << 4,
            _ => unreachable!(),
        }
    }

    #[inline]
    fn get(&self, index: usize) -> (Color, Piece, bool) {
        let entry = self.0[index / 2];
        let value = match index % 2 {
            0 => entry & 0xf,
            1 => entry >> 4,
            _ => unreachable!(),
        };
        let color = match value & COLOR_MASK {
            WHITE => Color::White,
            BLACK => Color::Black,
            _ => unreachable!(),
        };
        let (piece, is_castling_rook) = match value & PIECE_MASK {
            CASTLING_ROOK => (Piece::Rook, true),
            piece => (Piece::ALL[piece as usize - 1], false),
        };
        (color, piece, is_castling_rook)
    }
}

const NO_EVAL: i16 = i16::MIN;
const BLACK: u8 = 0b0000;
const WHITE: u8 = 0b1000;
const COLOR_MASK: u8 = 0b1000;
const PIECE_MASK: u8 = 0b0111;
const CASTLING_ROOK: u8 = 0b0111;

#[inline]
fn encode_color(color: Color) -> u8 {
    match color {
        Color::Black => BLACK,
        Color::White => WHITE,
    }
}

#[inline]
fn encode_piece(piece: Piece) -> u8 {
    piece as u8 + 1
}

#[cfg(test)]
mod tests {
    use super::Sample;
    use dama::{Color, Outcome, Position, SanMove};
    use rand::{seq::IndexedRandom, Rng, SeedableRng};
    use std::str::FromStr;

    #[test]
    fn pack_roundtrip_initial() {
        let sample = Sample {
            position: Position::new_initial(),
            outcome: Outcome::Draw,
            eval: None,
        };

        let packed = sample.pack().unwrap();
        let unpacked = packed.unpack().unwrap();
        assert_eq!(sample, unpacked);

        let sample = Sample {
            position: Position::new_initial(),
            outcome: Outcome::Winner(Color::White),
            eval: Some(-202),
        };

        let packed = sample.pack().unwrap();
        let unpacked = packed.unpack().unwrap();
        assert_eq!(sample, unpacked);
    }

    #[test]
    fn pack_roundtrip_random_moves() {
        let mut rng = rand_xoshiro::Xoroshiro128Plus::seed_from_u64(0x278B3E360F7ACC92);
        for _ in 0..1000 {
            let mut position = Position::new_initial();
            check_roundtrip(&mut rng, &position);

            for _ in 0..100 {
                let moves = position.legal_moves();
                if moves.is_empty() {
                    break;
                }
                position.play_unchecked(&moves.choose(&mut rng).unwrap());
                check_roundtrip(&mut rng, &position);
            }
        }
    }

    fn check_roundtrip(rng: &mut impl Rng, position: &Position) {
        let sample = Sample {
            position: position.clone(),
            outcome: random_outcome(rng),
            eval: random_eval(rng)
        };
        let packed = sample.pack().unwrap();
        let unpacked = packed.unpack().unwrap();
        assert_eq!(sample, unpacked);
    }

    fn random_eval(rng: &mut impl Rng) -> Option<i16> {
        match rng.random_range(0..100) {
            0..=10 => None,
            _ => Some(rng.random_range(-5000..=5000))
        }
    }

    fn random_outcome(rng: &mut impl Rng) -> Outcome {
        *[
            Outcome::Winner(Color::White),
            Outcome::Winner(Color::Black),
            Outcome::Draw,
        ]
        .choose(rng)
        .unwrap()
    }

    #[test]
    fn pack_roundtrip_game() {
        #[rustfmt::skip]
        let moves = [
            "Nf3", "d5",  "g3", "c5",  "Bg2", "Nc6",  "d4", "e6",  "O-O", "cxd4",  "Nxd4", "Nge7",  "c4",
            "Nxd4", "Qxd4", "Nc6",  "Qd1", "d4",  "e3", "Bc5",  "exd4", "Bxd4",  "Nc3", "O-O",  "Nb5",
            "Bb6", "b3", "a6",  "Nc3", "Bd4",  "Bb2", "e5",  "Qd2", "Be6",  "Nd5", "b5",  "cxb5",
            "axb5",  "Nf4", "exf4",  "Bxc6", "Bxb2",  "Qxb2", "Rb8",  "Rfd1", "Qb6",  "Bf3", "fxg3",  
            "hxg3", "b4",  "a4", "bxa3",  "Rxa3", "g6",  "Qd4", "Qb5",  "b4", "Qxb4",  "Qxb4", "Rxb4",  
            "Ra8", "Rxa8",  "Bxa8", "g5",  "Bd5", "Bf5",  "Rc1", "Kg7",  "Rc7", "Bg6",  "Rc4", "Rb1+",  
            "Kg2", "Re1",  "Rb4", "h5",  "Ra4", "Re5",  "Bf3", "Kh6",  "Kg1", "Re6",  "Rc4", "g4",  
            "Bd5", "Rd6",  "Bb7", "Kg5",  "f3", "f5",  "fxg4", "hxg4",  "Rb4", "Bf7",  "Kf2", "Rd2+", 
            "Kg1", "Kf6",  "Rb6+", "Kg5",  "Rb4", "Be6",  "Ra4", "Rb2",  "Ba8", "Kf6",  "Rf4", "Ke5", 
            "Rf2", "Rxf2",  "Kxf2", "Bd5",  "Bxd5", "Kxd5",  "Ke3", "Ke5"
        ];

        let mut sample = Sample {
            position: Position::new_initial(),
            outcome: Outcome::Draw,
            eval: None,
        };

        let packed = sample.pack().unwrap();
        let unpacked = packed.unpack().unwrap();
        assert_eq!(sample, unpacked);

        for mv in moves.map(SanMove::from_str).map(Result::unwrap) {
            sample.position.play(&mv).unwrap();
            let packed = sample.pack().unwrap();
            let unpacked = packed.unpack().unwrap();
            assert_eq!(
                sample,
                unpacked,
                "failed in move {}",
                sample.position.fullmove_number()
            );
        }
    }
}
