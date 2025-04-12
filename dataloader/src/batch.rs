use dama::{Color, Piece, Position};
use dataformat::Sample;
use crate::feature::{feature, MAX_ACTIVE_FEATURES};

#[derive(Clone, Debug)]
pub struct Batch {
    pub(crate) entries: usize,
    pub(crate) capacity: usize,
    pub(crate) total_features: usize,
    pub(crate) stm_features: Box<[u32]>,
    pub(crate) non_stm_features: Box<[u32]>,
    pub(crate) eval_centipawns: Box<[f32]>,
    pub(crate) outcomes: Box<[f32]>,
}

impl Batch {
    #[inline]
    pub fn new(capacity: usize) -> Batch {
        Batch {
            entries: 0,
            capacity,
            total_features: 0,
            stm_features: vec![0; 2 * MAX_ACTIVE_FEATURES * capacity].into(),
            non_stm_features: vec![0; 2 * MAX_ACTIVE_FEATURES * capacity].into(),
            eval_centipawns: vec![0.0; capacity].into(),
            outcomes: vec![0.0; capacity].into(),
        }
    }
        
    #[inline]
    pub fn clear(&mut self) {
        self.entries = 0; 
        self.total_features = 0;
    }

    #[inline]
    pub fn add(&mut self, sample: &Sample) {
        assert!(self.entries < self.capacity);

        let index = self.entries;
        self.eval_centipawns[index] = sample
            .eval
            .map(|e| e as f32)
            .unwrap_or(match sample.outcome.winner() {
                Some(color) if color == sample.position.side_to_move() => f32::INFINITY,
                Some(_) => f32::NEG_INFINITY,
                None => 0.0
            });
        self.outcomes[index] = match sample.outcome.winner() {
            Some(color) if color == sample.position.side_to_move() => 1.0,
            Some(_) => 0.0,
            None => 0.5
        };
        self.add_features(&sample.position);
        self.entries += 1;
    }

    #[inline]
    fn add_features(&mut self, position: &Position) {
        for color in Color::ALL {
            for piece in Piece::ALL {
                for square in position.pieces(piece) & position.colored(color) {
                    self.add_feature(
                        feature(position.side_to_move(), color, piece, square),
                        feature(!position.side_to_move(), color, piece, square),
                    );
                }
            }
        }
    }

    #[inline]
    fn add_feature(&mut self, stm: u32, non_stm: u32) {
        let index = 2 * self.total_features;
        self.stm_features[index] = self.entries as u32;
        self.non_stm_features[index] = self.entries as u32;
        self.stm_features[index + 1] = stm;
        self.non_stm_features[index + 1] = non_stm;
        self.total_features += 1;
    }
}
