import torch
import chess
import sys
import data
from dataclasses import dataclass

from model import OUTPUT_SCALING, NNUE
from feature import FEATURE_COUNT, feature

@dataclass
class Features:
    stm_features: torch.Tensor
    non_stm_features: torch.Tensor
    
def board_features(board: chess.Board) -> Features:
    stm_features = torch.zeros((1, FEATURE_COUNT))
    non_stm_features = torch.zeros((1, FEATURE_COUNT))

    for color in chess.COLORS:
        for piece in chess.PIECE_TYPES:
            for square in board.pieces(piece, color):
                stm_index = feature(board.turn, color, piece, square)
                non_stm_index = feature(not board.turn, color, piece, square)
                stm_features[(0, stm_index)] = 1.0
                non_stm_features[(0, non_stm_index)] = 1.0

    return Features(stm_features=stm_features, non_stm_features=non_stm_features)

def main():
    if len(sys.argv) < 3:
        print(f"expected model path and board FEN", file=sys.stderr)
        return

    board = chess.Board(sys.argv[2])
    features = board_features(board)

    model = NNUE(lr=0.0, eval_weight=0.0)
    model.load_state_dict(torch.load(sys.argv[1], weights_only=True))
    model.eval()

    score = model(features).item() * 3.0
    print(f"score: {score:+.2f}")

if __name__ == "__main__":
    main()

