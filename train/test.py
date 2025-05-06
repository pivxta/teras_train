import math
import torch
import chess
import sys
import data
from model import TerasNN
from dataclasses import dataclass

@dataclass
class Features:
    stm_features: torch.Tensor
    non_stm_features: torch.Tensor

def feature(
    perspective: chess.Color, 
    color: chess.Color, 
    piece: chess.PieceType,
    square: chess.Square
) -> int:
    square = chess.square_mirror(square) if perspective == chess.BLACK else square
    index = 0 if perspective == color else 1 
    index = index * 6 + piece - 1
    index = index * 64 + square
    return index
    
def board_features(board: chess.Board) -> Features:
    stm_features = torch.zeros((1, data.FEATURE_COUNT))
    non_stm_features = torch.zeros((1, data.FEATURE_COUNT))

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

    model = TerasNN()
    model.load_state_dict(torch.load(sys.argv[1], weights_only=True))
    model.eval()

    score = model(features).item()
    pawns = math.log(score / (1 - score)) * 1.73
    print(f"score: {score} ({pawns:+.2f})")

if __name__ == "__main__":
    main()

