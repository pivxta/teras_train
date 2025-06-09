import chess

FEATURE_COUNT = 768

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

def initial_psqt():
    piece_values = {
        chess.PAWN: 40,
        chess.KNIGHT: 300,
        chess.BISHOP: 310,
        chess.ROOK: 520,
        chess.QUEEN: 1000,
    }
    values = [0] * FEATURE_COUNT
    for square in range(64):
        for piece, value in piece_values.items():
            index_stm = feature(chess.WHITE, chess.WHITE, piece, square)
            index_non_stm = feature(chess.WHITE, chess.BLACK, piece, square)
            values[index_stm] = value
            values[index_non_stm] = -value

    return values


