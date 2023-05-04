NUM_EPOCHS = 2

RANKS_VBMF = [
    [29, 2],
    [29, 40],
    [24, 23],
    [21, 21],
    [20, 18],
    [25, 32],
    [40, 47],
    [31, 38],
    [29, 22],
    [77, 54],
    [73, 84],
    [61, 75],
    [68, 53],
    [126, 93],
    [97, 107],
    [114, 127],
    [307, 279],
]


RANKS_HARD = [[int(x/2) if x > 10 else x, int(y/2) if y > 10 else y]
              for [x, y] in RANKS_VBMF]

RANKS = RANKS_HARD
