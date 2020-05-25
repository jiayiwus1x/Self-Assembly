def make_pyramid():
    lists = [None] * (10 - 1)
    lists[0] = [1, 2, 3]

    lists[1] = [2, 3, 4, 5, 9]
    lists[2] = [3, 5, 6, 7]
    lists[3] = [7, 8, 9]

    lists[4] = [5, 9]
    lists[5] = [6, 7, 9]
    lists[6] = [7]
    lists[7] = [8, 9]
    lists[8] = [9]
    return lists


def make_pen():
    lists = [None] * (10 - 1)
    lists[0] = [1, 2, 3, 4]
    lists[1] = [2, 4, 5]
    lists[2] = [3, 6]
    lists[3] = [4, 7]
    lists[4] = [8]
    lists[5] = [6, 8, 9]
    lists[6] = [7, 9]
    lists[7] = [8, 9]
    lists[8] = [9]

    return lists
