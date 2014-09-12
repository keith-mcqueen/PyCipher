__author__ = 'keith'

import numpy as np

modulo = 0x11b
high_order_bit = 0x100
min_x_time_bit = 0x01
default_x_time_bit = 0x02


def mix_columns(state):
    # for each column, compute the new values for each cell by multiplying (ff_multiply) by the matrix:
    # [[ 0x02  0x03  0x01  0x01 ]
    #  [ 0x01  0x02  0x03  0x01 ]
    #  [ 0x01  0x01  0x02  0x03 ]
    #  [ 0x03  0x01  0x01  0x02 ]]

    for c in range(4):
        # get the column from the state
        col = state[c]

        # create a new column (initialized to all 0s)
        mixed_column = np.zeros(4, np.int64)

        # compute the new values doing the matrix multiply
        mixed_column[0] = ff_add(ff_multiply(col[0], 0x02), ff_multiply(col[1], 0x03), col[2], col[3])
        mixed_column[1] = ff_add(col[0], ff_multiply(col[1], 0x02), ff_multiply(col[2], 0x03), col[3])
        mixed_column[2] = ff_add(col[0], col[1], ff_multiply(col[2], 0x02), ff_multiply(col[3], 0x03))
        mixed_column[3] = ff_add(ff_multiply(col[0], 0x03), col[1], col[2], ff_multiply(col[3], 0x02))

        # set the new column in the state
        state[c] = mixed_column

    # return the modified state
    return state


def ff_multiply(operand, times):
    # if input or times <= 0, then return 0
    if operand <= 0 or times <= 0:
        return 0

    # if input or times is 1, then return the other
    if operand == 1:
        return times
    if times == 1:
        return operand

    # start with a 0 product that we'll accumulate
    product = 0

    # start at the lowest order bit
    current_bit = min_x_time_bit

    # loop until we've hit all the bits of 'times'
    while current_bit <= times:
        # if the current bit of 'times' is set, then do x_time for the current bit, adding the result to the product
        if times & current_bit == current_bit:
            product = ff_add(product, x_time(operand, current_bit))

        # increment the bit (left shift one place)
        current_bit <<= 1

    return product


def x_time(operand, bit=default_x_time_bit):
    # if we've hit the minimum bit, then just return the operand
    if bit <= min_x_time_bit:
        return operand

    # multiply the operand by 2 (left shift one place), while decrementing the bit (right shift one place)
    return x_time(normalize(operand << 1), bit >> 1)


def normalize(num):
    # if the product is too big, then modulo (just XOR) it by 0x11b
    if num & high_order_bit == high_order_bit:
        return num ^ modulo

    return num


def ff_add(*args):
    sum = 0

    for a in args:
        sum ^= a

    return sum


state = np.arange(16).reshape((4, 4))
print state
mix_columns(state)
print state