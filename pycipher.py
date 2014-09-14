__author__ = 'keith'

import numpy as np

N_B = 4
N_R = {4: 10, 6: 12, 8: 14}

MODULO = 0x11b
HIGH_ORDER_BIT = 0x100
MIN_X_TIME_BIT = 0x01
DEFAULT_X_TIME_BIT = 0x02

# these values are obtained directly from the FIPS 197 spec
S_BOX = np.array([[0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76],
                  [0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0],
                  [0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15],
                  [0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75],
                  [0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84],
                  [0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf],
                  [0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8],
                  [0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2],
                  [0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73],
                  [0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb],
                  [0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79],
                  [0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08],
                  [0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a],
                  [0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e],
                  [0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf],
                  [0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16]],
                 np.uint8)

INV_S_BOX = np.array([[0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb],
                      [0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb],
                      [0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e],
                      [0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25],
                      [0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92],
                      [0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84],
                      [0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06],
                      [0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b],
                      [0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73],
                      [0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e],
                      [0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b],
                      [0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4],
                      [0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f],
                      [0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef],
                      [0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61],
                      [0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d]],
                     np.uint8)

R_CON_CACHE = {}


def cipher(block, key_schedule, n_k):
    n_r = N_R[n_k]

    # copy the block to the state
    state = block.copy()

    # add the 0th round key
    state = add_round_key(state, key_schedule[0:N_B])

    # for each of the rounds...
    for i in range(1, n_r):
        # substitute the bytes
        state = sub_bytes(state)

        # shift the rows
        state = shift_rows(state)

        # mix the columns
        state = mix_columns(state)

        # add the round key
        state = add_round_key(state, key_schedule[i * N_B:(i + 1) * N_B])

    # do a final round of: substitute the bytes, shift the rows and add the last round key
    state = sub_bytes(state)
    state = shift_rows(state)
    state = add_round_key(state, key_schedule[n_r * N_B:(n_r + 1) * N_B])

    # return the encrypted block
    return state


def mix_columns(state):
    # for each column, compute the new values for each cell by multiplying (ff_multiply) by the matrix:
    # [[ 0x02  0x03  0x01  0x01 ]
    # [ 0x01  0x02  0x03  0x01 ]
    # [ 0x01  0x01  0x02  0x03 ]
    # [ 0x03  0x01  0x01  0x02 ]]

    # for each column in the state (there should only be 4)...
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


# def inv_mix_columns(state):
#     pass


def sub_bytes(state):
    copy = state.copy()

    # replace each byte in the array with it's corresponding value in the S_BOX
    for byte in np.nditer(copy, op_flags=['readwrite']):
        byte[...] = sub_byte(byte)

    return copy


# def inv_sub_bytes(state):
#     pass


def sub_byte(byte):
    # get the low-order value
    low = byte & 0x0f

    # get the high-order value
    high = (byte >> 4) & 0x0f

    # get the replacement value from the S_BOX
    return S_BOX[high, low]


def sub_word(word):
    return sub_bytes(word)


def rot_word(word):
    return np.roll(word, 3)


def shift_rows(state):
    # transpose (rotate) the state
    shifted = state.T

    # for each row (was a column) roll it by 4 - row index
    for i in range(4):
        shifted[i] = np.roll(shifted[i], 4 - i)

    return state


# def inv_shift_rows(state):
#     pass


def add_round_key(state, round_key):
    # just XOR the corresponding elements of the state and the round key
    return np.bitwise_xor(state, round_key)


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
    current_bit = MIN_X_TIME_BIT

    # loop until we've hit all the bits of 'times'
    while current_bit <= times:
        # if the current bit of 'times' is set, then do x_time for the current bit, adding the result to the product
        if times & current_bit == current_bit:
            product = ff_add(product, x_time(operand, current_bit))

        # increment the bit (left shift one place)
        current_bit <<= 1

    return product


def x_time(operand, bit=DEFAULT_X_TIME_BIT):
    # if we've hit the minimum bit, then just return the operand
    if bit <= MIN_X_TIME_BIT:
        return operand

    # multiply the operand by 2 (left shift one place), while decrementing the bit (right shift one place)
    return x_time(normalize(operand << 1), bit >> 1)


def normalize(val):
    # if the product is too big, then modulo (just XOR) it by 0x11b
    if val & HIGH_ORDER_BIT == HIGH_ORDER_BIT:
        return val ^ MODULO

    return val


def ff_add(*args):
    # start with 0
    result = 0

    # for each argument...
    for a in args:
        # XOR the argument with the sum
        result ^= a

    # return the sum
    return result


def key_expansion(cipher_key):
    key_schedule = np.copy(cipher_key)
    n_k = len(cipher_key)
    n_r = n_k + 6

    # print '               After     After      Rcon      XOR'
    # print ' i  Previous  RotWord   SubWord    Value    w/ Rcon   w[i-Nk]    Final'
    # print '=== ========  ========  ========  ========  ========  ========  ========'

    for i in range(n_k, N_B * (n_r + 1)):
        prev = key_schedule[i - 1]
        first = key_schedule[i - n_k]
        temp = prev
        # after_rot_word = None
        # after_sub_word = None
        # rcon_val = None
        # after_xor_rcon = None

        if i % n_k == 0:
            after_rot_word = rot_word(prev)
            after_sub_word = sub_word(after_rot_word)
            rcon_val = rcon(i / n_k)
            after_xor_rcon = np.bitwise_xor(after_sub_word, rcon_val)
            temp = after_xor_rcon
        elif n_k > 6 and i % n_k == 4:
            after_sub_word = sub_word(prev)
            temp = after_sub_word

        final = np.bitwise_xor(first, temp)

        key_schedule = np.append(key_schedule, final.reshape(1, 4), axis=0)

        # print '{:02}: {}  {}  {}  {}  {}  {}  {}'.format(i, w2s(prev), w2s(after_rot_word), w2s(after_sub_word),
        #                                                  w2s(rcon_val), w2s(after_xor_rcon), w2s(first), w2s(final))

    return key_schedule


def rcon(i):
    # check if this 'i' is in the cache already, if so then just use that
    if i in R_CON_CACHE:
        return R_CON_CACHE[i]

    # create a new word of all 0s
    result = np.zeros(4, np.uint16)

    # if i is too low, then just return the all 0s
    if i <= 0:
        return result

    # compute the 0th nibble of the word
    result[0] = x_time(1, 1 << (i - 1))

    # save the word in the cache
    R_CON_CACHE[i] = result

    return result


def w2s(word):
    if word is None:
        return '        '

    return '{:02x}{:02x}{:02x}{:02x}'.format(word[0], word[1], word[2], word[3])


def print_word(word):
    print w2s(word)


def print_state(state):
    transform = state.T

    for i in range(len(transform)):
        print_word(transform[i])


def do_128_bit_encrypt():
    # the plain 'text' is hard-coded from the FIPS-197 spec, Appendix B
    input_string = '32 43 f6 a8 88 5a 30 8d 31 31 98 a2 e0 37 07 34'

    # convert the input string to an array of numbers
    input = np.array([int(num, 16) for num in input_string.split()]).reshape(N_B, N_B)

    # the key string is hard-coded from the FIPS-197 spec, Appendix B
    key_string = '2b 7e 15 16 28 ae d2 a6 ab f7 15 88 09 cf 4f 3c'

    # convert key string to an array of numbers
    cipher_key = np.array([int(num, 16) for num in key_string.split()]).reshape(4, 4)

    # compute the key schedule
    key_schedule = key_expansion(cipher_key)

    # perform the encryption
    output = cipher(input, key_schedule, n_k=4)

    print_state(output)


# def do_192_bit_encrypt():
#     # the plain 'text' is hard-coded from the FIPS-197 spec, Appendix B
#     input_string = '32 43 f6 a8 88 5a 30 8d 31 31 98 a2 e0 37 07 34'
#
#     # convert the input string to an array of numbers
#     input = np.array([int(num, 16) for num in input_string.split()]).reshape(N_B, N_B)
#
#     # the key string is hard-coded from the FIPS-197 spec, Appendix B
#     key_string = '2b 7e 15 16 28 ae d2 a6 ab f7 15 88 09 cf 4f 3c'
#
#     # convert key string to an array of numbers
#     cipher_key = np.array([int(num, 16) for num in key_string.split()]).reshape(4, 4)
#
#     # compute the key schedule
#     key_schedule = key_expansion(cipher_key)
#
#     # perform the encryption
#     output = cipher(input, key_schedule, n_k = 4)
#
#     print_state(output)


do_128_bit_encrypt()