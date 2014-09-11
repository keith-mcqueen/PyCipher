__author__ = 'keith'

modulo = 0x11b
high_order_bit = 0x100
min_x_time_bit = 0x01
default_x_time_bit = 0x02


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
    current_bit = 0x01

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


def ff_add(num1, num2):
    # XOR the 2 numbers
    return num1 ^ num2

# this should be 0x57 - good
# print '0x57 . 0x01 = ' + hex(ff_multiply(0x57, 0x01))
# print

# this should be 0xae - good
# print '0x57 . 0x02 = ' + hex(ff_multiply(0x57, 0x02))
# print

# this should be 0x47 - good
# print '0x57 . 0x04 = ' + hex(ff_multiply(0x57, 0x04))
# print

# this should be 0x8e - good
# print '0x57 . 0x08 = ' + hex(ff_multiply(0x57, 0x08))
# print

# this should be 0x07 - good
# print '0x57 . 0x10 = ' + hex(ff_multiply(0x57, 0x10))
# print

# this should be 0x07 - good
# print '0x57 . 0x13 = ' + hex(ff_multiply(0x57, 0x13))
# print

# print hex(x_time(x_time(x_time(x_time(0x57)))))
