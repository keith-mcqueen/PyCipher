__author__ = 'keith'


def ff_multiply(operand, times):
    # if input or times <= 0, then return 0
    if operand <= 0 or times <= 0:
        return 0

    # if input or times is 1, then return the other
    if operand == 1:
        return times
    if times == 1:
        return operand

    product = x_time(operand) + ff_multiply(operand, times - 2)
    # product = x_time(operand) + ff_multiply(x_time(operand), times - 2)
    # product = x_time(operand)
    # product += ff_multiply(product, times - 2)

    return normalize(product)


def x_time(operand):
    # multiply by 2 (left shift bits one place)
    return normalize(operand << 1)


def normalize(num):
    # if the product is too big, then modulo it by 0x11b
    if num > 0x11b:
        #return normalize(num ^ 0x11b)
        return num ^ 0x11b

    return num


def ff_add(num1, num2):
    # XOR the 2 numbers
    return num1 ^ num2

# this should be 0x57 - good
print '0x57 . 0x01 = ' + hex(ff_multiply(0x57, 0x01))
print

# this should be 0xae - good
print '0x57 . 0x02 = ' + hex(ff_multiply(0x57, 0x02))
print

# this should be 0x47 - good
print '0x57 . 0x04 = ' + hex(ff_multiply(0x57, 0x04))
print

# there is something wrong with my logic on how I'm doing this.  It may be that the problem is in my recursion, but I'm
# really sure.
# this should be 0x8e - wrong: I'm getting 0xb8
print '0x57 . 0x08 = ' + hex(ff_multiply(0x57, 0x08))
print

# this should be 0x07 - wrong: I'm getting 0x97
print '0x57 . 0x10 = ' + hex(ff_multiply(0x57, 0x10))
print

print(hex(x_time(0x8e)))