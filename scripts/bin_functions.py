
def jump_bin_function(
        image,
        i,
        value
        func,
        bin_size
):
    mapped_jump = func(jump, bin_size)
    if mapped_jump < 1:
        key = int((mapped_jump * b) // 1)
        image[i, key] += 1