import numpy as np
import string

captcha_len = 5
image_size = (200, 50)

syms = list(string.ascii_lowercase) + list(string.digits)
charset_len = len(syms)

positions = range(0, len(syms))
syms_dict = dict(zip(syms, positions))
pos_dict = dict(zip(positions, syms))


def onehot_encode(captcha_str):
    onehot = np.zeros((len(syms), len(captcha_str)), dtype=np.float32)
    for i, letter in enumerate(captcha_str):
        onehot[syms_dict[letter], i] = 1
    return onehot.flatten()


def onehot_decode(vec):
    onehot_mat = vec.reshape((len(syms), -1))
    out = []
    for i in range(0, onehot_mat.shape[1]):
        ind = int(np.argmax(onehot_mat[:, i]))
        out.append(pos_dict[ind])
    return "".join(out)
