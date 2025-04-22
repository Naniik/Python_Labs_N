
import time
import os
import matplotlib.pyplot as plt
import numpy as np

Nb = 4
Nk_128, Nr_128 = 4, 10
Nk_256, Nr_256 = 8, 14

s_box = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5,
    0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0,
    0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc,
    0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a,
    0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0,
    0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b,
    0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85,
    0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5,
    0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17,
    0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88,
    0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c,
    0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9,
    0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6,
    0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e,
    0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94,
    0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68,
    0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
]


inv_s_box = [
    0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38,
    0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
    0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87,
    0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
    0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d,
    0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
    0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2,
    0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
    0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16,
    0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
    0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda,
    0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
    0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a,
    0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
    0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02,
    0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
    0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea,
    0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
    0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85,
    0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
    0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89,
    0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
    0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20,
    0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
    0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31,
    0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
    0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d,
    0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
    0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0,
    0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26,
    0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d
]

Rcon = [
    0x01, 0x02, 0x04, 0x08, 0x10,
    0x20, 0x40, 0x80, 0x1B, 0x36,
    0x6C, 0xD8, 0xAB, 0x4D, 0x9A
]


def sub_bytes(state):
    return [[s_box[byte] for byte in row] for row in state]

def inv_sub_bytes(state):
    return [[inv_s_box[byte] for byte in row] for row in state]

def shift_rows(state):
    return [
        state[0],
        state[1][1:] + state[1][:1],
        state[2][2:] + state[2][:2],
        state[3][3:] + state[3][:3]
    ]

def inv_shift_rows(state):
    return [
        state[0],
        state[1][-1:] + state[1][:-1],
        state[2][-2:] + state[2][:-2],
        state[3][-3:] + state[3][:-3]
    ]


def xtime(a):
    return ((a << 1) ^ 0x1B) & 0xFF if a & 0x80 else a << 1

def mix_single_column(col):
    t = col[0] ^ col[1] ^ col[2] ^ col[3]
    u = col[0]
    col[0] ^= t ^ xtime(col[0] ^ col[1])
    col[1] ^= t ^ xtime(col[1] ^ col[2])
    col[2] ^= t ^ xtime(col[2] ^ col[3])
    col[3] ^= t ^ xtime(col[3] ^ u)
    return col

def mix_columns(state):
    for i in range(4):
        col = [state[j][i] for j in range(4)]
        col = mix_single_column(col)
        for j in range(4):
            state[j][i] = col[j]
    return state


def gmul(a, b):
    p = 0
    for counter in range(8):
        if b & 1:
            p ^= a
        hi_bit_set = a & 0x80
        a <<= 1
        if hi_bit_set:
            a ^= 0x1b
        b >>= 1
    return p & 0xff


def inv_mix_columns(state):
    for i in range(4):
        s0 = gmul(0x0e, state[0][i]) ^ gmul(0x0b, state[1][i]) ^ gmul(0x0d, state[2][i]) ^ gmul(0x09, state[3][i])
        s1 = gmul(0x09, state[0][i]) ^ gmul(0x0e, state[1][i]) ^ gmul(0x0b, state[2][i]) ^ gmul(0x0d, state[3][i])
        s2 = gmul(0x0d, state[0][i]) ^ gmul(0x09, state[1][i]) ^ gmul(0x0e, state[2][i]) ^ gmul(0x0b, state[3][i])
        s3 = gmul(0x0b, state[0][i]) ^ gmul(0x0d, state[1][i]) ^ gmul(0x09, state[2][i]) ^ gmul(0x0e, state[3][i])
        state[0][i] = s0
        state[1][i] = s1
        state[2][i] = s2
        state[3][i] = s3
    return state


def add_round_key(state, round_key):
    for i in range(4):
        for j in range(4):
            state[j][i] ^= round_key[i][j]
    return state


def bytes_to_matrix(text_bytes):
    return [list(text_bytes[i:i + 4]) for i in range(0, len(text_bytes), 4)]


def matrix_to_bytes(matrix):
    return bytes(sum(matrix, []))


def key_expansion(key, Nk, Nr):
    key_schedule = [[] for _ in range(4)]
    for r in range(4):
        for c in range(Nk):
            key_schedule[r].append(key[r + 4 * c])

    for col in range(Nk, Nb * (Nr + 1)):
        temp = [key_schedule[row][col - 1] for row in range(4)]
        if col % Nk == 0:
            temp = temp[1:] + temp[:1]
            temp = [s_box[b] for b in temp]
            temp[0] ^= Rcon[(col // Nk) - 1]
        elif Nk > 6 and col % Nk == 4:
            temp = [s_box[b] for b in temp]
        for row in range(4):
            key_schedule[row].append(key_schedule[row][col - Nk] ^ temp[row])

    round_keys = []
    for r in range(0, len(key_schedule[0]), 4):
        round_key = [[key_schedule[row][r + c] for c in range(4)] for row in range(4)]
        round_keys.append(round_key)
    return round_keys


def encrypt_block(plaintext, round_keys, Nr):
    state = bytes_to_matrix(plaintext)
    state = add_round_key(state, round_keys[0])

    for rnd in range(1, Nr):
        state = sub_bytes(state)
        state = shift_rows(state)
        state = mix_columns(state)
        state = add_round_key(state, round_keys[rnd])

    state = sub_bytes(state)
    state = shift_rows(state)
    state = add_round_key(state, round_keys[Nr])

    return matrix_to_bytes(state)


def decrypt_block(ciphertext, round_keys, Nr):
    state = bytes_to_matrix(ciphertext)
    state = add_round_key(state, round_keys[Nr])

    for rnd in range(Nr - 1, 0, -1):
        state = inv_shift_rows(state)
        state = inv_sub_bytes(state)
        state = add_round_key(state, round_keys[rnd])
        state = inv_mix_columns(state)

    state = inv_shift_rows(state)
    state = inv_sub_bytes(state)
    state = add_round_key(state, round_keys[0])

    return matrix_to_bytes(state)

def pad(data, block_size=16):
    padding_len = block_size - (len(data) % block_size)
    padding = bytes([padding_len] * padding_len)
    return data + padding


def unpad(data):
    padding_len = data[-1]
    if padding_len > 16:
        raise ValueError("Помилка")
    for i in range(1, padding_len + 1):
        if data[-i] != padding_len:
            raise ValueError("Помилка")
    return data[:-padding_len]


class AES:
    MODE_ECB = 1
    MODE_CBC = 2
    block_size = 16

    def __init__(self, key, mode, iv=None):
        self.key = key
        self.mode = mode

        if mode == self.MODE_CBC and iv is None:
            raise ValueError("CBC режим потребує ІV")

        self.iv = iv

        key_len = len(key)
        if key_len == 16:  # AES-128
            self.Nk, self.Nr = Nk_128, Nr_128
        elif key_len == 32:  # AES-256
            self.Nk, self.Nr = Nk_256, Nr_256
        else:
            raise ValueError("Довжина ключа повина бути 16 (AES-128) чи 32 (AES-256) байти")

        self.round_keys = key_expansion(list(key), self.Nk, self.Nr)

    def encrypt(self, data):
        padded_data = pad(data, self.block_size)
        result = b''

        if self.mode == self.MODE_ECB:
            for i in range(0, len(padded_data), self.block_size):
                block = padded_data[i:i + self.block_size]
                result += encrypt_block(block, self.round_keys, self.Nr)

        elif self.mode == self.MODE_CBC:
            prev_block = self.iv
            for i in range(0, len(padded_data), self.block_size):
                block = padded_data[i:i + self.block_size]
                xored_block = bytes(x ^ y for x, y in zip(block, prev_block))
                encrypted_block = encrypt_block(xored_block, self.round_keys, self.Nr)
                result += encrypted_block
                prev_block = encrypted_block

        return result

    def decrypt(self, data):
        if len(data) % self.block_size != 0:
            raise ValueError("Довжина шифротексту має бути кратною розміру блоку")

        result = b''

        if self.mode == self.MODE_ECB:
            for i in range(0, len(data), self.block_size):
                block = data[i:i + self.block_size]
                result += decrypt_block(block, self.round_keys, self.Nr)

        elif self.mode == self.MODE_CBC:
            prev_block = self.iv
            for i in range(0, len(data), self.block_size):
                block = data[i:i + self.block_size]
                decrypted_block = decrypt_block(block, self.round_keys, self.Nr)
                # XOR with previous ciphertext block or IV
                result += bytes(x ^ y for x, y in zip(decrypted_block, prev_block))
                prev_block = block

        return unpad(result)



def get_random_bytes(n):
    return os.urandom(n)


def benchmark(key_len_bits):
    if key_len_bits == 128:
        Nk, Nr = Nk_128, Nr_128
    else:
        Nk, Nr = Nk_256, Nr_256

    key = list(get_random_bytes(key_len_bits // 8))
    data = get_random_bytes(16)
    round_keys = key_expansion(key, Nk, Nr)

    start = time.time()
    for _ in range(100):
        encrypt_block(data, round_keys, Nr)
    end = time.time()
    return (end - start) * 1000  # ms



def aes_demo():
    print("\n[ДЕМО] Шифрування та розшифрування AES-128 в режимі ECB")
    key = get_random_bytes(16)
    plaintext = b"This is demo AES!"

    aes = AES(key, AES.MODE_ECB)
    ciphertext = aes.encrypt(plaintext)
    print("Зашифрований текст (hex):", ciphertext.hex())

    aes_dec = AES(key, AES.MODE_ECB)
    decrypted = aes_dec.decrypt(ciphertext)
    print("Розшифрований текст:", decrypted.decode())

    print("\n[ДЕМО] Шифрування та розшифрування AES-128 в режимі CBC")
    iv = get_random_bytes(16)
    aes_cbc = AES(key, AES.MODE_CBC, iv)
    ciphertext_cbc = aes_cbc.encrypt(plaintext)
    print("Зашифрований текст CBC (hex):", ciphertext_cbc.hex())

    aes_cbc_dec = AES(key, AES.MODE_CBC, iv)
    decrypted_cbc = aes_cbc_dec.decrypt(ciphertext_cbc)
    print("Розшифрований текст CBC:", decrypted_cbc.decode())

    print("\n[ДЕМО] Шифрування та розшифрування AES-256 в режимі ECB")
    key_256 = get_random_bytes(32)
    aes_256 = AES(key_256, AES.MODE_ECB)
    ciphertext_256 = aes_256.encrypt(plaintext)
    print("Зашифрований текст (hex):", ciphertext_256.hex())

    aes_256_dec = AES(key_256, AES.MODE_ECB)
    decrypted_256 = aes_256_dec.decrypt(ciphertext_256)
    print("Розшифрований текст:", decrypted_256.decode())

    print("\n[ДЕМО] Шифрування та розшифрування AES-256 в режимі CBC")
    iv_256 = get_random_bytes(16)
    aes_256_cbc = AES(key_256, AES.MODE_CBC, iv_256)
    ciphertext_256_cbc = aes_256_cbc.encrypt(plaintext)
    print("Зашифрований текст CBC (hex):", ciphertext_256_cbc.hex())

    aes_256_cbc_dec = AES(key_256, AES.MODE_CBC, iv_256)
    decrypted_256_cbc = aes_256_cbc_dec.decrypt(ciphertext_256_cbc)
    print("Розшифрований текст CBC:", decrypted_256_cbc.decode())


def plot_results():
    trials = 10
    times_128 = [benchmark(128) for _ in range(5)]
    times_256 = [benchmark(256) for _ in range(5)]

    plt.figure(figsize=(8, 5))
    plt.plot(times_128, label="AES-128")
    plt.plot(times_256, label="AES-256")
    plt.ylabel("Час (мс)")
    plt.xlabel("Спроба")
    plt.title("Порівняння часу AES-128 і AES-256")
    plt.legend()
    plt.grid(True)

    plt.axhline(y=np.mean(times_128), color='b', linestyle='--',
                label=f"AES-128 середнє: {np.mean(times_128):.2f} мс")
    plt.axhline(y=np.mean(times_256), color='orange', linestyle='--',
                label=f"AES-256 середнє: {np.mean(times_256):.2f} мс")
    plt.legend()
    plt.show()

    # Виводимо дані для порівняння
    print(f"AES-128 середній час: {np.mean(times_128):.3f} мс")
    print(f"AES-256 середній час: {np.mean(times_256):.3f} мс")
    print(f"Співвідношення (AES-256/AES-128): {np.mean(times_256) / np.mean(times_128):.2f}x")



def benchmark_modes():
    text = b"Message to encrypt. Block AES test."
    key_128 = get_random_bytes(16)
    key_256 = get_random_bytes(16)
    iv = get_random_bytes(16)
    ecb_128_times, cbc_128_times = [], []
    ecb_256_times, cbc_256_times = [], []

    for _ in range(10):
        # AES-128 ECB режим
        cipher_ecb_enc = AES(key_128, AES.MODE_ECB)
        start = time.time()
        enc = cipher_ecb_enc.encrypt(text)
        cipher_ecb_dec = AES(key_128, AES.MODE_ECB)
        dec = cipher_ecb_dec.decrypt(enc)
        ecb_128_times.append((time.time() - start) * 1000)

        # AES-128 CBC режим
        cipher_cbc_enc = AES(key_128, AES.MODE_CBC, iv)
        start = time.time()
        enc = cipher_cbc_enc.encrypt(text)
        cipher_cbc_dec = AES(key_128, AES.MODE_CBC, iv)
        dec = cipher_cbc_dec.decrypt(enc)
        cbc_128_times.append((time.time() - start) * 1000)

        # AES-256 ECB режим
        cipher_ecb_enc = AES(key_256, AES.MODE_ECB)
        start = time.time()
        enc = cipher_ecb_enc.encrypt(text)
        cipher_ecb_dec = AES(key_256, AES.MODE_ECB)
        dec = cipher_ecb_dec.decrypt(enc)
        ecb_256_times.append((time.time() - start) * 1000)

        # AES-256 CBC режим
        cipher_cbc_enc = AES(key_256, AES.MODE_CBC, iv)
        start = time.time()
        enc = cipher_cbc_enc.encrypt(text)
        cipher_cbc_dec = AES(key_256, AES.MODE_CBC, iv)
        dec = cipher_cbc_dec.decrypt(enc)
        cbc_256_times.append((time.time() - start) * 1000)

    # Графік AES-128 ECB vs CBC
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(ecb_128_times, 'o-', label="ECB-128", linewidth=2)
    plt.plot(cbc_128_times, 's-', label="CBC-128", linewidth=2)
    plt.title("Порівняння швидкодії AES-128: ECB vs CBC")
    plt.xlabel("Спроба")
    plt.ylabel("Час (мс)")
    plt.legend()
    plt.grid(True)

    plt.axhline(y=np.mean(ecb_128_times), color='b', linestyle='--',label=f"ECB-128 середнє: {np.mean(ecb_128_times):.2f} мс")
    plt.axhline(y=np.mean(cbc_128_times), color='orange', linestyle='--',label=f"CBC-128 середнє: {np.mean(cbc_128_times):.2f} мс")
    plt.legend()
    # Графік AES-256 ECB vs CBC
    plt.subplot(2, 1, 2)
    plt.plot(ecb_256_times, 'o-', label="ECB-256", linewidth=2)
    plt.plot(cbc_256_times, 's-', label="CBC-256", linewidth=2)
    plt.title("Порівняння швидкодії AES-256: ECB vs CBC")
    plt.xlabel("Спроба")
    plt.ylabel("Час (мс)")
    plt.legend()
    plt.grid(True)

    plt.axhline(y=np.mean(ecb_256_times), color='b', linestyle='--',label=f"ECB-256 середнє: {np.mean(ecb_256_times):.2f} мс")
    plt.axhline(y=np.mean(cbc_256_times), color='orange', linestyle='--',label=f"CBC-256 середнє: {np.mean(cbc_256_times):.2f} мс")
    plt.legend()
    plt.tight_layout()
    plt.show()


    print(f"AES-128 ECB середній час: {np.mean(ecb_128_times):.3f} мс")
    print(f"AES-128 CBC середній час: {np.mean(cbc_128_times):.3f} мс")
    print(f"Різниця (CBC - ECB) для AES-128: {np.mean(cbc_128_times) - np.mean(ecb_128_times):.3f} мс")

    print(f"\nAES-256 ECB середній час: {np.mean(ecb_256_times):.3f} мс")
    print(f"AES-256 CBC середній час: {np.mean(cbc_256_times):.3f} мс")
    print(f"Різниця (CBC - ECB) для AES-256: {np.mean(cbc_256_times) - np.mean(ecb_256_times):.3f} мс")


if __name__ == "__main__":
    plot_results()
    aes_demo()
    benchmark_modes()