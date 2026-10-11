#include <cstdint>
#include <cstddef>
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__GNUC__) || defined(__clang__)
#define EXPORT extern "C" __attribute__((visibility("default")))
#else
#define EXPORT extern "C"
#endif

namespace {
constexpr int SPECK32_WORD_BITS = 16;
constexpr int SPECK32_KEY_WORDS = 4;
constexpr int SPECK32_MAX_ROUNDS = 22;

static inline uint16_t ror16(const uint16_t x, const unsigned r) noexcept {
    return static_cast<uint16_t>((x >> r) | (x << (SPECK32_WORD_BITS - r)));
}

static inline uint16_t rol16(const uint16_t x, const unsigned r) noexcept {
    return static_cast<uint16_t>((x << r) | (x >> (SPECK32_WORD_BITS - r)));
}

static inline void speck_round(uint16_t &x, uint16_t &y, const uint16_t k) noexcept {
    x = static_cast<uint16_t>(ror16(x, 7) + y);
    x ^= k;
    y = rol16(y, 2);
    y ^= x;
}

static inline bool valid_rounds(const int rounds) noexcept {
    return rounds >= 1 && rounds <= SPECK32_MAX_ROUNDS;
}
static inline void expand_round_keys(
    const uint16_t key[SPECK32_KEY_WORDS],
    const int rounds,
    uint16_t round_keys[SPECK32_MAX_ROUNDS]
) noexcept {
    uint16_t b = key[0];
    uint16_t a0 = key[1];
    uint16_t a1 = key[2];
    uint16_t a2 = key[3];

    for (int i = 0; i < rounds; ++i) {
        round_keys[i] = b;
        if (i + 1 < rounds) {
            uint16_t a = a0;
            speck_round(a, b, static_cast<uint16_t>(i));
            a0 = a1;
            a1 = a2;
            a2 = a;
        }
    }
}

static inline void encrypt_with_round_keys(
    const uint16_t plaintext[2],
    const uint16_t round_keys[SPECK32_MAX_ROUNDS],
    const int rounds,
    uint16_t ciphertext[2]
) noexcept {
    uint16_t x = plaintext[0];
    uint16_t y = plaintext[1];
    for (int r = 0; r < rounds; ++r) {
        speck_round(x, y, round_keys[r]);
    }
    ciphertext[0] = x;
    ciphertext[1] = y;
}
}
EXPORT int generate_round_keys(
    const uint16_t *key,
    const int rounds,
    uint16_t *round_keys
) {
    if (key == nullptr || round_keys == nullptr) {
        return -1;
    }
    if (!valid_rounds(rounds)) {
        return -2;
    }
    expand_round_keys(key, rounds, round_keys);
    return 0;
}

EXPORT void encrypt(
    const uint16_t plaintext[2],
    const uint16_t key[4],
    const int rounds,
    uint16_t ciphertext[2]
) {
    if (plaintext == nullptr || key == nullptr || ciphertext == nullptr || !valid_rounds(rounds)) {
        if (ciphertext != nullptr) {
            ciphertext[0] = 0;
            ciphertext[1] = 0;
        }
        return;
    }
    uint16_t round_keys[SPECK32_MAX_ROUNDS] = {0};
    expand_round_keys(key, rounds, round_keys);
    encrypt_with_round_keys(plaintext, round_keys, rounds, ciphertext);
}


EXPORT void encrypt_batch(
    const uint16_t plaintext[][2],
    const uint16_t key[][4],//每个样本一组独立的密钥
    const int rounds,
    uint16_t ciphertext[][2],
    const int batch_size
) {
    if (plaintext == nullptr || key == nullptr || ciphertext == nullptr ||
        batch_size <= 0 || !valid_rounds(rounds)) {
        return;
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < batch_size; ++i) {
        uint16_t round_keys[SPECK32_MAX_ROUNDS] = {0};
        expand_round_keys(key[i], rounds, round_keys);
        encrypt_with_round_keys(plaintext[i], round_keys, rounds, ciphertext[i]);
    }
}


EXPORT int encrypt_polytope_batch(
    const uint16_t *plaintext0,
    const uint16_t *diffs,
    const int num_diffs,
    const uint16_t *key,
    const int rounds,
    uint16_t *ciphertexts,
    const int batch_size
) {
    if (plaintext0 == nullptr || key == nullptr || ciphertexts == nullptr) {
        return -1;
    }
    if (num_diffs < 0 || (num_diffs > 0 && diffs == nullptr)) {
        return -2;
    }
    if (batch_size <= 0 || !valid_rounds(rounds)) {
        return -3;
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < batch_size; ++i) {
        const uint16_t *sample_key = key + static_cast<std::size_t>(i) * SPECK32_KEY_WORDS;
        const uint16_t *anchor = plaintext0 + static_cast<std::size_t>(i) * 2;

        uint16_t round_keys[SPECK32_MAX_ROUNDS] = {0};
        expand_round_keys(sample_key, rounds, round_keys);

        for (int vertex = 0; vertex <= num_diffs; ++vertex) {
            uint16_t plaintext[2] = {anchor[0], anchor[1]};
            if (vertex > 0) {
                const uint16_t *difference = diffs + static_cast<std::size_t>(vertex - 1) * 2;
                plaintext[0] ^= difference[0];
                plaintext[1] ^= difference[1];
            }

            uint16_t *output = ciphertexts +
                (static_cast<std::size_t>(vertex) * batch_size + i) * 2;
            encrypt_with_round_keys(plaintext, round_keys, rounds, output);
        }
    }

    return 0;
}

EXPORT int speck32_self_test() {
    const uint16_t key[4] = {0x0100, 0x0908, 0x1110, 0x1918};
    const uint16_t plaintext[2] = {0x6574, 0x694c};
    uint16_t ciphertext[2] = {0, 0};
    encrypt(plaintext, key, 22, ciphertext);
    return (ciphertext[0] == 0xa868 && ciphertext[1] == 0x42f2) ? 1 : 0;
}