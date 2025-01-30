#include <iostream>
#include <vector>
#include <cstring>
#include <random>
#include <chrono>
#include <iomanip>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Constantes para CUDA
#define BLOCK_SIZE 256
#define MAX_BLOCKS 65535

// Error checking macro for CUDA
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
}

// Constantes del dispositivo
__constant__ uint8_t d_SBOX[256];
__constant__ uint8_t d_RCON[10];
__constant__ uint8_t d_MIX_COLUMNS_MATRIX[4][4];

// S-box para SubBytes (host)
const uint8_t h_SBOX[256] = {
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e
};

// Rcon (host)
const uint8_t h_RCON[10] = {
    0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36
};

// Matriz MixColumns (host)
const uint8_t h_MIX_COLUMNS_MATRIX[4][4] = {
    {0x02, 0x03, 0x01, 0x01},
    {0x01, 0x02, 0x03, 0x01},
    {0x01, 0x01, 0x02, 0x03},
    {0x03, 0x01, 0x01, 0x02}
};

// Funciones device para operaciones AES
__device__ uint8_t gmul(uint8_t a, uint8_t b) {
    uint8_t p = 0;
    uint8_t counter;
    uint8_t hi_bit_set;
    for(counter = 0; counter < 8; counter++) {
        if((b & 1) == 1) 
            p ^= a;
        hi_bit_set = (a & 0x80);
        a <<= 1;
        if(hi_bit_set == 0x80) 
            a ^= 0x1b;
        b >>= 1;
    }
    return p;
}

__device__ void sub_bytes(uint8_t state[4][4]) {
    for(int i = 0; i < 4; i++)
        for(int j = 0; j < 4; j++)
            state[i][j] = d_SBOX[state[i][j]];
}

__device__ void shift_rows(uint8_t state[4][4]) {
    uint8_t temp;
    
    // Row 1
    temp = state[1][0];
    state[1][0] = state[1][1];
    state[1][1] = state[1][2];
    state[1][2] = state[1][3];
    state[1][3] = temp;
    
    // Row 2
    temp = state[2][0];
    state[2][0] = state[2][2];
    state[2][2] = temp;
    temp = state[2][1];
    state[2][1] = state[2][3];
    state[2][3] = temp;
    
    // Row 3
    temp = state[3][3];
    state[3][3] = state[3][2];
    state[3][2] = state[3][1];
    state[3][1] = state[3][0];
    state[3][0] = temp;
}

__device__ void mix_columns(uint8_t state[4][4]) {
    uint8_t temp[4];
    for(int c = 0; c < 4; c++) {
        for(int i = 0; i < 4; i++) {
            temp[i] = 0;
            for(int j = 0; j < 4; j++) {
                temp[i] ^= gmul(d_MIX_COLUMNS_MATRIX[i][j], state[j][c]);
            }
        }
        for(int i = 0; i < 4; i++) {
            state[i][c] = temp[i];
        }
    }
}

__device__ void add_round_key(uint8_t state[4][4], const uint32_t* round_key) {
    for(int c = 0; c < 4; c++) {
        uint32_t word = round_key[c];
        state[0][c] ^= (word >> 24) & 0xFF;
        state[1][c] ^= (word >> 16) & 0xFF;
        state[2][c] ^= (word >> 8) & 0xFF;
        state[3][c] ^= word & 0xFF;
    }
}

// Kernel principal para AES-CTR
__global__ void aes_ctr_kernel(const uint8_t* input, uint8_t* output, const uint32_t* expanded_key,
                              const uint8_t* nonce, unsigned long long len) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (len + 15) / 16) return;
    
    uint8_t counter[16];
    memcpy(counter, nonce, 12);
    
    // Set counter value for this block
    uint32_t block_counter = idx;
    counter[12] = (block_counter >> 24) & 0xFF;
    counter[13] = (block_counter >> 16) & 0xFF;
    counter[14] = (block_counter >> 8) & 0xFF;
    counter[15] = block_counter & 0xFF;
    
    uint8_t state[4][4];
    uint8_t keystream[16];
    
    // Initialize state from counter
    for(int i = 0; i < 4; i++)
        for(int j = 0; j < 4; j++)
            state[j][i] = counter[i*4 + j];
    
    // Initial round
    add_round_key(state, expanded_key);
    
    // Main rounds
    for(int round = 1; round < 10; round++) {
        sub_bytes(state);
        shift_rows(state);
        mix_columns(state);
        add_round_key(state, expanded_key + round*4);
    }
    
    // Final round
    sub_bytes(state);
    shift_rows(state);
    add_round_key(state, expanded_key + 40);
    
    // Copy state to keystream
    for(int i = 0; i < 4; i++)
        for(int j = 0; j < 4; j++)
            keystream[i*4 + j] = state[j][i];
    
    // XOR with input
    unsigned long long base = idx * 16;
    for(int i = 0; i < 16 && base + i < len; i++) {
        output[base + i] = input[base + i] ^ keystream[i];
    }
    
}

class AES_CTR_CUDA {
private:
    uint32_t* d_expanded_key;
    uint8_t* d_nonce;
    std::vector<uint32_t> h_expanded_key;
    static constexpr int NUM_STREAMS = 2;
    cudaStream_t streams[NUM_STREAMS];

    void expand_key(const uint8_t* key) {
        h_expanded_key.resize(44);

        // First 4 words are the original key
        for (int i = 0; i < 4; i++) {
            h_expanded_key[i] = (key[4 * i] << 24) | (key[4 * i + 1] << 16) |
                                (key[4 * i + 2] << 8) | key[4 * i + 3];
        }

        // Expand the rest of the key
        for (int i = 4; i < 44; i++) {
            uint32_t temp = h_expanded_key[i - 1];
            if (i % 4 == 0) {
                temp = ((temp << 8) | (temp >> 24)) & 0xFFFFFFFF;
                uint8_t* temp_bytes = reinterpret_cast<uint8_t*>(&temp);
                for (int j = 0; j < 4; j++) {
                    temp_bytes[j] = h_SBOX[temp_bytes[j]];
                }
                temp ^= h_RCON[i / 4 - 1] << 24;
            }
            h_expanded_key[i] = h_expanded_key[i - 4] ^ temp;
        }

        // Copy expanded key to device
        CHECK_CUDA(cudaMalloc(&d_expanded_key, 44 * sizeof(uint32_t)));
        CHECK_CUDA(cudaMemcpy(d_expanded_key, h_expanded_key.data(),
                              44 * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }

public:
    AES_CTR_CUDA(const uint8_t* key, const uint8_t* nonce) {
        // Copy constants to device
        CHECK_CUDA(cudaMemcpyToSymbol(d_SBOX, h_SBOX, sizeof(h_SBOX)));
        CHECK_CUDA(cudaMemcpyToSymbol(d_RCON, h_RCON, sizeof(h_RCON)));
        CHECK_CUDA(cudaMemcpyToSymbol(d_MIX_COLUMNS_MATRIX, h_MIX_COLUMNS_MATRIX, sizeof(h_MIX_COLUMNS_MATRIX)));

        // Expand key
        expand_key(key);

        // Copy nonce to device
        CHECK_CUDA(cudaMalloc(&d_nonce, 12));
        CHECK_CUDA(cudaMemcpy(d_nonce, nonce, 12, cudaMemcpyHostToDevice));

        // Create CUDA streams
        for (int i = 0; i < NUM_STREAMS; i++) {
            CHECK_CUDA(cudaStreamCreate(&streams[i]));
        }
    }

    ~AES_CTR_CUDA() {
        cudaFree(d_expanded_key);
        cudaFree(d_nonce);
        for (int i = 0; i < NUM_STREAMS; i++) {
            CHECK_CUDA(cudaStreamDestroy(streams[i]));
        }
    }

    static std::vector<uint8_t> generate_nonce() {
        std::vector<uint8_t> nonce(12);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 255);

        for (int i = 0; i < 12; i++) {
            nonce[i] = dis(gen);
        }
        return nonce;
    }

    void process(const uint8_t* input, uint8_t* output, unsigned long long total_len) {
        const unsigned long long CHUNK_SIZE = 1ULL * 1024 * 1024 * 1024; // 1GB
        uint8_t *d_input, *d_output;

        CHECK_CUDA(cudaMalloc(&d_input, CHUNK_SIZE));
        CHECK_CUDA(cudaMalloc(&d_output, CHUNK_SIZE));

        for (unsigned long long offset = 0; offset < total_len; offset += CHUNK_SIZE * NUM_STREAMS) {
            for (int i = 0; i < NUM_STREAMS && offset + i * CHUNK_SIZE < total_len; i++) {
                unsigned long long current_chunk_size = std::min(CHUNK_SIZE, total_len - (offset + i * CHUNK_SIZE));
                unsigned long long num_blocks = (current_chunk_size + 15) / 16;
                dim3 block_dim(BLOCK_SIZE);
                dim3 grid_dim((num_blocks + BLOCK_SIZE - 1) / BLOCK_SIZE);
                cudaStream_t current_stream = streams[i];

                CHECK_CUDA(cudaMemcpyAsync(d_input, input + offset + i * CHUNK_SIZE,
                                           current_chunk_size, cudaMemcpyHostToDevice,
                                           current_stream));

                aes_ctr_kernel<<<grid_dim, block_dim, 0, current_stream>>>(
                    d_input,
                    d_output,
                    d_expanded_key,
                    d_nonce,
                    current_chunk_size
                );

                CHECK_CUDA(cudaMemcpyAsync(output + offset + i * CHUNK_SIZE, d_output,
                                           current_chunk_size, cudaMemcpyDeviceToHost,
                                           current_stream));
            }
            for (int i = 0; i < NUM_STREAMS; i++) {
                CHECK_CUDA(cudaStreamSynchronize(streams[i]));
            }
        }

        CHECK_CUDA(cudaFree(d_input));
        CHECK_CUDA(cudaFree(d_output));
    }
};



int main() {
    try {
        // Initialize CUDA and get device properties
        cudaDeviceProp prop;
        CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
        std::cout << "Using GPU: " << prop.name << std::endl;
        
        // Key (128 bits)
        uint8_t key[16] = {
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
            0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F
        };

        // Generate random nonce
        auto nonce = AES_CTR_CUDA::generate_nonce();

        // Create CUDA AES-CTR instance
        AES_CTR_CUDA aes(key, nonce.data());

        // Test sizes
        std::vector<unsigned long long> message_sizes = {
            1ULL * 1024 * 1024,              // 1MB
            10ULL * 1024 * 1024,             // 10MB
            1ULL * 1024 * 1024 * 1024,       // 1GB
            6ULL * 1024 * 1024 * 1024        // 6GB
        };

        for (unsigned long long size : message_sizes) {
            try {
                std::cout << "\nProcessing " << (size / (1024.0 * 1024.0)) << " MB..." << std::endl;
                
                // Create test vectors with regular memory
                std::vector<uint8_t> message, ciphertext;
                message.resize(size, 'A');
                ciphertext.resize(size);

                // Measure encryption time
                auto start = std::chrono::high_resolution_clock::now();
                
                aes.process(message.data(), ciphertext.data(), size);
                
                // Ensure all GPU operations are complete
                CHECK_CUDA(cudaDeviceSynchronize());
                
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

                // Calculate speed
                double gbps = (size * 8.0 / 1000000000.0) / (duration.count() / 1000000.0);

                std::cout << "Encryption time: " << std::fixed << std::setprecision(2) 
                            << duration.count() / 1000.0 << " ms" << std::endl;
                std::cout << "Speed: " << gbps << " Gbps" << std::endl;
                std::cout << "-------------------" << std::endl;

                // Clear vectors to free memory immediately
                message.clear();
                message.shrink_to_fit();
                ciphertext.clear();
                ciphertext.shrink_to_fit();
                
            } catch (const std::bad_alloc& e) {
                std::cerr << "Memory allocation failed for size " << (size / (1024.0 * 1024.0)) 
                            << " MB: " << e.what() << std::endl;
                continue;
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

