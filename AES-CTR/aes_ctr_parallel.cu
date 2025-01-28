#include <iostream>
#include <vector>
#include <cstring>
#include <random>
#include <chrono>
#include <iomanip>
#include <cuda_runtime.h>

// Constantes del dispositivo
__constant__ uint8_t d_SBOX[256];
__constant__ uint8_t d_RCON[10];
__constant__ uint8_t d_MIX_COLUMNS_MATRIX[4][4];

// S-box y otras constantes en el host (igual que en el código original)
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

const uint8_t h_RCON[10] = {
    0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36
};

const uint8_t h_MIX_COLUMNS_MATRIX[4][4] = {
    {0x02, 0x03, 0x01, 0x01},
    {0x01, 0x02, 0x03, 0x01},
    {0x01, 0x01, 0x02, 0x03},
    {0x03, 0x01, 0x01, 0x02}
};

// Funciones auxiliares en el dispositivo
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
    
    // Fila 1
    temp = state[1][0];
    state[1][0] = state[1][1];
    state[1][1] = state[1][2];
    state[1][2] = state[1][3];
    state[1][3] = temp;
    
    // Fila 2
    temp = state[2][0];
    state[2][0] = state[2][2];
    state[2][2] = temp;
    temp = state[2][1];
    state[2][1] = state[2][3];
    state[2][3] = temp;
    
    // Fila 3
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
__global__ void aes_ctr_kernel(
    const uint8_t* input,
    uint8_t* output,
    const uint32_t* expanded_key,
    const uint8_t* nonce,
    unsigned long long total_blocks
) {
    unsigned long long block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= total_blocks) return;

    // Preparar el contador para este bloque
    uint8_t counter[16];
    memcpy(counter, nonce, 12);
    uint32_t block_counter = block_idx;
    counter[12] = (block_counter >> 24) & 0xFF;
    counter[13] = (block_counter >> 16) & 0xFF;
    counter[14] = (block_counter >> 8) & 0xFF;
    counter[15] = block_counter & 0xFF;

    // State array para el proceso de encriptación
    uint8_t state[4][4];
    for(int i = 0; i < 4; i++)
        for(int j = 0; j < 4; j++)
            state[j][i] = counter[i*4 + j];

    // Proceso de encriptación AES
    add_round_key(state, expanded_key);
    
    for(int round = 1; round < 10; round++) {
        sub_bytes(state);
        shift_rows(state);
        mix_columns(state);
        add_round_key(state, expanded_key + round*4);
    }
    
    sub_bytes(state);
    shift_rows(state);
    add_round_key(state, expanded_key + 40);

    // XOR con el input
    unsigned long long base_idx = block_idx * 16;
    for(int i = 0; i < 4; i++) {
        for(int j = 0; j < 4; j++) {
            unsigned long long idx = base_idx + i*4 + j;
            if(idx < total_blocks * 16) {
                output[idx] = input[idx] ^ state[j][i];
            }
        }
    }
}

class AES_CTR_CUDA {
private:
    std::vector<uint32_t> expanded_key;
    std::vector<uint8_t> nonce;
    uint32_t* d_expanded_key;
    uint8_t* d_nonce;

    void expand_key(const uint8_t* key) {
        expanded_key.resize(44);
        
        for(int i = 0; i < 4; i++) {
            expanded_key[i] = (key[4*i] << 24) | (key[4*i+1] << 16) | 
                             (key[4*i+2] << 8) | key[4*i+3];
        }
        
        for(int i = 4; i < 44; i++) {
            uint32_t temp = expanded_key[i-1];
            if(i % 4 == 0) {
                temp = ((temp << 8) | (temp >> 24)) & 0xFFFFFFFF;
                uint8_t* temp_bytes = (uint8_t*)&temp;
                for(int j = 0; j < 4; j++)
                    temp_bytes[j] = h_SBOX[temp_bytes[j]];
                temp ^= h_RCON[i/4 - 1] << 24;
            }
            expanded_key[i] = expanded_key[i-4] ^ temp;
        }
    }

public:
    AES_CTR_CUDA(const uint8_t* key, const uint8_t* init_nonce) : nonce(12) {
        // Inicializar constantes en el dispositivo
        cudaMemcpyToSymbol(d_SBOX, h_SBOX, sizeof(h_SBOX));
        cudaMemcpyToSymbol(d_RCON, h_RCON, sizeof(h_RCON));
        cudaMemcpyToSymbol(d_MIX_COLUMNS_MATRIX, h_MIX_COLUMNS_MATRIX, sizeof(h_MIX_COLUMNS_MATRIX));

        // Expandir clave y copiar al dispositivo
        expand_key(key);
        cudaMalloc(&d_expanded_key, expanded_key.size() * sizeof(uint32_t));
        cudaMemcpy(d_expanded_key, expanded_key.data(), expanded_key.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);

        // Copiar nonce al dispositivo
        memcpy(nonce.data(), init_nonce, 12);
        cudaMalloc(&d_nonce, 12);
        cudaMemcpy(d_nonce, nonce.data(), 12, cudaMemcpyHostToDevice);
    }

    ~AES_CTR_CUDA() {
        cudaFree(d_expanded_key);
        cudaFree(d_nonce);
    }

    void process(const uint8_t* input, uint8_t* output, unsigned long long len) {
        unsigned long long total_blocks = (len + 15) / 16;
        
        // Asignar memoria en el dispositivo
        uint8_t *d_input, *d_output;
        cudaMalloc(&d_input, len);
        cudaMalloc(&d_output, len);
        
        // Copiar datos de entrada al dispositivo
        cudaMemcpy(d_input, input, len, cudaMemcpyHostToDevice);
        
        // Configurar y lanzar kernel
        int block_size = 256;
        int grid_size = (total_blocks + block_size - 1) / block_size;
        
        aes_ctr_kernel<<<grid_size, block_size>>>(
            d_input, d_output, d_expanded_key, d_nonce, total_blocks
        );
        
        // Copiar resultado de vuelta al host
        cudaMemcpy(output, d_output, len, cudaMemcpyDeviceToHost);
        
        // Liberar memoria
        cudaFree(d_input);
        cudaFree(d_output);
    }
};

int main() {
    // Clave de ejemplo
    uint8_t key[16] = {
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
        0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F
    };

    // Generar nonce
    std::vector<uint8_t> nonce(12);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);
    for(int i = 0; i < 12; i++) {
        nonce[i] = dis(gen);
    }

    // Crear instancia de AES-CTR CUDA
    AES_CTR_CUDA aes(key, nonce.data());

    // Tamaños de prueba
    std::vector<unsigned long long> message_sizes = {
        1ULL * 1024 * 1024,              // 1MB
        10ULL * 1024 * 1024,             // 10MB
        1ULL * 1024 * 1024 * 1024,       // 1GB
        10ULL * 1024 * 1024 * 1024       // 10GB
    };

    for (unsigned long long size : message_sizes) {
        // Crear mensaje de prueba
        std::vector<uint8_t> message(size, 'A');
        std::vector<uint8_t> ciphertext(size);

        // Medir tiempo
        auto start = std::chrono::high_resolution_clock::now();
        
        aes.process(message.data(), ciphertext.data(), size);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        // Calcular y mostrar estadísticas
        double mbps = (size * 8.0 / 1000000.0) / (duration.count() / 1000000.0);

        std::cout << "Tamaño del mensaje: " << std::fixed << std::setprecision(2) 
                  << (size / 1024.0 / 1024.0) << " MB" << std::endl;
        std::cout << "Tiempo de encriptación: " << duration.count() / 1000.0 << " ms" << std::endl;
        std::cout << "Velocidad: " << mbps << " Mbps" << std::endl;
        std::cout << "-------------------" << std::endl;
    }

    return 0;
}