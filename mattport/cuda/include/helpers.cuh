/*
 * Helper code. 
 */

#define CUDA_NUM_BLOCKS_NEEDED(Q, CUDA_N_THREADS) ((Q - 1) / CUDA_N_THREADS + 1)