#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <cublas_v2.h>

inline int idx(int r, int c, int W) {
    return r * W + c;
}

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, const char *func, const char *file, int line) {
    if (result) {
        std::cerr << "CUDA error: " << cudaGetErrorString(result)
                  << " at " << file << ":" << line << std::endl;
        exit(1);
    }
}

#define CHECK_CUBLAS(call) \
    if ((call) != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error\n"; exit(1); \
    }

// ✅ im2col (FLOAT version)
__global__ void im2col(float* d_image, float* d_M, int W, int H, int K) {

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row > H - K || col > W - K) return;

    int out_W = W - K + 1;
    int patch_idx = row * out_W + col;

    for (int i = 0; i < K; i++) {
        for (int j = 0; j < K; j++) {
            int img_idx = (row + i) * W + (col + j);
            int m_idx = patch_idx * (K*K) + i*K + j;
            d_M[m_idx] = d_image[img_idx];
        }
    }
}

int main() {

    cv::Mat img = cv::imread("Image_created_with_a_mobile_phone.png", cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Image load failed\n";
        return -1;
    }

    int H = img.rows, W = img.cols;
    int K = 3;
    int C_filters = 2;

    int out_H = H - K + 1;
    int out_W = W - K + 1;
    int num_patches = out_H * out_W;
    int patch_size = K * K;

    // Host image
    std::vector<float> image(H * W);
    for(int i=0;i<H;i++)
        for(int j=0;j<W;j++)
            image[idx(i,j,W)] = (float)img.at<uchar>(i,j);

    // Filters (2 × 9)
    std::vector<float> F = {
         1,2,1, 0,0,0,-1,-2,-1,
         1,0,-1, 2,0,-2,1,0,-1
    };

    // Device memory
    float *d_image, *d_M, *d_F, *d_C;

    checkCudaErrors(cudaMalloc(&d_image, H*W*sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_M, patch_size*num_patches*sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_F, C_filters*patch_size*sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_C, C_filters*num_patches*sizeof(float)));

    cudaMemcpy(d_image, image.data(), H*W*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_F, F.data(), C_filters*patch_size*sizeof(float), cudaMemcpyHostToDevice);

    // Launch im2col
    dim3 threadsPerBlock(16, 16);
    dim3 grid(
        (W + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (H + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    cudaEvent_t start_im2col, stop_im2col;
    cudaEvent_t start_gemm, stop_gemm;

    cudaEventCreate(&start_im2col);
    cudaEventCreate(&stop_im2col);
    cudaEventCreate(&start_gemm);
    cudaEventCreate(&stop_gemm);

    cudaEventRecord(start_im2col);   // start event
    im2col<<<grid, threadsPerBlock>>>(d_image, d_M, W, H, K);
    cudaEventRecord(stop_im2col);    // stop event
    cudaEventSynchronize(stop_im2col); // wait for completion

    float time_im2col = 0;
    cudaEventElapsedTime(&time_im2col, start_im2col, stop_im2col);

    std::cout << "im2col kernel time: " << time_im2col << " ms\n";


    // cuBLAS GEMM
    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha=1.0f, beta=0.0f;

    int M = C_filters;
    int N = num_patches;
    int Kdim = patch_size;

    // C = F × M
    cudaEventRecord(start_gemm);    // start event

    cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, Kdim,
        &alpha,
        d_M, N,
        d_F, Kdim,
        &beta,
        d_C, N
    );

    cudaEventRecord(stop_gemm);     // stop event
    cudaEventSynchronize(stop_gemm);

    float time_gemm = 0;
    cudaEventElapsedTime(&time_gemm, start_gemm, stop_gemm);

    std::cout << "cuBLAS GEMM time: " << time_gemm << " ms\n";
    printf("Total Elapsed time : %.4f milliseconds\n", time_im2col + time_gemm);

    // Copy result
    std::vector<float> C(M*N);
    cudaMemcpy(C.data(), d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost);

    /*
    std::cout << "First 10 outputs:\n";
    for(int i=0;i<10;i++)
        std::cout << C[i] << " ";
    */
   
    // Cleanup
    cudaFree(d_image);
    cudaFree(d_M);
    cudaFree(d_F);
    cudaFree(d_C);
    cublasDestroy(handle);
    cudaEventDestroy(start_im2col);
    cudaEventDestroy(stop_im2col);
    cudaEventDestroy(start_gemm);
    cudaEventDestroy(stop_gemm);

    return 0;
}