#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <cublas_v2.h>

using namespace std;
using namespace cv;

inline int idx(int r, int c, int W) { return r * W + c; }

#define CHECK_CUDA(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, const char *func, const char *file, int line) {
    if (result) {
        cerr << "CUDA error: " << cudaGetErrorString(result)
             << " at " << file << ":" << line << endl;
        exit(1);
    }
}

#define CHECK_CUBLAS(call) \
    if ((call) != CUBLAS_STATUS_SUCCESS) { cerr << "cuBLAS error\n"; exit(1); }

__global__ void im2col_coalesced(float* d_image, float* d_M,
                                 int W, int H, int K,
                                 int row_offset, int out_W, int num_patches) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row_local = blockIdx.y * blockDim.y + threadIdx.y;

    int out_H = H - K + 1;
    if (col >= out_W || row_local >= out_H) return;

    int row = row_local + row_offset;
    if (row >= out_H) return;

    int patch_idx = row * out_W + col;
    for (int i = 0; i < K; i++)
        for (int j = 0; j < K; j++) {
            int img_idx = (row + i) * W + (col + j);
            int filter_idx = i * K + j;
            int m_idx = filter_idx * num_patches + patch_idx;
            d_M[m_idx] = d_image[img_idx];
        }
}

int main() {
    // ---------------- Load image ----------------
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return -1;
    }

    cv::Mat image = cv::imread(argv[1]);
    if (image.empty()) {
        std::cerr << "Error: Couldn't load input image." << std::endl;
        return -1;
    }
    cv::Mat img;

    cv::cvtColor(image, img, cv::COLOR_BGR2GRAY);
    // 1. Load a real image using OpenCV in Grayscale
    // std::string image_path = "Image_created_with_a_mobile_phone.png"; // Replace with your image
    // cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);


    int H = img.rows, W = img.cols;
    int K = 3, C_filters = 2;
    int out_H = H - K + 1, out_W = W - K + 1;
    int num_patches = out_H * out_W;
    int patch_size = K * K;

    cout << "Image size: " << W << "x" << H << endl;

    // ---------------- Pinned memory ----------------
    float *h_image, *h_output;
    CHECK_CUDA(cudaMallocHost(&h_image, H * W * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&h_output, C_filters * num_patches * sizeof(float)));

    for (int i = 0; i < H; i++)
        for (int j = 0; j < W; j++)
            h_image[idx(i,j,W)] = (float)img.at<uchar>(i,j);

    vector<float> F = {
         1, 2, 1, 0, 0, 0, -1,-2,-1,
         1, 0,-1, 2, 0,-2, 1, 0,-1
    };

    // ---------------- Streams ----------------
    int NSTREAMS = 5; // Fewer, bigger tiles → reduces overhead
    cout << "Using " << NSTREAMS << " streams\n";
    vector<cudaStream_t> streams(NSTREAMS);
    for (int i = 0; i < NSTREAMS; i++) CHECK_CUDA(cudaStreamCreate(&streams[i]));

    // ---------------- Device memory ----------------
    float *d_image, *d_M, *d_F, *d_C;
    CHECK_CUDA(cudaMalloc(&d_image, H * W * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_M, patch_size * num_patches * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_F, C_filters * patch_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, C_filters * num_patches * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_image, h_image, H * W * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_F, F.data(), C_filters * patch_size * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    float alpha = 1.f, beta = 0.f;

    // ---------------- Tiled pipeline ----------------
    int tile_h = (out_H + NSTREAMS - 1) / NSTREAMS;
    dim3 block(16,16);

    // Event for compute-only timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    
    for (int s = 0; s < NSTREAMS; s++) {
        int row_start = s * tile_h;
        if (row_start >= out_H) break;
        int rows = min(tile_h, out_H - row_start);
        int tile_patches = rows * out_W;

        dim3 grid((out_W + block.x - 1)/block.x, (rows + block.y - 1)/block.y);

        im2col_coalesced<<<grid, block, 0, streams[s]>>>(d_image, d_M, W, H, K, row_start, out_W, num_patches);

        cublasSetStream(handle, streams[s]);
        int patch_offset = row_start * out_W;
        CHECK_CUBLAS(cublasSgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            tile_patches,
            C_filters,
            patch_size,
            &alpha,
            d_M + patch_offset,
            num_patches,
            d_F,
            patch_size,
            &beta,
            d_C + patch_offset,
            num_patches
        ));
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float gpu_ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&gpu_ms, start, stop));
    cout << "GPU compute-only time: " << gpu_ms << " ms\n";

    // ---------------- Copy back ----------------
    CHECK_CUDA(cudaMemcpy(h_output, d_C, C_filters * num_patches * sizeof(float), cudaMemcpyDeviceToHost));

    // ---------------- Cleanup ----------------
    for (int i = 0; i < NSTREAMS; i++) cudaStreamDestroy(streams[i]);
    CHECK_CUDA(cudaFree(d_image));
    CHECK_CUDA(cudaFree(d_M));
    CHECK_CUDA(cudaFree(d_F));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFreeHost(h_image));
    CHECK_CUDA(cudaFreeHost(h_output));
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}
