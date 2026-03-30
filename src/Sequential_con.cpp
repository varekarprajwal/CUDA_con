#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <time.h>

double getCurrentTime() {
    struct timespec currentTime;
    clock_gettime(CLOCK_MONOTONIC, &currentTime);
    return (double)currentTime.tv_sec * 1000.0 + (double)currentTime.tv_nsec / 1000000.0;
}

// Helper function for 1D flat array indexing
inline int idx(int row, int col, int total_cols) {
    return row * total_cols + col;
}

// ---------------------------------------------------------
// 1. The im2col Transformation (Now using float)
// ---------------------------------------------------------
std::vector<float> im2col(const std::vector<float>& image, int H, int W, int K) {
    int out_H = H - K + 1;
    int out_W = W - K + 1;
    int num_patches = out_H * out_W;
    int patch_size = K * K;

    std::vector<float> M(patch_size * num_patches, 0.0f);

    int col_idx = 0;
    for (int y = 0; y < out_H; ++y) {
        for (int x = 0; x < out_W; ++x) {
            int row_idx = 0;
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    M[idx(row_idx, col_idx, num_patches)] = image[idx(y + ky, x + kx, W)];
                    row_idx++;
                }
            }
            col_idx++;
        }
    }
    return M;
}

// ---------------------------------------------------------
// 2. Matrix Multiplication (GEMM)
// ---------------------------------------------------------
std::vector<float> matmul(const std::vector<float>& A, const std::vector<float>& B, int m, int k_dim, int n) {
    std::vector<float> C(m * n, 0.0f);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int p = 0; p < k_dim; ++p) {
                sum += A[idx(i, p, k_dim)] * B[idx(p, j, n)];
            }
            C[idx(i, j, n)] = sum;
        }
    }
    return C;
}

// ---------------------------------------------------------
// Main Execution
// ---------------------------------------------------------
int main(int argc, char* argv[]) {
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


    int H = img.rows;
    int W = img.cols;
    int K = 3;
    int C_filters = 2;

    std::cout << "Loaded image size: " << W << "x" << H << std::endl;

    // 2. Bridge OpenCV Mat to our flat std::vector<float>
    std::vector<float> image_vec(H * W);
    for(int i = 0; i < H; ++i) {
        for(int j = 0; j < W; ++j) {
            image_vec[idx(i, j, W)] = static_cast<float>(img.at<uchar>(i, j));
        }
    }

    // 3. Define Filters (e.g., Sobel Horizontal and Vertical)
    std::vector<float> F_mat = {
         1.0f,  2.0f,  1.0f,  0.0f,  0.0f,  0.0f, -1.0f, -2.0f, -1.0f, // Filter 1
         1.0f,  0.0f, -1.0f,  2.0f,  0.0f, -2.0f,  1.0f,  0.0f, -1.0f  // Filter 2
    };

    // 4. Run the Pipeline
    std::cout << "Running im2col..." << std::endl;
    double startTime1 = getCurrentTime();
    std::vector<float> M = im2col(image_vec, H, W, K);
    double endTime1 = getCurrentTime();
    int num_patches = (H - K + 1) * (W - K + 1);

    std::cout << "Running GEMM..." << std::endl;
    double startTime2 = getCurrentTime();
    std::vector<float> O = matmul(F_mat, M, C_filters, K * K, num_patches);
    double endTime2 = getCurrentTime();

    // 5. Bridge back to OpenCV for visualization
    int out_H = H - K + 1;
    int out_W = W - K + 1;

    // Create an empty OpenCV Mat for the first filter's output
    cv::Mat out_img_f1(out_H, out_W, CV_32F);

    for(int i = 0; i < out_H; ++i) {
        for(int j = 0; j < out_W; ++j) {
            // Read from the first row (Filter 1) of our flat output matrix O
            out_img_f1.at<float>(i, j) = O[idx(0, idx(i, j, out_W), num_patches)];
        }
    }

    // Normalize the raw floating point values to 0-255 for saving/displaying
    cv::Mat display_img;
    cv::normalize(out_img_f1, display_img, 0, 255, cv::NORM_MINMAX, CV_8U);

    //cv::imwrite("convolution_output.jpg", display_img);
    std::cout << "Success! Saved output to convolution_output.jpg" << std::endl;

    double totalTime1 = endTime1 - startTime1;
    double totalTime2 = endTime2 - startTime2;

    std::cout << "\n";
    printf("Elapsed time for im2col Transformation : %.4f milliseconds\n", totalTime1);
    printf("Elapsed time for Matrix Multiplication : %.4f milliseconds\n", totalTime2);
    printf("Total Elapsed time : %.4f milliseconds\n", totalTime1 + totalTime2);

    return 0;
}