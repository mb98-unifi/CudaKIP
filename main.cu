#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <algorithm>
#include <chrono>

#define MASK_WIDTH 3
#define TILE_WIDTH 16
#define MASK_HALF_WIDTH (MASK_WIDTH/2)
#define NS_WIDTH (TILE_WIDTH + MASK_WIDTH - 1)
#define WITH_GPU 1

#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

static void CheckCudaErrorAux(const char *file, unsigned line, const char *statement, cudaError_t err) {
    if (err == cudaSuccess){return;}
    std::cerr << statement << " returned " << cudaGetErrorString(err) << "(" << err << ") at " << file << ":" << line << std::endl;
    exit(1);
}

void convolution_cpu(const unsigned char *input, unsigned char *output, const float *kernel, const int width, const int height) {
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            int tmpR = 0;
            int tmpG = 0;
            int tmpB = 0;
            for (int k = 0; k < MASK_WIDTH; k++) {
                for (int l = 0; l < MASK_WIDTH; l++) {
                    int x = i - (int) ceil(MASK_WIDTH / 2) + k;
                    int y = j - (int) ceil(MASK_WIDTH / 2) + l;
                    if (x >= 0 && x < width && y >= 0 && y < height) {
                        tmpB += ceil((float) input[(y * width + x) * 3] * kernel[l * MASK_WIDTH + k]);
                        tmpG += ceil((float) input[(y * width + x) * 3 + 1] * kernel[l * MASK_WIDTH + k]);
                        tmpR += ceil((float) input[(y * width + x) * 3 + 2] * kernel[l * MASK_WIDTH + k]);
                    }
                }
            }

            output[(j * width + i) * 3] = std::min(std::max(tmpB, 0), 255);
            output[(j * width + i) * 3 + 1] = std::min(std::max(tmpG, 0), 255);
            output[(j * width + i) * 3 + 2] = std::min(std::max(tmpR, 0), 255);
        }
    }
}

__global__ void convolution(unsigned char *I, unsigned char *O, const float *__restrict__ K, int width, int height) {
    __shared__ unsigned char Ns[NS_WIDTH * NS_WIDTH];

    //Loading first TILE_WIDTH*TILE_WIDTH elements
    for (int i = 0; i < 3; i++) {
        int Ns_Index = threadIdx.y * TILE_WIDTH + threadIdx.x;
        int Ns_Y = Ns_Index / NS_WIDTH;
        int Ns_X = Ns_Index % NS_WIDTH;
        int I_Y = blockIdx.y * TILE_WIDTH + Ns_Y - MASK_HALF_WIDTH;
        int I_X = blockIdx.x * TILE_WIDTH + Ns_X - MASK_HALF_WIDTH;
        int I_Index = (I_Y * width + I_X) * 3 + i;

        if (I_Y >= 0 && I_Y < height && I_X >= 0 && I_X < width) {
            Ns[Ns_Index] = I[I_Index];
        } else {
            Ns[Ns_Index] = 0;
        }

        //Loading last NS_WIDTH*NS_WIDTH - TILE_WIDTH*TILE_WIDTH elements
        Ns_Index = threadIdx.y * TILE_WIDTH + threadIdx.x + TILE_WIDTH * TILE_WIDTH;
        Ns_Y = Ns_Index / NS_WIDTH;
        Ns_X = Ns_Index % NS_WIDTH;
        I_Y = blockIdx.y * TILE_WIDTH + Ns_Y - MASK_HALF_WIDTH;
        I_X = blockIdx.x * TILE_WIDTH + Ns_X - MASK_HALF_WIDTH;
        I_Index = (I_Y * width + I_X) * 3 + i;

        if (Ns_Y < NS_WIDTH) {
            if (I_Y >= 0 && I_Y < height && I_X >= 0 && I_X < width) {
                Ns[Ns_Index] = I[I_Index];
            } else {
                Ns[Ns_Index] = 0;
            }
        }
        __syncthreads();

        int x,y;
        int pixelValue = 0;

        //Convolution
        for (y = 0; y < MASK_WIDTH; y++) {
            for (x = 0; x < MASK_WIDTH; x++) {
                pixelValue += Ns[(threadIdx.y + y) * NS_WIDTH + (threadIdx.x + x)] * K[y * MASK_WIDTH + x];
            }
        }
        y = blockIdx.y * TILE_WIDTH + threadIdx.y;
        x = blockIdx.x * TILE_WIDTH + threadIdx.x;
        if (y < height && x < width)
            //pixelValue is truncated to 0 if < 0 or to 255 if > 255
            O[(y * width + x) * 3 + i] = min(max(pixelValue, 0), 255);
        __syncthreads();
    }
}


int main() {
    unsigned char *deviceInput;
    unsigned char *deviceOutput;
    float *deviceKernel;
    float hostKernel[MASK_WIDTH * MASK_WIDTH] = {-1, -1, -1, -1, 8, -1, -1, -1, -1};   //Edge Detection Kernel

    cv::Mat input = cv::imread("INPUT FILE PATH", cv::IMREAD_COLOR);
    cv::Mat output = cv::Mat::zeros(input.rows, input.cols, CV_8UC3);

    if (!input.data) {
        std::cout << "Image not found!" << std::endl;
        return -1;
    }

    unsigned char *hostInput = input.ptr();
    unsigned char *hostOutput = output.ptr();
    int width = input.cols;
    int height = input.rows;


    CUDA_CHECK_RETURN(cudaMalloc((void **) &deviceInput, width * height * 3 * sizeof(unsigned char)));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &deviceOutput, width * height * 3 * sizeof(unsigned char)));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &deviceKernel, MASK_WIDTH * MASK_WIDTH * sizeof(float)));

    CUDA_CHECK_RETURN(cudaMemcpy(deviceInput, hostInput, width * height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(deviceKernel, hostKernel, MASK_WIDTH * MASK_WIDTH * sizeof(float), cudaMemcpyHostToDevice));

    dim3 dimGrid(ceil((float) width / TILE_WIDTH), ceil((float) height / TILE_WIDTH));
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    auto start = std::chrono::high_resolution_clock::now();

    if (WITH_GPU){
        std::cout << "Processing with GPU" << std::endl;
        convolution<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, deviceKernel, width, height);
        CUDA_CHECK_RETURN(cudaMemcpy(hostOutput, deviceOutput, width * height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    } else {
        std::cout << "Processing with CPU" << std::endl;
        convolution_cpu(hostInput, hostOutput, hostKernel, width, height);
    }

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "Time: " << duration.count() << "ms" << std::endl;

    cv::imwrite("OUTPUT FILE PATH", output);

    CUDA_CHECK_RETURN(cudaFree(deviceInput));
    CUDA_CHECK_RETURN(cudaFree(deviceOutput));
    CUDA_CHECK_RETURN(cudaFree(deviceKernel));

    return 0;
}
