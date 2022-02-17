#include <string>
#include <cstring>
#include <iostream>

#include <opencv2/opencv.hpp>
#include "opencv2/core.hpp"
#include "opencv2/core/cuda/common.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/imgcodecs.hpp"

#define BLOCK_SIZE (64)

void bubbleSort(uchar* buffer, size_t bufferSize)
{
    
    for(size_t i = 0; i <bufferSize; ++i){
        for(size_t j = i + 1; j < bufferSize; ++j){
            if(buffer[i] > buffer[j]){
                uchar tmp = buffer[i];
                buffer[i] = buffer[j];
                buffer[j] = tmp; 
            }
        }
    }
}

__global__ void medianFilter(unsigned char* inputImage, unsigned char* outputImage, int imageWidth, int imageHeight, size_t channels, int windowWidth, int windowsHeight)
{
    
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int x = blockDim.x * blockIdx.x + threadIdx.x;

    if( (x < (windowWidth/2)) || (y < (windowsHeight / 2)) || (x >= (imageWidth - (windowWidth/2))) || (y >= (imageHeight - (windowsHeight / 2))) ){
        for(size_t ch = 0; ch < 3; ++ch)
        {
            outputImage[y * imageWidth * channels + x * channels + ch] = inputImage[y * imageWidth * channels + x * channels + ch];
        }
    }
    else{
        uchar *window = new uchar[windowWidth * windowsHeight];
        for(size_t ch = 0; ch < channels; ++ch){
            size_t iterator = 0;
            for(size_t row = (y - (windowsHeight / 2)); row <= (y + (windowsHeight / 2)); ++row){
                for(size_t col = (x - (windowWidth / 2)); col <= (x + (windowWidth / 2)); ++col){
                    window[iterator] = inputImage[row * imageWidth * channels + col * channels + ch];
                    ++iterator;
                }
            }
            // bubble sort
            for(size_t i = 0; i < windowWidth * windowsHeight; ++i){
                for(size_t j = i + 1; j < windowWidth * windowsHeight; ++j){
                    if( window[i] > window[j] ){
                        uchar tmp = window[i];
                        window[i] = window[j];
                        window[j] = tmp; 
                    }
                }
            }
            outputImage[y * imageWidth * channels + x * channels + ch] = window[(windowWidth * windowsHeight) / 2];
        }
        delete window;
    }
}

__global__ void medianFilterShared(unsigned char* inputImage, unsigned char* outputImage, int imageWidth, int imageHeight, int channels, int windowWidth, int windowsHeight)
{
    
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int x = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ unsigned char sharedmem[BLOCK_SIZE + 2][BLOCK_SIZE + 2][3];

    bool is_x_left = ( 0 == threadIdx.x);
    bool is_x_right = ( (BLOCK_SIZE - 1) == threadIdx.x);
    bool is_y_top = ( 0 == threadIdx.y);
    bool is_y_bottom = ( (BLOCK_SIZE - 1 ) == threadIdx.y);

    if(is_x_left){
        if(x > 0)
        {
            for(size_t ch = 0; ch < channels; ++ch)
            {
                sharedmem[threadIdx.x][threadIdx.y + 1][ch] = inputImage[y * imageWidth * channels + (x - 1) * channels + ch];
            }
        }
        else{
            for(size_t ch = 0; ch < channels; ++ch)
            {
                sharedmem[threadIdx.x][threadIdx.y + 1][ch] = 0;
            }
        }
    }
    else if(is_x_right){
        if(x < (imageWidth - 1))
        {
            for(size_t ch = 0; ch < channels; ++ch)
            {
                sharedmem[threadIdx.x + 2][threadIdx.y + 1][ch] = inputImage[y * imageWidth * channels + (x + 1) * channels + ch];;
            }

        }
        else
        {
            for(size_t ch = 0; ch < channels; ++ch)
            {
                sharedmem[threadIdx.x + 2][threadIdx.y + 1][ch] = 0;
            }
        }
    }
    if (is_y_top){
        if(y > 0)
        {
            for(size_t ch = 0; ch < channels; ++ch)
            {
                sharedmem[threadIdx.x + 1][threadIdx.y][ch] = inputImage[(y - 1) * imageWidth * channels + x * channels + ch];
            }
            if(is_x_left)
            {
                for(size_t ch = 0; ch < channels; ++ch)
                {
                    sharedmem[threadIdx.x][threadIdx.y][ch] = inputImage[(y - 1) * imageWidth * channels + (x - 1) * channels + ch];    
                    
                }
            }
            else if(is_x_right)
            {
                for(size_t ch = 0; ch < channels; ++ch)
                {
                    sharedmem[threadIdx.x + 2][threadIdx.y][ch] = inputImage[(y - 1) * imageWidth * channels + (x + 1) * channels + ch];;
                }   
            }
        }
        else
        {
            for(size_t ch = 0; ch < channels; ++ch)
            {
                sharedmem[threadIdx.x + 1][threadIdx.y][ch] = 0;
            }
            if(is_x_left)
            {
                for(size_t ch = 0; ch < channels; ++ch)
                {
                    sharedmem[threadIdx.x][threadIdx.y][ch] = 0;
                }
            }
            else if(is_x_right)
            {
                for(size_t ch = 0; ch < channels; ++ch)
                {
                    sharedmem[threadIdx.x + 2][threadIdx.y][ch] = 0;
                }   
            }
        }
	}
	else if (is_y_bottom){
        if(y < (imageHeight - 1))
        {
            for(size_t ch = 0; ch < channels; ++ch)
            {
                sharedmem[threadIdx.x + 1][threadIdx.y + 2][ch] = inputImage[(y + 1) * imageWidth * channels + x * channels + ch];
            }
            if(is_x_left)
            {
                for(size_t ch = 0; ch < channels; ++ch)
                {
                    sharedmem[threadIdx.x][threadIdx.y + 2][ch] = inputImage[(y + 1) * imageWidth * channels + (x - 1)* channels + ch];
                }
            }
            else if(is_x_right)
            {
                for(size_t ch = 0; ch < channels; ++ch)
                {
                    sharedmem[threadIdx.x + 2][threadIdx.y + 2][ch] = inputImage[(y + 1) * imageWidth * channels + (x + 1)* channels + ch];;
                }
            }
        }
        else
        {
            for(size_t ch = 0; ch < channels; ++ch)
            {
                sharedmem[threadIdx.x + 1][threadIdx.y + 2][ch] = 0;
            }
            if(is_x_left)
            {
                for(size_t ch = 0; ch < channels; ++ch)
                {
                    sharedmem[threadIdx.x][threadIdx.y + 2][ch] = 0;
                }
            }
            else if(is_x_right)
            {
                for(size_t ch = 0; ch < channels; ++ch)
                {
                    sharedmem[threadIdx.x + 2][threadIdx.y + 2][ch] = 0;
                }
            }
        }
	}

    for(size_t ch = 0; ch < channels; ++ch)
    {
        sharedmem[threadIdx.x + 1][threadIdx.y + 1][ch] = inputImage[y * imageWidth * channels + x * channels + ch];
    }
    
    __syncthreads();

    uchar *window = new uchar[windowWidth * windowsHeight];

    for(size_t ch = 0; ch < channels; ++ch)
    {
        size_t iterator = 0;
        for (size_t col = threadIdx.x; col <= threadIdx.x + 2; ++col)
        {
            for(size_t row = threadIdx.y; row <= threadIdx.y + 2; ++row)
            {
                window[iterator] = sharedmem[col][row][ch]; 
                ++iterator;
            }
        }
        // bubble sort
        for(size_t i = 0; i < windowWidth * windowsHeight; ++i){
            for(size_t j = i + 1; j < windowWidth * windowsHeight; ++j){
                if( window[i] > window[j] ){
                    uchar tmp = window[i];
                    window[i] = window[j];
                    window[j] = tmp; 
                }
            }
        }
        outputImage[y * imageWidth * channels + x * channels + ch] = window[(windowWidth * windowsHeight) / 2];
    }

    delete window;
    __syncthreads();
}

int main(int argc, char** argv)
{
    if(argc >= 4)
    {
        cv::Mat img;
        img = cv::imread(argv[3]);

        if(img.empty())
        {
            std::cout << "Could not open or find the image" << std::endl;
            return -1;
        }

        unsigned int window_rows{(unsigned int) std::stoul(argv[1])};
        unsigned int window_cols{(unsigned int) std::stoul(argv[2])};

        int width = img.cols;
        int height = img.rows;
        int channels = img.channels();

        cudaError_t status;
        cudaEvent_t start, stop;
        float time;

        uint64_t imageSize = width * height * channels * sizeof(uchar);

        unsigned char *outputImageHost = (unsigned char *) malloc(imageSize);
        unsigned char *inputImageDevice;
        unsigned char *outputImageDevice;

        cudaMalloc<unsigned char>(&inputImageDevice, imageSize);
        status = cudaGetLastError();
        if (status != cudaSuccess) {                     
            std::cout << "Kernel failed for cudaMalloc : " << cudaGetErrorString(status) << std::endl;
            return -1;
        }

        cudaMalloc<unsigned char>(&outputImageDevice, imageSize);
        status = cudaGetLastError();
        if (status != cudaSuccess) {                     
            std::cout << "Kernel failed for cudaMalloc : " << cudaGetErrorString(status) << std::endl;
            return -1;
        }

        cudaMemcpy(inputImageDevice, img.ptr(), imageSize, cudaMemcpyHostToDevice);
        status = cudaGetLastError();
        if (status != cudaSuccess) {                     
            std::cout << "Kernel failed for cudaMemcpy cudaMemcpyHostToDevice: " << cudaGetErrorString(status) << std::endl;
            cudaFree(inputImageDevice);
            return -1;
        }
        // cudaMalloc((void**)&outputImageDevice, imageSize);
        // const dim3 grid (((width % BLOCK_WIDTH) != 0) ? (width / BLOCK_WIDTH + 1) : (width / BLOCK_WIDTH), ((height % BLOCK_HEIGHT) != 0) ? (height / BLOCK_HEIGHT + 1) : (height / BLOCK_HEIGHT), 1);
        // const dim3 block ((int)ceil((float)width / (float)BLOCK_WIDTH), (int)ceil((float)height / (float)BLOCK_HEIGHT));

        const dim3 block(BLOCK_SIZE,BLOCK_SIZE);
        const dim3 grid(cv::cuda::device::divUp(width, block.x), cv::cuda::device::divUp(height, block.y));

        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        medianFilter<<<grid, block>>>(inputImageDevice, outputImageDevice, width, height, channels, window_cols,  window_rows);
        status = cudaGetLastError(); 

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        
        std::cout << "GPU time in milliseconds: " << time << " ms" << std::endl;

        cudaMemcpy(inputImageDevice, img.ptr(), imageSize, cudaMemcpyHostToDevice);
        status = cudaGetLastError();
        if (status != cudaSuccess) {                     
            std::cout << "Kernel function failed: " << cudaGetErrorString(status) << std::endl;
            cudaFree(inputImageDevice);
            return -1;
        }

        
        if(3 == window_cols && 3 == window_rows && channels <= 3)
        {
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start, 0);
    
            medianFilterShared<<<grid, block>>>(inputImageDevice, outputImageDevice, width, height, channels, window_cols,  window_rows);
            status = cudaGetLastError(); 
    
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&time, start, stop);
    
            cudaMemcpy(inputImageDevice, img.ptr(), imageSize, cudaMemcpyHostToDevice);
            status = cudaGetLastError();
            if (status != cudaSuccess) {                     
                std::cout << "Shared memory kernel function failed: " << cudaGetErrorString(status) << std::endl;
                cudaFree(inputImageDevice);
                return -1;
            }
    
            std::cout << "GPU time for shared memory in milliseconds: " << time << " ms" << std::endl;
            
            cudaMemcpy(outputImageHost, outputImageDevice, imageSize, cudaMemcpyDeviceToHost);
            status = cudaGetLastError();
            if (status != cudaSuccess) {                     
                std::cout << "Kernel failed for cudaMemcpy cudaMemcpyHostToDevice: " << cudaGetErrorString(status) << std::endl;
                cudaFree(inputImageDevice);
                return -1;
            }
        }

        cv::Mat output_image = cv::Mat(height, width, img.type(), outputImageHost);

        if(5 == argc)
        {
            cv::imwrite(argv[4], output_image);
        }
        else
        {
            cv::imwrite("output.png", output_image);
        }

        cudaFree(inputImageDevice);
        cudaFree(outputImageDevice);
        free(outputImageHost);

        return 0;
    }
    else
    {
        std::cout << "usage: median_filter_normal wh ww input_image <output_directory>" << std::endl << std::endl;
        std::cout << "Applies median filter to the image" << std::endl << std::endl;
        std::cout << "Positional arguments:" << std::endl;
        std::cout << std::left << std::setw(30) << "  wh" << "kernel height, odd number" << std::endl;
        std::cout << std::left << std::setw(30) << "  ww" << "kernel width, odd number" << std::endl;
        std::cout << std::left << std::setw(30) << "  input_image" << "input image directory" << std::endl;
        std::cout << std::left << std::setw(30) << "  <output_directory>" << "optional output directory, if not present then image is saved to output.png" << std::endl;
        return -1;
    }
}