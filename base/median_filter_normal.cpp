#include <iostream>
#include <string>
#include <cstring>

#include <opencv2/opencv.hpp>
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
// #include "FreeImage.h"

int main(int argc, char** argv)
{

    // if(1 == argc)
    // {
    //     std::cout << "No path to image" << std::endl;
    // }

    cv::Mat img;
    img = cv::imread("lena_noise3.png");

    if(img.empty())
    {
        std::cout << "Could not open or find the image" << std::endl;
        return 1;
    }

    unsigned int window_rows{3};
    unsigned int window_cols{3};
    // FREE_IMAGE_FORMAT imageFormat = FreeImage_GetFileType("lena.png",0);
    // FIBITMAP* inputImage = FreeImage_Load(imageFormat, "lena.png");

    int width = img.cols;
    int height = img.rows;
    int channels = img.channels();

    std::cout << "channels: " << channels << std::endl;
    uchar *image_uchar = img.data;
    std::cout << (int)img.data[width * channels * 0 + channels * 2 + 0 ] << "\t" << width * channels * 0 + channels * 2 + 0  << std::endl;
    std::cout << (int)img.data[width * channels * 0 + channels * 2 + 1 ] << "\t" << width * channels * 0 + channels * 2 + 1  << std::endl;
    std::cout << (int)img.data[width * channels * 0 + channels * 2 + 2 ] << "\t" << width * channels * 0 + channels * 2 + 2  << std::endl;
    // uchar temp[window_rows * window_cols] = {};

    bool flag = true;

    for(size_t row = window_rows / 2; row < height - window_rows / 2; ++row)
    {
        for(size_t col = window_cols / 2; col < width - window_cols / 2; ++col)
        {
            // std::memset(temp, 0, window_rows * window_cols);
            // uchar temp[window_rows * window_cols] = {0};
            for(size_t ch = 0; ch < channels; ++ch)
            {
                uchar temp[window_rows * window_cols] = {0};
                 for(size_t x = 0; x < window_rows; ++x)
                {
                    for(size_t y = 0; y < window_cols; ++y)
                    {
                        temp[x * window_cols + y] = image_uchar[(row + x - 1) * width * channels + (col + y - 1) * channels + ch];
                    }
                }
                    // std::cout << std::setw(4) << (unsigned int)temp << std::endl;

                for(size_t i = 0; i < window_rows * window_cols && flag; ++i)
                {
                    std::cout << std::setw(4) << (int) temp[i];
                }
                if(flag) std::cout << std::endl;
                for(size_t i = 0; i < window_rows * window_cols; ++i)
                {
                    for(size_t j = i + 1; j < window_rows * window_cols; ++j)
                    {
                        if(temp[i] > temp[j])
                        {
                            uchar tmp = temp[i];
                            temp[i] = temp[j];
                            temp[j] = tmp; 
                        }
                    }
                }

                for(size_t i = 0; i < window_rows * window_cols && flag; ++i)
                {
                    std::cout << std::setw(4) << (int) temp[i];
                }
                flag = false;
                image_uchar[row * width * channels + col * channels + ch] = temp[(window_rows * window_cols) / 2];
            }
        }
    }
    std::cout << std::endl;
    // std::cout << "format: " << imageFormat << " width: " << width << " height: " << height << std::endl;  
    std::cout << "width: " << img.cols << " height: " << img.rows << std::endl; 
    std::cout << "argc: " << argc << " argv: ";
    for(int i = 0; i < argc; ++i)
    {
        std::cout << argv[i] << " ";
    }
    std::cout << std::endl;

    cv::Mat output_image = cv::Mat(height, width, img.type(), image_uchar);

    cv::imwrite("lena_denoised.png", output_image);
    return 0;
}