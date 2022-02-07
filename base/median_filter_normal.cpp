#include <iostream>
#include <string>
#include <cstring>

#include <opencv2/opencv.hpp>
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"

int main(int argc, char** argv)
{

    if(argc > 4)
    {
        cv::Mat img;
        img = cv::imread(argv[3]);

        if(img.empty())
        {
            std::cout << "Could not open or find the image" << std::endl;
            return -1;
        }

        unsigned int window_rows{std::stoul(argv[1])};
        unsigned int window_cols{std::stoul(argv[2])};

        int width = img.cols;
        int height = img.rows;
        int channels = img.channels();

        uchar *image_uchar = img.data;
        // uchar temp[window_rows * window_cols] = {};
        for(size_t row = 0; row < height; ++row)
        {
            for(size_t col = 0; col < width; ++col)
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
                    image_uchar[row * width * channels + col * channels + ch] = temp[(window_rows * window_cols) / 2];
                }
            }
        }
        cv::Mat output_image = cv::Mat(height, width, img.type(), image_uchar);

        if(5 == argc)
        {
            cv::imwrite(argv[4], output_image);
        }
        else
        {
            cv::imwrite("output.png", output_image);
        }
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