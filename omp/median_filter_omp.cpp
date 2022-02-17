#include <omp.h>
#include <string>
#include <cstring>
#include <chrono>

#include <opencv2/opencv.hpp>
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"


#define NUM_THREADS 8
#define CHUNK 16000

int main(int argc, char** argv)
{

    if(argc >= 2)
    {
        cv::Mat img;
        img = cv::imread(argv[1]);

        if(img.empty())
        {
            std::cout << "Could not open or find the image" << std::endl;
            return -1;
        }

        unsigned int window_rows{3};
        unsigned int window_cols{3};

        int width = img.cols;
        int height = img.rows;
        int channels = img.channels();

        size_t imageSize = width * height * channels;

        uchar *image_uchar = img.data;

        uchar window[9];

        size_t windowSize = window_cols * window_rows;

        size_t row = 0;
        size_t col = 0;
        size_t ch = 0;
        size_t x = 0;
        size_t y = 0;
        int minval;
        auto start = std::chrono::steady_clock::now();

        omp_set_num_threads(NUM_THREADS);
        #pragma omp parallel private(window, row, col, ch, x, y, minval)
        {
            #pragma omp parallel for 
            for(row = window_rows / 2; row < height - window_rows / 2; ++row){
                for(col = window_cols / 2; col < width - window_cols / 2; ++col){
                    for(ch = 0; ch < channels; ++ch){
                    // fill window with data from input image
                        for(y = 0; y < window_rows; ++y){
                            for(x = 0; x < window_cols; ++x){
                                window[y * window_cols + x] = image_uchar[(row + y - (window_rows / 2) ) * width * channels + (col + x - (window_cols / 2)) * channels + ch];
                            }
                        }
                        
                        for (int i = 0; i < (1 + (windowSize / 2)); ++i) {
                            // --- Find the position of the minimum element
                            int minval = i;
                            for (int l = i + 1; l < (windowSize); ++l) 
                            {
                                if (window[l] < window[minval]) minval=l;
                            }
                            // --- Put found minimum element in its place
                            uchar temp = window[i];
                            window[i] = window[minval];
                            window[minval] = temp;
                        }
                        image_uchar[row * width * channels + col * channels + ch] = window[4];
                    }
                }
            }
        }

        auto end = std::chrono::steady_clock::now();
        std::cout << "Elapsed time in milliseconds: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
        cv::Mat output_image = cv::Mat(height, width, img.type(), image_uchar);

        if(5 == argc)
        {
            cv::imwrite(argv[2], output_image);
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
        std::cout << std::left << std::setw(30) << "  input_image" << "input image directory" << std::endl;
        std::cout << std::left << std::setw(30) << "  <output_directory>" << "optional output directory, if not present then image is saved to output.png" << std::endl;
        return -1;
    }
}