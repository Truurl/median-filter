#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
// #include "FreeImage.h"

int main(int argc, char** argv)
{

    // FreeImage_Initialise();
    // if(1 == argc)
    // {
    //     std::cout << "No path to image" << std::endl;
    // }

    cv::Mat img, Bands[3];
    img = cv::imread("lena.png");

    if(img.empty())
    {
        std::cout << "Could not open or find the image" << std::endl;
        return 1;
    }

    cv::split(img, Bands);

    unsigned int windows_rows{3}, windows_cols{3};
    // FREE_IMAGE_FORMAT imageFormat = FreeImage_GetFileType("lena.png",0);
    // FIBITMAP* inputImage = FreeImage_Load(imageFormat, "lena.png");

    int width = img.cols;
    int height = img.rows;

    for(auto i = )

    // std::cout << "format: " << imageFormat << " width: " << width << " height: " << height << std::endl;  
    std::cout << "width: " << img.cols << " height: " << img.rows << std::endl; 
    std::cout << "argc: " << argc << " argv: ";
    for(int i = 0; i < argc; ++i)
    {
        std::cout << argv[i] << " ";
    }
    std::cout << std::endl;
    return 0;
}