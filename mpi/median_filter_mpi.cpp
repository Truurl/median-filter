#include <mpi.h>
#include <iostream>
#include <string>
#include <cstring>
#include <chrono>
#include <stdio.h>

#include <opencv2/opencv.hpp>
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"


typedef struct jobPartition
{
    int height;
    int width;
    int channels;
    int windowHeight;
    int windowWidth;
    int rowOffset;
    int colOffset;
    // int yStart;
    // int xStart;
} JobPartition;

void sort(uchar* buffer, size_t bufferSize)
{                        
    for (int i = 0; i < (1 + (bufferSize / 2)); ++i) {
        // --- Find the position of the minimum element
        int minval = i;
        for (int l = i + 1; l < (bufferSize); ++l) 
        {
            if (buffer[l] < buffer[minval]) minval=l;
        }
        // --- Put found minimum element in its place
        uchar temp = buffer[i];
        buffer[i] = buffer[minval];
        buffer[minval] = temp;
    }
}

void scheduleJob(int height, int width, int channels, int windowHeight, int windowWidth, int size, JobPartition* jobs){
    
    if(height >= width){
        for(size_t i = 0; i < size; ++i)
        {
            jobs[i].colOffset = 0;
            jobs[i].width = width;
            jobs[i].channels = channels;
            jobs[i].windowHeight = windowHeight;
            jobs[i].windowWidth = windowWidth;
            if( 0 == i || (size - 1 ) == i){
                jobs[i].height = (height / size) + (windowHeight / 2);
                jobs[i].rowOffset = ( 0 == i ? 0 : i * (height / size) - (windowHeight / 2));

            }
            else{
                jobs[i].rowOffset = i * (height / size) - (windowHeight / 2);
                jobs[i].height = (height / size) +  2 * (windowHeight / 2);
            }

            if(jobs[i].height < windowHeight)
            {
                jobs[i].height = windowHeight;
            }
        }
    }else{
        for(size_t i = 0; i < size; ++i)
        {
            jobs[i].rowOffset = 0;
            jobs[i].height = height;
            jobs[i].channels = channels;
            jobs[i].windowHeight = windowHeight;
            jobs[i].windowWidth = windowWidth;
            if(0 == i || (size - 1) == i){
                jobs[i].width = (width % size) + windowWidth / 2;
                jobs[i].colOffset = ( 0 == i ? 0 : i * (width / size) - (windowWidth / 2));
            }
            else{
                jobs[i].colOffset = i * (height / size) - (windowWidth / 2);
                jobs[i].width = (width % size) +  2 * (windowWidth / 2);
            }

            if(jobs[i].width < windowWidth)
            {
                jobs[i].width = windowWidth;
            }
        }
    }
}

void portionData(uchar *data, uchar *srcData, JobPartition *job)
{
    int height = job->height;
    int width = job->width;
    int channels = job->channels;

    for(size_t ch = 0; ch < channels; ++ch){
        for(size_t h = 0; h < height; ++h){
            for(size_t w = 0; w < width; ++w){
                data[h * width * channels + w * channels + ch] = srcData[(h + job->rowOffset) * width * channels + (w + job->colOffset) * channels + ch];
            }
        }
    }
}

void concatenateData(uchar *data, uchar * dstData, JobPartition *job){
    
    int height = job->height;
    int width = job->width;
    int channels = job->channels;

    for(size_t ch = 0; ch < channels; ++ch){
        for(size_t h = 0; h < height; ++h){
            for(size_t w = 0; w < width; ++w){
                dstData[(h + job->rowOffset) * width * channels + (w + job->colOffset) * channels + ch] = data[(h + job->windowHeight / 2)* width * channels + (w + job->windowWidth / 2) * channels + ch];
            }
        }
    }

}

void medianFilterJob(uchar* buff, JobPartition* job)
{
    int width = job->width;
    int height = job->height;
    int channels = job->channels;

    int windowHeight = job->windowHeight;
    int windowWidth = job->windowWidth;

    uchar *window = new uchar[windowHeight * windowWidth];

    for(size_t ch = 0; ch < channels; ++ch){
        for(size_t row = windowHeight / 2; row < height - windowHeight / 2; ++row){
            for(size_t col = windowWidth / 2; col < width - windowWidth / 2; ++col){
                for(size_t y = 0; y < windowHeight; ++y)
                    for(size_t x = 0; x < windowWidth; ++x){
                    {
                        window[y * windowWidth + x] = buff[(row + y - windowWidth / 2) * width * channels + (col + x - windowHeight / 2) * channels + ch];
                    }
                }
                sort(window, windowWidth * windowHeight);
                buff[row * width * channels + col * channels + ch] = window[(windowWidth * windowHeight) / 2];
            }
        }
    }

    delete window;
}

int main(int argc, char** argv)
{
    JobPartition *jobs;
    int size, rank;
    int tag = 1;
    MPI_Status status;

    cv::Mat img;
    int width;
    int height;
    int channels;
    int shape[3];
    size_t buffSize;

    double start, end;

    uchar *imageData;
    uchar *outputData;
    uchar *dataBuff;
    unsigned int window_rows{3};
    unsigned int window_cols{3};

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(0 == rank)
    {
        std::cout << "Root Starts here" << std::endl;
        img = cv::imread("../lena_noise1.png");

        if(img.empty())
        {
            std::cout << "Could not open or find the image" << std::endl;
            return -1;
        }
        imageData = img.data;
        width = img.cols;
        height = img.rows;
        channels = img.channels();
        outputData = new uchar[width * height * channels];
        // shape = {width, height, channels};
    }

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();

    // master sends structure with information about job parttion
    if(0 == rank)
    {
        // std::cout << "Root sends data" << std::endl;
        jobs = new JobPartition[size];
        scheduleJob(height, width, channels, window_rows, window_cols, size, jobs); // partition job
        // send job partition to nodes 
        for(size_t i = 1; i < size; ++i){
            MPI_Send(&jobs[i], sizeof(JobPartition), MPI_BYTE, i, tag, MPI_COMM_WORLD); 
        }
    }
    else{
        jobs = new JobPartition;
        MPI_Recv(jobs, sizeof(JobPartition), MPI_BYTE, 0, tag, MPI_COMM_WORLD, &status);
        buffSize = jobs->channels * jobs->width * jobs-> height; // calulate job size
        dataBuff = new uchar[buffSize]; // create buffer for part of image
    }

    if(0 == rank){
        // std::cout << "Root sends patrioned picture" << std::endl;
        for(size_t i = 1; i < size; ++i ){
            // allocate temporary buffer for job data
            buffSize = jobs[i].width * jobs[i].height * jobs[i].channels;
            dataBuff = new uchar[buffSize];
            // copy data from image to temporarty buffer
            portionData(dataBuff, imageData, &jobs[i]);
            // send data to slave
            MPI_Send(dataBuff, buffSize, MPI_UINT8_T, i, tag, MPI_COMM_WORLD);
            free(dataBuff);
        }
        // get data partiion for master
        buffSize = jobs[0].width * jobs[0].height * jobs[0].channels;
        dataBuff = new uchar[buffSize];
        portionData(dataBuff, imageData, &jobs[0]);
        // std::cout << "Root starts own job" << std::endl;
        // do median job
        medianFilterJob(dataBuff, &jobs[0]);
        // merge master parttion with output
        concatenateData(dataBuff, outputData, &jobs[0]);
        free(dataBuff);
    }else{
        MPI_Recv(dataBuff, buffSize, MPI_UINT8_T, 0, tag, MPI_COMM_WORLD, &status);
        // do median job
        medianFilterJob(dataBuff, jobs);
        MPI_Send(dataBuff, buffSize, MPI_UINT8_T, 0, tag, MPI_COMM_WORLD);
    }


    if(0 == rank){
        // std::cout << "Root waits for finished jobs" << std::endl;
        // wait for other nodes to finish job
        for(size_t i = 1; i < size; ++i)
        {
            buffSize = jobs[i].width * jobs[i].height * jobs[i].channels;
            dataBuff = new uchar[buffSize];
            MPI_Recv(dataBuff, buffSize, MPI_UINT8_T, i, tag, MPI_COMM_WORLD, &status);
            // merge slave data partion with image data
            concatenateData(dataBuff, outputData, &jobs[i]);
            free(dataBuff);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();

    MPI_Finalize();

    if(0 == rank){
        // save image
        cv::Mat output_image = cv::Mat(height, width, img.type(), outputData);
        cv::imwrite("output.png", output_image);
        // std::cout << "Root finish job" << std::endl;
        std::cout << "Elapsed time: " << end - start << " seconds" << std::endl;
    }

    return 0;
}