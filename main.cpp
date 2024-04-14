#include <fstream>
#include <sstream>

#include <iostream>
#include <opencv2/opencv.hpp>
#include <time.h>

using namespace cv;
using namespace dnn;
using namespace std;

std::vector<std::string> classes;
std::vector<Vec3b> colors;

/*"{ backend     | 0 | Choose one of computation backends: "
                        "0: automatically (by default), "
                        "1: Halide language (http://halide-lang.org/), "
                        "2: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), "
                        "3: OpenCV implementation, "
                        "4: VKCOM, "
                        "5: CUDA }"
    "{ target      | 0 | Choose one of target computation devices: "
                        "0: CPU target (by default), "
                        "1: OpenCL, "
                        "2: OpenCL fp16 (half-float precision), "
                        "3: VPU, "
                        "4: Vulkan, "
                        "6: CUDA, "
                        "7: CUDA fp16 (half-float preprocess) }";*/

void showLegend();

void colorizeSegmentation(const Mat &score, Mat &segm);

int main(/*int argc, char** argv*/)
{
    int64 t1 = getTickCount();

    float scale = 1 / 255.0;
    Scalar mean(0.485 * 255.0, 0.456 * 255.0, 0.406 * 255.0);
    //bool swapRB = parser.get<bool>("rgb");
    int inpWidth = 256;
    int inpHeight =256;
    String model = "./fcn_resnet50.onnx";
    int backendId = 0; //automatically
    int targetId = 0;  //cpu default
    String file_path = "./room_3.jpg";


    // Open file with classes names.
    std::string file_c = "./classes.txt";
    std::ifstream ifs(file_c.c_str());
    if (!ifs.is_open())
        CV_Error(Error::StsError, "File " + file_c + " not found");
    std::string line;
    while (std::getline(ifs, line))
    {
        classes.push_back(line);
    }

    //classes check
    /*for (int i = 0; i < 38; i++) {
        cout << classes[i] << endl;
    }*/

    //Load colors

    ifstream ifp("./colors.txt");
    if (!ifp.is_open()) {
        cerr << "colors file load failed!" << endl;
        return -1;
    }
    else {
        string line;
        while (getline(ifp, line)) {
            istringstream colorStr(line);

            int num, i = 0;
            Vec3b color;

            while (colorStr >> num) {
                color[i] = num;
                i++;
            }
            colors.push_back(color);
        }
    }

    //check colors
    /*for (int i = 0; i < 38; i++) {
        cout << "colors " << i << " :" << colors[i] << endl;
    }*/
    

    
    CV_Assert(!model.empty());
    //! [Read and initialize network]
    Net net = readNet(model);
    net.setPreferableBackend(0);
    net.setPreferableTarget(0);

    //! [Read and initialize network]

    // Create a window
    /*static const std::string kWinName = "Deep learning semantic segmentation in OpenCV";
    namedWindow(kWinName, WINDOW_NORMAL);*/

    //! [Open a video file or an image file or a camera stream]
    //! camera stream cap(device_number)
    /*VideoCapture cap;
    cap.open(file_path);

    if (!cap.isOpened()) {
        cerr << "file is not open!" << endl;
        //return -1;
    }*/

    // Process frames.
    Mat frame, blob;

    frame = imread(file_path);
    if (frame.empty()) {
        cerr << "image load failed" << endl;
        return -1;
    }

    //! [Create a 4D blob from a frame]
    blobFromImage(frame, blob, scale, Size(inpWidth, inpHeight), mean, false, false);
    //! [Create a 4D blob from a frame]

    //! [Set input blob]
    net.setInput(blob);
    //! [Set input blob]
    //! [Make forward pass]
    Mat score = net.forward();

    Mat segm;
    colorizeSegmentation(score, segm);

    resize(segm, segm, frame.size(), 0, 0, INTER_NEAREST);
    addWeighted(frame, 0, segm, 0.9, 0.0, frame);

    int64 t2 = getTickCount();
    double ms = (t2 - t1) * 1000 / getTickFrequency();
    cout << "time: " << ms <<"ms"<< endl;

    imwrite("./output2.jpg",frame);
    imshow("mask", frame);
    waitKey();

    return 0;
}

void colorizeSegmentation(const Mat &score, Mat &segm)
{
    const int rows = score.size[2];
    const int cols = score.size[3];
    const int chns = score.size[1];

    if (colors.empty())
    {
        // Generate colors.
        colors.push_back(Vec3b());
        for (int i = 1; i < chns; ++i)
        {
            Vec3b color;
            for (int j = 0; j < 3; ++j)
                color[j] = (colors[i - 1][j] + rand() % 256) / 2;
            colors.push_back(color);
        }
    }
    else if (chns != (int)colors.size())
    {
        CV_Error(Error::StsError, format("Number of output classes does not match "
                                         "number of colors (%d != %zu)", chns, colors.size()));
    }

    Mat maxCl = Mat::zeros(rows, cols, CV_8UC1);
    Mat maxVal(rows, cols, CV_32FC1, score.data);
    for (int ch = 1; ch < chns; ch++)
    {
        for (int row = 0; row < rows; row++)
        {
            const float *ptrScore = score.ptr<float>(0, ch, row);
            uint8_t *ptrMaxCl = maxCl.ptr<uint8_t>(row);
            float *ptrMaxVal = maxVal.ptr<float>(row);
            for (int col = 0; col < cols; col++)
            {
                if (ptrScore[col] > ptrMaxVal[col])
                {
                    ptrMaxVal[col] = ptrScore[col];
                    ptrMaxCl[col] = (uchar)ch;
                }
            }
        }
    }

    segm.create(rows, cols, CV_8UC3);
    for (int row = 0; row < rows; row++)
    {
        const uchar *ptrMaxCl = maxCl.ptr<uchar>(row);
        Vec3b *ptrSegm = segm.ptr<Vec3b>(row);
        for (int col = 0; col < cols; col++)
        {
            ptrSegm[col] = colors[ptrMaxCl[col]];
        }
    }
}

void showLegend()
{
    static const int kBlockHeight = 30;
    static Mat legend;
    if (legend.empty())
    {
        const int numClasses = (int)classes.size();
        if ((int)colors.size() != numClasses)
        {
            CV_Error(Error::StsError, format("Number of output classes does not match "
                                             "number of labels (%zu != %zu)", colors.size(), classes.size()));
        }
        legend.create(kBlockHeight * numClasses, 200, CV_8UC3);
        for (int i = 0; i < numClasses; i++)
        {
            Mat block = legend.rowRange(i * kBlockHeight, (i + 1) * kBlockHeight);
            block.setTo(colors[i]);
            putText(block, classes[i], Point(0, kBlockHeight / 2), FONT_HERSHEY_SIMPLEX, 0.5, Vec3b(255, 255, 255));
        }
        namedWindow("Legend", WINDOW_NORMAL);
        imshow("Legend", legend);
    }
}
//! [Make forward pass]
    /*while (waitKey(20) < 0)
    {
        cap >> frame;
        if (frame.empty())
        {
            waitKey();
            break;
        }
        imshow(kWinName, frame);

        //! [Create a 4D blob from a frame]
        blobFromImage(frame, blob, scale, Size(inpWidth, inpHeight), mean, false, false);
        //! [Create a 4D blob from a frame]

        //! [Set input blob]
        net.setInput(blob);
        //! [Set input blob]
        //! [Make forward pass]
        Mat score = net.forward();
        //! [Make forward pass]

        /*
        Mat segm;
        colorizeSegmentation(score, segm);

        resize(segm, segm, frame.size(), 0, 0, INTER_NEAREST);
        addWeighted(frame, 0.1, segm, 0.9, 0.0, frame);

        // Put efficiency information.
        std::vector<double> layersTimes;
        double freq = getTickFrequency() / 1000;
        double t = net.getPerfProfile(layersTimes) / freq;
        std::string label = format("Inference time: %.2f ms", t);
        putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));

        imshow(kWinName, frame);
        if (!classes.empty())
            showLegend();

    }*/