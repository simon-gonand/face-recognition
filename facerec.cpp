#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core/types.hpp"

#include <iostream>
#include <fstream>
#include <filesystem>
namespace fs = std::filesystem;

// g++ -std=c++1z 1_simple_facerec_eigenfaces.cpp -lopencv_face -lopencv_core -lopencv_imgcodecs -lstdc++fs

int main(int argc, char *argv[])
{
  std::vector<cv::Mat> images;
  std::vector<int>     labels;

  // Iterate through all subdirectories, looking for .pgm files
  fs::path p(argc > 1 ? argv[1] : "src/att_faces");
  for (const auto &entry : fs::recursive_directory_iterator{ p }) {
    if (fs::is_regular_file(entry.status())) { // Was once always (wrongly) false in VS
      if (entry.path().extension() == ".pgm") {
        std::string str = entry.path().parent_path().stem().string(); // s26 s27 etc.
        int label = atoi(str.c_str() + 1); // s1 -> 1 (pointer arithmetic)
        images.push_back(cv::imread(entry.path().string().c_str(), cv::IMREAD_GRAYSCALE));
        labels.push_back(label);
      }
    }
  }

  // Randomly choose an image, and remove it from the main collection
  /*std::srand(std::time(0));
  int rand_image_id = 350;
  cv::Mat testSample = images[rand_image_id];
  int     testLabel  = labels[rand_image_id];
  images.erase(images.begin() + rand_image_id);
  labels.erase(labels.begin() + rand_image_id);
  std::cout << "Actual class    = " << testLabel << '\n';*/
  std::cout << " training..." << std::endl;

  cv::Ptr<cv::face::BasicFaceRecognizer> model = cv::face::EigenFaceRecognizer::create();
  model->train(images, labels);

  cv::Mat frame;
  double fps = 30;
  const char win_name[] = "Face Recognition";
  
  cv::Mat gray_frame;
  cv::Mat resize_frame;
  cv::Size size(92, 112);

  cv::VideoCapture vid_in(0);   // argument is the camera id
  if (!vid_in.isOpened()) {
      std::cout << "error: Camera 0 could not be opened for capture.\n";
      return -1;
  }
  cv::namedWindow(win_name);

  while (true) {
      vid_in >> frame;
      

      cv::Rect myROI(250, 115, 180, 230);
      cv::Mat cropped_frame = frame(myROI);

      imshow(win_name, cropped_frame);

      cvtColor(cropped_frame, gray_frame, CV_BGR2GRAY);
      cv::resize(gray_frame, resize_frame, size);

      int predictedLabel = model->predict(resize_frame);
      std::cout << "\nPredicted class = " << predictedLabel << '\n';

      if (cv::waitKeyEx(1000 / fps) >= 0) // how long to wait for a key (milliseconds)
          break;
  }

  vid_in.release();
  return 0;
}
