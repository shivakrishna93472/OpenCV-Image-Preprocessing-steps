// #include <opencv2/opencv.hpp>
// #include <iostream>

//code to display or read the image
// using namespace cv;
// int main(){
//     std::string image_path = "/home/kpit/opencv/samples/data/HappyFish.jpg";
//     cv::Mat img = cv::imread(image_path, cv::IMREAD_ANYCOLOR);
//     cv::imshow("Display window", img);
//     cv::imwrite("myImage.jpg",img);
//     cv::waitKey(0);
//     return 0;
// }
// #include <opencv2/opencv.hpp>
// #include <iostream>

// int main() {
//     // Load the image
//     cv::Mat image = cv::imread("/home/kpit/opencv/samples/data/HappyFish.jpg");

//     // Check if the image is loaded successfully
//     if (image.empty()) {
//         std::cout << "Failed to load the image" << std::endl;
//         return -1;
//     }

//     // Apply Gaussian blur
//     cv::GaussianBlur(image, image, cv::Size(5, 5), 0);

//     // Display the blurred image
//     cv::imshow("Blurred Image", image);
//     cv::waitKey(0);

//     return 0;
// }


//code to the pixel intensity at that particular location
//
// using namespace cv;
// int main(){
//     std::string image_path = "/home/kpit/opencv/samples/data/HappyFish.jpg";
//     cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
//     cv::imshow("Display window", img);
//     unsigned int y=80, x =90;
//     int intensity = img.at<uchar>(y, x);
//     // cv::imwrite("myImage.jpg",img);
//     cv::waitKey(0);//for printing the image in a window
//     std::cout<<img<<std::endl;//here we are printing the image location pixel values
//     std::cout<< "intensity"<<intensity<<std::endl;//here we are finding the particular index intensity
//     return 0;
// }

//this code is used to change the color image grayscale to binary like binary means 0 or 1 means 0 ==0 && 1==255 convert the values of pixel into like if pixel value at the particular location
//the value < 127 then covert into 0 if value > 127 then convert the value into 255
// using namespace cv;
// int main(){
//     std::string image_path = "/home/kpit/opencv/samples/data/HappyFish.jpg";
//     cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
//     unsigned int y=80, x =90;
//     int intensity = img.at<uchar>(y, x);
////img.rows and img.cols is used to find the number of rows and columns in a matrix
//     for(int i=0;i<img.rows;i++){
//         for(int j=0;j<img.cols;j++){
//             if(img.at<uchar>(i, j)<127){
//                 img.at<uchar>(i, j) = 0;
//             }else{
//                 img.at<uchar>(i, j) = 255;
//             }
//         }
//     }
//     cv::imshow("Display window", img);
//     cv::imwrite("myImage.jpg",img);
//     cv::waitKey(0);
//     // std::cout<<img<<std::endl;//here we are printing the image location pixel values
//     // std::cout<< "intensity"<<intensity<<std::endl;//here we are finding the particular index intensity
//     return 0;
// }


// #include<opencv2/opencv.hpp>
// #include <iostream>
// using namespace std;
// using namespace cv;

// int main()
// {
// 	string img_path = "/home/kpit/opencv/samples/data/HappyFish.jpg";
// 	Mat img = imread(img_path, IMREAD_COLOR);
// 	imshow("HappyFish Window", img);
	
// 	unsigned int y = 120, x = 100;
	
// 	Vec3b intensity = img.at<Vec3b> (y,x);
// 	unsigned int blue = intensity.val[0];
// 	unsigned int green = intensity.val[1];
// 	unsigned int red = intensity.val[2];
	
// 	cout << " Blue: " << blue << " Green: " << green  << " Red: " << red << endl;
// 	waitKey(0);
// 	return 0;
// }


// #include<opencv2/opencv.hpp>
// #include <iostream>
// using namespace std;
// using namespace cv;

// int main()
// {
// 	string img_path = "/home/kpit/opencv/samples/data/HappyFish.jpg";
// 	Mat color_image = imread(img_path, IMREAD_COLOR);
//     Mat channels[3];
//     cv::split(color_image, channels);
//     Mat blue_channel = channels[0],green_channel=channels[1],red_channel=channels[2];
// 	// imshow("HappyFish Window",red_channel);
//     // cv::imwrite("myImage.jpg",red_channel);

//     Mat color_image_new;
//     merge(std::vector<cv::Mat>{blue_channel,green_channel,red_channel},color_image_new);
//     imshow("HappyFish Window",color_image_new);
//     cv::imwrite("myImage.jpg",color_image_new);
	
// 	// unsigned int y = 120, x = 100;
	
// 	// Vec3b intensity = img.at<Vec3b> (y,x);
// 	// unsigned int blue = intensity.val[0];
// 	// unsigned int green = intensity.val[1];
// 	// unsigned int red = intensity.val[2];
	
// 	// cout << " Blue: " << blue << " Green: " << green  << " Red: " << red << endl;
// 	waitKey(0);
// 	// return 0;
// }



// #include <iostream>
// #include <opencv2/opencv.hpp>
// using namespace std;
// using namespace cv;
// int main(){
//     Mat img = imread("/home/kpit/opencv/samples/data/lena.jpg",IMREAD_COLOR);
//     if(img.empty()){
//         cout<<"Image not found"<<endl;
//         return -1;
//     }
//     imshow("Display Image",img);
//     waitKey(0);
// }



// #include <iostream>
// #include <opencv2/opencv.hpp>
// using namespace std;
// using namespace cv;
// int main(){
//     Mat img = imread("/home/kpit/opencv/samples/data/messi5.jpg",IMREAD_GRAYSCALE);
//     if(img.empty()){
//         cout<<"Image not found"<<endl;
//         return -1;
//     }
//     unsigned int x=100,y=120;
//     int intensity = img.at<uchar>(x,y);
//     cout<<"the value of intensity is: "<<intensity<<endl;

// }

// #include <iostream>
// #include <opencv2/opencv.hpp>
// using namespace std;
// using namespace cv;
// int main(){
//     Mat img = imread("/home/kpit/opencv/samples/data/messi5.jpg",IMREAD_GRAYSCALE);
//     if(img.empty()){
//         cout<<"Image not found"<<endl;
//         return -1;
//     }
//     imshow("Original Image",img);
//     for(int i=0;i<img.rows;i++){
//         for(int j=0;j<img.cols;j++){
//             if(img.at<uchar>(i,j)<127){
//                 img.at<uchar>(i,j) = 0;
//             }else{
//                 img.at<uchar>(i,j) = 255;
//             }
//         }
//     }
//     imshow("display Image",img);
//     waitKey(0);
// }

// #include <iostream>
// #include <opencv2/opencv.hpp>
// using namespace std;
// using namespace cv;
// int main(){
//     Mat img = imread("/home/kpit/opencv/samples/data/messi5.jpg",IMREAD_COLOR);
//     if(img.empty()){
//         cout<<"Image not found"<<endl;
//         return -1;
//     }
//     unsigned int x=120,y=120;
//     Vec3b intensity =img.at<Vec3b>(x,y);
//     int blue = intensity.val[0];
//     int green = intensity.val[1];
//     int red = intensity.val[2];
//     cout<<"Blue "<<blue<<"Green "<<green<<"red "<<red<<endl;
// }


// #include <iostream>
// #include <vector>
// #include <opencv2/opencv.hpp>
// using namespace std;
// using namespace cv;
// int main(){
//     Mat img = imread("/home/kpit/opencv/samples/data/messi5.jpg",IMREAD_COLOR);
//     if(img.empty()){
//         cout<<"Image not found"<<endl;
//         return -1;
//     }
//     vector<Mat> channels;
//     split(img,channels);
//     Mat blue = channels[0];
//     Mat green = channels[1];
//     Mat red = channels[2];
//     Mat colorimage;
//     merge(vector<Mat>{blue,green,red},colorimage);
//     imshow("COlored Image",colorimage);
//     waitKey(0);

// }

// #include <iostream>
// #include <opencv2/opencv.hpp>
// using namespace std;
// using namespace cv;
// int main(){
//     Mat img = imread("/home/kpit/opencv/samples/data/messi5.jpg",IMREAD_GRAYSCALE);
//     if(img.empty()){
//         cout<<"Image not found"<<endl;
//         return -1;
//     }
//     Mat dest;
//     cvtColor(img,dest,COLOR_GRAY2BGR);
//     cout<<"the number of channels before conversion of image: "<<img.channels()<<endl;
//     cout<<"after conversion"<<dest.channels()<<endl;
//     imshow("color image",img);
//     imshow("Gray image",dest);
//     waitKey(0);

// }


// #include <iostream>
// #include <opencv2/opencv.hpp>
// using namespace std;
// using namespace cv;
// int main(){
    // Mat img = imread("/home/kpit/opencv/samples/data/messi5.jpg",IMREAD_GRAYSCALE);
    // if(img.empty()){
    //     cout<<"Image not found"<<endl;
    //     return -1;
    // }
//     Rect r(0,0,300,300);
//     Mat dest = img(r);
//     imshow("Transformed Image",dest);
//     waitKey(0);
// }


// #include <iostream>
// #include <opencv2/opencv.hpp>
// using namespace std;
// using namespace cv;
// int main(){
//     Mat img = imread("/home/kpit/opencv/samples/data/messi5.jpg",IMREAD_COLOR);
//     if(img.empty()){
//         cout<<"Image not found"<<endl;
//         return -1;
//     }
//     Mat dest;
//     int width = 2*img.cols;
//     int height = 2*img.rows;
//     resize(img,dest,Size(width,height),INTER_LINEAR);
//     imshow("transformed image",dest);
//     waitKey(0);
// }


// #include <iostream>
// #include <opencv2/opencv.hpp>
// using namespace std;
// using namespace cv;
// int main(){
//     Mat img = imread("/home/kpit/opencv/samples/data/messi5.jpg",IMREAD_GRAYSCALE);
//     if(img.empty()){
//         cout<<"Image not found"<<endl;
//         return -1;
//     }
//     Mat dest=Mat::zeros(img.size(),img.type());
//     double alpha = 2;
//     double beta = 5;
//     for(int i=0;i<img.rows;i++){
//         for(int j=0;j<img.cols;j++){
//             dest.at<uchar>(i,j) = saturate_cast<uchar>(alpha*img.at<uchar>(i,j)+beta);
//         }
//     }
//     imshow("Bright Image",dest);
//     waitKey(0);
// }

// #include <iostream>
// #include <opencv2/opencv.hpp>
// using namespace std;
// using namespace cv;
// int main(){
//     Mat img = imread("/home/kpit/opencv/samples/data/messi5.jpg",IMREAD_COLOR);
//     if(img.empty()){
//         cout<<"Image not found"<<endl;
//         return -1;
//     }
//     Mat dest = Mat::zeros(img.size(),img.type());
//     double alpha = 4.5;
//     double beta = 5.7;
//     for(int i=0;i<img.rows;i++){
//         for(int j=0;j<img.cols;j++){
//             for(int k=0;k<img.channels();k++){
//                 dest.at<Vec3b>(i,j).val[k] = saturate_cast<uchar>(alpha*img.at<Vec3b>(i,j).val[k]+beta);
//             }
//         }
//     }
//     imshow("colour bright&contrast",dest);
//     waitKey(0);
// }


// #include <iostream>
// #include <opencv2/opencv.hpp>
// using namespace std;
// using namespace cv;
// int main(){
//     Mat img = imread("/home/kpit/opencv/samples/data/messi5.jpg",IMREAD_GRAYSCALE);
//     if(img.empty()){
//         cout<<"Image not found"<<endl;
//         return -1;
//     }
//     float gamma =2.5;
//     float inv_gamma = 1.0f/gamma;
//     Mat dest= img.clone();
//     for(int i=0;i<img.rows;i++){
//         for(int j=0;j<img.cols;j++){
            
//                 dest.at<uchar>(i,j) = saturate_cast<uchar>(pow(img.at<uchar>(i,j)/255.0f,inv_gamma)*255.0f);
            
//         }
//     }
//     imshow("Original Image",img);
//     imshow("Transformed Image",dest);
//     waitKey(0);
// }

// #include <iostream>
// #include <opencv2/opencv.hpp>
// using namespace std;
// using namespace cv;
// int main(){
//     Mat img = imread("/home/kpit/opencv/samples/data/messi5.jpg",IMREAD_COLOR);
//     Mat img1 = imread("/home/kpit/opencv/samples/data/lena.jpg",IMREAD_COLOR);\
//     resize(img,img,img1.size());
//     if(img.empty()||img1.empty()){
//         cout<<"Image not found"<<endl;
//         return -1;
//     }
//     Mat dest;
//     dest = img+img1;
//     imshow("Source1",img);
//     imshow("Source2",img1);
//     imshow("Transformed",dest);
//     waitKey(0);
// }

// #include <iostream>
// #include <opencv2/opencv.hpp>
// using namespace std;
// using namespace cv;
// int main(){
//     Mat img = imread("/home/kpit/opencv/samples/data/messi5.jpg",IMREAD_GRAYSCALE);
//     if(img.empty()){
//         cout<<"Image not found"<<endl;
//         return -1;
//     }
//     Mat dest;
//     Mat Affine = getRotationMatrix2D(Point(img.cols/2,img.rows/2),90,1);
//     warpAffine(img,dest,Affine,img.size());
//     imshow("Transformed",dest);
//     waitKey(0);
// }


// #include <iostream>
// #include <vector>
// #include <opencv2/opencv.hpp>
// using namespace std;
// using namespace cv;
// int main(){
//     Mat img = imread("/home/kpit/opencv/samples/data/messi5.jpg",IMREAD_COLOR);
//     if(img.empty()){
//         cout<<"Image not found"<<endl;
//         return -1;
//     }
//     double array[3][3] ={{0,300,300},{200,150,300},{200,100,1}};
//     vector<Point2f> srcpts;
//     vector<Point2f> destpts;
//     srcpts.push_back(Point2f(100,100));
//     srcpts.push_back(Point2f(100,500));
//     srcpts.push_back(Point2f(500,100));
//     srcpts.push_back(Point2f(500,500));
//     destpts.push_back(Point2f(100,270));
//     destpts.push_back(Point2f(400,170));
//     destpts.push_back(Point2f(300,270));
//     destpts.push_back(Point2f(200,470));
//     Mat pers = getPerspectiveTransform(srcpts,destpts);


//     Mat perspective(3,3,CV_64F,array);
//     Mat dest;
//     warpPerspective(img,dest,pers,img.size());
//     imshow("Original",img);
//     imshow("transformed",dest);
//     waitKey(0);
// }

// #include <iostream>
// #include <opencv2/opencv.hpp>
// using namespace std;
// using namespace cv;
// int main(){
//     Mat img = imread("/home/kpit/opencv/samples/data/messi5.jpg",IMREAD_COLOR);
//     Mat img1 = imread("/home/kpit/opencv/samples/data/lena.jpg",IMREAD_COLOR);\
//     resize(img,img,img1.size());
//     if(img.empty()||img1.empty()){
//         cout<<"Image not found"<<endl;
//         return -1;
//     }
//     Mat b_and,b_or,b_not,b_xor;
//     bitwise_and(img,img1,b_and);
//     bitwise_or(img,img1,b_or);
//     bitwise_not(img,b_not);
//     bitwise_xor(img,img1,b_xor);
//     imshow("AND",b_and);
//     imshow("Or",b_or);
//     imshow("NOT",b_not);
//     imshow("XOR",b_xor);
//     waitKey(0);
// }

//Displaying the image
// #include <iostream>
// #include <opencv2/opencv.hpp>
// using namespace std;
// using namespace cv;
// int main(){
//     Mat img = imread("/home/kpit/opencv/samples/data/messi5.jpg",IMREAD_GRAYSCALE);
//     if(img.empty()){
//         cout<<"Image Not Found"<<endl;
//         return -1;
//     }
//     imshow("Display Image1",img);
//     waitKey(0);
// }

//display the value of intensity at particular location
// #include <iostream>
// #include <opencv2/opencv.hpp>
// using namespace std;
// using namespace cv;
// int main(){
//     Mat img = imread("/home/kpit/opencv/samples/data/messi5.jpg",IMREAD_GRAYSCALE);
//     if(img.empty()){
//         cout<<"Image Not Found"<<endl;
//         return -1;
//     }
//     unsigned int x=120,y=120;
//     int intensity = img.at<uchar>(x,y);
//     cout<<"the intensity value is: "<<intensity<<endl;
// }


//converting the grayscale image into full balck and white image(binary)
// #include <iostream>
// #include <opencv2/opencv.hpp>
// using namespace std;
// using namespace cv;
// int main(){
//     Mat img = imread("/home/kpit/opencv/samples/data/messi5.jpg",IMREAD_GRAYSCALE);
//     if(img.empty()){
//         cout<<"Image Not Found"<<endl;
//         return -1;
//     }
//     for(int i=0;i<img.rows;i++){
//         for(int j=0;j<img.cols;j++){
//             if(img.at<uchar>(i,j)<127){
//                 img.at<uchar>(i,j) = 0;
//             }else{
//                 img.at<uchar>(i,j)=255;
//             }
//         }
//     }
//     imshow("Image",img);
//     waitKey(0);
// }

//Displaying the intensity value of colour image intensity at par loc
// #include <iostream>
// #include <opencv2/opencv.hpp>
// using namespace std;
// using namespace cv;
// int main(){
//     Mat img = imread("/home/kpit/opencv/samples/data/messi5.jpg",IMREAD_GRAYSCALE);
//     if(img.empty()){
//         cout<<"Image Not Found"<<endl;
//         return -1;
//     }
//     unsigned int x=120,y=120;
//     Vec3b intensity = img.at<Vec3b>(x,y);
//     int blue = intensity.val[0];
//     int green = intensity.val[1];
//     int red =intensity.val[2];
//     cout<<"Blue "<<blue<<" Green "<<green<<" Red "<<red<<endl;
// }


//split and merge of an image into another image;
// #include <iostream>
// #include <vector>
// #include <opencv2/opencv.hpp>
// using namespace std;
// using namespace cv;
// int main(){
//     Mat img = imread("/home/kpit/opencv/samples/data/messi5.jpg",IMREAD_COLOR);
//     if(img.empty()){
//         cout<<"Image Not Found"<<endl;
//         return -1;
//     }
//     vector<Mat> channels;
//     split(img,channels);
//     Mat blue = channels[0];
//     Mat green = channels[1];
//     Mat red = channels[2];
//     Mat colorphoto;
//     merge(vector<Mat>{blue,green,red},colorphoto);
//     imshow("Image conversion",colorphoto);
//     waitKey(0);
// }


//working of cvtColor function
// #include <iostream>
// #include <opencv2/opencv.hpp>
// using namespace std;
// using namespace cv;
// int main(){
//     Mat img = imread("/home/kpit/opencv/samples/data/messi5.jpg",IMREAD_COLOR);
//     if(img.empty()){
//         cout<<"Image Not Found"<<endl;
//         return -1;
//     }
//     Mat dest;

//     cvtColor(img,dest,COLOR_BGR2GRAY);
//     imshow("Original",img);
//     imshow("transformed",dest);
//     waitKey(0);
//     cout<<"Before color conversion"<<img.channels()<<endl;
//     cout<<"After color conversion"<<dest.channels()<<endl;
// }


//Cropping an image using Rect function
// #include <iostream>
// #include <opencv2/opencv.hpp>
// using namespace std;
// using namespace cv;
// int main(){
//     Mat img = imread("/home/kpit/opencv/samples/data/messi5.jpg",IMREAD_GRAYSCALE);
//     if(img.empty()){
//         cout<<"Image Not Found"<<endl;
//         return -1;
//     }
//     Rect r(20,20,500,320);
//     Mat dest = img(r);
//     imshow("Original Image",img);
//     imshow("transformed Image",dest);
//     waitKey(0);
// }


//scaling an image using resize function
// #include <iostream>
// #include <opencv2/opencv.hpp>
// using namespace std;
// using namespace cv;
// int main(){
//     Mat img = imread("/home/kpit/opencv/samples/data/messi5.jpg",IMREAD_GRAYSCALE);
//     if(img.empty()){
//         cout<<"Image Not Found"<<endl;
//         return -1;
//     }
//     Mat dest;
//     int height = 2*img.rows;
//     int width = 2*img.cols;
//     double scale_fcator = 2;
//     // resize(img,dest,Size(width,height),INTER_LINEAR);
//     resize(img,dest,Size(),scale_fcator,scale_fcator,INTER_LINEAR);
//     imshow("Transformed Image",dest);
//     waitKey(0);
// }


//Brightness and contrast of an image
// #include <iostream>
// #include <opencv2/opencv.hpp>
// using namespace std;
// using namespace cv;
// int main(){
//     Mat img = imread("/home/kpit/opencv/samples/data/messi5.jpg",IMREAD_COLOR);
//     if(img.empty()){
//         cout<<"Image Not Found"<<endl;
//         return -1;
//     }
//     Mat dest = Mat::zeros(img.size(),img.type());
//     double alpha = 3;
//     double beta = 5;
//     for(int i=0;i<img.rows;i++){
//         for(int j=0;j<img.cols;j++){
//             for(int k=0;k<img.channels();k++){
//             dest.at<Vec3b>(i,j).val[k] = saturate_cast<uchar>(alpha*img.at<Vec3b>(i,j).val[k]+beta);
//             }
//         }
//     }
//     imshow("Original ",img);
//     imshow("Transformed ",dest);
//     waitKey(0);
// }

//gamma correction
// #include <iostream>
// #include <opencv2/opencv.hpp>
// using namespace std;
// using namespace cv;
// int main(){
//     Mat img = imread("/home/kpit/opencv/samples/data/messi5.jpg",IMREAD_COLOR);
//     if(img.empty()){
//         cout<<"Image Not Found"<<endl;
//         return -1;
//     }
//     double gamma = 4.9;
//     double inv_gamma = 1.0/gamma;
//     Mat dest = img.clone();
//     for(int i=0;i<img.rows;i++){
//         for(int j=0;j<img.cols;j++){
//             for(int k=0;k<img.channels();k++){
//             dest.at<Vec3b>(i,j).val[k] = saturate_cast<uchar>(pow(img.at<Vec3b>(i,j).val[k]/255.0f,inv_gamma)*255.0f);
//             }
//         }
//     }
//     imshow("Original",img);
//     imshow("Transformed",dest);
//     waitKey(0);
// }

//affine for rotation of image and perspective for vision of seeing
// #include <iostream>
// #include <opencv2/opencv.hpp>
// using namespace std;
// using namespace cv;
// int main(){
//     Mat img = imread("/home/kpit/opencv/samples/data/messi5.jpg",IMREAD_GRAYSCALE);
//     if(img.empty()){
//         cout<<"Image Not Found"<<endl;
//         return -1;
//     }
//     Mat Affine = getRotationMatrix2D(Point(img.cols/2,img.rows/2),90,1.0);
//     Mat dest;
//     warpAffine(img,dest,Affine,img.size());
//     imshow("Transformed ",dest);
//     waitKey(0);
// }


// #include <iostream>
// #include <opencv2/opencv.hpp>
// using namespace std;
// using namespace cv;
// int main(){
//     Mat img = imread("/home/kpit/opencv/samples/data/messi5.jpg",IMREAD_COLOR);
//     if(img.empty()){
//         cout<<"Image Not Found"<<endl;
//         return -1;
//     }
//     double array[3][3] = {{0.0,1.0,0.5},{0.9,1.0,2.0},{0.5,0.8,1}};
//     Mat perspective(3,3,CV_64F,array);
//     Mat dest;
//     warpPerspective(img,dest,perspective,img.size());
//     imshow("Transformed Image",dest);
//     waitKey(0);
// }

//multiplication of pixel intensities
// #include <iostream>
// #include <opencv2/opencv.hpp>
// using namespace std;
// using namespace cv;
// int main(){
//     Mat src1 = imread("/home/kpit/opencv/samples/data/messi5.jpg",IMREAD_COLOR);
//     Mat src2 = imread("/home/kpit/opencv/samples/data/lena.jpg",IMREAD_COLOR);
//     resize(src2,src2,src1.size());
//     if(src1.empty() || src2.empty() ){
//         cout<<"Image Not Found"<<endl;
//         return -1;
//     }
//     Mat dest;
//     dest = src2.mul(src1);
//     imshow("Original1",src1);
//     imshow("Original2",src2);
//     imshow("Transformed",dest);
//     waitKey(0);
// }


// #include <iostream>
// #include <opencv2/opencv.hpp>
// using namespace std;
// using namespace cv;
// int main(){
//     Mat src1 = imread("/home/kpit/opencv/samples/data/messi5.jpg",IMREAD_COLOR);
//     Mat src2 = imread("/home/kpit/opencv/samples/data/lena.jpg",IMREAD_COLOR);
//     resize(src2,src2,src1.size());
//     if(src1.empty() || src2.empty() ){
//         cout<<"Image Not Found"<<endl;
//         return -1;
//     }
//     Mat dest;
//     dest = src1-src2;
//     imshow("Original1",src1);
//     imshow("Original2",src2);
//     imshow("Transformed",dest);
//     waitKey(0);
// }



//logical operations on images
// #include <iostream>
// #include <opencv2/opencv.hpp>
// using namespace std;
// using namespace cv;
// int main(){
//     Mat src1 = imread("/home/kpit/opencv/samples/data/messi5.jpg",IMREAD_GRAYSCALE);
//     Mat src2 = imread("/home/kpit/opencv/samples/data/lena.jpg",IMREAD_GRAYSCALE);
//     resize(src2,src2,src1.size());
//     if(src1.empty() || src2.empty() ){
//         cout<<"Image Not Found"<<endl;
//         return -1;
//     }
//     Mat bit_and,bit_or,bit_not,bit_xor;
//     bitwise_and(src1,src2,bit_and);
//     bitwise_or(src1,src2,bit_or);
//     bitwise_not(src2,bit_not);
//     bitwise_xor(src1,src2,bit_xor);
//     imshow("AND operation",bit_and);
//     imshow("OR operation",bit_or);
//     imshow("NOT Operation",bit_not);
//     imshow("Xor Operation",bit_xor);
//     waitKey(0);
// }


//histogram on grayscale image
// #include <iostream>
// #include <opencv2/opencv.hpp>
// using namespace std;
// using namespace cv;
// int main(){
//     Mat img = imread("/home/kpit/opencv/samples/data/lena.jpg",IMREAD_GRAYSCALE);
//     if(img.empty()){
//         cout<<"Image Not Found"<<endl;
//         return -1;
//     }
//     int histSize = 256;
//     float range[] = {0,256};
//     const float* histRange[]= {range};
//     Mat hist;
//     calcHist(&img,1,0,Mat(),hist,1,&histSize,histRange);
//     int width =256,height=200;
//     Mat histImage(height,width,CV_8UC3,Scalar(0,0,0));
//     normalize(hist,hist,0,histImage.rows,NORM_MINMAX);
//     for(int i=0;i<histSize;i++){
//         line(histImage,Point((i-1),(height-hist.at<float>(i-1))),Point((i),(height - hist.at<float>(i))),Scalar(255,0,0),2,8,0);
//     }
//     imshow("original",img);
//     imshow("Histogram",histImage);
//     waitKey(0);
// }

//Histogram on color image
// #include <iostream>
// #include <vector>
// #include <opencv2/opencv.hpp>
// using namespace std;
// using namespace cv;
// int main(){
//     Mat img = imread("/home/kpit/opencv/samples/data/lena.jpg",IMREAD_COLOR);
//     if(img.empty()){
//         cout<<"Image Not found"<<endl;
//         return -1;
//     }
    // vector<Mat> channels;
    // split(img,channels);
    // int histSize = 256;
    // float range[] = {0,256};
    // const float* histRange[] = {range};
    // Mat b_hist,g_hist,r_hist;
    // calcHist(&channels[0],1,0,Mat(),b_hist,1,&histSize,histRange);
    // calcHist(&channels[1],1,0,Mat(),g_hist,1,&histSize,histRange);
    // calcHist(&channels[2],1,0,Mat(),r_hist,1,&histSize,histRange);
    // int width = 512,height = 400;
    // int b_mul = cvRound((double)width/histSize);
    // Mat histImage(height,width,CV_8UC3,Scalar(0,0,0));
    // normalize(b_hist,b_hist,0,histImage.rows,NORM_MINMAX);
    // normalize(g_hist,g_hist,0,histImage.rows,NORM_MINMAX);
    // normalize(r_hist,r_hist,0,histImage.rows,NORM_MINMAX);

    // for(int i=0;i<histSize;i++){
    //     line(histImage,Point(b_mul*(i-1),(height- cvRound(b_hist.at<float>(i-1)))),Point(b_mul*(i),(height - cvRound(b_hist.at<float>(i)))),Scalar(255,0,0),2,8,0);
    //     line(histImage,Point(b_mul*(i-1),(height- cvRound(g_hist.at<float>(i-1)))),Point(b_mul*(i),(height - cvRound(g_hist.at<float>(i)))),Scalar(0,255,0),2,8,0);
    //     line(histImage,Point(b_mul*(i-1),(height- cvRound(r_hist.at<float>(i-1)))),Point(b_mul*(i),(height - cvRound(r_hist.at<float>(i)))),Scalar(0,0,255),2,8,0);
    // }
//     imshow("Original",img);
//     imshow("Histogram color",histImage);
//     waitKey(0);    
// }

//equalizeHist function it will increase the contrast of the image
// #include <iostream>
// #include <opencv2/opencv.hpp>
// using namespace std;
// using namespace cv;
// int main(){
//     Mat img = imread("/home/kpit/opencv/samples/data/lena.jpg",IMREAD_GRAYSCALE);
//     if(img.empty()){
//         cout<<"Image Not Found"<<endl;
//         return -1;
//     }
//     Mat dest;
//     equalizeHist(img,dest);
//     int histSize = 256;
//     float range[] = {0,256};
//     const float* histRange[]= {range};
//     Mat hist,hist1;
//     calcHist(&img,1,0,Mat(),hist,1,&histSize,histRange);
//     calcHist(&dest,1,0,Mat(),hist1,1,&histSize,histRange);
//     int width =256,height=200;
//     Mat histImage(height,width,CV_8UC3,Scalar(0,0,0));
//     Mat histImage1(height,width,CV_8UC3,Scalar(0,0,0));
//     normalize(hist,hist,0,histImage.rows,NORM_MINMAX);
//     normalize(hist1,hist1,0,histImage1.rows,NORM_MINMAX);
//     for(int i=0;i<histSize;i++){
//         line(histImage,Point((i-1),(height-hist.at<float>(i-1))),Point((i),(height - hist.at<float>(i))),Scalar(255,0,0),2,8,0);
//         line(histImage1,Point((i-1),(height-hist1.at<float>(i-1))),Point((i),(height - hist1.at<float>(i))),Scalar(255,0,0),2,8,0);
//     }
//     imshow("Original",img);
//     imshow("Histogram original",histImage);
//     imshow("Transformed",dest);
//     imshow("Histogram Tarnsformed",histImage1);
//     waitKey(0);
// }

// #include <iostream>
// #include <opencv2/opencv.hpp>
// using namespace std;
// using namespace cv;
// int main(){
//     Mat img = imread("/home/kpit/opencv/samples/data/lena.jpg",IMREAD_GRAYSCALE);
//     if(img.empty()){
//         cout<<"Image Not Found"<<endl;
//         return -1;
//     }
//     int histSize = 256;
//     float range[] = {0,256};
//     const float* histRange[] = {range};
//     Mat hist;
//     calcHist(&img,1,0,Mat(),hist,1,&histSize,histRange);
//     Mat dest;
//     matchHistogram(img,hist,dest);
//     imshow("jbjb",img);
//     imshow("dest",dest);
//     waitKey(0);
// }


//spatial filters
// #include <iostream>
// #include <opencv2/opencv.hpp>
// using namespace std;
// using namespace cv;
// int main(){
//     Mat img = imread("/home/kpit/opencv/samples/data/messi5.jpg",IMREAD_COLOR);
//     if(img.empty()){
//         cout<<"Image Not Found"<<endl;
//         return -1;
//     }
//     Mat dest,opening,closing,gradient,tophat,blackhat;
//     // boxFilter(img,dest,-1,Size(20,20),Point(-1,-1),true,BORDER_ISOLATED);
//     // blur(img,dest,Size(20,20),Point(-1,-1),BORDER_CONSTANT);
//     // GaussianBlur(img,dest,Size(7,7),20,20,BORDER_CONSTANT);
//     // medianBlur(img,dest,15);
//     Mat kernal = getStructuringElement(MORPH_RECT,Size(3,2));
//     Mat kernal1 = (Mat_<char>(3,3)<<-1,-1,-1,-1,9,-1,-1,-1,-1);

//     // erode(img,dest,kernal1);
//     // dilate(img,dest,kernal1);
//     morphologyEx(img,opening,MORPH_OPEN,kernal);
//     morphologyEx(img,closing,MORPH_CLOSE,kernal);
//     morphologyEx(img,gradient,MORPH_GRADIENT,kernal);
//     morphologyEx(img,tophat,MORPH_TOPHAT,kernal);
//     morphologyEx(img,blackhat,MORPH_BLACKHAT,kernal);
//     imshow("Original",img);
//     imshow("opening",opening);
//     imshow("closing",closing);
//     imshow("Gradient",gradient);
//     imshow("Tophat",tophat);
//     imshow("blackhat",blackhat);
//     waitKey(0);
// }






#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
void SpatialFilter1(const Mat& src){
    Mat bfilter,blurf,gfilter,mfilter;
    boxFilter(src,bfilter,-1,Size(20,20),Point(-1,-1),true,BORDER_CONSTANT);
    blur(src,blurf,Size(20,20),Point(-1,-1),BORDER_CONSTANT);
    GaussianBlur(src,gfilter,Size(7,7),20,20,BORDER_CONSTANT);
    medianBlur(src,mfilter,15);
    imshow("BOX FILTER",bfilter);
    imshow("BLUR",blurf);
    imshow("Gaussian Blur",gfilter);
    imshow("Median Blur",mfilter);
    }
    void MorphologicalImages(Mat src){
        Mat erodedest;
        Mat dilatedest;
        Mat opening;
        Mat closing;
        if(src.empty()){
            cout<<"image not found"<<endl;
        }
        erode(src,erodedest,getStructuringElement(MORPH_RECT,Size(3,3)));
        dilate(src,dilatedest,getStructuringElement(MORPH_RECT,Size(3,3)));
        imshow("Erode",erodedest);
        imshow("Dilate",dilatedest);
    }
    void Brightandcontrast(Mat src){
        int alpha = 3;
        int beta = 1.5;
        Mat output = Mat::zeros(src.size(),src.type());
        for(int i=0;i<src.rows;i++){
            for(int j=0;j<src.cols;j++){
                for(int k=0;k<src.channels();k++){
                    output.at<Vec3b>(i,j).val[k] = alpha * src.at<Vec3b>(i,j).val[k]+beta;
                }
            }
        }
        imshow("Brightness&Contrast",output);
    }
    void edgedetection(Mat src){
        if(src.empty()){
            cout<<"image not found"<<endl;
        }
        Mat gray;
        cvtColor(src,gray,COLOR_BGR2GRAY);
        Mat blur;
        GaussianBlur(gray,blur,Size(3,3),0);
        Mat edges;
        Mat lap;
        Laplacian(blur,lap,CV_16S,3);
        Mat conver;
        convertScaleAbs(lap,conver);
        Canny(blur,edges,50,150,3);
        imshow("Final output",conver);
        imshow("Edge Image Detection",edges);
    }

    void objectRotation(Mat img){
    Mat Affine = getRotationMatrix2D(Point(img.cols/2,img.rows/2),90,1.0);
    Mat dest;
    warpAffine(img,dest,Affine,img.size());
    imshow("Transformed ",dest);
    }
    void histogram1(const Mat src){
        vector<Mat> channels;
    split(src,channels);
    int histSize = 256;
    float range[] = {0,256};
    const float* histRange[] = {range};
    Mat b_hist,g_hist,r_hist;
    calcHist(&channels[0],1,0,Mat(),b_hist,1,&histSize,histRange);
    calcHist(&channels[1],1,0,Mat(),g_hist,1,&histSize,histRange);
    calcHist(&channels[2],1,0,Mat(),r_hist,1,&histSize,histRange);
    int width = 512,height = 400;
    int b_mul = cvRound((double)width/histSize);
    Mat histImage(height,width,CV_8UC3,Scalar(0,0,0));
    normalize(b_hist,b_hist,0,histImage.rows,NORM_MINMAX);
    normalize(g_hist,g_hist,0,histImage.rows,NORM_MINMAX);
    normalize(r_hist,r_hist,0,histImage.rows,NORM_MINMAX);

    for(int i=0;i<histSize;i++){
        line(histImage,Point(b_mul*(i-1),(height- cvRound(b_hist.at<float>(i-1)))),Point(b_mul*(i),(height - cvRound(b_hist.at<float>(i)))),Scalar(255,0,0),2,8,0);
        line(histImage,Point(b_mul*(i-1),(height- cvRound(g_hist.at<float>(i-1)))),Point(b_mul*(i),(height - cvRound(g_hist.at<float>(i)))),Scalar(0,255,0),2,8,0);
        line(histImage,Point(b_mul*(i-1),(height- cvRound(r_hist.at<float>(i-1)))),Point(b_mul*(i),(height - cvRound(r_hist.at<float>(i)))),Scalar(0,0,255),2,8,0);
    }
    imshow("histogram",histImage);
    }
int main(){
    VideoCapture cap(0);
    if(!cap.isOpened()){
        cout<<"web cam not found"<<endl;
        return -1;
    }
    Mat photo;
    while(true){
        Mat frame,blurcam,grad;
        cap>>frame;
        if(frame.empty()){
            cout<<"frame not found"<<endl;
            break;
        }
        imshow("Original Webcam",frame);
        SpatialFilter1(frame);
        histogram1(frame);
        // MorphologicalImages(frame);
        edgedetection(frame);
        Brightandcontrast(frame);
        objectRotation(frame);

        int key = waitKey(1);
        if(key == ' '){
            frame.copyTo(photo);
        
        }
        if(key == 'q'||key == 'Q'){
            break;
        }
    }
    cap.release();
    SpatialFilter1(photo);
    imshow("Photo",photo);
    waitKey(0);
    
}
