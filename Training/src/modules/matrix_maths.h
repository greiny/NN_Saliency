#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

// Conv2 parameter
#define CONV_FULL 0
#define CONV_SAME 1
#define CONV_VALID 2
// Pooling methods
#define POOL_MAX 0
#define POOL_MEAN 1
#define POOL_STOCHASTIC 2
// non-linearity
#define NL_SIGMOID 0
#define NL_TANH 1
#define NL_RELU 2
#define NL_LEAKY_RELU 3

#define ATD at<double>
#define AT3D at<cv::Vec3d>
#define elif else if

double Reciprocal(const double &);
Mat Reciprocal(const Mat &);
Mat sigmoid(const Mat &);
Mat dsigmoid_a(const Mat &);
Mat dsigmoid(const Mat &);
Mat ReLU(const Mat& );
Mat dReLU(const Mat& );
Mat Tanh(const Mat &);
Mat dTanh(const Mat &);
Mat nonLinearity(const Mat &, int);
Mat dnonLinearity(const Mat &, int);
// Mimic rot90() in Matlab/GNU Octave.
Mat rot90(const Mat &, int);
// A Matlab/Octave style 2-d convolution function.
// from http://blog.timmlinder.com/2011/07/opencv-equivalent-to-matlabs-conv2-function/
Mat conv2(const Mat&, const Mat&, int);
Mat convCalc(const Mat&, const Mat&, int);
// get KroneckerProduct 
// for upsample
// see function kron() in Matlab/Octave
Mat kron(const Mat&, const Mat&);
Mat getBernoulliMatrix(int, int, double);

// Follows are OpenCV maths
Mat exp(const Mat&);
Mat log(const Mat&);
Mat reduce(const Mat&, int, int);
Mat divide(const Mat&, const Mat&);
Mat pow(const Mat&, double);
double sum1(const Mat&);
double max(const Mat&);
double min(const Mat&);
