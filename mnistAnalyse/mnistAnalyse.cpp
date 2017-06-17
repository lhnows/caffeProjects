
#define USE_OPENCV 1
#define CPU_ONLY 1
//#define USE_ACCELERATE

#include <iostream>
#include <string>
#include <fstream> 
   #include <stdlib.h>  
#include <caffe/caffe.hpp>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "head.h"

#include <algorithm>
#include <iosfwd>
#include <memory>
#include <utility>
#include <vector>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <google/protobuf/text_format.h>

#if defined(USE_LEVELDB) && defined(USE_LMDB)
#include <leveldb/db.h>
#include <leveldb/write_batch.h>
#include <lmdb.h>
#endif

#include <stdint.h>
#include <sys/stat.h>

#include <fstream> // NOLINT(readability/streams)
#include <string>

#include "boost/scoped_ptr.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"

//#if defined(USE_LEVELDB) && defined(USE_LMDB)

using boost::scoped_ptr;
using std::string;
using namespace caffe;
using namespace cv;
//using namespace std;

//GFLAGS工具定义命令后选项backend，默认值为lmdb，即--backend=lmdb
DEFINE_string(backend, "lmdb", "The backend for storing the result");

//大小端转换。Mnist原始数据文件中32位整形值为大端存储，C/C++变量为小端存储，因此需要加入转换机制
uint32_t swap_endian(uint32_t val)
{
	val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
	return (val << 16) | (val >> 16);
}

cv::Point previousPoint(-1, -1), nowPoint(-1, -1);
Mat srcimage = Mat::zeros(280, 280, CV_8UC1);
Mat srcimageori = Mat::zeros(280, 280, CV_8UC1);
	//打开输出文件
std::ofstream out("out.txt");  

class Classifier
{
  public:
	Classifier(const string &model_file,
			   const string &trained_file);

	int Classify(const cv::Mat &img);

  private:
	std::vector<int> Predict(const cv::Mat &img);

  private:
	shared_ptr<Net<float> > net_;
	cv::Size input_geometry_;
	int num_channels_;
};

Classifier::Classifier(const string &model_file,
					   const string &trained_file)
{
#ifdef CPU_ONLY
	Caffe::set_mode(Caffe::CPU);
#else
	Caffe::set_mode(Caffe::GPU);
#endif

	/* Load the network. */
	net_.reset(new Net<float>(model_file, TEST));
	net_->CopyTrainedLayersFrom(trained_file);

	CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
	CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

	Blob<float> *input_layer = net_->input_blobs()[0];
	num_channels_ = input_layer->channels();
	CHECK(num_channels_ == 3 || num_channels_ == 1)
		<< "Input layer should have 1 or 3 channels.";
	input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
}

/* Return the top N predictions. */
int Classifier::Classify(const cv::Mat &img)
{
	std::vector<int> output = Predict(img);
	std::vector<int>::iterator iter = find(output.begin(), output.end(), 1);
	int prediction = distance(output.begin(), iter);
	return prediction < 10 ? prediction : 0;
}
std::vector<int> Classifier::Predict(const cv::Mat &aImage)
{
	Blob<float> *tInput = net_->input_blobs()[0];
	//tInput->Reshape(1, 1,28, 28);
	/* Forward dimension change to all layers. */
	//net_->Reshape();
	float *tInPtr = tInput->mutable_cpu_data();

	//获取图像信息
	int tChannel = tInput->channels();
	int tHeight = tInput->height();
	int tWidth = tInput->width();

	//检查参数
	if (!(tChannel == 1 && tHeight == 28 && tWidth == 28))
	{
		std::cout << "exit with check params" << std::endl;
		exit(0);
	}
	if (!(aImage.channels() == 1 && aImage.rows == 28 && aImage.cols == 28))
	{
		std::cout << "exit with check params" << std::endl;
		exit(0);
	}

	//检查图像
	if (aImage.empty())
	{
		std::cout << "exit with aImage empty" << std::endl;
		exit(0);
	}

	//获取图像指针
	unsigned char *tImagePtr = aImage.data;

	//拷贝图像

	for (int y = 0; y < tHeight; y++)
	{
		for (int x = 0; x < tWidth; x++)
		{
			//计算偏移量
			int tOffset = y * tWidth + x;

			//设置像素值
			tInPtr[tOffset] = tImagePtr[tOffset]; // / 255.0;
		}
	}
	//memcpy(tInPtr,tImagePtr,tHeight*tWidth);
	//前向传播
	net_->Forward();

	/* Copy the output layer to a std::vector */
	Blob<float> *output_layer = net_->output_blobs()[0];
	const float *begin = output_layer->cpu_data();
	const float *end = begin + output_layer->channels();

	for (int i = 0; i < output_layer->channels(); i++)
	{
		//std::cout << begin[i];
		out<<begin[i]<<",";
	}
	

	//std::cout << std::endl;
	return std::vector<int>(begin, end);
}

static void on_Mouse(int event, int x, int y, int flags, void *)
{

	if (event == EVENT_LBUTTONUP || !(flags & EVENT_FLAG_LBUTTON))
	{
		previousPoint = cv::Point(-1, -1);
	}
	else if (event == EVENT_LBUTTONDOWN)
	{
		previousPoint = cv::Point(x, y);
	}
	else if (event == EVENT_MOUSEMOVE || (flags & EVENT_FLAG_LBUTTON))
	{
		cv::Point pt(x, y);
		if (previousPoint.x < 0)
		{
			previousPoint = pt;
		}
		line(srcimage, previousPoint, pt, Scalar(255), 16, 8, 0);
		previousPoint = pt;
		imshow("result", srcimage);
	}
}

int main(int argc, char **argv)
{

	::google::InitGoogleLogging(argv[0]);

#ifdef CPU_ONLY
	Caffe::set_mode(Caffe::CPU);
#else
	Caffe::set_mode(Caffe::GPU);
#endif

	string model_file = "../lenet.prototxt";
	string trained_file = "../lenet_2features_iter_10000.caffemodel";
	Classifier classifier(model_file, trained_file);

	//std::cout << "------directed by watersink------" << std::endl;
	//std::cout << "------------esc:退出-----------" << std::endl;
	//std::cout << "--------------1:还原-------------" << std::endl;
	//std::cout << "-------------2:写数字------------" << std::endl;
	//std::cout << "-----lhnows@qq.com-----" << std::endl;




     
	imshow("result", srcimage);
	//setMouseCallback("result", on_Mouse, 0);

	///////////
	// Open files
	string image_filename = "../t10k-images-idx3-ubyte";
  	string label_filename = "../t10k-labels-idx1-ubyte";
	std::ifstream image_file(image_filename, std::ios::in | std::ios::binary);
	std::ifstream label_file(label_filename, std::ios::in | std::ios::binary);
	CHECK(image_file) << "Unable to open file " << image_filename;
	CHECK(label_file) << "Unable to open file " << label_filename;
	// Read the magic and the meta data
	uint32_t magic; //魔数 2051-数据，2049-标记
	uint32_t num_items;
	uint32_t num_labels;
	uint32_t rows;
	uint32_t cols;

	//获取魔数，样本图像宽高标记，进行魔数验证
	image_file.read(reinterpret_cast<char *>(&magic), 4);
	magic = swap_endian(magic);
	CHECK_EQ(magic, 2051) << "Incorrect image file magic.";
	label_file.read(reinterpret_cast<char *>(&magic), 4);
	magic = swap_endian(magic);
	CHECK_EQ(magic, 2049) << "Incorrect label file magic.";
	image_file.read(reinterpret_cast<char *>(&num_items), 4);
	num_items = swap_endian(num_items);
	label_file.read(reinterpret_cast<char *>(&num_labels), 4);
	num_labels = swap_endian(num_labels);
	CHECK_EQ(num_items, num_labels);
	image_file.read(reinterpret_cast<char *>(&rows), 4);
	rows = swap_endian(rows);
	image_file.read(reinterpret_cast<char *>(&cols), 4);
	cols = swap_endian(cols);

	// Storing to db
	char label;
	char *pixels = new char[rows * cols];
	int count = 0;

	LOG(INFO) << "A total of " << num_items << " items.";
	LOG(INFO) << "Rows: " << rows << " Cols: " << cols;
	for (int item_id = 0; item_id < num_items; ++item_id)
	{
		char c;// = (char)waitKey(1);
		if (c == 27)
			break;
		//读取样本数据、标记
		image_file.read(pixels, rows * cols);
		label_file.read(&label, 1);
		Mat img(rows, cols, CV_8UC1, pixels);
		int  prediction = classifier.Classify(img);
		out<<(int)label<<std::endl;
		//std::cout << "prediction:" << prediction << std::endl;
		//imshow("img", img);
		if(item_id%100==0)
			std::cout<<item_id<<std::endl;
	}
	out.close();
	LOG(INFO) << "Processed " << count << " files.";
	delete[] pixels;

	waitKey();
	return 0;
}