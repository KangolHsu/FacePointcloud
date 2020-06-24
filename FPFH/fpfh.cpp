#include <fstream>
#include <iostream>

#include <pcl/features/fpfh.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/histogram_visualizer.h>

#include <pcl/visualization/pcl_plotter.h>

#include <pcl/common/transforms.h>
#include <pcl/range_image/range_image.h>
#include <pcl/visualization/range_image_visualizer.h> 

#include <pcl/features/normal_3d_omp.h>

#include <opencv2/opencv.hpp>

static const float fx = 382.544;
static const float fy = 382.544;
static const float cx = 317.78;
static const float cy = 236.709;
static const float scale = 1000; //d is m  scale = 1, d is mm scale = 1000


typedef pcl::PointXYZ PointXYZ;
typedef pcl::PointXYZRGB PointRGB;
typedef pcl::PointCloud<PointXYZ> PointCloudXYZ;
typedef pcl::PointCloud<PointRGB> PointCloudRGB;


void StringSplit(const std::string source,std::vector<std::string> &targets)
{
	//std::cout << line << std::endl;
	char* p = const_cast<char*>(source.c_str());
	const char* delim = " ";

	char* token = strtok(p, delim);
	targets.push_back(token);
	//std::cout << token << std::endl;
	while (token != NULL)
	{
		//std::cout << token << std::endl;
		targets.push_back(token);
		token = strtok(NULL, delim);
	}
	//std::cout << "split size :"<<targets.size() << std::endl;
}

void FilterDepth(const cv::Mat& depth,cv::Mat& depth_filter)
{
	cv::medianBlur(depth, depth_filter, 5);
	cv::Mat depth_float, depth_bilater;
	depth_filter.convertTo(depth_float, CV_32FC1);
	cv::bilateralFilter(depth_float, depth_bilater, 3, 3 * 2, 3 / 2);
	depth_bilater.convertTo(depth_filter, CV_16U);
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr ConvertDepthToCloudRGB(const cv::Mat &depth,const cv::Mat &color,const cv::Point2d nosetip_2d,cv::Point3f &nosetip_3d)
{
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
	assert(depth.cols() == color.cols());
	assert(depth.rows() == color.rows());
	for (int row = 0; row < depth.rows; row++)
	{
		for (int col = 0; col < depth.cols; col++)
		{
			ushort d = depth.ptr<ushort>(row)[col];
			if (d == 0)
				continue;
			pcl::PointXYZRGB point;
			point.z = d;
			point.x = d*(col - cx) / fx;
			point.y = d*(row - cy) / fy;
			point.b = color.ptr<uchar>(row)[col*3];
			point.g = color.ptr<uchar>(row)[col * 3 + 1];
			point.r = color.ptr<uchar>(row)[col * 3 + 2];

			if (col == nosetip_2d.x && row == nosetip_2d.y)
			{
				nosetip_3d.x = point.x;
				nosetip_3d.y = point.y;
				nosetip_3d.z = point.z;
			}
			cloud->push_back(point);
		}
	}
	cloud->height = 1;
	cloud->width = cloud->points.size();
	cloud->is_dense = false;
	return cloud;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr ConvertDepthToCloud(const cv::Mat &depth, const cv::Mat &color, const cv::Point2d nosetip_2d, cv::Point3f &nosetip_3d)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	assert(depth.cols() == color.cols());
	assert(depth.rows() == color.rows());
	for (int row = 0; row < depth.rows; row++)
	{
		for (int col = 0; col < depth.cols; col++)
		{
			ushort d = depth.ptr<ushort>(row)[col];
			if (d == 0)
				continue;
			pcl::PointXYZ point;
			point.z = d;
			point.x = d*(col - cx) / fx;
			point.y = d*(row - cy) / fy;

			if (col == nosetip_2d.x && row == nosetip_2d.y)
			{
				nosetip_3d.x = point.x;
				nosetip_3d.y = point.y;
				nosetip_3d.z = point.z;
			}
			cloud->push_back(point);
		}
	}
	cloud->height = 1;
	cloud->width = cloud->points.size();
	cloud->is_dense = false;
	return cloud;
}

void estimateNormals(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &cloud, float radius,pcl::PointCloud<pcl::Normal> &cloud_normals)
{
	std::cout << "Estimate PointCloud Normals start" << std::endl;
	// Create the normal estimation class, and pass the input dataset to it
	pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
	ne.setNumberOfThreads(12);  // 手动设置线程数，否则提示错误
	ne.setInputCloud(cloud);
	// Create an empty kdtree representation, and pass it to the normal estimation object.
	// Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
	ne.setSearchMethod(tree);
	// Use all neighbors in a sphere of radius 
	ne.setRadiusSearch(radius);
	// Compute the features
	ne.compute(cloud_normals);
	//cloud_normals->points.size () should have the same size as the input cloud->points.size ()*
	//std::cout << cloud_normals.size() << std::endl;
	std::cout << "Estimate PointCloud Normals done" << std::endl;
}

template <typename T>
void getFeatures(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, const pcl::PointCloud<pcl::Normal>::ConstPtr normals,float radius , pcl::PointCloud<T> &fpfhs)
{
	//创建FPFH估计对象fpfh，并把输入数据集cloud和法线normals传递给它。
	pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, T> fpfh;
	fpfh.setInputCloud(cloud);
	fpfh.setInputNormals(normals);
	std::cout << "....................." << std::endl;
	//如果点云是类型为PointNormal，则执行fpfh.setInputNormals (cloud);
	//创建一个空的kd树对象tree，并把它传递给FPFH估计对象。
	//基于已知的输入数据集，建立kdtree
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
	fpfh.setSearchMethod(tree);
	//输出数据集
	//pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs(new pcl::PointCloud<pcl::FPFHSignature33>());
	//使用所有半径在5厘米范围内的邻元素
	//注意：此处使用的半径必须要大于估计表面法线时使用的半径!!!
	fpfh.setRadiusSearch(radius);

	//计算获取特征向量
	fpfh.compute(fpfhs);
	// fpfhs->points.size ()应该和input cloud->points.size ()有相同的大小，即每个点有一个特征向量
	std::cout << "Estimate PointCloud FPFHSignature33 done" << std::endl;
}

int main()
{
	std::string image_list_file = "E:/TRAIN_DATA/A200m_output_labels.txt";
	std::ifstream file(image_list_file);
	std::string line;
	while (std::getline(file, line))
	{

		std::vector<std::string> splits;
		StringSplit(line, splits);
		std::cout << splits.size() << std::endl;

		//float nosetip_2d[2] = {0,0};
		cv::Point2d nosetip_2d;
		cv::Point3f nosetip_3d;
		nosetip_2d.x = atof(splits[4 + 55 * 2 - 1].c_str());
		nosetip_2d.y = atof(splits[4 + 55 * 2 ].c_str());

		std::cout <<"nosetip_2d : "<< nosetip_2d << std::endl;
		std::string image_color_path = splits[0] + "_color.jpg";
		std::string image_depth_path = splits[0] + "_depth.png";
		cv::Mat color = cv::imread(image_color_path);
		cv::Mat depth = cv::imread(image_depth_path, cv::IMREAD_ANYDEPTH);
		if (color.empty() || depth.empty()) { std::cout << " Load image failed .."; }
		cv::circle(color,nosetip_2d, 2, cv::Scalar(0, 0, 255));
		cv::imshow("color", color);
		cv::imshow("depth", depth);
		cv::waitKey(0);

		cv::Mat depth_filter;
		FilterDepth(depth, depth_filter);


		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
		cloud = ConvertDepthToCloud(depth_filter, color,nosetip_2d,nosetip_3d);
		std::cout << cloud->points.size() << std::endl;
		std::cout << "nosetip_3d : " << nosetip_3d << std::endl;

		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_RGB(new pcl::PointCloud<pcl::PointXYZRGB>);
		cloud_RGB = ConvertDepthToCloudRGB(depth_filter, color, nosetip_2d, nosetip_3d);

		pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
		estimateNormals(cloud, 5.0, *cloud_normals);
		std::cout <<"Cloud Normals Size :"<< cloud_normals->size() << std::endl;

		pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs(new pcl::PointCloud<pcl::FPFHSignature33>);
		getFeatures(cloud, cloud_normals, 5.1, *fpfhs);
		std::cout << "Cloud FPFH Size :" << fpfhs->points.size() << std::endl;

		


		//pcl::visualization::PCLHistogramVisualizer features_viewer;
		//features_viewer.setBackgroundColor(255, 0, 0);
		//features_viewer.addFeatureHistogram<pcl::FPFHSignature33>(*fpfhs, "fpfh", 1000);   //对下标为1000的元素可视化
		////view.spinOnce(10000);  //循环的次数
		//features_viewer.spin();  //无限循环
		
		pcl::visualization::PCLPlotter *plotter = new pcl::visualization::PCLPlotter("Histogram Plotter");
		plotter->setShowLegend(true);
		std::cout << pcl::getFieldsList<pcl::FPFHSignature33>(*fpfhs);
		plotter->addFeatureHistogram<pcl::FPFHSignature33>(*fpfhs, pcl::getFieldsList<pcl::FPFHSignature33>(*fpfhs),30000);
		plotter->setWindowSize(800, 600);
		plotter->spinOnce(30000);
		plotter->clearPlots();



		//Transofrm PointCloud
		//方法一 #1: 使用 Matrix4f 这个是“手工方法”，可以完美地理解，但容易出错!*/
		Eigen::Matrix4f transform_1 = Eigen::Matrix4f::Identity();// 定义一个旋转矩阵 (见 https://en.wikipedia.org/wiki/Rotation_matrix)
		float theta = M_PI / 4; // 弧度角
		transform_1(0, 0) = cos(theta);
		transform_1(0, 1) = -sin(theta);
		transform_1(1, 0) = sin(theta);
		transform_1(1, 1) = cos(theta);

		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_trasnformed(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::transformPointCloud(*cloud, *cloud_trasnformed, transform_1);

		//pcl::PointCloud<pcl::PointXYZ> cloud_zbuffer;
		//cloud_zbuffer = pcl::RangeImage::doZBuffer(const_cast<pcl::PointCloud<pcl::PointXYZ>&>(*cloud_trasnformed), 0.5,0,0,0,100,100);
		
		

		//// Create Range image from PointCloud
		//float angularResolution = (float)(1.0f * (M_PI / 180.0f));  //   1.0 degree in radians
		//float maxAngleWidth = (float)(360.0f * (M_PI / 180.0f));  // 360.0 degree in radians
		//float maxAngleHeight = (float)(180.0f * (M_PI / 180.0f));  // 180.0 degree in radians
		//Eigen::Affine3f sensorPose = (Eigen::Affine3f)Eigen::Translation3f(0.0f, 0.0f, 0.0f);
		//float noiseLevel = 0.50;
		//float minRange = 0.0f;
		//int borderSize = 1;
		//pcl::RangeImage::CoordinateFrame coordinate_frame = pcl::RangeImage::CAMERA_FRAME;
		//pcl::RangeImage rangeImage;
		//rangeImage.createFromPointCloud(*cloud_trasnformed, angularResolution, maxAngleWidth, maxAngleHeight,
		//	sensorPose, coordinate_frame, noiseLevel, minRange, borderSize);

		//pcl::visualization::RangeImageVisualizer range_image_widget("Range image");
		//range_image_widget.showRangeImage(rangeImage);

		//pcl::visualization::PCLVisualizer viewer_pcl("3D Viewer");
		//while (!viewer_pcl.wasStopped())
		//{
		//	range_image_widget.spinOnce();
		//	viewer_pcl.spinOnce();
		//	pcl_sleep(0.01);
		//}




		//for (auto p : *cloud_normals)
		//	std::cout << p.normal_x << " " << p.normal_y << " " << p.normal_z << std::endl;
		//for (auto p : *fpfhs)
		//{
		//	std::cout << p.descriptorSize() << " *************** "<< std::endl;
		//	for(int i=0;i<p.descriptorSize();i++)
		//		std::cout << p.histogram[i] << std::endl;
		//}
		// 可视化
		boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Normals"));
		viewer->addPointCloud<pcl::PointXYZ>(cloud, "cloud");//显示点云

		viewer->addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(cloud, cloud_normals,5, 5, "normals");//显示法向量
		while (!viewer->wasStopped())
		{
			viewer->spinOnce(100);
			boost::this_thread::sleep(boost::posix_time::microseconds(100000));
		}

	}
	system("pause");
}

