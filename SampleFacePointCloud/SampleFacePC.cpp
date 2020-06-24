#include <fstream>
#include <iostream>
#include <algorithm>

#include <io.h>
#include <direct.h>

#include <pcl/features/fpfh.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/histogram_visualizer.h>

#include <pcl/surface/mls.h>
#include <pcl/filters/random_sample.h>

#include <pcl/common/transforms.h>
#include <pcl/range_image/range_image.h>
#include <pcl/visualization/range_image_visualizer.h> 

#include <pcl/features/normal_3d_omp.h>

#include <opencv2/opencv.hpp>


#include "ini.h"


struct intrisic//相机内参
{
	float fx;
	float fy;
	float cx;
	float cy;
	float scale;
};

typedef pcl::PointXYZ PointXYZ;
typedef pcl::PointXYZRGB PointRGB;
typedef pcl::PointCloud<PointXYZ> PointCloudXYZ;
typedef pcl::PointCloud<PointRGB> PointCloudRGB;


void StringSplit(const std::string source, const char* delim,std::vector<std::string> &targets)
{
	//std::cout << line << std::endl;
	char* p = const_cast<char*>(source.c_str());
	//const char* delim = " ";

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

pcl::PointCloud<pcl::PointXYZRGB>::Ptr ConvertDepthToCloudRGB(const cv::Mat &depth,const cv::Mat &color,const cv::Point2d nosetip_2d,cv::Point3f &nosetip_3d, const intrisic intr_param)
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
			point.x = d*(col - intr_param.cx) / intr_param.fx;
			point.y = d*(row - intr_param.cy) / intr_param.fy;
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

pcl::PointCloud<pcl::PointXYZ>::Ptr ConvertDepthToCloud(const cv::Mat &depth, const cv::Mat &color, const cv::Point2d nosetip_2d, cv::Point3f &nosetip_3d,const intrisic intr_param)
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
			point.x = d*(col - intr_param.cx) / intr_param.fx;
			point.y = d*(row - intr_param.cy) / intr_param.fy;

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


void SampleCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, const float radius,pcl::PointCloud<pcl::PointNormal> &smoothedCloud)//表面重建
{
	// Smoothing object (we choose what point types we want as input and output).
	pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointNormal> filter;
	filter.setInputCloud(cloud);
	// Use all neighbors in a radius 
	filter.setSearchRadius(radius);
	// If true, the surface and normal are approximated using a polynomial estimation
	// (if false, only a tangent one).
	filter.setPolynomialFit(true);
	// We can tell the algorithm to also compute smoothed normals (optional).
	filter.setComputeNormals(true);
	// kd-tree object for performing searches.
	pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree;
	filter.setSearchMethod(kdtree);
	filter.process(smoothedCloud);
}

void UpSampleCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, const float radius,const float step_size, pcl::PointCloud<pcl::PointXYZ> &filteredCloud)//表面重建
{
	pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointXYZ> filter;
	filter.setInputCloud(cloud);
	//建立搜索对象
	pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree;
	filter.setSearchMethod(kdtree);
	//设置搜索邻域的半径为3cm
	filter.setSearchRadius(radius);
	// Upsampling 采样的方法有 DISTINCT_CLOUD, RANDOM_UNIFORM_DENSITY
	filter.setUpsamplingMethod(pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointXYZ>::SAMPLE_LOCAL_PLANE);
	// 采样的半径是
	filter.setUpsamplingRadius(radius);
	// 采样步数的大小
	filter.setUpsamplingStepSize(step_size);

	filter.process(filteredCloud);
}

void Write_PointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,std::string file_name)
{
	std::ofstream file;
	file.open(file_name);
	for (auto p : cloud->points)
		file << p.x << " " << p.y << " " << p.z << "\n";
	file.close();
}

//Prepare Data type ： x,y,z,nx,ny,nz   
int main()
{
	ini_t *config = ini_load("../../SampleFacePointCloud/SampleFacePC.ini");
	if (config == NULL)
		std::cout << "load config file failed" << std::endl;
	std::string  image_list_file = ini_get(config, "Input", "image_list_file");
	std::string  input_dir = ini_get(config, "Input", "input_dir");
	std::string  output_dir = ini_get(config, "Output", "output_dir");

	intrisic intr_param;
	intr_param.fx = atof(ini_get(config, "Input", "fx"));
	intr_param.fy = atof(ini_get(config, "Input", "fy"));
	intr_param.cx = atof(ini_get(config, "Input", "cx"));
	intr_param.cy = atof(ini_get(config, "Input", "cy"));
	intr_param.scale = atof(ini_get(config, "Input", "scale"));

	//std::string image_list_file = "E:/TRAIN_DATA/A200m_output_labels.txt";
	//std::string image_list_file = "E:/TRAIN_DATA/Realsense/RS300_output_labels_all.txt";
	std::ifstream file(image_list_file);
	std::string line;
	while (std::getline(file, line))
	{
		std::vector<std::string> splits;
		StringSplit(line, " ", splits);
		std::cout << splits.size() << std::endl;

		std::vector<std::string> directorys;
		StringSplit(splits[0], "/", directorys);
		std::cout << directorys.size() << std::endl;
		std::string current_dir = output_dir;
		if (_access(current_dir.c_str(), 0) == -1)
			_mkdir(current_dir.c_str());
		for (int i = 1; i < directorys.size()-1; i++)
		{
			current_dir = current_dir + "/" + directorys[i];
			if (_access(current_dir.c_str(), 0) == -1)
			{
				std::cout << "mkdir : " << current_dir << std::endl;
				_mkdir(current_dir.c_str());
			}
		}
		
		cv::Point2d nosetip_2d;
		cv::Point3f nosetip_3d;
		nosetip_2d.x = atof(splits[4 + 55 * 2 - 1].c_str());
		nosetip_2d.y = atof(splits[4 + 55 * 2 ].c_str());

		std::cout <<"nosetip_2d : "<< nosetip_2d << std::endl;
		//std::string image_color_path = input_dir + "/" + splits[0] + "_color.jpg";
		//std::string image_depth_path = input_dir + "/" + splits[0] + "_depth.png";

		std::string image_color_path = input_dir + "/" + splits[0] + "_rgb.jpg";
		//std::string image_depth_path = input_dir + "/" + splits[0] + ".dep.png";//BJ
		std::string image_depth_path = input_dir + "/" + splits[0] + "_dep.png";  //NJ

		cv::Mat color = cv::imread(image_color_path);
		cv::Mat depth = cv::imread(image_depth_path, cv::IMREAD_ANYDEPTH);
		if (color.empty() || depth.empty()) { std::cout << " Load image failed .."; continue; }
		cv::circle(color,nosetip_2d, 3, cv::Scalar(0, 0, 255));
#if 0  //crop face by landmark
		std::vector<int> landamrk106;
		for (int i = 5; i < splits.size(); i++)
		{
			landamrk106.push_back(atoi(splits[i].c_str()));
			std::cout << atoi(splits[i].c_str()) << std::endl;
		}
		std::cout << landamrk106.size() << std::endl;
		std::vector<int> x_cords, y_cords;
		for (int i = 0; i < landamrk106.size() / 2; i++)
		{
			x_cords.push_back(landamrk106[2 * i]);
			y_cords.push_back(landamrk106[2 * i + 1]);
			cv::circle(color, cv::Point2i(landamrk106[2 * i], landamrk106[2 * i + 1]), 2, cv::Scalar(255, 0, 0));
		}
		auto left_right = std::minmax_element(x_cords.begin(), x_cords.end());
		auto top_bottom = std::minmax_element(y_cords.begin(), y_cords.end());
		auto left = left_right.first;
		auto right = left_right.first;
		auto top = top_bottom.second;
		auto bottom = top_bottom.second;
		std::cout << *left << " " << *top << " " << *right << " " << *bottom << std::endl;
#endif // 0
		int show_rgb_depth= atoi(ini_get(config, "Input", "show_rgb_depth"));
		if (1==show_rgb_depth)
		{
			cv::imshow("color", color);
			cv::imshow("depth", depth);
			cv::waitKey(0);
		}

		cv::Mat depth_filter;
		FilterDepth(depth, depth_filter);


		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
		cloud = ConvertDepthToCloud(depth_filter, color,nosetip_2d,nosetip_3d, intr_param);
		std::cout << cloud->points.size() << std::endl;
		std::cout << "nosetip_3d : " << nosetip_3d << std::endl;

		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_crop_face(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_crop_final(new pcl::PointCloud<pcl::PointXYZ>);
		for(auto p:cloud->points)
		{
			float distance = std::sqrt(std::pow((p.x - nosetip_3d.x),2) 
									 + std::pow((p.y - nosetip_3d.y),2) 
									 + std::pow((p.z - nosetip_3d.z),2));
			//std::cout << distance << std::endl;
			if (distance < 1000)// a200:100  rs_beijing:100    rs_nanjing:1000
				cloud_crop_face->push_back(p);
		}
		std::cout << "Crop face Point Num :"<<cloud_crop_face->size()<<std::endl;
		if (cloud_crop_face->size() == 0)
			continue;




		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_RGB(new pcl::PointCloud<pcl::PointXYZRGB>);
		cloud_RGB = ConvertDepthToCloudRGB(depth_filter, color, nosetip_2d, nosetip_3d,intr_param);

		float compute_normal_radius= atof(ini_get(config, "Input", "compute_normal_radius"));
		pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
		estimateNormals(cloud, compute_normal_radius, *cloud_normals);
		std::cout <<"Cloud Normals Size :"<< cloud_normals->size() << std::endl;

		float sample_cloud_radius = atof(ini_get(config, "Input", "sample_cloud_radius"));
		float step_size = atof(ini_get(config, "Input", "step_size"));
		//pcl::PointCloud<pcl::PointNormal>::Ptr smoothedCloud(new pcl::PointCloud<pcl::PointNormal>);
		//SampleCloud(cloud_crop_face, sample_cloud_radius, *smoothedCloud);
		pcl::PointCloud<pcl::PointXYZ>::Ptr smoothedCloud(new pcl::PointCloud<pcl::PointXYZ>);
		UpSampleCloud(cloud_crop_face, sample_cloud_radius, step_size, *smoothedCloud);
		std::cout << "smoothedCloud Size :" << smoothedCloud->size() << std::endl;


		int output_cloud_num = atof(ini_get(config, "Input", "output_cloud_num"));
		//if (smoothedCloud->size() < output_cloud_num)
		//	continue;// 获取的点云过少时，抛弃该人脸cloud
		pcl::RandomSample<pcl::PointXYZ> rs;
		rs.setInputCloud(smoothedCloud);
		//设置输出点的数量   
		rs.setSample(output_cloud_num);
		//下采样并输出到cloud_sample
		rs.filter(*cloud_crop_final);
		std::cout << "cloud_crop_final Size :" << cloud_crop_final->size() << std::endl;



#if 0
		// 创建存储点云重心的对象  点云缺失会导致质心偏移？影响点云识别？
		Eigen::Vector4f centroid;
		pcl::compute3DCentroid(*cloud_crop_final, centroid);
		std::cout << "Centroid : " << centroid[0] << " " << centroid[1] << " " << centroid[2] << std::endl;
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_mean(new pcl::PointCloud<pcl::PointXYZ>);
		float furthest_distance = 0;
		for (auto p : cloud_crop_final->points)
		{
			float x = p.x - centroid[0];
			float y = p.y - centroid[1];
			float z = p.z - centroid[2];
			float distance = std::sqrt(x*x + y*y + z*z);
			cloud_mean->push_back(pcl::PointXYZ(x, y, z));
			if (distance > furthest_distance)
				furthest_distance = distance;
		}
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_normlized(new pcl::PointCloud<pcl::PointXYZ>);
		for (auto p : cloud_mean->points)
		{
			float x = p.x / furthest_distance;
			float y = p.y / furthest_distance;
			float z = p.z / furthest_distance;
			cloud_normlized->push_back(pcl::PointXYZ(x, y, z));
			//std::cout << p.x << " " << p.y << " " << p.z << std::endl;
		}
#endif
		int write_flag = atoi(ini_get(config, "Input", "save_txt"));
		if (write_flag == 1)
		{
			std::string save_file = output_dir + "/" + splits[0] + ".txt";
			std::cout << save_file << std::endl;
			Write_PointCloud(cloud_crop_final, save_file);
		}
		int show_cloud_final = atoi(ini_get(config, "Input", "show_cloud_final"));
		if (1 == show_cloud_final)
		{// 可视化
			boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Normals"));
			viewer->addPointCloud<pcl::PointXYZ>(cloud, "cloud");//显示点云
			//viewer->addPointCloudNormals<pcl::PointNormal>(smoothedCloud,"smoothedCloud");
			//viewer->addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(cloud_crop_face, cloud_normals,5, 5, "normals");//显示法向量
			while (!viewer->wasStopped())
			{
				viewer->spinOnce(100);
				boost::this_thread::sleep(boost::posix_time::microseconds(100000));
			}
		}

	}
	system("pause");
}

