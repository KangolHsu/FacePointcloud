# FacePointcloud
该工程是用来提取3D点云FPFH特征以及进行人脸点云采样



## 1.利用cmake 构建工程（只在windows环境下进行了测试，其他平台可能需要修改下代码）
	依赖项：PCL1.8.0 及 opencv3.2.0
## 2.FPFH文件夹是用来提取3D点云
## 3.SampleFacePointCloud文件夹是用来根据已经检测到68个人脸特征点的基础之上进行人脸区域点云提取和下采样
具体的输入参数配置在SampleFacePC.ini文件中
