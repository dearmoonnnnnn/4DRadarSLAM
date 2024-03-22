// SPDX-License-Identifier: BSD-2-Clause

#include <string>
#include <fstream>
#include <functional>

#include <ros/ros.h>
#include <ros/time.h>
#include <pcl_ros/transforms.h>
#include <pcl_ros/point_cloud.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>

#include <std_msgs/String.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointCloud.h>
#include <geometry_msgs/TwistWithCovarianceStamped.h>
#include <nav_msgs/Odometry.h>

#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/fast_bilateral.h>
#include <pcl/filters/filter.h>

#include <opencv2/imgproc/imgproc.hpp>

#include <Eigen/Dense>

#include "radar_ego_velocity_estimator.h"
#include "rio_utils/radar_point_cloud.h"
#include "utility_radar.h"

using namespace std;

//定义命名空间，里面只有三个类：PreprocessingNodelet、radar_graph_slam_nodelet、scan_matching_odometry_nodelet
namespace radar_graph_slam {

//定义预处理节点
class PreprocessingNodelet : public nodelet::Nodelet, public ParamServer {
public: 
  // typedef pcl::PointXYZI PointT;
  typedef pcl::PointXYZI PointT;

  PreprocessingNodelet() {}
  virtual ~PreprocessingNodelet() {}

  virtual void onInit() {
    nh = getNodeHandle();                         // 获取节点句柄
    private_nh = getPrivateNodeHandle();          // 获取私有节点句柄

    initializeTransformation();                   // 调用初始化变换的函数
    initializeParams();                           // 调用初始化参数的函数

    // 订阅点云数据、IMU数据和命令数据
    points_sub = nh.subscribe(pointCloudTopic, 64, &PreprocessingNodelet::cloud_callback, this);
    imu_sub = nh.subscribe(imuTopic, 1, &PreprocessingNodelet::imu_callback, this);
    command_sub = nh.subscribe("/command", 10, &PreprocessingNodelet::command_callback, this);

    // 发布器，发布过滤后的点云、带颜色的点云、IMU数据和Aft-mapped到初始位姿的里程计数据
    points_pub = nh.advertise<sensor_msgs::PointCloud2>("/filtered_points", 32);
    colored_pub = nh.advertise<sensor_msgs::PointCloud2>("/colored_points", 32);
    imu_pub = nh.advertise<sensor_msgs::Imu>("/imu", 32);
    gt_pub = nh.advertise<nav_msgs::Odometry>("/aftmapped_to_init", 16);
  
    // 从ROS参数服务器中获取名为 "topic_twist"等字符串参数的值，如果该参数不存在，则使用默认值 "/eagle_data/twist"等值
    std::string topic_twist = private_nh.param<std::string>("topic_twist", "/eagle_data/twist");  
    std::string topic_inlier_pc2 = private_nh.param<std::string>("topic_inlier_pc2", "/eagle_data/inlier_pc2");
    std::string topic_outlier_pc2 = private_nh.param<std::string>("topic_outlier_pc2", "/eagle_data/outlier_pc2");
    pub_twist = nh.advertise<geometry_msgs::TwistWithCovarianceStamped>(topic_twist, 5);      // Twist
    pub_inlier_pc2 = nh.advertise<sensor_msgs::PointCloud2>(topic_inlier_pc2, 5);             // 内点点云
    pub_outlier_pc2 = nh.advertise<sensor_msgs::PointCloud2>(topic_outlier_pc2, 5);           // 外点点云
    pc2_raw_pub = nh.advertise<sensor_msgs::PointCloud2>("/eagle_data/pc2_raw",1);            // 原始点云
    enable_dynamic_object_removal = private_nh.param<bool>("enable_dynamic_object_removal", false); //是否启动动态物体去除
    power_threshold = private_nh.param<float>("power_threshold", 0);                                //功率阈值
  }

private:
  void initializeTransformation(){

    /*************** 作者原数据  ****************/
    // livox_to_RGB = (cv::Mat_<double>(4,4) << 
    // -0.006878330000, -0.999969000000, 0.003857230000, 0.029164500000,  
    // -7.737180000000E-05, -0.003856790000, -0.999993000000, 0.045695200000,
    //  0.999976000000, -0.006878580000, -5.084110000000E-05, -0.19018000000,
    // 0,  0,  0,  1);
    // RGB_to_livox =livox_to_RGB.inv();
    // Thermal_to_RGB = (cv::Mat_<double>(4,4) <<
    // 0.9999526089706319, 0.008963747151337641, -0.003798822163962599, 0.18106962419014,  
    // -0.008945181135788245, 0.9999481006917174, 0.004876439015823288, -0.04546324090016857,
    // 0.00384233617405678, -0.004842226763999368, 0.999980894463835, 0.08046453079998771,
    // 0,0,0,1);
    // Radar_to_Thermal = (cv::Mat_<double>(4,4) <<
    // 0.999665,    0.00925436,  -0.0241851,  -0.0248342,
    // -0.00826999, 0.999146,    0.0404891,   0.0958317,
    // 0.0245392,   -0.0402755,  0.998887,    0.0268037,
    // 0,  0,  0,  1);
    // Change_Radarframe=(cv::Mat_<double>(4,4) <<
    // 0,-1,0,0,
    // 0,0,-1,0,
    // 1,0,0,0,
    // 0,0,0,1);
    // Radar_to_livox=RGB_to_livox*Thermal_to_RGB*Radar_to_Thermal*Change_Radarframe;
    // std::cout << "Radar_to_livox = "<< std::endl << " "  << Radar_to_livox << std::endl << std::endl;

    /************** 自己采集的数据 ***************/
    Radar_to_livox=(cv::Mat_<double>(4,4) <<
    0.9987420694356727, -0.02154593184251807, 0.04527979957349116, 0.1060016945940701,
    0.02057017793626469, 0.9995486686923027, 0.02190458926318335, -0.1298868374575176,
    -0.04573127973037173, -0.02094561026271845, 0.9987339684250071, -0.154543126256660535,
    0, 0, 0, 1);

    }
  
  void initializeParams() {
    // 降采样方法、分辨率
    std::string downsample_method = private_nh.param<std::string>("downsample_method", "VOXELGRID");
    double downsample_resolution = private_nh.param<double>("downsample_resolution", 0.1);

    if(downsample_method == "VOXELGRID") {
      std::cout << "downsample: VOXELGRID " << downsample_resolution << std::endl;
      auto voxelgrid = new pcl::VoxelGrid<PointT>();                                                  // 创建pcl::VoxelGrid对象，用于进行体素网格降采样
      voxelgrid->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);    // 设置体素网格的叶子大小，即降采样的分辨率
      downsample_filter.reset(voxelgrid);                                                             // 将 downsample_filter 重置为新创建的 pcl::VoxelGrid 对象
    } else if(downsample_method == "APPROX_VOXELGRID") {
      std::cout << "downsample: APPROX_VOXELGRID " << downsample_resolution << std::endl;
      pcl::ApproximateVoxelGrid<PointT>::Ptr approx_voxelgrid(new pcl::ApproximateVoxelGrid<PointT>()); // 用于进行近似体素网格降采样
      approx_voxelgrid->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
      downsample_filter = approx_voxelgrid;                                                             // 将downsample_filer赋值为approx_voxelgrid
    } else {
      if(downsample_method != "NONE") {
        std::cerr << "warning: unknown downsampling type (" << downsample_method << ")" << std::endl;
        std::cerr << "       : use passthrough filter" << std::endl;
      }
      std::cout << "downsample: NONE" << std::endl;
    }

    // 去除离群点的方法，可选值为"STATISTICAL"、"RADIUS",即"统计方法"、"半径方法"
    std::string outlier_removal_method = private_nh.param<std::string>("outlier_removal_method", "STATISTICAL");
    if(outlier_removal_method == "STATISTICAL") {
      int mean_k = private_nh.param<int>("statistical_mean_k", 20);                   // 计算邻域均值的点数
      double stddev_mul_thresh = private_nh.param<double>("statistical_stddev", 1.0); // 标准差的倍数阈值
      std::cout << "outlier_removal: STATISTICAL " << mean_k << " - " << stddev_mul_thresh << std::endl;
      
      //创建统计离群点移除对象
      pcl::StatisticalOutlierRemoval<PointT>::Ptr sor(new pcl::StatisticalOutlierRemoval<PointT>());
      sor->setMeanK(mean_k);
      sor->setStddevMulThresh(stddev_mul_thresh);
      outlier_removal_filter = sor;   // 将智能指针指向统计离群点移除对象
    } else if(outlier_removal_method == "RADIUS") {
      double radius = private_nh.param<double>("radius_radius", 0.8);                   // 半径
      int min_neighbors = private_nh.param<int>("radius_min_neighbors", 2);             // 领域最小点数
      std::cout << "outlier_removal: RADIUS " << radius << " - " << min_neighbors << std::endl;
      
      // 创建半径离群点移除对象
      pcl::RadiusOutlierRemoval<PointT>::Ptr rad(new pcl::RadiusOutlierRemoval<PointT>());
      rad->setRadiusSearch(radius);
      rad->setMinNeighborsInRadius(min_neighbors);
      outlier_removal_filter = rad;   // 将智能指针指向半径离群点移除对象
    } 
    // else if (outlier_removal_method == "BILATERAL")
    // {
    //   double sigma_s = private_nh.param<double>("bilateral_sigma_s", 5.0);
    //   double sigma_r = private_nh.param<double>("bilateral_sigma_r", 0.03);
    //   std::cout << "outlier_removal: BILATERAL " << sigma_s << " - " << sigma_r << std::endl;

    //   pcl::FastBilateralFilter<PointT>::Ptr fbf(new pcl::FastBilateralFilter<PointT>());
    //   fbf->setSigmaS (sigma_s);
    //   fbf->setSigmaR (sigma_r);
    //   outlier_removal_filter = fbf;
    // }
     else {
      std::cout << "outlier_removal: NONE" << std::endl;
    }

    // 距离过滤相关参数
    use_distance_filter = private_nh.param<bool>("use_distance_filter", true);    // 表示是否使用距离过滤
    distance_near_thresh = private_nh.param<double>("distance_near_thresh", 1.0); // 距离过滤的近的阈值
    distance_far_thresh = private_nh.param<double>("distance_far_thresh", 100.0); // 距离过滤的远的阈值
    // 点云在z轴上的高度范围
    z_low_thresh = private_nh.param<double>("z_low_thresh", -5.0);                
    z_high_thresh = private_nh.param<double>("z_high_thresh", 20.0);

    // 从参数服务器获取ground truth文件路径和是否进行tf发布的参数
    std::string file_name = private_nh.param<std::string>("gt_file_location", "");
    publish_tf = private_nh.param<bool>("publish_tf", false);

    ifstream file_in(file_name);
    if (!file_in.is_open()) {
        cout << "Can not open this gt file" << endl;
    }
    else{
      //将文件中的每一行存储到vector中
      std::vector<std::string> vectorLines;
      std::string line;
      while (getline(file_in, line)) {
          vectorLines.push_back(line);
      }
      
      for (size_t i = 1; i < vectorLines.size(); i++) {
          std::string line_ = vectorLines.at(i);                    // 获取vector中的每一行
          double stamp,tx,ty,tz,qx,qy,qz,qw;                        // 定义相关变量
          stringstream data(line_);                                 // 使用字符串流读取当前行的数据
          data >> stamp >> tx >> ty >> tz >> qx >> qy >> qz >> qw;  // 解析当前行的数据
          nav_msgs::Odometry odom_msg;                              // 创建Odometry消息对象
          // odom_msg的头部信息
          odom_msg.header.frame_id = mapFrame;                      // 该消息所在的坐标系的名称            
          odom_msg.child_frame_id = baselinkFrame;                  // 表示一个相对于frame_id的子坐标系。如激光雷达数据的child_frame_id可能是base_laser,表示激光雷达相对于机器人底盘坐标系的位置。
          odom_msg.header.stamp = ros::Time().fromSec(stamp);
          // odom_msg的姿态信息
          odom_msg.pose.pose.orientation.w = qw;
          odom_msg.pose.pose.orientation.x = qx;
          odom_msg.pose.pose.orientation.y = qy;
          odom_msg.pose.pose.orientation.z = qz;
          // odom_msg的位置信息
          odom_msg.pose.pose.position.x = tx;
          odom_msg.pose.pose.position.y = ty;
          odom_msg.pose.pose.position.z = tz;
          std::lock_guard<std::mutex> lock(odom_queue_mutex);
          odom_msgs.push_back(odom_msg);
      }
    }
    file_in.close();
  }

  // 处理Imu数据并发布，同时发布和Imu数据时间戳对应的Ground Truth数据(Odom消息)
  void imu_callback(const sensor_msgs::ImuConstPtr& imu_msg) {
    sensor_msgs::Imu imu_data;
    // imu_data的头部信息
    imu_data.header.stamp = imu_msg->header.stamp;
    imu_data.header.seq = imu_msg->header.seq;
    imu_data.header.frame_id = "imu_frame";
    Eigen::Quaterniond q_ahrs(imu_msg->orientation.w,   // 四元数的姿态信息
                              imu_msg->orientation.x,
                              imu_msg->orientation.y,
                              imu_msg->orientation.z);
    Eigen::Quaterniond q_r =                            // 绕Z轴旋转180度，然后绕Y轴旋转180度，最后绕X轴旋转0度的旋转
        Eigen::AngleAxisd( M_PI, Eigen::Vector3d::UnitZ()) * 
        Eigen::AngleAxisd( M_PI, Eigen::Vector3d::UnitY()) * 
        Eigen::AngleAxisd( 0.00000, Eigen::Vector3d::UnitX());
    Eigen::Quaterniond q_rr =                           // 绕Z轴旋转0度，然后绕Y轴旋转0度，最后绕X轴旋转180度的旋转
        Eigen::AngleAxisd( 0.00000, Eigen::Vector3d::UnitZ()) * 
        Eigen::AngleAxisd( 0.00000, Eigen::Vector3d::UnitY()) * 
        Eigen::AngleAxisd( M_PI, Eigen::Vector3d::UnitX());
    Eigen::Quaterniond q_out =  q_r * q_ahrs * q_rr;    // 校正原始的IMU数据的姿态信息
    imu_data.orientation.w = q_out.w();
    imu_data.orientation.x = q_out.x();
    imu_data.orientation.y = q_out.y();
    imu_data.orientation.z = q_out.z();
    imu_data.angular_velocity.x = imu_msg->angular_velocity.x;
    imu_data.angular_velocity.y = -imu_msg->angular_velocity.y;
    imu_data.angular_velocity.z = -imu_msg->angular_velocity.z;
    imu_data.linear_acceleration.x = imu_msg->linear_acceleration.x;
    imu_data.linear_acceleration.y = -imu_msg->linear_acceleration.y;
    imu_data.linear_acceleration.z = -imu_msg->linear_acceleration.z;
    imu_pub.publish(imu_data);                          // 发布新的Imu消息
    // imu_queue.push_back(imu_msg);


    double time_now = imu_msg->header.stamp.toSec();    // 当前Imu数据的时间戳
    bool updated = false;
    if (odom_msgs.size() != 0) {
      // 循环处理Odometry队列
      // 如果队列的第一个元素的时间戳小于当前时间戳减去偏移，将队列的第一个元素弹出，表示已经处理了这个时间戳对应的 Odometry 消息。
      while (odom_msgs.front().header.stamp.toSec() + 0.001 < time_now) {
        // 锁定互斥锁，弹出队列的第一个元素
        std::lock_guard<std::mutex> lock(odom_queue_mutex);
        odom_msgs.pop_front();
        updated = true;

        // 如果队列为空，退出循环
        if (odom_msgs.size() == 0)
          break;
      }
    }

    if (updated == true && odom_msgs.size() > 0){       // 队列更新且不为空
      if (publish_tf) {                                 // 如果启用了发布TF，则创建并发布TransformStamped消息
        geometry_msgs::TransformStamped tf_msg;
        tf_msg.child_frame_id = baselinkFrame;
        tf_msg.header.frame_id = mapFrame;
        tf_msg.header.stamp = odom_msgs.front().header.stamp;
        // tf_msg.header.stamp = ros::Time().now();
        tf_msg.transform.rotation = odom_msgs.front().pose.pose.orientation;
        tf_msg.transform.translation.x = odom_msgs.front().pose.pose.position.x;
        tf_msg.transform.translation.y = odom_msgs.front().pose.pose.position.y;
        tf_msg.transform.translation.z = odom_msgs.front().pose.pose.position.z;
        tf_broadcaster.sendTransform(tf_msg);           // 使用TF Broadcaster发布Transform
      }
      
      gt_pub.publish(odom_msgs.front());
    }
  }

  // 回调函数，处理sensor_msgs::PointCloud消息类型的数据
  void cloud_callback(const sensor_msgs::PointCloud::ConstPtr&  eagle_msg) { // const pcl::PointCloud<PointT>& src_cloud_r

    // std::cout << "------------------------------cloud_callback--------------------------------" << std::endl;
    // std::cout << "channels[0].name: " << eagle_msg-> channels[0].name << std::endl;
    // std::cout << "channels[1].name: " << eagle_msg-> channels[1].name << std::endl;
    
    // 定义两种不同的点云类型和它们的指针
    RadarPointCloudType radarpoint_raw;            // 原始点云，带有x、y、z、强度和多普勒速度信息
    PointT radarpoint_xyzi;                        // 原始点云，带有x、y、z、强度信息
    pcl::PointCloud<RadarPointCloudType>::Ptr radarcloud_raw( new pcl::PointCloud<RadarPointCloudType> );
    pcl::PointCloud<PointT>::Ptr radarcloud_xyzi( new pcl::PointCloud<PointT> );

    radarcloud_xyzi->header.frame_id = baselinkFrame;
    radarcloud_xyzi->header.seq = eagle_msg->header.seq;        // 序列号，用于标识消息的顺序
    radarcloud_xyzi->header.stamp = eagle_msg->header.stamp.toSec() * 1e6;

    // 遍历每一个点，i表示点
    for(int i = 0; i < eagle_msg->points.size(); i++)
    {
        // // 官方数据
        // // channels[0].value[i]表示点i的多普勒速度，channels[2].value[i]表示点i的信号强度
        // // cout << i << ":    " <<eagle_msg->points[i].x<<endl;
        // if(eagle_msg->channels[2].values[i] > power_threshold) //"Power"
        // {
        //     // 检查点的坐标是否无效(NaN或无穷大)
        //     if (eagle_msg->points[i].x == NAN || eagle_msg->points[i].y == NAN || eagle_msg->points[i].z == NAN) continue;
        //     if (eagle_msg->points[i].x == INFINITY || eagle_msg->points[i].y == INFINITY || eagle_msg->points[i].z == INFINITY) continue;
           
        //     // 将点从雷达坐标系转换到Livox坐标系
        //     cv::Mat ptMat, dstMat;
        //     ptMat = (cv::Mat_<double>(4, 1) << eagle_msg->points[i].x, eagle_msg->points[i].y, eagle_msg->points[i].z, 1);    
        //     // Perform matrix multiplication and save as Mat_ for easy element access
        //     dstMat= Radar_to_livox * ptMat;

        //     // 对点进行赋值
        //     radarpoint_raw.x = dstMat.at<double>(0,0);
        //     radarpoint_raw.y = dstMat.at<double>(1,0);
        //     radarpoint_raw.z = dstMat.at<double>(2,0);
        //     radarpoint_raw.intensity = eagle_msg->channels[2].values[i];
        //     radarpoint_raw.doppler = eagle_msg->channels[0].values[i];
        //     radarpoint_xyzi.x = dstMat.at<double>(0,0);
        //     radarpoint_xyzi.y = dstMat.at<double>(1,0);
        //     radarpoint_xyzi.z = dstMat.at<double>(2,0);
        //     radarpoint_xyzi.intensity = eagle_msg->channels[2].values[i];

        //     // 将点添加到点云中
        //     radarcloud_raw->points.push_back(radarpoint_raw);
        //     radarcloud_xyzi->points.push_back(radarpoint_xyzi);
        // }

        // 处理自己采集的点云数据
        // channels[0].value[i]表示点i的多普勒速度，channels[1].value[i]表示点i的信号强度
        // cout << i << ":    " <<eagle_msg->points[i].x<<endl;
        if(eagle_msg->channels[1].values[i] > power_threshold) //"Power"
        {
            // 检查点的坐标是否无效(NaN或无穷大)
            if (eagle_msg->points[i].x == NAN || eagle_msg->points[i].y == NAN || eagle_msg->points[i].z == NAN) continue;
            if (eagle_msg->points[i].x == INFINITY || eagle_msg->points[i].y == INFINITY || eagle_msg->points[i].z == INFINITY) continue;
           
            // 将点从雷达坐标系转换到Livox坐标系
            // ptMat:雷达坐标系的点
            // dstMat:livox坐标系的点
            cv::Mat ptMat, dstMat;
            // ptMat为四行一列的矩阵，值分别为点云的xyz坐标和一个额外的1，便于矩阵乘法运算
            ptMat = (cv::Mat_<double>(4, 1) << eagle_msg->points[i].x, eagle_msg->points[i].y, eagle_msg->points[i].z, 1); 
            // std::cout << "points[i].x :" << eagle_msg->points[i].x << std::endl;
            // std::cout << "points[i].y :" << eagle_msg->points[i].y << std::endl;
            // std::cout << "points[i].z :" << eagle_msg->points[i].z << std::endl; 
            // std::cout << "ptMAt :" << ptMat << std::endl;  

            // Perform matrix multiplication and save as Mat_ for easy element access
            dstMat= Radar_to_livox * ptMat;

            // 对点进行赋值
            radarpoint_raw.x = dstMat.at<double>(0,0);
            radarpoint_raw.y = dstMat.at<double>(1,0);
            radarpoint_raw.z = dstMat.at<double>(2,0);
            radarpoint_raw.intensity = eagle_msg->channels[1].values[i];
            radarpoint_raw.doppler = eagle_msg->channels[0].values[i];
            radarpoint_xyzi.x = dstMat.at<double>(0,0);
            radarpoint_xyzi.y = dstMat.at<double>(1,0);
            radarpoint_xyzi.z = dstMat.at<double>(2,0);
            radarpoint_xyzi.intensity = eagle_msg->channels[1].values[i];

            // 将点添加到点云中
            radarcloud_raw->points.push_back(radarpoint_raw);
            radarcloud_xyzi->points.push_back(radarpoint_xyzi);
        }
        
    }

    //********** Publish PointCloud2 Format Raw Cloud **********
    sensor_msgs::PointCloud2 pc2_raw_msg;                       // 定义PointCloud2消息对象
    pcl::toROSMsg(*radarcloud_raw, pc2_raw_msg);                // 将pcl的点云数据(radarcloud_raw)转换为ROS中的PointCloud2格式(pc2_raw_msg)
    pc2_raw_msg.header.stamp = eagle_msg->header.stamp;         // 时间戳
    pc2_raw_msg.header.frame_id = baselinkFrame;                // 消息所在坐标系
    pc2_raw_pub.publish(pc2_raw_msg);                           // 发布消息到指定的话题/eagle_data/pc2_raw中

    //********** Ego Velocity Estimation **********
    Eigen::Vector3d v_r, sigma_v_r;                               // 雷达的自我线速度和线速度的不确定性
    sensor_msgs::PointCloud2 inlier_radar_msg, outlier_radar_msg; // 内点点云数据(运动信息)和外点点云数据(噪声或运动干扰) 
    clock_t start_ms = clock();                                   // 记录开始估计自我运动的时刻
    // 调用estimate函数进行自我运动估计，如果估计成功，返回true，并将并将估计得到的线速度和不确定性保存在 v_r 和 sigma_v_r 中
    // 同时将内点和外点的点云数据保存在 inlier_radar_msg 和 outlier_radar_msg 中
    if (estimator.estimate(pc2_raw_msg, v_r, sigma_v_r, inlier_radar_msg, outlier_radar_msg))
    {
        clock_t end_ms = clock();                                       // 自我运动估计结束的时刻
        double time_used = double(end_ms - start_ms) / CLOCKS_PER_SEC;  // 自我运动估计所用的时间，单位为秒
        egovel_time.push_back(time_used);                               // 将自我运动估计的所用时间添加到时间数组中
        
        // 将估计得到的线速度和不确定性信息赋值给twist对象
        geometry_msgs::TwistWithCovarianceStamped twist;
        twist.header.stamp         = pc2_raw_msg.header.stamp;
        twist.twist.twist.linear.x = v_r.x();
        twist.twist.twist.linear.y = v_r.y();
        twist.twist.twist.linear.z = v_r.z();
        // 三个方向上的协方差矩阵，分别表示三个方向上的不确定性
        twist.twist.covariance.at(0)  = std::pow(sigma_v_r.x(), 2);
        twist.twist.covariance.at(7)  = std::pow(sigma_v_r.y(), 2);
        twist.twist.covariance.at(14) = std::pow(sigma_v_r.z(), 2);

        pub_twist.publish(twist);
        pub_inlier_pc2.publish(inlier_radar_msg);
        pub_outlier_pc2.publish(outlier_radar_msg);

    }
    else{;}

    // 创建一个指向pcl::PointCloud模板类（类型为PointT）的智能指针，该点云用于存储雷达点云中的内点。
    pcl::PointCloud<PointT>::Ptr radarcloud_inlier( new pcl::PointCloud<PointT> );
    pcl::fromROSMsg (inlier_radar_msg, *radarcloud_inlier);           // 从ROS消息转换为pcl点云类型，包含了内点信息。
    
    // src_cloud指针，用于选择处理的源点云
    pcl::PointCloud<PointT>::ConstPtr src_cloud;
    if (enable_dynamic_object_removal)    // 若选择启用了动态物体去除，指向雷达点云的内点
      src_cloud = radarcloud_inlier;
    else                                  // 未启用，指向原始的雷达点云
      src_cloud = radarcloud_xyzi;        

    // 没有有效数据或者数据获取不成功，直接返回
    if(src_cloud->empty()) {
      return;
    }

    // 去除点云中的扭曲
    src_cloud = deskewing(src_cloud);

    // if baselinkFrame is defined, transform the input cloud to the frame
    if(!baselinkFrame.empty()) {      // 检查是否定义了基准坐标系
      // 检查是否能够获取从src_cloud的坐标系到baselinkFrame的变换
      if(!tf_listener.canTransform(baselinkFrame, src_cloud->header.frame_id, ros::Time(0))) {
        std::cerr << "failed to find transform between " << baselinkFrame << " and " << src_cloud->header.frame_id << std::endl;
      }

      // 在最多等待2秒的时间内等待坐标变换的可用性
      // 获取从src_cloud坐标系到baselinkFrame的变换，并将其存储在transform中
      tf::StampedTransform transform;
      tf_listener.waitForTransform(baselinkFrame, src_cloud->header.frame_id, ros::Time(0), ros::Duration(2.0));
      tf_listener.lookupTransform(baselinkFrame, src_cloud->header.frame_id, ros::Time(0), transform);

      pcl::PointCloud<PointT>::Ptr transformed(new pcl::PointCloud<PointT>());      // 坐标变换后的点云
      pcl_ros::transformPointCloud(*src_cloud, *transformed, transform);            // 转换坐标
      transformed->header.frame_id = baselinkFrame;
      transformed->header.stamp = src_cloud->header.stamp;
      src_cloud = transformed;                                                      // 更新src_cloud，使其代表baselinkFrame中的点云
    }

    // 对点云依次进行距离过滤、下采样、离群点去除
    pcl::PointCloud<PointT>::ConstPtr filtered = distance_filter(src_cloud);     
    // filtered = passthrough(filtered);
    filtered = downsample(filtered);                
    // filtered = outlier_removal(filtered);

    // 此处输出为0，由于点云数量稀疏，所有的点都被当成离群点
    // ROS_INFO("After outlier_removal, Received point cloud message with %lu points", filtered->points.size());

    // Distance Histogram      计算点云中不同距离范围内点的数量，形成一个距离直方图
    static size_t num_frame = 0;            // 帧数
    if (num_frame % 10 == 0) {              // 每隔10帧执行操作
      Eigen::VectorXi num_at_dist = Eigen::VectorXi::Zero(100);       // 长度为100的整数向量，并将其所有元素初始化为0
      for (int i = 0; i < filtered->size(); i++){                     // 遍历filetred中的每个点
        int dist = floor(filtered->at(i).getVector3fMap().norm());    // 计算当前点到原点的距离，结果取整
        if (dist < 100)                                               // 如果距离小于100，则增加相应距离范围内点的数量
          num_at_dist(dist) += 1;
      }
      num_at_dist_vec.push_back(num_at_dist);                         // 将当前帧计算得到的距离直方图添加到num_at_dist_vec向量中
    }

    points_pub.publish(*filtered);
    
  }

  // 点云的区域截取，输入点云中 z 坐标在 -2 和 10 之间的点截取出来，返回一个新的点云
  pcl::PointCloud<PointT>::ConstPtr passthrough(const pcl::PointCloud<PointT>::ConstPtr& cloud) const {
    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
    PointT pt;
    for(int i = 0; i < cloud->size(); i++){
      if (cloud->at(i).z < 10 && cloud->at(i).z > -2){
        pt.x = (*cloud)[i].x;
        pt.y = (*cloud)[i].y;
        pt.z = (*cloud)[i].z;
        pt.intensity = (*cloud)[i].intensity;
        filtered->points.push_back(pt);
      }
    }
    filtered->header = cloud->header;
    return filtered;
  }

  // 下采样函数，用于减少点云的密度
  pcl::PointCloud<PointT>::ConstPtr downsample(const pcl::PointCloud<PointT>::ConstPtr& cloud) const {
    if(!downsample_filter) {      // 若不存在下采样滤波器，移除NaN/Inf点并返回
      // Remove NaN/Inf points
      pcl::PointCloud<PointT>::Ptr cloudout(new pcl::PointCloud<PointT>());
      std::vector<int> indices;
      pcl::removeNaNFromPointCloud(*cloud, *cloudout, indices);
      
      return cloudout;
    }

    // 若存在下采样滤波器，创建一个新的点云对象用于存储下采样后的点云
    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
    downsample_filter->setInputCloud(cloud);
    downsample_filter->filter(*filtered);
    filtered->header = cloud->header;

    return filtered;
  }

  // 离群点移除函数
  pcl::PointCloud<PointT>::ConstPtr outlier_removal(const pcl::PointCloud<PointT>::ConstPtr& cloud) const {
    if(!outlier_removal_filter) {
      return cloud;
    }

    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
    outlier_removal_filter->setInputCloud(cloud);
    outlier_removal_filter->filter(*filtered);
    filtered->header = cloud->header;

    return filtered;
  }

  // 距离过滤函数
  pcl::PointCloud<PointT>::ConstPtr distance_filter(const pcl::PointCloud<PointT>::ConstPtr& cloud) const {
    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());

    filtered->reserve(cloud->size());                     // 为过滤后的点云预先分配空间


    std::copy_if(cloud->begin(), cloud->end(), std::back_inserter(filtered->points), [&](const PointT& p) {
      double d = p.getVector3fMap().norm();               // 计算点到原点的距离
      double z = p.z;                                     // 获取点的z坐标
      // 根据设定的距离阈值和Z坐标范围过滤点云
      return d > distance_near_thresh && d < distance_far_thresh && z < z_high_thresh && z > z_low_thresh;
    });
    // for (size_t i=0; i<cloud->size(); i++){
    //   const PointT p = cloud->points.at(i);
    //   double d = p.getVector3fMap().norm();
    //   double z = p.z;
    //   if (d > distance_near_thresh && d < distance_far_thresh && z < z_high_thresh && z > z_low_thresh)
    //     filtered->points.push_back(p);
    // }

    // 设置过滤后的点云的宽度、高度和稠密属性
    filtered->width = filtered->size();
    filtered->height = 1;
    filtered->is_dense = false;

    // 将过滤后的点云的头信息设置为与输入点云相同
    filtered->header = cloud->header;

    return filtered;
  }

  pcl::PointCloud<PointT>::ConstPtr deskewing(const pcl::PointCloud<PointT>::ConstPtr& cloud) {
    ros::Time stamp = pcl_conversions::fromPCL(cloud->header.stamp);
    if(imu_queue.empty()) {           // 如果IMU队列为空，直接返回原始点云(无法进行去畸变)
      return cloud;
    }

    // the color encodes the point number in the point sequence
    if(colored_pub.getNumSubscribers()) {     // 如果有节点订阅了带颜色的点云，将带有颜色的点云发布出去
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored(new pcl::PointCloud<pcl::PointXYZRGB>());
      colored->header = cloud->header;
      colored->is_dense = cloud->is_dense;
      colored->width = cloud->width;
      colored->height = cloud->height;
      colored->resize(cloud->size());

      for(int i = 0; i < cloud->size(); i++) {
        double t = static_cast<double>(i) / cloud->size();
        colored->at(i).getVector4fMap() = cloud->at(i).getVector4fMap();
        colored->at(i).r = 255 * t;
        colored->at(i).g = 128;
        colored->at(i).b = 255 * (1 - t);
      }
      colored_pub.publish(*colored);
    }

    // 从IMU队列中获取最早的IMU数据，用于去畸变
    sensor_msgs::ImuConstPtr imu_msg = imu_queue.front();

    // 寻找IMU队列中最早的时间戳，以与当前点云对齐
    auto loc = imu_queue.begin();
    for(; loc != imu_queue.end(); loc++) {
      imu_msg = (*loc);
      if((*loc)->header.stamp > stamp) {
        break;
      }
    }

    // 从IMU队列中移除已使用的IMU数据
    imu_queue.erase(imu_queue.begin(), loc);

    // 获取IMU的角度速度，并反转
    Eigen::Vector3f ang_v(imu_msg->angular_velocity.x, imu_msg->angular_velocity.y, imu_msg->angular_velocity.z);
    ang_v *= -1;

    // deskewed存储去畸变后的点云
    pcl::PointCloud<PointT>::Ptr deskewed(new pcl::PointCloud<PointT>());
    deskewed->header = cloud->header;
    deskewed->is_dense = cloud->is_dense;
    deskewed->width = cloud->width;
    deskewed->height = cloud->height;
    deskewed->resize(cloud->size());

    // 获取扫描周期
    double scan_period = private_nh.param<double>("scan_period", 0.1);
    // 遍历每个点，进行去畸变操作
    for(int i = 0; i < cloud->size(); i++) {
      const auto& pt = cloud->at(i);

      // TODO: transform IMU data into the LIDAR frame
      double delta_t = scan_period * static_cast<double>(i) / cloud->size();          // 计算每个点在扫描周期内的时间偏移
      Eigen::Quaternionf delta_q(1, delta_t / 2.0 * ang_v[0], delta_t / 2.0 * ang_v[1], delta_t / 2.0 * ang_v[2]);  // 在delta_t内发生的旋转
      Eigen::Vector3f pt_ = delta_q.inverse() * pt.getVector3fMap();                  // 将点的三维坐标乘以四元数的逆来应用去畸变变换，并转换成 Vector3f 类型

      // 将去畸变后的点添加到新的点云中
      deskewed->at(i) = cloud->at(i);
      deskewed->at(i).getVector3fMap() = pt_;
    }

    return deskewed;
  }

  bool RadarRaw2PointCloudXYZ(const pcl::PointCloud<RadarPointCloudType>::ConstPtr &raw, pcl::PointCloud<pcl::PointXYZ>::Ptr &cloudxyz)
  {
      pcl::PointXYZ point_xyz;
      for(int i = 0; i < raw->size(); i++)
      {
          point_xyz.x = (*raw)[i].x;
          point_xyz.y = (*raw)[i].y;
          point_xyz.z = (*raw)[i].z;
          cloudxyz->points.push_back(point_xyz);
      }
      return true;
  }
  bool RadarRaw2PointCloudXYZI(const pcl::PointCloud<RadarPointCloudType>::ConstPtr &raw, pcl::PointCloud<pcl::PointXYZI>::Ptr &cloudxyzi)
  {
      pcl::PointXYZI radarpoint_xyzi;
      for(int i = 0; i < raw->size(); i++)
      {
          radarpoint_xyzi.x = (*raw)[i].x;
          radarpoint_xyzi.y = (*raw)[i].y;
          radarpoint_xyzi.z = (*raw)[i].z;
          radarpoint_xyzi.intensity = (*raw)[i].intensity;
          cloudxyzi->points.push_back(radarpoint_xyzi);
      }
      return true;
  }

  void command_callback(const std_msgs::String& str_msg) {
    if (str_msg.data == "time") {                                           // 如果收到的命令是time
      std::sort(egovel_time.begin(), egovel_time.end());                    // 对存储着时间的容器进行排序
      double median = egovel_time.at(size_t(egovel_time.size() / 2));       // 计算排序后的中值
      cout << "Ego velocity time cost (median): " << median << endl;
    }
    else if (str_msg.data == "point_distribution") {                        // 如果收到的命令是point_distribution           
      Eigen::VectorXi data(100);
      for (size_t i = 0; i < num_at_dist_vec.size(); i++){                  // num_at_dist_vec存储着点分布数据
        Eigen::VectorXi& nad = num_at_dist_vec.at(i);
        for (int j = 0; j< 100; j++){
          data(j) += nad(j);                                                // 将每个位置上的数据相加，得到累计的点分布数据
        }
      }
      data /= num_at_dist_vec.size();                                       // 计算平均值
      for (int i=0; i<data.size(); i++){
        cout << data(i) << ", ";
      }
      cout << endl;
    }
  }

private:
  ros::NodeHandle nh;
  ros::NodeHandle private_nh;

  ros::Subscriber imu_sub;
  std::vector<sensor_msgs::ImuConstPtr> imu_queue;
  ros::Subscriber points_sub;
  
  ros::Publisher points_pub;
  ros::Publisher colored_pub;
  ros::Publisher imu_pub;
  ros::Publisher gt_pub;

  tf::TransformListener tf_listener;
  tf::TransformBroadcaster tf_broadcaster;


  bool use_distance_filter;
  double distance_near_thresh;
  double distance_far_thresh;
  double z_low_thresh;
  double z_high_thresh;

  pcl::Filter<PointT>::Ptr downsample_filter;
  pcl::Filter<PointT>::Ptr outlier_removal_filter;

  cv::Mat Radar_to_livox; // Transform Radar point cloud to LiDAR Frame
  cv::Mat Thermal_to_RGB,Radar_to_Thermal,RGB_to_livox,livox_to_RGB,Change_Radarframe;
  rio::RadarEgoVelocityEstimator estimator;
  ros::Publisher pub_twist, pub_inlier_pc2, pub_outlier_pc2, pc2_raw_pub;

  float power_threshold;
  bool enable_dynamic_object_removal = false;

  std::mutex odom_queue_mutex;
  std::deque<nav_msgs::Odometry> odom_msgs;
  bool publish_tf;

  ros::Subscriber command_sub;
  std::vector<double> egovel_time;

  std::vector<Eigen::VectorXi> num_at_dist_vec;
};

}  // namespace radar_graph_slam

PLUGINLIB_EXPORT_CLASS(radar_graph_slam::PreprocessingNodelet, nodelet::Nodelet)
