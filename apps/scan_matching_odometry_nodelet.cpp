// SPDX-License-Identifier: BSD-2-Clause

#include <ctime>
#include <chrono>
#include <mutex>
#include <atomic>
#include <memory>
#include <iomanip>
#include <iostream>
#include <cmath>
#include <unordered_map>
#include <boost/format.hpp>
#include <boost/thread.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

#include <ros/ros.h>
#include <ros/time.h>
#include <ros/duration.h>

#include <tf_conversions/tf_eigen.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <std_msgs/String.h>
#include <std_msgs/Time.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/TwistWithCovarianceStamped.h>

#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

#include <pcl_ros/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/octree/octree_search.h>

#include <Eigen/Dense>

#include <radar_graph_slam/ros_utils.hpp>
#include <radar_graph_slam/registrations.hpp>
#include <radar_graph_slam/ScanMatchingStatus.h>
#include <radar_graph_slam/keyframe.hpp>
#include <radar_graph_slam/keyframe_updater.hpp>
#include <radar_graph_slam/graph_slam.hpp>
#include <radar_graph_slam/information_matrix_calculator.hpp>

#include "utility_radar.h"

using namespace std;

namespace radar_graph_slam {

class ScanMatchingOdometryNodelet : public nodelet::Nodelet, public ParamServer {
public:
  typedef pcl::PointXYZI PointT;
  typedef message_filters::sync_policies::ApproximateTime<geometry_msgs::TwistWithCovarianceStamped, sensor_msgs::PointCloud2> ApproxSyncPolicy;
  // typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, geometry_msgs::TransformStamped> ApproxSyncPolicy2;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ScanMatchingOdometryNodelet() {}
  virtual ~ScanMatchingOdometryNodelet() {}

  virtual void onInit() {
    NODELET_DEBUG("initializing scan_matching_odometry_nodelet...");
    nh = getNodeHandle();
    mt_nh = getMTNodeHandle();
    private_nh = getPrivateNodeHandle();

    initialize_params(); // this

    if(private_nh.param<bool>("enable_imu_frontend", false)) {
      msf_pose_sub = nh.subscribe<geometry_msgs::PoseWithCovarianceStamped>("/msf_core/pose", 1, boost::bind(&ScanMatchingOdometryNodelet::msf_pose_callback, this, _1, false));
      msf_pose_after_update_sub = nh.subscribe<geometry_msgs::PoseWithCovarianceStamped>("/msf_core/pose_after_update", 1, boost::bind(&ScanMatchingOdometryNodelet::msf_pose_callback, this, _1, true));
    }
    //******** Subscribers **********
    ego_vel_sub.reset(new message_filters::Subscriber<geometry_msgs::TwistWithCovarianceStamped>(mt_nh, "/eagle_data/twist", 256));
    points_sub.reset(new message_filters::Subscriber<sensor_msgs::PointCloud2>(mt_nh, "/filtered_points", 32));
    sync.reset(new message_filters::Synchronizer<ApproxSyncPolicy>(ApproxSyncPolicy(32), *ego_vel_sub, *points_sub));
    sync->registerCallback(boost::bind(&ScanMatchingOdometryNodelet::pointcloud_callback, this, _1, _2));
    imu_sub = nh.subscribe("/imu", 1024, &ScanMatchingOdometryNodelet::imu_callback, this);
    command_sub = nh.subscribe("/command", 10, &ScanMatchingOdometryNodelet::command_callback, this);

    //******** Publishers **********
    read_until_pub = nh.advertise<std_msgs::Header>("/scan_matching_odometry/read_until", 32);
    // Odometry of Radar scan-matching_
    odom_pub = nh.advertise<nav_msgs::Odometry>(odomTopic, 32);
    // Transformation of Radar scan-matching_
    trans_pub = nh.advertise<geometry_msgs::TransformStamped>("/scan_matching_odometry/transform", 32);
    status_pub = private_nh.advertise<ScanMatchingStatus>("/scan_matching_odometry/status", 8);
    aligned_points_pub = nh.advertise<sensor_msgs::PointCloud2>("/aligned_points", 32);
    submap_pub = nh.advertise<sensor_msgs::PointCloud2>("/radar_graph_slam/submap", 2);
  }

private:
  /**
   * @brief initialize parameters
   */
  void initialize_params() {
    auto& pnh = private_nh;
    // points_topic = pnh.param<std::string>("points_topic", "/radar_enhanced_pcl");           // 点云话题
    points_topic = pnh.param<std::string>("points_topic", "/ars548_process/detection_point_cloud");           // 点云话题
    use_ego_vel = pnh.param<bool>("use_ego_vel", false);                                    // 是否使用车辆自我速度

    // The minimum tranlational distance and rotation angle between keyframes_.
    // If this value is zero, frames are always compared with the previous frame
    keyframe_delta_trans = pnh.param<double>("keyframe_delta_trans", 0.25);                 // 关键帧之间的最小平移距离
    keyframe_delta_angle = pnh.param<double>("keyframe_delta_angle", 0.15);                 // 关键帧之间的最小旋转角度
    keyframe_delta_time = pnh.param<double>("keyframe_delta_time", 1.0);                    // 关键帧之间的最小时间间隔

    // Registration validation by thresholding
    enable_transform_thresholding = pnh.param<bool>("enable_transform_thresholding", false);// 是否启用通过阈值判断进行配准验证
    enable_imu_thresholding = pnh.param<bool>("enable_imu_thresholding", false);            // 是否启用IMU数据进行阈值判断
    max_acceptable_trans = pnh.param<double>("max_acceptable_trans", 1.0);                  // 允许的最大平移阈值
    max_acceptable_angle = pnh.param<double>("max_acceptable_angle", 1.0);                  // 允许的最大旋转阈值
    max_diff_trans = pnh.param<double>("max_diff_trans", 1.0);                              // 最大平移差异阈值，默认1米
    max_diff_angle = pnh.param<double>("max_diff_angle", 1.0);                              // 最大旋转阈值，默认1弧度
    max_egovel_cum = pnh.param<double>("max_egovel_cum", 1.0);                              // 最大累积自身车辆速度阈值

    map_cloud_resolution = pnh.param<double>("map_cloud_resolution", 0.05);                 // 点云地图的分辨率，默认为0.05米
    keyframe_updater.reset(new KeyframeUpdater(pnh));                                       // 初始化关键帧更新器

    enable_scan_to_map = pnh.param<bool>("enable_scan_to_map", false);                      // 是否启用从激光雷达扫描到地图的配准
    max_submap_frames = pnh.param<int>("max_submap_frames", 5);                             // 子地图包含的最大帧数

    enable_imu_fusion = private_nh.param<bool>("enable_imu_fusion", false);                 // 是否启用Imu数据融合
    imu_debug_out = private_nh.param<bool>("imu_debug_out", false);                         // 是否启用IMU调试输出
    cout << "enable_imu_fusion = " << enable_imu_fusion << endl;
    imu_fusion_ratio = private_nh.param<double>("imu_fusion_ratio", 0.1);                   // IMU数据融合比例

    // graph_slam.reset(new GraphSLAM(pnh.param<std::string>("g2o_solver_type", "lm_var")));

    // select a downsample method (VOXELGRID, APPROX_VOXELGRID, NONE)
    std::string downsample_method = pnh.param<std::string>("downsample_method", "VOXELGRID");// 下采样方法
    double downsample_resolution = pnh.param<double>("downsample_resolution", 0.1);          
    if(downsample_method == "VOXELGRID") {
      std::cout << "downsample: VOXELGRID " << downsample_resolution << std::endl;
      auto voxelgrid = new pcl::VoxelGrid<PointT>();
      voxelgrid->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
      downsample_filter.reset(voxelgrid);
    } else if(downsample_method == "APPROX_VOXELGRID") {
      std::cout << "downsample: APPROX_VOXELGRID " << downsample_resolution << std::endl;
      pcl::ApproximateVoxelGrid<PointT>::Ptr approx_voxelgrid(new pcl::ApproximateVoxelGrid<PointT>());
      approx_voxelgrid->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
      downsample_filter = approx_voxelgrid;
    } else {
      if(downsample_method != "NONE") {
        std::cerr << "warning: unknown downsampling type (" << downsample_method << ")" << std::endl;
        std::cerr << "       : use passthrough filter" << std::endl;
      }
      std::cout << "downsample: NONE" << std::endl;
      pcl::PassThrough<PointT>::Ptr passthrough(new pcl::PassThrough<PointT>());
      downsample_filter = passthrough;
    }
    registration_s2s = select_registration_method(pnh);                                     // 选择扫描到扫描的配准方法
    registration_s2m = select_registration_method(pnh);                                     // 选择扫描到地图的配准方法
  }

  void imu_callback(const sensor_msgs::ImuConstPtr& imu_msg) {
    
    Eigen::Quaterniond imu_quat_from(imu_msg->orientation.w, imu_msg->orientation.x, imu_msg->orientation.y, imu_msg->orientation.z);
    Eigen::Quaterniond imu_quat_deskew = imu_quat_from * extQRPY;            // 通过预定义的extQRPY对IMU方向进行去扰动
    imu_quat_deskew.normalize();                                             // 归一化去扰动后的四元数

    double roll, pitch, yaw;                                                
    // tf::quaternionMsgToTF(imu_odom_msg->orientation, orientation); 
    tf::Quaternion orientation = tf::Quaternion(imu_quat_deskew.x(),imu_quat_deskew.y(),imu_quat_deskew.z(),imu_quat_deskew.w()); // 将去扰动后的四元数转换为TF库中的四元数。
    tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);                      // 使用TF库计算RPY角
    imuPointerLast = (imuPointerLast + 1) % imuQueLength;                     // 更新IMU队列的指针
    imuTime[imuPointerLast] = imu_msg->header.stamp.toSec();                  // IMU消息的时间戳
    imuRoll[imuPointerLast] = roll;                                           // IMU消息的滚转角
    imuPitch[imuPointerLast] = pitch;                                         // IMU消息的俯仰角
    // cout << "get imu rp: " << roll << " " << pitch << endl;

    sensor_msgs::ImuPtr imu(new sensor_msgs::Imu);
    imu->header = imu_msg->header;
    imu->angular_velocity = imu_msg->angular_velocity; imu->linear_acceleration = imu_msg->linear_acceleration;
    imu->angular_velocity_covariance = imu_msg->angular_velocity_covariance;
    imu->linear_acceleration_covariance, imu_msg->linear_acceleration_covariance;
    imu->orientation_covariance = imu_msg->orientation_covariance;
    imu->orientation.w=imu_quat_deskew.w(); imu->orientation.x = imu_quat_deskew.x(); imu->orientation.y = imu_quat_deskew.y(); imu->orientation.z = imu_quat_deskew.z();
    { // 将新的IMU消息推送到IMU队列中，使用互斥锁确保线程安全
      std::lock_guard<std::mutex> lock(imu_queue_mutex);
      imu_queue.push_back(imu); 
    }

    static int cnt = 0;                             // 静态计数器，用于跟踪处理IMU消息的次数
    if(cnt == 0) {                                  // 在第一次处理IMU消息时执行
      geometry_msgs::Quaternion imuQuat = tf::createQuaternionMsgFromRollPitchYaw(roll, pitch, 0);              // 使用TF库创建包含给定RPY角的四元数
      global_orient_matrix = Eigen::Quaterniond(imuQuat.w, imuQuat.x, imuQuat.y, imuQuat.z).toRotationMatrix(); // 将四元数转换为旋转矩阵，并设置为全局变量
      ROS_INFO_STREAM("Initial IMU euler angles (RPY): "
            << RAD2DEG(roll) << ", " << RAD2DEG(pitch) << ", " << RAD2DEG(yaw));
      cnt = 1;
    }
    
  }


  // 根据时间戳的相似度，关联IMU和关键帧数据
  bool flush_imu_queue() {
    std::lock_guard<std::mutex> lock(imu_queue_mutex);
    if(keyframes.empty() || imu_queue.empty()) {// 关键帧队列或IMU队列为空，没有进行更新
      return false;
    }
    bool updated = false;                       // 标志是否有新的IMU数据与关键帧相关联
    auto imu_cursor = imu_queue.begin();

    for(size_t i=0; i < keyframes.size(); i++) {           // 遍历关键帧队列
      auto keyframe = keyframes.at(i);                     // 当前关键帧
      if(keyframe->stamp < (*imu_cursor)->header.stamp) {  // 关键帧的时间戳小于当前IMU数据的时间戳，跳过当前迭代
        continue;
      }
      if(keyframe->stamp > imu_queue.back()->header.stamp) {// 如果关键帧的时间戳大于IMU队列中最后一条数据的时间戳，跳出循环。
        break;
      }
      // find the imu data which is closest to the keyframe_
      auto closest_imu = imu_cursor;                        
      for(auto imu = imu_cursor; imu != imu_queue.end(); imu++) {                   
        auto dt = ((*closest_imu)->header.stamp - keyframe->stamp).toSec(); // 当前最接近的IMU数据与关键帧时间戳的时间差
        auto dt2 = ((*imu)->header.stamp - keyframe->stamp).toSec();
        if(std::abs(dt) < std::abs(dt2)) {                                  // 如果当前最接近的IMU数据时间差小于当前迭代的时间差，跳出循环
          break;
        }
        closest_imu = imu;
      }
      // if the time residual between the imu and keyframe_ is too large, skip it
      imu_cursor = closest_imu;
      if(0.2 < std::abs(((*closest_imu)->header.stamp - keyframe->stamp).toSec())) {
        continue;
      }
      sensor_msgs::Imu imu_;
      imu_.header = (*closest_imu)->header; imu_.orientation = (*closest_imu)->orientation;
      imu_.angular_velocity = (*closest_imu)->angular_velocity; imu_.linear_acceleration = (*closest_imu)->linear_acceleration;
      imu_.angular_velocity_covariance = (*closest_imu)->angular_velocity_covariance;
      imu_.linear_acceleration_covariance = (*closest_imu)->linear_acceleration_covariance;
      imu_.orientation_covariance = (*closest_imu)->orientation_covariance;
      keyframe->imu = imu_;                                                   // 关联两者数据
      updated = true;                                                         // 标记为有新的IMU数据与关键帧相关联。
    }
    // 从imu_queue中删除那些已经与关键帧相关联的IMU数据
    auto remove_loc = std::upper_bound(imu_queue.begin(), imu_queue.end(), keyframes.back()->stamp, [=](const ros::Time& stamp, const sensor_msgs::ImuConstPtr& imupoint) { return stamp < imupoint->header.stamp; });
    imu_queue.erase(imu_queue.begin(), remove_loc);
    return updated;                                       
  }

  // 获取与给定时间戳最接近的IMU数据
  std::pair<bool, sensor_msgs::Imu> get_closest_imu(ros::Time frame_stamp) {
    sensor_msgs::Imu imu_;
    std::pair<bool, sensor_msgs::Imu> false_result {false, imu_};
    if(keyframes.empty() || imu_queue.empty())      // 关键帧队列或IMU队列为空，直接返回false_result
      return false_result;
    bool updated = false;                           // 标记是否成功获取IMU数据(与给定时间戳最近的)
    auto imu_cursor = imu_queue.begin();            // IMU队列的迭代器
    
    // find the imu data which is closest to the keyframe_
    auto closest_imu = imu_cursor;
    for(auto imu = imu_cursor; imu != imu_queue.end(); imu++) {
      auto dt = ((*closest_imu)->header.stamp - frame_stamp).toSec();
      auto dt2 = ((*imu)->header.stamp - frame_stamp).toSec();
      if(std::abs(dt) < std::abs(dt2)) {
        break;
      }
      closest_imu = imu;
    }
    // if the time residual between the imu and keyframe_ is too large, skip it
    imu_cursor = closest_imu;
    if(0.2 < std::abs(((*closest_imu)->header.stamp - frame_stamp).toSec()))
      return false_result;

    imu_.header = (*closest_imu)->header; imu_.orientation = (*closest_imu)->orientation;
    imu_.angular_velocity = (*closest_imu)->angular_velocity; imu_.linear_acceleration = (*closest_imu)->linear_acceleration;
    imu_.angular_velocity_covariance = (*closest_imu)->angular_velocity_covariance; 
    imu_.linear_acceleration_covariance = (*closest_imu)->linear_acceleration_covariance;
    imu_.orientation_covariance = (*closest_imu)->orientation_covariance;

    updated = true;
    // cout << (*closest_imu)->orientation <<endl;
    std::pair<bool, sensor_msgs::Imu> result {updated, imu_};
    return result;
  }

  // 将IMU的姿态信息融合到激光雷达里程计的变换矩阵中，以实现更准确的姿态估计
  void transformUpdate(Eigen::Matrix4d& odom_to_update) // IMU
  {
		if (imuPointerLast >= 0) // 检查是否有IMU数据可用，imuPointerLast指向IMU队列的最新数据
    {
      // cout << "    ";
      float imuRollLast = 0, imuPitchLast = 0;                              // 用于存储最近的IMU Roll 和 Pitch 角度
      while (imuPointerFront != imuPointerLast) {
        if (timeLaserOdometry + scanPeriod < imuTime[imuPointerFront]) {    // 如果激光雷达的时间戳加上扫描周期小于当前IMU数据的时间戳，表示找到了最近的IMU数据，退出循环
          break;
        }
        imuPointerFront = (imuPointerFront + 1) % imuQueLength;             // 如果还没有找到最近的IMU数据，继续迭代
      }
      cout << "    ";
      if (timeLaserOdometry + scanPeriod > imuTime[imuPointerFront]) {      // 如果激光雷达的时间戳加上扫描周期大于最近的IMU数据的时间戳，表示找到了最近的IMU数据
        imuRollLast = imuRoll[imuPointerFront];
        imuPitchLast = imuPitch[imuPointerFront];
        cout << "    ";
      }
      else {                 // 如果没有刚好找到匹配的IMU数据，通过插值计算最近的IMU数据姿态
        cout << "    ";
        int imuPointerBack = (imuPointerFront + imuQueLength - 1) % imuQueLength;                     // 计算前一个IMU数据的索引
        float ratioFront = (timeLaserOdometry + scanPeriod - imuTime[imuPointerBack])                 // 计算插值的权重，用于前一个IMU数据
                          / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
        float ratioBack = (imuTime[imuPointerFront] - timeLaserOdometry - scanPeriod)                 // 计算插值的权重，用于当前IMU数据
                        / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);

        imuRollLast = imuRoll[imuPointerFront] * ratioFront + imuRoll[imuPointerBack] * ratioBack;    // 通过线性插值计算最近IMU数据的Roll角度
        imuPitchLast = imuPitch[imuPointerFront] * ratioFront + imuPitch[imuPointerBack] * ratioBack; // 通过线性插值计算最近IMU数据的Pitch角度
      }
      
      Eigen::Matrix3d matr = odom_to_update.block<3, 3>(0, 0);                    // 从激光雷达里程计的变换矩阵中提取旋转部分
      // Eigen::Vector3d xyz = odom_to_update.block<3, 1>(0, 3);
      Eigen::Vector3d ypr_odom = R2ypr(matr.block<3,3>(0,0));                     // 将旋转矩阵转换为欧拉角表示，R2ypr是自定义的函数
      // 根据IMU Roll、Pitch 和激光雷达里程计的Yaw构造IMU的四元数。
      geometry_msgs::Quaternion imuQuat = tf::createQuaternionMsgFromRollPitchYaw(imuRollLast, imuPitchLast, ypr_odom(0));
      Eigen::Matrix3d imu_rot = Eigen::Matrix3d(Eigen::Quaterniond(imuQuat.w, imuQuat.x, imuQuat.y, imuQuat.z));
      Eigen::Vector3d ypr_imu = R2ypr(imu_rot);                                   // 将IMU的旋转矩阵转换为欧拉角表示
      // IMU orientation transformed from world coordinate to map coordinate
      // 将IMU的旋转矩阵从世界坐标系变换到地图坐标系，通过地图坐标系的全局方向矩阵 global_orient_matrix 实现
      Eigen::Matrix3d imu_rot_transed = global_orient_matrix.inverse() * imu_rot;
      Eigen::Vector3d ypr_imu_trans = R2ypr(imu_rot_transed);
      double& yaw_ = ypr_odom(0);                                                 // 获取激光雷达里程计的yaw角度
      double pitch_fused = (1 - imu_fusion_ratio) * ypr_odom(1) + imu_fusion_ratio * ypr_imu_trans(1);      // 融合IMU数据和激光雷达里程计的Pitch角度
      double roll_fused = (1 - imu_fusion_ratio) * ypr_odom(2) + imu_fusion_ratio * ypr_imu_trans(2);       // 融合IMU数据和激光雷达里程计的Roll角度
      // 根据融合后的Roll、Pitch和原始的Yaw构造更新后的四元数
      geometry_msgs::Quaternion rosQuat = tf::createQuaternionMsgFromRollPitchYaw(roll_fused, pitch_fused, yaw_);
      Eigen::Quaterniond quat_updated = Eigen::Quaterniond(rosQuat.w, rosQuat.x, rosQuat.y, rosQuat.z);     // 将更新后的四元数转换为Eigen库的Quaterniond类型
      odom_to_update.block<3, 3>(0, 0) = quat_updated.toRotationMatrix();                                   // 将更新后的旋转矩阵写回激光雷达里程计的变换矩阵

      // 如果启用了IMU的调试输出，打印IMU的Roll和Pitch角度以及其他信息
      if (imu_debug_out)
        cout << "IMU rp: " << RAD2DEG(ypr_imu(2)) << " " << RAD2DEG(ypr_imu(1))
            << ". IMU transed rp: " << RAD2DEG(ypr_imu_trans(2)) << " " << RAD2DEG(ypr_imu_trans(1))
            // << ". Odom rp: " << RAD2DEG(ypr_odom(2)) << " " << RAD2DEG(ypr_odom(1))
            // << ". Updated rp: " << RAD2DEG(roll_fused) << " " << RAD2DEG(pitch_fused)
            // << ". Roll Pitch increment: " << RAD2DEG(roll_fused - ypr_odom(2)) << " " << RAD2DEG(pitch_fused - ypr_odom(1)) 
            << endl;
		}
  }

  /**
   * @brief callback for point clouds
   * @param cloud_msg  point cloud msg
   */
  void pointcloud_callback(const geometry_msgs::TwistWithCovarianceStampedConstPtr& twistMsg, const sensor_msgs::PointCloud2ConstPtr& cloud_msg) {
    if(!ros::ok()) {
      return;
    }

    // std::cout << "-----------point_callback-scan_matching_odometry_nodelet.cpp-----------------" << std::endl;
    // if (cloud_msg == nullptr)
    // {
    //   ROS_WARN("Received empty point cloud message");
    // }
    // else
    // {
    //   ROS_INFO("Received point cloud message with %u points", cloud_msg->width * cloud_msg->height);
    //   // Received point cloud message with 0 points???
    // }

    // 从点云信息提取时间戳
    timeLaserOdometry = cloud_msg->header.stamp.toSec();
    double this_cloud_time = cloud_msg->header.stamp.toSec();
    static double last_cloud_time = this_cloud_time;

    // 计算累积的自我速度
    double dt = this_cloud_time - last_cloud_time;                      // 连续点云之间的时间差 = 当前点云时间 - 之前的点云时间
    double egovel_cum_x = twistMsg->twist.twist.linear.x * dt;
    double egovel_cum_y = twistMsg->twist.twist.linear.y * dt;
    double egovel_cum_z = twistMsg->twist.twist.linear.z * dt;
    // If too large, set 0
    if (pow(egovel_cum_x,2)+pow(egovel_cum_y,2)+pow(egovel_cum_z,2) > pow(max_egovel_cum, 2));
    else egovel_cum.block<3, 1>(0, 3) = Eigen::Vector3d(egovel_cum_x, egovel_cum_y, egovel_cum_z);
    
    last_cloud_time = this_cloud_time;

    // 将ROS点云消息转换为PCL点云消息
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(*cloud_msg, *cloud);

    // Matching  执行点云配准来估计传感器位姿
    Eigen::Matrix4d pose = matching(cloud_msg->header.stamp, cloud);
    geometry_msgs::TwistWithCovariance twist = twistMsg->twist;
    // publish map to odom frame  
    publish_odometry(cloud_msg->header.stamp, mapFrame, odometryFrame, pose, twist);

    // In offline estimation, point clouds will be supplied until the published time
    // 使用read_until机制，处理特定时间戳之前的数据
    std_msgs::HeaderPtr read_until(new std_msgs::Header());
    read_until->frame_id = points_topic;
    read_until->stamp = cloud_msg->header.stamp + ros::Duration(1, 0);
    read_until_pub.publish(read_until);

    read_until->frame_id = "/filtered_points";
    read_until_pub.publish(read_until);
  }


  void msf_pose_callback(const geometry_msgs::PoseWithCovarianceStampedConstPtr& pose_msg, bool after_update) {
    if(after_update) {
      msf_pose_after_update = pose_msg;
    } else {
      msf_pose = pose_msg;
    }
  }

  /**
   * @brief downsample a point cloud
   * @param cloud  input cloud
   * @return downsampled point cloud
   */
  pcl::PointCloud<PointT>::ConstPtr downsample(const pcl::PointCloud<PointT>::ConstPtr& cloud) const {
    if(!downsample_filter) {
      return cloud;
    }

    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
    downsample_filter->setInputCloud(cloud);
    downsample_filter->filter(*filtered);

    return filtered;
  }

  /**
   * @brief estimate the relative pose between an input cloud and a keyframe_ cloud
   * @param stamp  the timestamp of the input cloud
   * @param cloud  the input cloud
   * @return the relative pose between the input cloud and the keyframe_ cloud
   */
  Eigen::Matrix4d matching(const ros::Time& stamp, const pcl::PointCloud<PointT>::ConstPtr& cloud) {
    if(!keyframe_cloud_s2s) {                               // 若关键帧点云不为空
      prev_time = ros::Time();                              // 将上一次处理的时间戳prev_time设置为零
      prev_trans_s2s.setIdentity();                         // 将上一次的相对姿态变换prev_trans_s2s设置为单位矩阵
      keyframe_pose_s2s.setIdentity();                      // 将关键帧的位姿设置为单位矩阵
      keyframe_stamp = stamp;                               // 将关键帧的时间戳设置为当前输入点云的时间戳
      keyframe_cloud_s2s = cloud;//downsample(cloud);       // 将关键帧的点云设置为当前输入的点云数据
      // keyframe_cloud_s2s: 点云配准目标
      // registration_s2s: 点云配准对象
      registration_s2s->setInputTarget(keyframe_cloud_s2s); // Scan-to-scan
      if (enable_scan_to_map){                              // 如果启用了扫描到地图的配准
        prev_trans_s2m.setIdentity();
        keyframe_pose_s2m.setIdentity();
        keyframe_cloud_s2m = cloud;
        registration_s2m->setInputTarget(keyframe_cloud_s2m);
      }
      return Eigen::Matrix4d::Identity();                   // 由于当前是第一帧，相对姿态尚未计算，因此直接返回单位矩阵
    }
    // auto filtered = downsample(cloud);
    auto filtered = cloud;
    // Set Source Cloud
    registration_s2s->setInputSource(filtered);              // 设置filtered为配准的源点云，用于扫描到扫描的配准
    if (enable_scan_to_map)
      registration_s2m->setInputSource(filtered);            // 设置filtered为配准的源点云，用于扫描到地图的配准

    std::string msf_source;
    Eigen::Isometry3d msf_delta = Eigen::Isometry3d::Identity();    // msf_delta用于存储扫描匹配的相对位姿变换
    
    pcl::PointCloud<PointT>::Ptr aligned(new pcl::PointCloud<PointT>());    // aligned 存储配准后的点云
    Eigen::Matrix4d odom_s2s_now;                                           // 变换矩阵，存储扫描到扫描配准后的位姿
    Eigen::Matrix4d odom_s2m_now;                                           // 变换矩阵，存储扫描到地图配准后的位姿

    // **********  Matching  **********
    Eigen::Matrix4d guess;                                                  // 初始猜测变换矩阵
    if (use_ego_vel)                                                        // 使用车辆速度信息，则乘以累积的车辆速度变换
      guess = prev_trans_s2s * egovel_cum * msf_delta.matrix();             // 则 guess = 上一次扫描到扫描的变换 * 累积的车辆速度变换 * 扫描匹配的相对位姿变换   
    else
      guess = prev_trans_s2s * msf_delta.matrix();

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    registration_s2s->align(*aligned, guess.cast<float>());                 // 进行点云配准，aligned存储配准后的点云
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    double time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1).count();    // 配准所需的时间
    s2s_matching_time.push_back(time_used);

    // 发布扫描配准的状态信息，包括时间戳、点云的坐标系、配准后的电源、扫描匹配的来源msf_source、扫描匹配的相对位姿变换
    publish_scan_matching_status(stamp, cloud->header.frame_id, aligned, msf_source, msf_delta);

    // If not converged, use last transformation
    if(!registration_s2s->hasConverged()) {                               // 扫描到扫描的匹配未收敛,忽略当前帧，返回上一次的变换矩阵
      NODELET_INFO_STREAM("scan matching_ has not converged!!");          
      NODELET_INFO_STREAM("ignore this frame(" << stamp << ")");
      if (enable_scan_to_map) return keyframe_pose_s2m * prev_trans_s2m;
      else return keyframe_pose_s2s * prev_trans_s2s;
    }
    // 匹配收敛，获取最终的扫描到扫描的变换矩阵trans_s2s
    Eigen::Matrix4d trans_s2s = registration_s2s->getFinalTransformation().cast<double>();   
    odom_s2s_now = keyframe_pose_s2s * trans_s2s;                         // 得到当前时刻的扫描到扫描的位姿变换矩阵

    Eigen::Matrix4d trans_s2m;
    if (enable_scan_to_map){
      registration_s2m->align(*aligned, guess.cast<float>());
      if(!registration_s2m->hasConverged()) {                             // 扫描到地图的匹配未收敛
        NODELET_INFO_STREAM("scan matching_ has not converged!!");
        NODELET_INFO_STREAM("ignore this frame(" << stamp << ")");
        return keyframe_pose_s2m * prev_trans_s2m;
      }
      trans_s2m = registration_s2m->getFinalTransformation().cast<double>();// 变换矩阵
      odom_s2m_now = keyframe_pose_s2m * trans_s2m;                         // 当前时刻的地图到扫描的位姿变换矩阵 odom_s2m_now
    }

    // Add abnormal judgment, that is, if the difference between the two frames matching point cloud 
    // transition matrix is too large, it will be discarded
    bool thresholded = false;                                                                     // 表示是否超过阈值
    if(enable_transform_thresholding) {                                                           // 启用变换阈值判断
      Eigen::Matrix4d radar_delta;                                                                // 两帧之间变换矩阵的差异
      if(enable_scan_to_map) radar_delta = prev_trans_s2m.inverse() * trans_s2m;
      else radar_delta = prev_trans_s2s.inverse() * trans_s2s;
      double dx_rd = radar_delta.block<3, 1>(0, 3).norm();                                        // 平移部分的欧几里得范数(即平移量的长度)
      // double da_rd = std::acos(Eigen::Quaterniond(radar_delta.block<3, 3>(0, 0)).w())*180/M_PI;
      Eigen::AngleAxisd rotation_vector;
      rotation_vector.fromRotationMatrix(radar_delta.block<3, 3>(0, 0));
      double da_rd = rotation_vector.angle();
      Eigen::Matrix3d rot_rd = radar_delta.block<3, 3>(0, 0).cast<double>();
      bool too_large_trans = dx_rd > max_acceptable_trans || da_rd > max_acceptable_angle;
      double da, dx, delta_rot_imu = 0;
      Eigen::Matrix3d matrix_rot; Eigen::Vector3d delta_trans_egovel;

      if (enable_imu_thresholding) {
        // Use IMU orientation to determine whether the matching result is good or not
        sensor_msgs::Imu frame_imu;
        Eigen::Matrix3d rot_imu = Eigen::Matrix3d::Identity();
        auto result = get_closest_imu(stamp);
        if (result.first) {
          frame_imu = result.second;
          Eigen::Quaterniond imu_quat(frame_imu.orientation.w, frame_imu.orientation.x, frame_imu.orientation.y, frame_imu.orientation.z);
          Eigen::Quaterniond prev_imu_quat(last_frame_imu.orientation.w, last_frame_imu.orientation.x, last_frame_imu.orientation.y, last_frame_imu.orientation.z);
          rot_imu = (prev_imu_quat.inverse() * imu_quat).toRotationMatrix();
          Eigen::Vector3d eulerAngle_imu = rot_imu.eulerAngles(0,1,2); // roll pitch yaw
          Eigen::Vector3d eulerAngle_rd = last_radar_delta.block<3,3>(0,0).eulerAngles(0,1,2);
          Eigen::AngleAxisd rollAngle(Eigen::AngleAxisd(restrict_rad(eulerAngle_imu(0)),Eigen::Vector3d::UnitX()));
          Eigen::AngleAxisd pitchAngle(Eigen::AngleAxisd(restrict_rad(eulerAngle_imu(1)),Eigen::Vector3d::UnitY()));
          Eigen::AngleAxisd yawAngle(Eigen::AngleAxisd(restrict_rad(eulerAngle_rd(2)),Eigen::Vector3d::UnitZ()));
          matrix_rot = yawAngle * pitchAngle * rollAngle;
          da = fabs(std::acos(Eigen::Quaterniond(rot_rd.inverse() * rot_imu).w()))*180/M_PI;
          delta_rot_imu = fabs(std::acos(Eigen::Quaterniond(rot_imu).w()))*180/M_PI;
          last_frame_imu = frame_imu;
        }
        delta_trans_egovel = egovel_cum.block<3,1>(0,3).cast<double>();
        Eigen::Vector3d delta_trans_radar = radar_delta.block<3,1>(0,3).cast<double>();
        dx = (delta_trans_egovel - delta_trans_radar).norm();

        if (dx > max_diff_trans || da > max_diff_angle || too_large_trans) {
          Eigen::Matrix4d mat_est(Eigen::Matrix4d::Identity());
          mat_est.block<3, 3>(0, 0) = matrix_rot;
          mat_est.block<3, 1>(0, 3) = delta_trans_egovel;
          if (too_large_trans) cout << "Too large transform! " << dx_rd << "[m] " << da_rd << "[deg] ";
          cout << "Difference of Odom and IMU/EgoVel too large " << dx << "[m] " << da << "[deg] (" << stamp << ")" << endl;
          prev_trans_s2s = prev_trans_s2s * mat_est;
          thresholded = true;
          if (enable_scan_to_map){
            prev_trans_s2m = prev_trans_s2m * mat_est;
            odom_s2m_now = keyframe_pose_s2m * prev_trans_s2m;
          }
          else odom_s2s_now = keyframe_pose_s2s * prev_trans_s2s;
        }
      }
      else {
        if (too_large_trans) {
          cout << "Too large transform!!  " << dx_rd << "[m] " << da_rd << "[degree]"<<
            " Ignore this frame (" << stamp << ")" << endl;
          prev_trans_s2s = trans_s2s;
          thresholded = true;
          if (enable_scan_to_map){
            prev_trans_s2m = trans_s2m;
            odom_s2m_now = keyframe_pose_s2m * prev_trans_s2m * radar_delta;
          }
          else odom_s2s_now = keyframe_pose_s2s * prev_trans_s2s * radar_delta;
        }
      }
      last_radar_delta = radar_delta;
      
      if(0){
        cout << "radar trans:" << dx_rd << " m rot:" << da_rd << " degree. EgoVel " << delta_trans_egovel.norm() << " m. "
        << "IMU rot " << delta_rot_imu << " degree." << endl;
        cout << "dx " << dx << " m. da " << da << " degree." << endl;
      }
    }
    prev_time = stamp;
    if (!thresholded) {
      prev_trans_s2s = trans_s2s;
      prev_trans_s2m = trans_s2m;
    }
    
    //********** Decided whether to accept the frame as a key frame or not **********
    if(keyframe_updater->decide(Eigen::Isometry3d(odom_s2s_now), stamp)) {
      // Loose Coupling the IMU roll & pitch
      if (enable_imu_fusion){
        if(enable_scan_to_map) transformUpdate(odom_s2m_now);
        else transformUpdate(odom_s2s_now);
      }

      keyframe_cloud_s2s = filtered;
      registration_s2s->setInputTarget(keyframe_cloud_s2s);
      keyframe_pose_s2s = odom_s2s_now;
      keyframe_stamp = stamp;
      prev_time = stamp;
      prev_trans_s2s.setIdentity();

      double accum_d = keyframe_updater->get_accum_distance();
      KeyFrame::Ptr keyframe(new KeyFrame(keyframe_index, stamp, Eigen::Isometry3d(odom_s2s_now.cast<double>()), accum_d, cloud));
      keyframe_index ++;
      keyframes.push_back(keyframe);

      // record keyframe's imu
      flush_imu_queue();

      if (enable_scan_to_map){
        pcl::PointCloud<PointT>::Ptr submap_cloud(new pcl::PointCloud<PointT>());
        pcl::PointCloud<PointT>::ConstPtr submap_cloud_downsampled;
        for(size_t i=std::max(0, (int)keyframes.size()-max_submap_frames); i < keyframes.size()-1; i++){
          Eigen::Matrix4d rel_pose = keyframes.at(i)->odom_scan2scan.matrix().inverse() * keyframes.back()->odom_scan2scan.matrix();
          pcl::PointCloud<PointT>::Ptr cloud_transformed(new pcl::PointCloud<PointT>());
          pcl::transformPointCloud(*keyframes.at(i)->cloud, *cloud_transformed, rel_pose);
          *submap_cloud += *cloud_transformed;
        }
        submap_cloud_downsampled = downsample(submap_cloud);
        keyframe_cloud_s2m = submap_cloud_downsampled;
        registration_s2m->setInputTarget(keyframe_cloud_s2m);
        
        keyframes.back()->odom_scan2map = Eigen::Isometry3d(odom_s2m_now);
        keyframe_pose_s2m = odom_s2m_now;
        prev_trans_s2m.setIdentity();
      }
    }
    
    if (aligned_points_pub.getNumSubscribers() > 0)
    {
      pcl::transformPointCloud (*cloud, *aligned, odom_s2s_now);
      aligned->header.frame_id = odometryFrame;
      aligned_points_pub.publish(*aligned);
    }

    if (enable_scan_to_map)
      return odom_s2m_now;
    else
      return odom_s2s_now;
  }


  /**
   * @brief publish odometry
   * @param stamp  timestamp
   * @param pose   odometry pose to be published
   */
  void publish_odometry(const ros::Time& stamp, const std::string& father_frame_id, const std::string& child_frame_id, const Eigen::Matrix4d& pose_in, const geometry_msgs::TwistWithCovariance twist_in) {
    // publish transform stamped for IMU integration
    geometry_msgs::TransformStamped odom_trans = matrix2transform(stamp, pose_in, father_frame_id, child_frame_id); //"map" 
    trans_pub.publish(odom_trans);

    // broadcast the transform over TF
    map2odom_broadcaster.sendTransform(odom_trans);

    // publish the transform
    nav_msgs::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = father_frame_id;   // frame: /odom
    odom.child_frame_id = child_frame_id;

    odom.pose.pose.position.x = pose_in(0, 3);
    odom.pose.pose.position.y = pose_in(1, 3);
    odom.pose.pose.position.z = pose_in(2, 3);
    odom.pose.pose.orientation = odom_trans.transform.rotation;
    odom.twist = twist_in;

    odom_pub.publish(odom);
  }

  /**
   * @brief publish scan matching_ status
   */
  void publish_scan_matching_status(const ros::Time& stamp, const std::string& frame_id, pcl::PointCloud<pcl::PointXYZI>::ConstPtr aligned, const std::string& msf_source, const Eigen::Isometry3d& msf_delta) {
    if(!status_pub.getNumSubscribers()) {
      return;
    }

    ScanMatchingStatus status;
    status.header.frame_id = frame_id;
    status.header.stamp = stamp;
    status.has_converged = registration_s2s->hasConverged();
    status.matching_error = registration_s2s->getFitnessScore();

    const double max_correspondence_dist = 0.5;

    int num_inliers = 0;
    std::vector<int> k_indices;
    std::vector<float> k_sq_dists;
    for(int i=0; i<aligned->size(); i++) {
      const auto& pt = aligned->at(i);
      registration_s2s->getSearchMethodTarget()->nearestKSearch(pt, 1, k_indices, k_sq_dists);
      if(k_sq_dists[0] < max_correspondence_dist * max_correspondence_dist) {
        num_inliers++;
      }
    }
    status.inlier_fraction = static_cast<float>(num_inliers) / aligned->size();

    status.relative_pose = isometry2pose(Eigen::Isometry3d(registration_s2s->getFinalTransformation().cast<double>()));

    if(!msf_source.empty()) {
      status.prediction_labels.resize(1);
      status.prediction_labels[0].data = msf_source;

      status.prediction_errors.resize(1);
      Eigen::Isometry3d error = Eigen::Isometry3d(registration_s2s->getFinalTransformation().cast<double>()).inverse() * msf_delta;
      status.prediction_errors[0] = isometry2pose(error.cast<double>());
    }

    status_pub.publish(status);
  }

  void command_callback(const std_msgs::String& str_msg) {
    if (str_msg.data == "time") {
      std::sort(s2s_matching_time.begin(), s2s_matching_time.end());
      double median = s2s_matching_time.at(size_t(s2s_matching_time.size() / 2));
      cout << "Scan Matching time cost (median): " << median << endl;
    }
  }

private:
  // ROS topics
  ros::NodeHandle nh;
  ros::NodeHandle mt_nh;
  ros::NodeHandle private_nh;

  // ros::Subscriber points_sub;
  ros::Subscriber msf_pose_sub;
  ros::Subscriber msf_pose_after_update_sub;
  ros::Subscriber imu_sub;

  std::mutex imu_queue_mutex;
  std::deque<sensor_msgs::ImuConstPtr> imu_queue;
  sensor_msgs::Imu last_frame_imu;

  bool enable_imu_fusion;
  bool imu_debug_out;
  Eigen::Matrix3d global_orient_matrix;  // The rotation matrix with initial IMU roll & pitch measurement (yaw = 0)
    double timeLaserOdometry = 0;
    int imuPointerFront;
    int imuPointerLast;
    double imuTime[imuQueLength];
    float imuRoll[imuQueLength];
    float imuPitch[imuQueLength];
    double imu_fusion_ratio;

  std::unique_ptr<message_filters::Subscriber<geometry_msgs::TwistWithCovarianceStamped>> ego_vel_sub;
  std::unique_ptr<message_filters::Subscriber<sensor_msgs::PointCloud2>> points_sub;
  std::unique_ptr<message_filters::Synchronizer<ApproxSyncPolicy>> sync;

  // Submap
  ros::Publisher submap_pub;
  std::unique_ptr<KeyframeUpdater> keyframe_updater;
  std::vector<KeyFrame::Ptr> keyframes;
  size_t keyframe_index = 0;
  double map_cloud_resolution;
  int  max_submap_frames;
  bool enable_scan_to_map;

  // std::unique_ptr<GraphSLAM> graph_slam;
  // std::unique_ptr<InformationMatrixCalculator> inf_calclator;
  
  ros::Publisher odom_pub;
  ros::Publisher trans_pub;
  // ros::Publisher keyframe_trans_pub;
  ros::Publisher aligned_points_pub;
  ros::Publisher status_pub;
  ros::Publisher read_until_pub;
  tf::TransformListener tf_listener;
  tf::TransformBroadcaster map2odom_broadcaster; // map => odom_frame

  std::string points_topic;


  // keyframe_ parameters
  double keyframe_delta_trans;  // minimum distance between keyframes_
  double keyframe_delta_angle;  //
  double keyframe_delta_time;   //

  // registration validation by thresholding
  bool enable_transform_thresholding;  //
  bool enable_imu_thresholding;
  double max_acceptable_trans;  //
  double max_acceptable_angle;
  double max_diff_trans;
  double max_diff_angle;
  double max_egovel_cum;
  Eigen::Matrix4d last_radar_delta = Eigen::Matrix4d::Identity();

  // odometry calculation
  geometry_msgs::PoseWithCovarianceStampedConstPtr msf_pose;
  geometry_msgs::PoseWithCovarianceStampedConstPtr msf_pose_after_update;

  Eigen::Matrix4d egovel_cum = Eigen::Matrix4d::Identity();
  bool use_ego_vel;

  ros::Time prev_time;
  Eigen::Matrix4d prev_trans_s2s;                  // previous relative transform from keyframe_
  Eigen::Matrix4d keyframe_pose_s2s;               // keyframe_ pose
  Eigen::Matrix4d prev_trans_s2m;
  Eigen::Matrix4d keyframe_pose_s2m;               // keyframe_ pose
  ros::Time keyframe_stamp;                    // keyframe_ time
  pcl::PointCloud<PointT>::ConstPtr keyframe_cloud_s2s;  // keyframe_ point cloud
  pcl::PointCloud<PointT>::ConstPtr keyframe_cloud_s2m;  // keyframe_ point cloud

  // Registration
  pcl::Filter<PointT>::Ptr downsample_filter;
  pcl::Registration<PointT, PointT>::Ptr registration_s2s;    // Scan-to-Scan Registration
  pcl::Registration<PointT, PointT>::Ptr registration_s2m;    // Scan-to-Submap Registration

  // Time evaluation
  std::vector<double> s2s_matching_time;
  ros::Subscriber command_sub;
};

}  // namespace radar_graph_slam

PLUGINLIB_EXPORT_CLASS(radar_graph_slam::ScanMatchingOdometryNodelet, nodelet::Nodelet)
