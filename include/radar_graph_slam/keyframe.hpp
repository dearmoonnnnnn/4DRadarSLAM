// SPDX-License-Identifier: BSD-2-Clause

#ifndef KEYFRAME_HPP
#define KEYFRAME_HPP

#include <ros/ros.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <boost/optional.hpp>

#include <geometry_msgs/Transform.h>
#include <sensor_msgs/Imu.h>

#include <g2o/edge_se3_priorz.hpp>

namespace g2o {
class VertexSE3;
class HyperGraph;
class SparseOptimizer;
}  // namespace g2o

namespace radar_graph_slam {

/**
 * @brief KeyFrame (pose node)
 */
struct KeyFrame {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using PointT = pcl::PointXYZI;
  using Ptr = std::shared_ptr<KeyFrame>;

  // 通过索引、时间戳、位姿、累计距离、点云构建 KeyFrame
  KeyFrame(const size_t index, const ros::Time& stamp, const Eigen::Isometry3d& odom_scan2scan, double accum_distance, const pcl::PointCloud<PointT>::ConstPtr& cloud);
  // 从目录中加载关键帧，并将其添加到图优化中
  KeyFrame(const std::string& directory, g2o::HyperGraph* graph);
  virtual ~KeyFrame();

  // 将关键帧数据保存到指定目录中
  void save(const std::string& directory);
  // 从指定目录中加载关键帧数据，并将其添加到位姿图中
  bool load(const std::string& directory, g2o::HyperGraph* graph);

  long id() const;                      // 返回关键帧的唯一标识符
  Eigen::Isometry3d estimate() const;   // 返回当前关键帧的位姿

public:
  size_t index;
  ros::Time stamp;                                // timestamp
  Eigen::Isometry3d odom_scan2scan;               // odometry (estimated by scan_matching_odometry)
  Eigen::Isometry3d odom_scan2map;
  double accum_distance;                          // accumulated distance from the first node (by scan_matching_odometry)
  pcl::PointCloud<PointT>::ConstPtr cloud;        // point cloud
  boost::optional<Eigen::Vector4d> floor_coeffs;  // detected floor's coefficients
  boost::optional<Eigen::Vector3d> utm_coord;     // UTM coord obtained by GPS
  boost::optional<Eigen::Vector1d> altitude;      // Altitude (Filtered) obtained by Barometer

  boost::optional<Eigen::Vector3d> acceleration;    //
  boost::optional<Eigen::Quaterniond> orientation;  //

  geometry_msgs::Transform trans_integrated; // relative transform obtained by imu preintegration

  boost::optional<sensor_msgs::Imu> imu; // the IMU message close to keyframe_

  g2o::VertexSE3* node;  // node instance
};

/**
 * @brief KeyFramesnapshot for map cloud generation
 */
struct KeyFrameSnapshot {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using PointT = KeyFrame::PointT;
  using Ptr = std::shared_ptr<KeyFrameSnapshot>;

  KeyFrameSnapshot(const KeyFrame::Ptr& key);
  KeyFrameSnapshot(const Eigen::Isometry3d& pose, const pcl::PointCloud<PointT>::ConstPtr& cloud);

  ~KeyFrameSnapshot();

public:
  Eigen::Isometry3d pose;                   // pose estimated by graph optimization
  pcl::PointCloud<PointT>::ConstPtr cloud;  // point cloud
};

}  // namespace radar_graph_slam

#endif  // KEYFRAME_HPP
