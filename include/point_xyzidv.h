#ifndef POINT_XYZIDV_H
#define POINT_XYZIDV_H

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

// 定义自定义点类型
namespace pcl
{
  struct PointXYZIDV
  {
    PCL_ADD_POINT4D;            // 这个宏定义将会添加成员 x, y, z 和 padding
    float intensity;            // 强度信息
    float doppler_velocity;     // 多普勒速度信息

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // 确保内存对齐
  } EIGEN_ALIGN16;                    // 强制16字节对齐
}

// 注册自定义点类型
POINT_CLOUD_REGISTER_POINT_STRUCT(pcl::PointXYZIDV,
                                  (float, x, x)
                                  (float, y, y)
                                  (float, z, z)
                                  (float, intensity, intensity)
                                  (float, doppler_velocity, doppler_velocity))

#endif // CUSTOM_POINT_TYPES_H
