// 防止头文件重复包含，确保头文件只被编译一次
#ifndef DBSCAN_KDTREE_H
#define DBSCAN_KDTREE_H

// 包含了PointT类型的定义和DBSCANSimpleCluster类的声明或定义
#include <pcl/point_types.h>
#include "DBSCAN_simple.h"

// 定义模板类，模板参数是PointT，即数据点的类型
template <typename PointT>
class DBSCANKdtreeCluster: public DBSCANSimpleCluster<PointT> { // 继承自'DBSCANSimpleCluster<PointT>'，使得DBSCANKdtreeCluster可以重用父类的成员函数和数据成员
protected:
    // 重写虚函数，用于在k-d树中进行半径搜索
    virtual int radiusSearch (
        int index, double radius, std::vector<int> &k_indices,
        std::vector<float> &k_sqr_distances) const
    {
        // this->search_method->radiusSearch调用父类DBSCANSimpleCluster的同名函数，实现了对父类函数的重写
        return this->search_method_->radiusSearch(index, radius, k_indices, k_sqr_distances);
    }

}; // class DBSCANCluster

#endif // DBSCAN_KDTREE_H
