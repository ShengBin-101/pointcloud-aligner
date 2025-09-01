#ifndef SB_COLOR_ICP_H
#define SB_COLOR_ICP_H

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <Eigen/Core>
#include <pcl/search/kdtree.h>
#include <memory>

namespace Eigen {
    using Matrix4d = Eigen::Matrix<double, 4, 4>;
    using Vector6d = Eigen::Matrix<double, 6, 1>;
}

namespace sb {

class ColorICP {
public:
    using PointT = pcl::PointXYZRGBNormal;
    using PointCloudPtr = pcl::PointCloud<PointT>::Ptr;
    
    ColorICP();
    ~ColorICP() {}
    
    void setSourceCloud(const PointCloudPtr& source);
    void setTargetCloud(const PointCloudPtr& target);
    void setParameters(bool downsample = false, bool estimate_normal = true,
                      double voxel_resolution = 0.01, double normal_est_radius = 0.05,
                      double search_radius = 0.04, double color_icp_lambda = 0.968,
                      double icp_max_corres_dist = 0.05);
                      
    Eigen::Matrix4d perform(const PointCloudPtr& source, const PointCloudPtr& target);
    Eigen::Matrix4d getTransformation();
    
private:
    pcl::search::KdTree<PointT>::Ptr kdtree_;
    pcl::PointCloud<PointT>::Ptr source_cloud_;
    pcl::PointCloud<PointT>::Ptr target_cloud_;
    std::vector<Eigen::Vector3d> target_color_gradient_;
    Eigen::Vector3d rgb_to_intensity_weight_;
    Eigen::Matrix4d transformation_;
    
    bool downsample_;
    bool estimate_normal_;
    double voxel_resolution_;
    double normal_est_radius_;
    double search_radius_;
    std::string icp_method_;
    double icp_max_corres_dist_;
    double icp_transformation_epsilon_;
    int icp_max_iterations_;
    double color_icp_lambda_;
    std::string color_weight_;
    bool color_visualization_;

    void removeNaNPoints(const PointCloudPtr& cloud);
    void downSampleVoxelGrids(const PointCloudPtr& cloud);
    void estimateNormals(const PointCloudPtr& cloud);
    void copyPointCloud(const PointCloudPtr& cloud_in, const std::vector<int>& indices, PointCloudPtr& cloud_out);
    void visualizeRegistration(const PointCloudPtr& source, const PointCloudPtr& source_transformed, const PointCloudPtr& target);
    void visualizeRegistrationWithColor(const PointCloudPtr& source_transformed, const PointCloudPtr& target);
    void prepareColorGradient(const PointCloudPtr& target);
    Eigen::Matrix4d ColorICPRegistration(const PointCloudPtr& source, const PointCloudPtr& target);
    Eigen::Matrix4d PCLICP(const PointCloudPtr& source, const PointCloudPtr& target);
    Eigen::Matrix4d ClassicICPRegistration(const PointCloudPtr& source, const PointCloudPtr& target);
    Eigen::Matrix4d GaussNewton(const PointCloudPtr& source, const PointCloudPtr& target);
    Eigen::Matrix4d GaussNewtonWithColor(const PointCloudPtr& source, const PointCloudPtr& target, 
                                          const std::vector<Eigen::Vector3d>& target_gradient);
    Eigen::Matrix4d TransformVector6dToMatrix4d(const Eigen::Vector6d& input);
};

} // namespace sb

#endif // SB_COLOR_ICP_H