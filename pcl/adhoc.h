#ifndef _ADHOC_H_
#define _ADHOC_H_

//#include <pcl/point_types.h>
//#include <pcl/segmentation/sac_segmentation.h>
#include<vector>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
//#include <pcl/surface/mls.h>
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/feature.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/pfh.h>

//void SimpleResample(pcl::PointCloud<pcl::PointXYZ> &input,
//                    pcl::PointCloud<pcl::PointXYZ> &output,
//                    double searchRadius,
//                    bool computeNormals)

void detection_and_correspondence(pcl::PointCloud<pcl::PointXYZ>::Ptr &points1,
                                  pcl::PointCloud<pcl::PointXYZ>::Ptr &points2,
                                  int num1, int num2,
                                  float *intensities1, float *intensities2,
                                  int normal_num_neighs, float normal_radius,
                                  int feature_num_neighs, float feature_radius,
                                  float min_scale, int nr_octaves, int nr_scales_per_octave, bool usecontrast, float min_contrast,
                                  bool useOldMethod,
                                  std::vector<int> &correspondences_out,
                                  std::vector<float> &correspondence_scores_out1,
                                  std::vector<float> &correspondence_scores_out2);

void detect_keypoints(pcl::PointCloud<pcl::PointXYZI>::Ptr &points, float min_scale,
                      int nr_octaves, int nr_scales_per_octave, bool usecontrast, float min_contrast,
                      pcl::PointCloud<pcl::PointWithScale>::Ptr &keypoints_out);


void compute_surface_normals (pcl::PointCloud<pcl::PointXYZI>::Ptr &points, int normal_num_neighs, float normal_radius,
                              pcl::PointCloud<pcl::Normal>::Ptr &normals_out);

void compute_PFH_features_at_keypoints(pcl::PointCloud<pcl::PointXYZI>::Ptr &points,
                                       pcl::PointCloud<pcl::Normal>::Ptr &normals,
                                       pcl::PointCloud<pcl::PointWithScale>::Ptr &keypoints, int feature_num_neighs, float feature_radius,
                                       pcl::PointCloud<pcl::PFHSignature125>::Ptr &descriptors_out);

void
find_feature_correspondences (pcl::PointCloud<pcl::PFHSignature125>::Ptr &source_descriptors,
                              pcl::PointCloud<pcl::PFHSignature125>::Ptr &target_descriptors,
                              std::vector<int> &correspondences_out, std::vector<float> &correspondence_scores_out);

void
find_feature_correspondences2(pcl::PointCloud<pcl::PFHSignature125>::Ptr &source_descriptors,
                              pcl::PointCloud<pcl::PFHSignature125>::Ptr &target_descriptors,
                              std::vector<int> &correspondences_out,
                              std::vector<float> &correspondence_scores_out1,
                              std::vector<float> &correspondence_scores_out2);


#endif
