#include "adhoc.h"

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
                                  std::vector<float> &correspondence_scores_out2)
{


  pcl::PointCloud<pcl::PointXYZI>::Ptr p1 (new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::Normal>::Ptr normals1 (new pcl::PointCloud<pcl::Normal>);
  pcl::PointCloud<pcl::PointWithScale>::Ptr keypoints1 (new pcl::PointCloud<pcl::PointWithScale>);
  pcl::PointCloud<pcl::PFHSignature125>::Ptr descriptors1 (new pcl::PointCloud<pcl::PFHSignature125>);

  
  pcl::PointCloud<pcl::PointXYZI>::Ptr p2 (new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::Normal>::Ptr normals2 (new pcl::PointCloud<pcl::Normal>);
  pcl::PointCloud<pcl::PointWithScale>::Ptr keypoints2 (new pcl::PointCloud<pcl::PointWithScale>);
  pcl::PointCloud<pcl::PFHSignature125>::Ptr descriptors2 (new pcl::PointCloud<pcl::PFHSignature125>);


  pcl::copyPointCloud (*points1, *p1);
  pcl::copyPointCloud (*points2, *p2);
  for (int i=0; i<num1; i++) {
    p1->points[i].intensity = intensities1[i];
  }
  for (int i=0; i<num2; i++) {
    p2->points[i].intensity = intensities2[i];
  }


  compute_surface_normals (p1, normal_num_neighs, normal_radius, normals1);
  compute_surface_normals (p2, normal_num_neighs, normal_radius, normals2);


  detect_keypoints(p1, min_scale, nr_octaves, nr_scales_per_octave, usecontrast, min_contrast, keypoints1);
  detect_keypoints(p2, min_scale, nr_octaves, nr_scales_per_octave, usecontrast, min_contrast, keypoints2);


  compute_PFH_features_at_keypoints(p1, normals1, keypoints1, feature_num_neighs, feature_radius, descriptors1);
  compute_PFH_features_at_keypoints(p2, normals2, keypoints2, feature_num_neighs, feature_radius, descriptors2);

  if (useOldMethod) {
    find_feature_correspondences2(descriptors1, descriptors2, correspondences_out, correspondence_scores_out1, correspondence_scores_out2);
  } else {
     find_feature_correspondences(descriptors1, descriptors2, correspondences_out, correspondence_scores_out1);
  }

}


void detect_keypoints(pcl::PointCloud<pcl::PointXYZI>::Ptr &points, float min_scale,
                      int nr_octaves, int nr_scales_per_octave, bool usecontrast, float min_contrast,
                      pcl::PointCloud<pcl::PointWithScale>::Ptr &keypoints_out)
{
  pcl::SIFTKeypoint<pcl::PointXYZI, pcl::PointWithScale> sift_detect;
  // Use a FLANN-based KdTree to perform neighborhood searches
  //sift_detect.setSearchMethod (pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr (new pcl::KdTreeFLANN<pcl::PointXYZI>));
  sift_detect.setSearchMethod (pcl::search::KdTree<pcl::PointXYZI>::Ptr (new pcl::search::KdTree<pcl::PointXYZI>));
  // Set the detection parameters
  sift_detect.setScales (min_scale, nr_octaves, nr_scales_per_octave);
  if (usecontrast) {
    sift_detect.setMinimumContrast (min_contrast);
  }
  // Set the input
  sift_detect.setInputCloud (points);
  // Detect the keypoints and store them in "keypoints_out"
  sift_detect.compute (*keypoints_out);
}

void compute_surface_normals (pcl::PointCloud<pcl::PointXYZI>::Ptr &points, int normal_num_neighs, float normal_radius,
                              pcl::PointCloud<pcl::Normal>::Ptr &normals_out)
{
  pcl::NormalEstimation<pcl::PointXYZI, pcl::Normal> norm_est;

  // Use a FLANN-based KdTree to perform neighborhood searches
  //norm_est.setSearchMethod (pcl::KdTreeFLANN<pcl::PointXYZRGB>::Ptr (new pcl::KdTreeFLANN<pcl::PointXYZRGB>));
  norm_est.setSearchMethod (pcl::search::KdTree<pcl::PointXYZI>::Ptr (new pcl::search::KdTree<pcl::PointXYZI>));


  // Specify the size of the local neighborhood to use when computing the surface normals
  if (normal_num_neighs >= 0) {
    norm_est.setKSearch(normal_num_neighs);
  } else {
    norm_est.setRadiusSearch (normal_radius);
  }

  // Set the input points
  norm_est.setInputCloud (points);

  // Estimate the surface normals and store the result in "normals_out"
  norm_est.compute (*normals_out);
}

void compute_PFH_features_at_keypoints(pcl::PointCloud<pcl::PointXYZI>::Ptr &points,
                                       pcl::PointCloud<pcl::Normal>::Ptr &normals,
                                       pcl::PointCloud<pcl::PointWithScale>::Ptr &keypoints, int feature_num_neighs, float feature_radius,
                                       pcl::PointCloud<pcl::PFHSignature125>::Ptr &descriptors_out)
{
  // Create a PFHEstimation object
  pcl::PFHEstimation<pcl::PointXYZI, pcl::Normal, pcl::PFHSignature125> pfh_est;
  // Set it to use a FLANN-based KdTree to perform its
  // neighborhood searches
  //pfh_est.setSearchMethod(pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr (new pcl::KdTreeFLANN<pcl::PointXYZI>));
  pfh_est.setSearchMethod (pcl::search::KdTree<pcl::PointXYZI>::Ptr (new pcl::search::KdTree<pcl::PointXYZI>));

  // Specify the radius of the PFH feature
  if (feature_num_neighs >= 0) {
    pfh_est.setKSearch(feature_num_neighs);
  } else {
    pfh_est.setRadiusSearch (feature_radius);
  }
  // Convert the keypoints cloud from PointWithScale to PointXYZ
  // so that it will be compatible with our original point cloud
  pcl::PointCloud<pcl::PointXYZI>::Ptr keypoints_xyzrgb
  (new pcl::PointCloud<pcl::PointXYZI>);
  pcl::copyPointCloud (*keypoints, *keypoints_xyzrgb);
  // Use all of the points for analyzing the local structure of the cloud
  pfh_est.setSearchSurface (points);
  pfh_est.setInputNormals (normals);
  // But only compute features at the keypoints
  pfh_est.setInputCloud (keypoints_xyzrgb);
  // Compute the features
  pfh_est.compute (*descriptors_out);
}

void
find_feature_correspondences (pcl::PointCloud<pcl::PFHSignature125>::Ptr &source_descriptors,
                              pcl::PointCloud<pcl::PFHSignature125>::Ptr &target_descriptors,
                              std::vector<int> &correspondences_out, std::vector<float> &correspondence_scores_out)
{
  // Resize the output vector
  correspondences_out.resize (source_descriptors->size ());
  correspondence_scores_out.resize (source_descriptors->size ());

  // Use a KdTree to search for the nearest matches in feature space
  pcl::search::KdTree<pcl::PFHSignature125> descriptor_kdtree;
  descriptor_kdtree.setInputCloud (target_descriptors);

  // Find the index of the best match for each keypoint, and store it in "correspondences_out"
  const int k = 1;
  std::vector<int> k_indices (k);
  std::vector<float> k_squared_distances (k);
  for (size_t i = 0; i < source_descriptors->size (); ++i)
  {
    descriptor_kdtree.nearestKSearch (*source_descriptors, i, k, k_indices, k_squared_distances);
    correspondences_out[i] = k_indices[0];
    correspondence_scores_out[i] = k_squared_distances[0];
  }
}

void
find_feature_correspondences2(pcl::PointCloud<pcl::PFHSignature125>::Ptr &source_descriptors,
                              pcl::PointCloud<pcl::PFHSignature125>::Ptr &target_descriptors,
                              std::vector<int> &correspondences_out,
                              std::vector<float> &correspondence_scores_out1,
                              std::vector<float> &correspondence_scores_out2)
{
  // Resize the output vector
  correspondences_out.resize (source_descriptors->size ());
  correspondence_scores_out1.resize (source_descriptors->size ());
  correspondence_scores_out2.resize (source_descriptors->size ());

  // Use a KdTree to search for the nearest matches in feature space
  pcl::search::KdTree<pcl::PFHSignature125> descriptor_kdtree;
  descriptor_kdtree.setInputCloud (target_descriptors);

  // Find the index of the best match for each keypoint, and store it in "correspondences_out"
  const int k = 2;
  std::vector<int> k_indices (k);
  std::vector<float> k_squared_distances (k);
  for (size_t i = 0; i < source_descriptors->size (); ++i)
  {
    descriptor_kdtree.nearestKSearch (*source_descriptors, i, k, k_indices, k_squared_distances);
    correspondences_out[i] = k_indices[0];
    correspondence_scores_out1[i] = k_squared_distances[0];
    correspondence_scores_out2[i] = k_squared_distances[1];
  }
}


/*// Output has the PointNormal type in order to store the normals calculated by MLS
void SimpleResample(pcl::PointCloud<pcl::PointXYZ> &input, pcl::PointCloud<pcl::PointXYZ> &output, double searchRadius, bool computeNormals) {

  // Create a KD-Tree
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);

  // Init object 
  pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointXYZ> mls;

  // Set parameters
  mls.setComputeNormals (computeNormals);
  mls.setInputCloud (input);
  mls.setPolynomialFit (true);
  mls.setSearchMethod (tree);
  mls.setSearchRadius (searchRadius);

  //The default resampling method is NONE, which justs projects the original points onto the estimated surface
  //see http://docs.pointclouds.org/trunk/classpcl_1_1_moving_least_squares.html#ac29ad97b98353d64ce64e2ff924f7d20

  // Reconstruct
  mls.process (output);

}*/


//VERSION WITH POINTNORMALS, VERY INCONVENIENT FOR THE EXISTING PYTHON BINDINGS
/*#include <pcl/common/io.h>

// Output has the PointNormal type in order to store the normals calculated by MLS
void SimpleResample(pcl::PointCloud<pcl::PointXYZ> &input, pcl::PointCloud<pcl::PointXYZ> &output, double searchRadius, bool computeNormals) {

  // Create a KD-Tree
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);

  // Init object (second point type is for the normals, even if unused)
  pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointNormal> mls;

  // Output has the PointNormal type in order to store the normals calculated by MLS
  pcl::PointCloud<pcl::PointNormal> mls_points;

  // Set parameters
  mls.setComputeNormals (computeNormals);
  mls.setInputCloud (input);
  mls.setPolynomialFit (true);
  mls.setSearchMethod (tree);
  mls.setSearchRadius (searchRadius);

  //The default resampling method is NONE, which justs projects the original points onto the estimated surface
  //see http://docs.pointclouds.org/trunk/classpcl_1_1_moving_least_squares.html#ac29ad97b98353d64ce64e2ff924f7d20

  // Reconstruct
  mls.process (mls_points);

  //transfer to output
  //http://answers.ros.org/question/61642/pointnormal-to-pointxyz-transfer-problem/
  pcl::copyPointCloud(mls_points, output);

}*/


/*int
main (int argc, char** argv)
{
  // Load input file into a PointCloud<T> with an appropriate type
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ> ());
  // Load bun0.pcd -- should be available with the PCL archive in test 
  pcl::io::loadPCDFile ("bun0.pcd", *cloud);

  // Create a KD-Tree
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);

  // Output has the PointNormal type in order to store the normals calculated by MLS
  pcl::PointCloud<pcl::PointNormal> mls_points;

  // Init object (second point type is for the normals, even if unused)
  pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointNormal> mls;
 
  mls.setComputeNormals (true);

  // Set parameters
  mls.setInputCloud (cloud);
  mls.setPolynomialFit (true);
  mls.setSearchMethod (tree);
  mls.setSearchRadius (0.03);

  // Reconstruct
  mls.process (mls_points);

  // Save output
  pcl::io::savePCDFile ("bun0-mls.pcd", mls_points);
}*/