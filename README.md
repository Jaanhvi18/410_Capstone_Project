# 410_Capstone_Project

Submissions are evaluated on the mean Average Accuracy (mAA) of the registered camera centers C=−RTT
. Given the set of cameras of a scene, parameterized by their rotation matrices R
 and translation vectors T
, and the hidden ground truth, we compute the best similarity transformation T
 (i.e. scale, rotation and translation altogether) that is able to register onto the ground-truth the highest number of cameras starting from triplets of corresponding camera centers.

A camera is registered if ||Cg−T(C)||<t
, where Cg
 is the ground-truth camera center corresponding to C
 and t
 is a given threshold. Using a RANSAC-like approach, all the possible (N3)
 feasible similarity transformations T′
 that can be derived by the Horn's method on triplets of corresponding camera centers (C,Cg)
 are verified exhaustively, where N
 is the number of cameras in the scene. Each transformation T′
 is further refined into T′′
 by registering again the camera centers by the Horn's method, but including at this time the previous registered cameras together with the initial triplets. The best model T
 among all T′′
 with the highest number of registered cameras is finally returned.

Assuming that ri
 is the percentage of the cameras in a scene, excluding the original camera center triplets, successfully registered by Ti
 setting a threshold ti
, the mAA for that scene is computed by averaging ri
 among several thresholds ti
. The thresholds ti
 employed range from roughly 1 cm to 1 m according to the kind of scene. The final score is obtained by averaging the mAA among all the scenes in the dataset.

Submission File
For each image ID in the test set, you must predict its pose. The file should contain a header and have the following format:

image_path,dataset,scene,rotation_matrix,translation_vector
da1/sc1/images/im1.png,da1,sc1,0.1;0.2;0.3;0.4;0.5;0.6;0.7;0.8;0.9,0.1;0.2;0.3
da1/sc2/images/im2.png,da1,sc1,0.1;0.2;0.3;0.4;0.5;0.6;0.7;0.8;0.9,0.1;0.2;0.3
etc
The rotation_matrix (a 3x3 matrix) and translation_vector (a 3-D vector) are written as ;-separated vectors. Matrices are flattened into vectors in row-major order. Note that this metric does not require the intrinsics (the calibration matrix K
), usually estimated along with R
 and T
 during the 3D reconstruction process.

Rows that correspond to images without a predicted pose must contain at least one nan value in the corresponding rotation_matrix or translation_vector columns:

dax/scy/images/im_unregistered.png,dax,scy,0.1;0.2;0.3;0.4;0.5;0.6;0.7;0.8;nan,0.1;0.2;0.3
