import open3d as o3d
import numpy as np
import os
import cv2
import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser(description=" use a rgb and depth image to get a criteria judgement standard")
    parser.add_argument("--rgb", default=" ", help=" run mask to choose points in depth ")
    parser.add_argument("--depth", default=" ", help=" get a pointcloud from mask ")
    parser.add_argument("--pc", default=" ", help=" target pointcloud ")

    args = parser.parse_args()
    return args


def get_pc(depth, rgb):
    w, h,_= depth.shape
    x, y = np.meshgrid(np.array([i for i in range(h)]), np.array([i for i in range(w)]))
    x = x - 320 + 0.5
    y = y -240 + 0.5
    xf = depth[:,:,0]/523.627
    yf = depth[:,:,0]/523.627
    x = x * xf
    y = y * yf
    cloud = np.zeros((w,h,3))
    cloud[:,:,0] = x
    cloud[:,:,1] = y
    cloud[:,:,2] = depth[:,:,0]
    cloud = cloud.reshape(-1, 3)
    cloud = cloud/60
    D = np.zeros((w,h,4))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    ab = lab[:,:,1:3]
    numColors = 2  
    data = ab.reshape((-1,2))
    data = np.float32(data)
    criteria = (cv2.TERM_CRITERIA_EPS +cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_PP_CENTERS
    compactness, L2, centers = cv2.kmeans(data,numColors,None,criteria, 10, flags)
    if L2[0]==1:
        L2[L2==1] = 2
        L2[L2==0] = 1
        L2[L2==2] = 0 
    cloud_l = cloud[(L2 == 1)[:,0], :]
    points_norm = np.linalg.norm(cloud_l, axis=1, keepdims=True)
    index = (points_norm < 1.1*np.mean(points_norm)) & (points_norm >0.7*np.mean(points_norm))
    cloud = cloud_l[index[:,0], :]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud)
    #pcd.paint_uniform_color([1.0, 0.5, 0.0])
    return pcd

def pc_reg(pc1,pc2):
    threshold = 1.0
    trans_init = np.array([[1,0,0,0],
                           [0,1,0,0],   
                           [0,0,1,0],   
                           [0,0,0,1]])

    reg_p2p = o3d.pipelines.registration.registration_icp(
            pc1, pc2, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())

    print(reg_p2p)
    pc1.transform(reg_p2p.transformation)
    return pc1


def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
 
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def prepare_dataset(source, target, voxel_size):
    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh

def main(args):
    if args.rgb == " ":
        print("  Error: please input a rgb image")
        sys.exit(1)
    if args.depth == " ":
        print("  Error: please input a depth image")
        sys.exit(1)
    if args.pc == " ":
        print("  Error: please input a pointcloud")
        sys.exit(1)

    depth = cv2.imread(args.depth, -1)
    rgb   = cv2.imread(args.rgb, -1)
    lego_cam  = get_pc(depth, rgb)
    lego_mesh = o3d.io.read_point_cloud(args.pc)
    o3d.visualization.draw_geometries([lego_cam])
    
    voxel =0.005
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(lego_cam, lego_mesh, voxel)
    distance_threshold = voxel * 1.5

    result_fast = o3d.pipelines.registration.registration_fast_based_on_feature_matching(source, target,
                                                                                    source_fpfh, target_fpfh,
                                                                                    o3d.pipelines.registration.FastGlobalRegistrationOption(
                                                                                        maximum_correspondence_distance=distance_threshold))
    #print(result_fast)
    lego_cam.transform(result_fast.transformation)
    lego_mesh.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.08, max_nn=100))
    distance_threshold = 0.05 * 4
    result_icp = o3d.pipelines.registration.registration_icp(lego_cam, lego_mesh, distance_threshold, np.identity(4),
                                                         o3d.pipelines.registration.TransformationEstimationPointToPlane())
    #print(result_icp)
    #print(" The score of match: (lower and better) ", result_icp.inlier_rmse)
    lego_cam.transform(result_icp.transformation)
    o3d.visualization.draw_geometries([lego_mesh])
    o3d.visualization.draw_geometries([lego_cam])
    o3d.visualization.draw_geometries([lego_cam,lego_mesh])
    return result_icp.inlier_rmse



if __name__ == "__main__":
    args = parse_args()
    score = main(args)
    print(" The score of match: (lower and better) ", score)