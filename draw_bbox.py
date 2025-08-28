import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
import argparse
import os
import sys

def generate_bounding_boxes_from_labels(point_cloud_path, label_file_path, target_label=81, eps=0.75, min_samples=15):
    """
    포인트 클라우드와 라벨 파일로부터 특정 클래스의 3D 바운딩 박스를 생성하고 시각화합니다.

    :param point_cloud_path: 원본 .bin 파일 경로
    :param label_file_path: 예측 결과 .label 파일 경로
    :param target_label: 바운딩 박스를 생성할 목표 클래스 ID (SemanticKITTI 'traffic-sign' = 81)
    :param eps: DBSCAN 군집화를 위한 이웃 탐색 거리
    :param min_samples: 하나의 군집을 구성하기 위한 최소 포인트 수
    """
    # 1. 파일 존재 여부 확인
    if not os.path.exists(point_cloud_path):
        print(f"오류: 포인트 클라우드 파일을 찾을 수 없습니다 - {point_cloud_path}")
        return
    if not os.path.exists(label_file_path):
        print(f"오류: 라벨 파일을 찾을 수 없습니다 - {label_file_path}")
        return

    # 2. 포인트 클라우드 및 라벨 파일 로드
    try:
        # 포인트 클라우드 로드 (Nx4 형태)
        points = np.fromfile(point_cloud_path, dtype=np.float32).reshape(-1, 4)
        
        # 라벨 파일 로드 및 파싱 (시맨틱 라벨만 추출)
        pred_labels_with_instance = np.fromfile(label_file_path, dtype=np.uint32)
        # print(pred_labels_with_instance)
        # sys.exit()
        semantic_labels = pred_labels_with_instance & 0xFFFF  # 하위 16비트가 시맨틱 라벨
        
        print(f"성공: '{os.path.basename(point_cloud_path)}' ({len(points)} 포인트) 로드 완료")
        print(f"성공: '{os.path.basename(label_file_path)}' 로드 완료")

    except Exception as e:
        print(f"파일 로드 중 오류 발생: {e}")
        return

    # 3. 목표 라벨에 해당하는 포인트 필터링
    target_indices = np.where(semantic_labels == target_label)[0]
    if len(target_indices) == 0:
        print(f"'traffic sign'(ID: {target_label})으로 예측된 포인트가 없습니다.")
        # 표지판이 없어도 원본 포인트 클라우드는 시각화
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        pcd.paint_uniform_color([0.8, 0.8, 0.8])
        o3d.visualization.draw_geometries([pcd], window_name="No Traffic Signs Detected")
        return

    target_points = points[target_indices]
    print(f"{len(target_points)}개의 'traffic sign' 포인트를 찾았습니다.")

    # 4. DBSCAN을 이용한 군집화
    print("DBSCAN 군집화 진행 중...")
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(target_points[:, :3])
    cluster_labels = clustering.labels_
    
    unique_labels = set(cluster_labels)
    bounding_boxes = []

    # 5. 각 군집에 대한 바운딩 박스 생성
    for label in unique_labels:
        # -1은 노이즈로 간주되므로 제외
        if label == -1:
            continue
        
        cluster_mask = (cluster_labels == label)
        object_points = target_points[cluster_mask]
        
        # Open3D 포인트 클라우드 객체 생성
        pcd_cluster = o3d.geometry.PointCloud()
        pcd_cluster.points = o3d.utility.Vector3dVector(object_points[:, :3])
        
        # 방향성 바운딩 박스(OBB) 계산
        obb = pcd_cluster.get_oriented_bounding_box()
        obb.color = (1, 0, 0)  # 바운딩 박스는 빨간색으로 표시
        bounding_boxes.append(obb)
    
    print(f"총 {len(bounding_boxes)}개의 'traffic sign' 객체를 검출했습니다.")

    # 6. 시각화
    # 원본 포인트 클라우드 시각화 객체
    pcd_original = o3d.geometry.PointCloud()
    pcd_original.points = o3d.utility.Vector3dVector(points[:, :3])
    pcd_original.paint_uniform_color([0.8, 0.8, 0.8])  # 원본은 회색으로 표시
    
    # Ego Pose (차량 위치 및 방향) 좌표계 생성 (size: 좌표축의 길이)
    # X축: 빨간색(전방), Y축: 초록색(좌측), Z축: 파란색(상단)
    ego_pose_vis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])

    # 시각화 목록에 원본과 바운딩 박스 추가
    geometries_to_visualize = [pcd_original, ego_pose_vis] + bounding_boxes
    
    print("결과 시각화...")
    o3d.visualization.draw_geometries(geometries_to_visualize, 
                                      window_name="Traffic Sign Detection from Semantic Labels",
                                      width=1280, 
                                      height=720)


if __name__ == '__main__':
    root_dir = os.path.dirname(os.path.realpath(__file__))
    
    # ArgumentParser를 사용하여 커맨드 라인에서 파일 경로를 쉽게 입력받음
    parser = argparse.ArgumentParser(description="Visualize 3D Bounding Boxes for Traffic Signs from 2DPASS predictions.")
    parser.add_argument('--pcd_file', type=str, help='Path to the .bin point cloud file.', 
                        default=f'./semantickitti/dataset/sequences/11/velodyne/000327.bin')
    parser.add_argument('--label_file', type=str, help='Path to the .label prediction file.', 
                        default=f'./subproblem2_lidar_camera/2DPASS/checkpoints/submit_2025_08_26/sequences/11/predictions/000327.label')
    '''
    subproblem1_lidar_only/LSK3DNet/output_skitti
    subproblem2_lidar_camera/2DPASS/checkpoints/submit_2025_08_26
    '''
    
    # DBSCAN 파라미터도 조절 가능하게 추가
    parser.add_argument('--eps', type=float, default=0.75, help='DBSCAN epsilon parameter.')
    parser.add_argument('--min_samples', type=int, default=15, help='DBSCAN min_samples parameter.')

    args = parser.parse_args()

    # 메인 함수 실행
    generate_bounding_boxes_from_labels(
        point_cloud_path=os.path.join(args.pcd_file),
        label_file_path=os.path.join(args.label_file),
        eps=args.eps,
        min_samples=args.min_samples
    )