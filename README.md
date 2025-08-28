LiDAR-Camera based 3D Object Detection for Traffic Sign
===================================================

본 프로젝트는 자율주행 환경에서 LiDAR-Camera 센서 융합을 통해 교통 표지판(traffic sign) 객체를 3D로 검출하는 것을 목표로 하며, SemanticKITTI 데이터셋, 최신 3D Semantic Segmentation 모델(LSK3DNet, 2DPASS), 바운딩 박스 생성 및 시각화 코드를 활용한 실험 및 분석을 포함합니다.   

목차
---
- 프로젝트 개요
- 아키텍처 다이어그램
- 개발 환경 및 의존성
- 프로젝트 실행 방법
- 세부 구현 내용
- 참고 논문 및 Repo   

프로젝트 개요
----------
과제의 목표는 LiDAR 및 LiDAR-Camera 센서 데이터를 활용하여 SemanticKITTI 데이터셋 내의 '교통 표지판(Traffic Sign)' 객체를 3D로 검출하는 것입니다.   

과제에서 제시된 세부 문제들을 해결하기 위해 다음과 같이 두 가지 접근 방식을 사용했습니다.
- 세부문제 1(LiDAR Only): LiDAR 포인트 클라우드만을 사용하여 객체를 탐지하기 위해 LSK3DNet 모델을 활용했습니다.   
- 세부문제 2(LiDAR-Camera Fusion): LiDAR와 Camera 데이터를 함께 사용하기 위해 2DPASS 모델을 활용했습니다.

본 프로젝트