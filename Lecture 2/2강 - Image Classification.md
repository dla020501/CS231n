# 키워드<br>  
* 이미지 분류, 컴퓨터 비전, 딥러닝, 기계 학습, 의미론적 간극, 픽셀 값, 관점 변화, 클래스 내 변동, 세분화된 카테고리, 배경 혼란, 조명 변화, 변형, 가려짐, 객체 탐지, 이미지 캡셔닝, 데이터셋, MNIST, CIFAR-10, CIFAR-100, ImageNet, MIT Places, Omniglot, K-최근접 이웃 (K-Nearest Neighbor), L1 거리, L2 거리, 맨해튼 거리, 유클리드 거리, 하이퍼파라미터, 훈련 세트, 검증 세트, 테스트 세트, 교차 검증, 차원의 저주, 컨볼루션 신경망(CNN), 특징 벡터.  
# Image Classification<br>  
: 이미지를 input으로 받아서 카테고리 집합들 중 하나의 카테고리로 분류하는 과정  
* 우리 눈에는 딱 봐도 고양이 사진 보고 고양이라고 판단할 수 있지만…  
* 컴퓨터 입장에서는 쉽지 않은 task이다  
## Problem: Semantic Gap(의미 격차)<br>  
![IMAGE](https://raw.githubusercontent.com/nogi-bot/resources/main/dla020501/images/86aef4d5-9d49-42ab-95ac-c24e35aa7a43-image.png)  
* 컴퓨터가 보는 이미지는 단순히 big grid of numbers between 0~255 이다  
* ex) 800 x 600 x 3 (3 channels RGB)  
* 이 grid만 보고 고양이라는 정보를 얻기란 쉽지 않다  
* 우리는 다양한 상황에 의해 변화하는 픽셀 값으로부터 견고한 분류 알고리즘을 설계할 필요가 있다  
# Challenges<br>  
### Viewpoint Variation<br>  
![IMAGE](https://raw.githubusercontent.com/nogi-bot/resources/main/dla020501/images/2a124bbe-9788-43f0-8de0-a610318b3f66-image.png)  
* 이미지를 촬영하는 시점이 바뀌기만 해도 모든 pixel 값이 달라진다  
* 즉, 원시 grid와 전혀 다른 새로운 grid가 생긴다  
### Intraclass Variation(클래스 내 변이)<br>  
![IMAGE](https://raw.githubusercontent.com/nogi-bot/resources/main/dla020501/images/9c6953d0-8df0-4188-b04d-0250dcbd9e3f-image.png)  
* 같은 고양이이더라도 종류가 다양하고 픽셀 값도 다 다름  
### Fine-Grained Categories(세분화된 카테고리)<br>  
![IMAGE](https://raw.githubusercontent.com/nogi-bot/resources/main/dla020501/images/9e12bc90-36e8-47c1-b943-9fbb3a178721-image.png)  
* 고양이 카테고리 내에서 세분화해서 고양이 품종을 분류하고 싶을 수도 있다  
* 시각적으로 비슷한 것들 내에서 다른 것으로 분류하기? 쉽지 않다  
### Background Clutter(배경 클러터)<br>  
![IMAGE](https://raw.githubusercontent.com/nogi-bot/resources/main/dla020501/images/e2614eba-fe39-4af4-9cd9-b63a5cdc6e2b-image.png)  
* 객체와 배경이 비슷하다면 어떻게 객체와 배경을 분리할 수 있을까  
* 배경이 선명할 수도, 뿌옇게 되어 있을 수도 있다  
### Illumination Changes(조명 변경)<br>  
![IMAGE](https://raw.githubusercontent.com/nogi-bot/resources/main/dla020501/images/44a97658-3b39-4fd6-b1c6-e86a1b9276e0-image.png)  
* 조명에 따라서 픽셀 값이 달라질 수도 있다  
### Deformation(변형)<br>  
![IMAGE](https://raw.githubusercontent.com/nogi-bot/resources/main/dla020501/images/b5ec7106-3ca0-4eb1-86e1-d2d05bf09b44-image.png)  
* 각 사진마다 고양이가 취하는 자세나 어디에 위치해 있는지 등등… 다 고려해야 한다  
* 고양이를 귀의 위치를 보고 판단하고자 한다면  
### Occlusion(폐색)<br>  
![IMAGE](https://raw.githubusercontent.com/nogi-bot/resources/main/dla020501/images/a5b4b9dd-1274-4bfa-8f0b-2f5639a379c0-image.png)  
* 인식하려는 객체가 잘 안보일 수도 있다  
* 소파 쿠션 밑에 있는 객체를 인식하기 위해서 때로는 인간의 지식이 필요할 수도 있다  
# Image Classification: Very Useful!<br>  
![IMAGE](https://raw.githubusercontent.com/nogi-bot/resources/main/dla020501/images/0a3ed891-e10f-44b7-a86d-f3baf8b366af-image.png)  
## Medical Imaging<br>  
* 피부 병변 사진을 통해 악성/비악성 종양 판단  
* X-ray 사진을 통해 어떤 문제가 있는지 판단  
## Galaxy Classification<br>  
* 망원경 같은 것으로 사진 찍어서 어떤 유형의 현상이 있는지 분류  
## Whale recognition<br>  
* 고래 인식 또는 그 외의 다양한 생물 인식  
# Image Classfication: Building Block for other tasks!<br>  
## Example: Object Detection<br>  
![IMAGE](https://raw.githubusercontent.com/nogi-bot/resources/main/dla020501/images/f7c507ec-1520-45db-82de-2ab59792af2c-image.png)  
* 객체를 탐지해서 Bounding Box를 그린다  
* R-CNN, Fast R-CNN, Faster R-CNN  
## Example: Image Captioning<br>  
![IMAGE](https://raw.githubusercontent.com/nogi-bot/resources/main/dla020501/images/085f0882-23b2-4022-a7e5-398ea2444ae3-image.png)  
* 이미지를 보고 텍스트와 매핑한다  
* CLIP과 같은 VLM  
## Example: Playing Go<br>  
![IMAGE](https://raw.githubusercontent.com/nogi-bot/resources/main/dla020501/images/1cad73a1-82d3-4456-931d-545ef485b04c-image.png)  
# Image Classifier<br>  
## classifier 함수 정의?<br>  
* 우리는 과연 이미지를 분류하는 함수를 명시적으로 만들 수 있을까?  
* 이건 단순히 리스트 내 숫자들을 정렬하는 알고리즘 같이 쉽게 만들 수 있는 문제가 아니다  
* 고양이나 다른 클래스를 인식하는 알고리즘을 하드 코딩할 방법 없다  
* 그래서 예전에 사람들은 Edges나 Corners를 검출하는 필터를 만들어서 적용해 보기도 했다  
## Machine Learning: Data-Driven Approach<br>  
### 1. Collect a dataset of images and labels<br>  
* MNIST: 숫자 손 글씨 데이터셋. 클래스 10개: 0~9. 28x28 grayscale images  
* CIFAR10: 클래스 10개 분류하는 32x32 RGB 이미지 task  
* CIFAR100: 클래스 100개 분류하는 32x32 RGB 이미지 task  
* ImageNet: 클래스 1000개 분류하는 다양한 크기의 RGB 이미지 task  
* MIT Places: MIT 공간 클래스 365개 분류하는 다양한 크기의 RGB 이미지 task  
* Omniglot: 클래스 1623개 분류하는 알파벳 문자 이미지 분류 task. few shot learning 할 때 사용  
### 2. Use Machine Learning to train a classifier<br>  
* Nearest Neighbor: 가장 가까운 거리의 벡터에 해당하는 label로 예측하는 방식  
* K-Nearest Neighbors: 가장 가까운 거리의 벡터 K개를 골라서 가장 많은 label로 예측하는 방식  
### 3. Evaluate the classifier on new images<br>  
* Distance Metric 예제: tf-idf  
* Hyperparameters  
* Universal Approximation (보편적 근사)  
* Problem: Curse of Dimensionality (차원의 저주)  
* 픽셀들 간의 K-Nearest Neighbor 잘 사용하지 않는다  
# Nearest Neighbor with ConvNet features works well!<br>  
![IMAGE](https://raw.githubusercontent.com/nogi-bot/resources/main/dla020501/images/c3c29938-1139-4885-871b-277831f4ae7b-image.png)  
![IMAGE](https://raw.githubusercontent.com/nogi-bot/resources/main/dla020501/images/fd0bd816-2e03-4f4e-8ab4-f22c141a44e6-image.png)  
  
