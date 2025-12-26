# 2강 - Image Classification

nogiCategory: Lecture 2
nogiCommitDate: 2025/12/26 10:17 (GMT+9)
nogiStatus: 완료

# 키워드

- 이미지 분류, 컴퓨터 비전, 딥러닝, 기계 학습, 의미론적 간극, 픽셀 값, 관점 변화, 클래스 내 변동, 세분화된 카테고리, 배경 혼란, 조명 변화, 변형, 가려짐, 객체 탐지, 이미지 캡셔닝, 데이터셋, MNIST, CIFAR-10, CIFAR-100, ImageNet, MIT Places, Omniglot, K-최근접 이웃 (K-Nearest Neighbor), L1 거리, L2 거리, 맨해튼 거리, 유클리드 거리, 하이퍼파라미터, 훈련 세트, 검증 세트, 테스트 세트, 교차 검증, 차원의 저주, 컨볼루션 신경망(CNN), 특징 벡터.

# Image Classification

: 이미지를 input으로 받아서 카테고리 집합들 중 하나의 카테고리로 분류하는 과정

- 우리 눈에는 딱 봐도 고양이 사진 보고 고양이라고 판단할 수 있지만…
- 컴퓨터 입장에서는 쉽지 않은 task이다

## Problem: Semantic Gap(의미 격차)

![image.png](image/image.png)

- 컴퓨터가 보는 이미지는 단순히 big grid of numbers between 0~255 이다
- ex) 800 x 600 x 3 (3 channels RGB)
- 이 grid만 보고 고양이라는 정보를 얻기란 쉽지 않다
- 우리는 다양한 상황에 의해 변화하는 픽셀 값으로부터 견고한 분류 알고리즘을 설계할 필요가 있다

# Challenges

### Viewpoint Variation

![image.png](image/image%201.png)

- 이미지를 촬영하는 시점이 바뀌기만 해도 모든 pixel 값이 달라진다
- 즉, 원시 grid와 전혀 다른 새로운 grid가 생긴다

### Intraclass Variation(클래스 내 변이)

![image.png](image/image%202.png)

- 같은 고양이이더라도 종류가 다양하고 픽셀 값도 다 다름

### Fine-Grained Categories(세분화된 카테고리)

![image.png](image/image%203.png)

- 고양이 카테고리 내에서 세분화해서 고양이 품종을 분류하고 싶을 수도 있다
- 시각적으로 비슷한 것들 내에서 다른 것으로 분류하기? 쉽지 않다

### Background Clutter(배경 클러터)

![image.png](image/image%204.png)

- 객체와 배경이 비슷하다면 어떻게 객체와 배경을 분리할 수 있을까
- 배경이 선명할 수도, 뿌옇게 되어 있을 수도 있다

### Illumination Changes(조명 변경)

![image.png](image/image%205.png)

- 조명에 따라서 픽셀 값이 달라질 수도 있다

### Deformation(변형)

![image.png](image/image%206.png)

- 각 사진마다 고양이가 취하는 자세나 어디에 위치해 있는지 등등… 다 고려해야 한다
- 고양이를 귀의 위치를 보고 판단하고자 한다면
    - 단순히 이미지의 상단에 귀가 2개 있으면 고양이다 라고 분류해도 괜찮을까?
    - 이처럼 다양한 포즈의 고양이 이미지에 대해서는 귀의 위치를 가지고 분류하기 까다로움
    - CNN의 특징: Inductive bias를 잘 포착한다 → 앉아 있는 고양이 이미지 데이터셋이라면 귀가 이미지의 상단에 위치한다 라는 정형화된 편향을 포착할 수 있다면 학습하기 편해짐
        - 비슷한 특징은 인접한 영역에 있다
        - Inductive bias: 모델을 설계 시 설계자가 고정 관념으로 넣은 개념 ex. LSTM은 RNN에서 메모리 기능을 추가하면 잘 될 것이라는 설계 이념에 따라 만들어짐
    - VIT의 장점: inductive bias가 없다 → 이미지를 flatten해서 입력으로 넣기 때문에 공간 정보 사라짐 → 정형화된 편향 포착 못한다 → 그만큼 모델이 포착하는 특징이 유연함 → 잘 되는 이유는 층 여러 개 쌓고 데이터셋도 대규모로 때려 박기 때문에 잘 된다
    - VIT의 단점: 너무 오래 걸리고 데이터셋 규모가 작다면 성능이 생각보다 잘 안 나옴
    - 따라서 같은 데이터셋 양일 때 CNN이 VIT보다 성능이 잘 나오고 현업에서는 CNN 많이 사용한다
    - Q. Vertical Flip으로 augmentation 하면 cnn과 vit 잘 학습할까?

### Occlusion(폐색)

![image.png](image/image%207.png)

- 인식하려는 객체가 잘 안보일 수도 있다
- 소파 쿠션 밑에 있는 객체를 인식하기 위해서 때로는 인간의 지식이 필요할 수도 있다
    - ex) 고양이는 보통 집에 살고, 너구리는 소파 쿠션 밑에 잘 안 들어가니까 이건 고양이겠구나

# Image Classification: Very Useful!

![image.png](image/image%208.png)

## Medical Imaging

- 피부 병변 사진을 통해 악성/비악성 종양 판단
- X-ray 사진을 통해 어떤 문제가 있는지 판단

## Galaxy Classification

- 망원경 같은 것으로 사진 찍어서 어떤 유형의 현상이 있는지 분류

## Whale recognition

- 고래 인식 또는 그 외의 다양한 생물 인식

# Image Classfication: Building Block for other tasks!

## Example: Object Detection

![image.png](image/image%209.png)

- 객체를 탐지해서 Bounding Box를 그린다
- R-CNN, Fast R-CNN, Faster R-CNN

## Example: Image Captioning

![image.png](image/image%2010.png)

- 이미지를 보고 텍스트와 매핑한다
- CLIP과 같은 VLM

## Example: Playing Go

![image.png](image/image%2011.png)

# Image Classifier

## classifier 함수 정의?

- 우리는 과연 이미지를 분류하는 함수를 명시적으로 만들 수 있을까?
- 이건 단순히 리스트 내 숫자들을 정렬하는 알고리즘 같이 쉽게 만들 수 있는 문제가 아니다
- 고양이나 다른 클래스를 인식하는 알고리즘을 하드 코딩할 방법 없다
- 그래서 예전에 사람들은 Edges나 Corners를 검출하는 필터를 만들어서 적용해 보기도 했다
    - 엣지: 프리윗 필터, 소벨 필터
    - 코너: 헤리스 코너

## Machine Learning: Data-Driven Approach

### 1. Collect a dataset of images and labels

- MNIST: 숫자 손 글씨 데이터셋. 클래스 10개: 0~9. 28x28 grayscale images
- CIFAR10: 클래스 10개 분류하는 32x32 RGB 이미지 task
- CIFAR100: 클래스 100개 분류하는 32x32 RGB 이미지 task
- ImageNet: 클래스 1000개 분류하는 다양한 크기의 RGB 이미지 task
    - 성능 지표는 Top-5 accuracy → 5개 중 하나가 label과 같으면 정답으로 인정
    - 이미지 상태도 별로 안 좋고 1000개 클래스를 Top-1만 가지고 평가하기엔 너무나도 어렵다
- MIT Places: MIT 공간 클래스 365개 분류하는 다양한 크기의 RGB 이미지 task
- Omniglot: 클래스 1623개 분류하는 알파벳 문자 이미지 분류 task. few shot learning 할 때 사용

### 2. Use Machine Learning to train a classifier

- Nearest Neighbor: 가장 가까운 거리의 벡터에 해당하는 label로 예측하는 방식
    
    ```python
    import numpy as np
    
    class NearestNeighbor:
    	def __init__(self):
    		pass
    	
    	def train(self, X, y):
    		# X는 NxD 크기, 각 row마다 샘플을 의미함. y는 Nx1 크기
    		# 모델은 각 벡터마다 label이 무엇인지 알고 있음
    		self.Xtr = X
    		self.ytr = y
    	
    	def predict(self, X):
    		# X: 우리가 예측하고 싶은 NxD 크기의 데이터
    		num_test = X.shape[0]
    		# 예측한 값 저장
    		Ypred = np.zeros(num_test, dtype=self.ytr.dtype)
    		
    		for i in range(num_test):
    			# L1 (Manhattan) distance
    			distances = np.sum(np.abs(self.Xtr - X[i,:]), axis=1)
    			# L2 (Euclidean) distance
    			distances = np.sqrt(np.sum(np.square(self.Xtr - X[i,:]), axis=1))
    			# 거리 차이가 가장 작은 값의 인덱스
    			min_index = np.argmin(distances)
    			# i번째 샘플의 예측 값
    			Ypred[i] = self.ytr[min_index]
    ```
    
    - train 시간 = O(1)
    - test 시간 = O(n)
    - Bad: training 시간은 느려도 되지만, testing 시간은 빨라야 한다 → 실시간을 위해서
    
    ![image.png](image/image%2012.png)
    
- K-Nearest Neighbors: 가장 가까운 거리의 벡터 K개를 골라서 가장 많은 label로 예측하는 방식
    
    ![image.png](image/image%2013.png)
    
    - K가 많아질 수록 결정 경계가 부드러워 진다
    - K가 많아질 수록 outlier에 강해진다
    - K가 많아질 수록 빈 공간이 생긴다 → 가장 가까운 K개의 지점이 전부 다 다른 label인 경우
        - 이를 해결할 필요가 있음 → 만약 저 빈 공간에 데이터가 들어오면 하나의 label로 예측 못함
    
    ![image.png](image/image%2014.png)
    
    ![image.png](image/image%2015.png)
    
    - 그래프는 가로가 $I_1$축, 세로가 $I_2$축일 때 원점으로부터 거리가 1인 모든 점들의 집합
        - L1 distance: $|x|+|y|=1$
        - L2 distance: $\sqrt{x^2+y^2}=1$
    - L1 distance보다 L2 distance가 결정 경계가 더 세밀함 → 왜? 제곱 때문에

### 3. Evaluate the classifier on new images

- Distance Metric 예제: tf-idf
    - tf(d,t): 특정 문서 d에서 특정 단어 t의 등장 횟수
    - df(t): 특정 단어 t가 등장한 문서의 수
    - idf(t) = $log(\frac{n}{1+df(t)})$, n: 전체 문서 개수
    - tf-idf = tf * idf
    
    ```python
    docs = [
    	'나는 사과가 좋아',
    	'나는 바나나가 좋아',
    	'이건 전혀 다른 문서야'
    	]
    
    vocab = ['나는', '사과가', '좋아', '바나나가', '이건', '전혀', '다른', '문서야']
    
    '''
    tf
    ---
    |문서 번호|나는|사과가|좋아|바나나가|이건|전혀|다른|문서야|
    |   0   | 1 |  1  | 1 |  0   | 0 | 0 | 0 |  0  |
    |   1   | 1 |  0  | 1 |  1   | 0 | 0 | 0 |  0  |
    |   2   | 0 |  0  | 0 |  0   | 1 | 1 | 1 |  1  |
    
    df
    ---
    |단어|등장한 문서 수|
    |나는|2|
    |사과가|1|
    |좋아|2|
    |바나나가|1|
    |이건|1|
    |전혀|1|
    |다른|1|
    |문서야|1|
    
    tf-idf
    ---
    [
    [1*log(3/3), 1*log(3/1), 1*log(3/3), 0, 0, 0, 0, 0],
    [1*log(3/3), 0, 1*log(3/2), 1*log(3/3), 0, 0, 0, 0],
    [0, 0, 0, 0, 1*log(3/2), 1*log(3/2), 1*log(3/2), 1*log(3/2)],
    ]
    
    문서0과 문서1의 유사도?
    문서0과 문서2의 유사도?
    이 계산을 Distance Metric을 통해 진행한다
    '''
    ```
    
- Hyperparameters
    - 가장 좋은 K는 무엇인가? 가장 좋은 distance metric은 무엇인가?
    - 학습되는 값이 아닌 사용자가 모델을 학습하기 전에 미리 설정하는 값이 하이퍼파라미터
    - 문제마다 가장 좋은 하이퍼파라미터는 전부 다르다
    - 어떻게 하이퍼파라미터 잘 설정하나?
    
    ![image.png](image/image%2016.png)
    
    ![image.png](image/image%2017.png)
    
- Universal Approximation (보편적 근사)
    
    ![image.png](image/image%2018.png)
    
    - 샘플 수가 많아지면 많아질 수록 점점 더 실제 function과 근접해진다
- Problem: Curse of Dimensionality (차원의 저주)
    - 차원이 커지면 커질수록 그만큼 필요한 메모리 양도 지수적으로 증가한다
- 픽셀들 간의 K-Nearest Neighbor 잘 사용하지 않는다
    - AutoEncoder가 VAE보다 성능이 잘 안 나왔던 이유
    
    ![image.png](image/image%2019.png)
    

# Nearest Neighbor with ConvNet features works well!

![image.png](image/image%2020.png)

![image.png](image/image%2021.png)