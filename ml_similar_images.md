# similar images 

## 이미지 변환
1. 이미지를 벡터화
	- im = mh.imread(img)
	- img는 이미지의 경로
	- 이미지파일의 형식까지 포함
2. 벡터화환 이미지를 gray로 변환 
	- im = mh.color.rgb2gray(img,dtype=np.uint8)
3. gray로 변환한 이미지를 haralick 알고리즘으로 변환 후 features에 저장
	- features.append(mh.features.haralick(im).ravel())
	- 이미지의 날카로운 부분을 잘 나타내줄 수 있는 좌표변환? 알고리즘
4. 최종 이미지 벡터를 np.array()로 변환
	- np.array(features)
	
## 라벨 데이터
1. 이미지의 파일명에서 라벨이 될 수 있는 부분만 labels에 따로 저장
	- labels.append(이미지경로[라벨이름 시작:-len("제외할부분")])
	- 또는 다른 방법으로 라벨 저장
2. 최종 라벨을 np.array()로 변환
	- np.array(labels)
	
## 예측
1. pipeline 만들기
	- 파이프라인 임포트 : from sklearn.pipeline import Pipeline
	- 스케일러 임포트 : from sklearn.preprocessing import StandardScaler
	- 분류 모델 임포트 : from sklearn.linear_model import LogisticRegression
	- clf = Pipeline([("proproc", StandardScaler()), ("classifier", LogisticRegression)])
2. 예측 정확도 확인
	- from sklearn.model_selection import cross_val_score
	- scores = cross_val_score(clf, features, labels)

## 이미지 벡터간 거리를 계산하여 유사한 이미지 찾기
1. 이미지 벡터를 스케일러를 사용하여 스케일링 
	- from sklearn.preprocessing import StandardScaler
	- sc = StandardScaler()
	- features = sc.fit_transform(features)
	- features는 list안에 이미지의 벡터가 각각 numpy.ndarray로 저장 되어 있다.
2. 이미지 벡터를 각 이미지 벡터와의 거리를 계산
	- from scipy.spatial import distance
	- distance의 squareform 서브패키지 사용
	- dists = distance.squareform(distance.pdist(features))
	- 이미지의 갯수 M이면 M X M의 정방행렬로 반환된다. 각 이미지마다 다른 이미지와의 거리값이 저장된다.
3. dists에서 가장 유사한 이미지 선택
	- sim_img_position = dists[n].argsort()[m]
	- dists에서 원하는 이미지 A를 인덱스로 선택
	- A에는 A와 나머지 이미지들간의 거리값이 저장되어 있다. 
		- array([    0.        ,  5261.64361451, 10201.11126381, 10605.19152115, ...])
	- argsort() 명령어를 사용하면 가장 작은 값 순서데로 인덱스가 나열된다. (거리값이 정렬되는 것이아니다.)
		- array([ 0, 16,  5, 52,  9,  8, 28,...])
		- 가장 거리값이 작은 것은 0번째, 두번쨰로 거리값이 작은것은 16번째 이미지라는 의미이다.
	- 이 배열에서 인덱스 0, 1, 2, 3, 4 는 가장 거리값이 가까운 이미지이다.
	- 이미지의 경로가 저장된 images에서 이 인덱스값의 위치에 있는 이미지가 유사한 이미지가 된다.

![sim_img_process.png](./images/sim_img_process.png)









