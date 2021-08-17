# Machine-Learning
Description and Summary of the use of the Machine-Learning


### 전처리
> 간략한 설명, 여러가지 종류 위주로 정리, pca 는 자세하게
- Scaler : 수치형 데이터의 차이를 맞춰주는 작업
    - MinMaxScaler, StandardScaler, LogScaler 등
- PCA : principle component analysis : 차원축소 (중요)
- mpimg : 이미지 처리 도구, 이미지 분석 때 필요함
- Regular Expression : 정규표현식 : 문자열 데이터를 분석하는데 필요한 도구
- Label Encoder : 라벨인코더 : 문자열 데이터를 수치형 데이터로 변환해주는 도구
- SMOTE oversampling : 데이터의 불균형을 맞추기 위한 도구, 데이터를 생성해준다.
- tgdm : 반복문의 실행 과정을 시각화해서 보여주는 도구

### 모델
> 이론 + 옵션 + 사용법
- Decision Tree : 의사결정 나무
- Cost Function : 선형회귀모델의 분류 방법
- Logistic Regression : 로지스틱 함수를 사용한 회귀모델의 분류 방법
    - cost function 과 연관되어 있음
    - Decision Boundary
    - 다변수 방정식
- Ensemble : 여러개의 분류기를 생성하고 분류기들의 예측값을 결합하여 더 정확한 최종 예측을 도출하는 기법
    - 단일 분류기보다 신뢰성이 높은 예측값을 얻고자 함
    - voting
    - bagging - bootstrapping
    - 하드보팅, 소프트보팅
    - Random Forest Classifier
- Boosting : 앙상블 기법 중 부스팅 방식을 사용한 모델들
    - Adaboost
    - Gradientboosting : GBM
    - XGBoost
    - LightGBM
- kNN : k Nearest Neighbor : 가까운 이웃 데이터 분류 모델, k 는 가까운 거리
- FBProphet : 페이스북에서 제작한 시계열 데이터 분석 모델
- Natural language processing : 자연어 처리
    - nltk : natural language toolkit
    - KoNLPy : Korean natural language processing in Python : 한국어 정보처리 파이썬 패키지
    - wordcloud
- MultiNomial Naive Bayes Classifier : 나이브 베이즈 분류
    - 베이즈 정리르 적용한 확률 분류기
- Suppoert Vector Machine : SVM, 서포트 벡터 머신

### 모델 성능 향상 도구
- Hyperparameter Tunning : 모델 성능 향상을 위한 설정값 조절
    - GridSearchCV : 교차검증의 종류, 모델의 설정값을 일괄적으로 조절하는 도구
- PipeLine : 모델을 만드는 과정의 여러 단계들의 순서를 연결해주는 도구

### 모델평가
- Cross validation : 교차검증
    - 훈련용 데이터를 5개로 나누고 1개의 테스트데이터와 4개의 훈련데이터로 세분화하여 모델의 성능의 정확도를 높이는 검증과정
    - k-Fold cross validation
    - stratified cross validation
- Model Verification : 모델 평가
    - 정확도, 오차행렬, 정밀도, 재현율, FPR-TPR, F1score, AUC score, ROC curve
    - threshold 값에 따른 평가지표의 변화
    - classification report, confusion matrix
