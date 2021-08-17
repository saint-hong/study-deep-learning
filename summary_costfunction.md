# cost funnction : 비용함수

### 가설함수 Hypothesis Function
- 기계학습에서 훈련용 데이터를 통해서 적절한 예측값을 도출하기 위한 함수
    - 집값데이터에서 집의 특성과 가격이라는 훈련데이터를 학습 알고리즘으로 학습한 후 어떤 새로운 집의 특성에 대한 집값을 예측하고자할 때 필요한 가설 또는 함수
    - $h_{\theta}(x) = \theta_{0} + \theta_{1}(x)$
    - 쎼타 = 파라미터_
    - $y=ax+b$ 라는 직선과 같다. 이때 y절편과 x절편의 값을 찾는 것과 같다.
    - 이와 같이 특징 x가 하나인 경우, 선현회귀문제는 주어진 학습데이터(정답, label, 실제값)와 가장 잘 맞는 가설 함수 Hypothesis function h를 찾는 문제가 된다. 
- 훈련용 데이터를 기반으로 가설함수를 만들고 가장 작은 비용함수 값에 해당하는 지점을 찾는 과정이다.
    
### 비용함수 Cost Function
- 예측값과 실제 결과값 사이의 차이를 나타내는 함수
- 머신러닝에서 훈련용 데이터의 학습 알고리즘을 거친 후 적절한 예측을 위해 사용하는 함수
    - 예측의 결과가 가장 좋은 가설함수는 cost function 의 값을 가장 작게만드는 가설함수이다.
    - $J(\theta_{0}, \theta_{1}) = \dfrac{1}{2m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)})- y^{(i)})^2$
    - 1~m 까지의 예측값 H(x^i) - 실제값 (y^i) 의 차이들을 모두 더한 후 평균을 구하는 방식
    - 차이의 부호를 양수로 맞추기 위해 제곱을 함
    - 이 평균값이 최소가 되는 쎼타값을 적절하게 조절하여 구해야한다.
- 파라미터 값을 조정하면서 가장 최소인 파라미터 값을 찾는 과정
    - 2차 방정식의 해 = 기울기가 0인 지점 = 최소값 (항상 2차방정식의 형태는 아니다)
- 그러나 실제 데이터는 특징 변수 x가 매우 많은 다차원이므로 단순한 방식으로 cost function 의 최소값을 찾기 어렵다.
    - 경사하강 알고리즘 gradient decent algorithm 방식을 사용 
    - 비용 함수의 그래프 위에 임의의 점 선택 -> 해당 점에서 미분 또는 편미분 값을 계산 -> 미분값 즉 기울기가 양수인지 음수인지 판단하여 학습률 Learning Rate 업데이트 -> 새로운 임의의 지점 이동 후 반복
- 학습률 Learning Rate
    - $\theta = \theta - \alpha \dfrac{d}{dt} J_{\theta} (x)$
    - _경사하강 기법에서 임의의 점 선택할 때 다음 지점을 선택하는 기준값
    - 알파를 어떻게 설정하느냐에 따라서 theta 를 갱신하게 된다.
    - 학습률이 작으면 최소값을 찾기위한 임의의 점간의 간격이 작게된다. 여러번 시행을 해야하지만 최솟값을 잘 찾을 수 있음
    - 학습률이 크면 최소값을 찾기위한 임의의 점간의 간격이 크게 된다. 시행 횟수가 상대적으로 작으나 최소값에 수렴하지 않고 진동할 수 있다.
- 로지스틱 회귀 Logistic Regression 문제에서 cost function 을 사용하면 그래프가 울퉁불퉁한 형태가 되므로 gradient descent algorithm 으로 최소값을 구할 수 없다.
    - 가장 작은 지점으로 가기전에 상대적으로 작은 지점을 최소값으로 파악하게 된다.
- 여러개의 특성 x가 있는 경우
    - 행렬식으로 표현
    - $h_{\theta}(x) = \theta_{0} + \theta_{1}x_{1} + \theta_{2}x_{2} + \theta_{3}x_{3} + \theta_{4}x_{4}$
    - $h_{\theta}(x) = \theta^T x$
    
### 비용함수의 종류
- _머신러닝의 예측값(prediction)과 실제값(label)의 차이를 줄여 실제값에 가까운 예측값을 도출하는 모델을 만들기 위해 필요함. 차이를 조절하는 하는 몇가지 비용함수들
- 성능 측정 지표
- MAE (Mean Absolute Error) : L1 loss function
    - $MAE=\dfrac{1}{n} \sum |y- \hat y|$
    - y = 실제값, y^ = 예측값
    - 실제값과 예측값의 차이값의 절대값의 평균
    - 절대값이므로 예측값이 실제값보다 큰지 작은지 판단이 어렵다.
    - 특잇값이 많은 경우에 유용함
    - 에러에 따른  손실이 선형적으로 올라갈 때 적합
- MSE (Mean Squared Error) : L2 loss function
    - $MSE = \dfrac{1}{n} \sum (y - \hat y)^2$
    - 예측값과 실제값의 차이의 면적의 합
    - 제곱을 하므로 예측값이 실제값보다 큰지 작은지 판단 어렵다.
    - 항상 양수
    - 오차값의 크기가 클수록 오차의 정도가 커진다. outlier 의 영향을 받는다. 이상치가 존재하면 수치가 많이 늘어난다.
- RMSE (Root Mean Squared Error)
    - $RMSE = \sqrt {\dfrac {\sum_{i=1}^{N} (y_{i} - \hat y_{i})^2}{N}} $
    - _오차에 제곱을 한 MSE 값은 실제 오류 평균보다 커지는 특성이 있으므로, MSE에 루트를 적용
    - 에러에 따른 손실이 기하급수저긍로 올라갈때에 적합
- MSLE (Mean Squared Log Error)
    - $MSLE(y, \hat y) = \dfrac{1}{n_{samples}} 
    \sum_{i=0}^{n_{smaples}-1}(log_{e}(1+y_{i}) - log_{e}(1+\hat y_{i}))^2$
    - _MSE 에 로그를 적용
    - 실제값과 예측값에 자연로그를 취한 형태
    - 기하급수적으로 증가하는 데이터의 예측에 사용하기 유용하다
- MLE, MAPE, MPE, cross entropy 등이 있음

# cost function 계산

- cost function 식에 예측값과 실제값 대입 후 2차 방정식 계산
```
import numpy as np

np.poly1d([2, -1])**2 + np.poly1d([3, -5]) **2 + np.poly1d([5, -6]) **2


===== print =====

poly1d([ 38, -94,  62])
```

- 미분을 위한 심파이의 심볼릭 연산 적용
```
import sympy as sp

theta = sp.symbols('theta')
diff_theta = sp.diff(38 * theta ** 2 - 94 * theta + 64, theta)
diff_theta

===== print =====

$76th - 94$
```

- 도함수의 해 : 비용함수의 2차 방정식의 최소값은 theta = 1.237 인 지점
```
sp.solve(diff_th)

===== print =====

[47/38]

round(47/38, 3)

===== print =====

1.237
```

# 성능 측정 지표의 사용

- 보스턴 데이터 로드 및 X, y 데이터 구분
```
data = load_boston()
boston_df = pd.DataFrame(data.data, columns=data.feature_names)
boston_df['PRICE'] = data.target

X = boston_df.drop(['PRICE'], axis=1)
y = boston_df['PRICE']
```

- train, test 데이터 분리
```
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                   random_state=13)
```

- 선형회귀 모델 생성
```
from sklearn.linear_model import LinearRegression

linear_regression = LinearRegression()
linear_regression.fit(X_train, y_train)

y_pred = linear_regression.predict(X_test)
```

- MAE, MSE, RMSE, MSLE 지표 확인
```
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error

print("MAE : ", mean_absolute_error(y_test, y_pred))
print("MSE : ", mean_squared_error(y_test, y_pred))
print("RMSE : ", (np.sqrt(mean_squared_error(y_test, y_pred))))
print("MSLE : ", mean_squared_log_error(y_test.drop(414), np.delete(y_pred, 32)))

===== print =====

MAE :  3.626783994958872
MSE :  24.318238309170447
RMSE :  4.931352584146711
MSLE :  0.11209850477548165
```
