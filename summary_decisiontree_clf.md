# DecisionTreeClassifier()

## 분석 패턴
- 1) 분석 데이터 로드
- 2) 분석 데이터 탐색
    - 컬럼, 벨류 분석
    - 그래프로 분석
- 3) 분석 데이터 전처리
    - 스케일러
- 4) 의미있는 분석 지점 선택
    - 데이터에서 분류모델을 적용할 수 있는 의미 찾기
- 5) 예측 데이터 X, 라벨 데이터 y 구분
    - 라벨 데이터와 예측 데이터가 겹치는 부분이 있으면 제거.
- 6) 모델 구현 (옵션에 따라서 결과가 달라진다.)
    - 훈련, 테스트 데이터 분리
    - 모델 정의
    - 모델 학습
    - 모델로 예측값 계산
    - 라벨 데이터와 예측값을 비교하여 정확도 계산
- 7) 모델 시각화
    - 결정나무 모델을 graphviz 를 사용하여 시각화 한다.
- 8) 모델의 성능을 높이기 위한 추가 작업들
    - 여러 옵션들을 한번에 처리하여 결과값을 비교하여, 좋은 성능의 모델을 선택.
    - gridsearchcv
    - pipeline
    
## DecisionTreeClassifier 속성값들
```

DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini', max_depth=None, max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, presort='deprecated', random_state=None, splitter='best')

```

- 특히 중요한 속성들
```
max_depth, criterion, max_leaf_nodes, random_state
```

# 1. 그래프

## 1) boxplot
```
plt.figure(figsize=(12, 6))
sns.boxplot(x='sepal length (cm)', y='species', data=iris_pd, orient='h')
```

## 2) pairplot
```
sns.pairplot(iris_pd, hue='species', vars=['sepal lenght (cm)'.'sepal width (cm)'])
plst.show()
```

## 3) jointplot
```
sns.set_style("whitegrid")
sns.jointplot(x='sepal length (cm)', y='sepal width (cm)', data=iris_pd)
plt.show()
```

## 4) scatterplot
```
plt.figrue(figsize=(12,6))
sns.scatterplot(x='sepal lenght (cm)', y='sepal width (cm)', data=iris_pd, hue='species', palette="Set2")
```

## 5) tree model visualization
- 그 전에 결정나무 모델을 먼저 만들어야 한다.
- 결정나무 모델, 구분할 피쳐의 명칭, 구분 된 데이터의 클래스 명칭 을 넣는다.

```
from graphviz import Source
from sklearn.tree import export_graphviz

Source(export_graphviz(iris_tree, feature_names=['length','width'], class_names=iris.target_names, rounded=True, filled=True))
```

## 6) mlxtend 로 분류 지점(결정경계) 확인
```
from mlxtend.plottin import plot_decision_regions

plt.figure(figsize=(12, 6))
plot_decision_regions(X=iris.data[:, 2:], y=iris.target, clf=iris_tree, legend=2)
plt.show()
```

## 7) px 그래프로 count 확인
- plotly.express 패키지 필요함.
- 그래프에서 돋보기, 범위설정, 해당 그래프의 요약내용 등을 쉽게 확인할 수 있음
- color 옵션을 넣어주면 red, white 를 색으로 구분하여 그래프를 그려준다.
```
import plotly.express as px

fig = px.histogram(wine, x='quality', color='color') 
fig.show()
```

## 8) go 그래프로 분류 비율 확인
- plotly.graph_objects 패키지 사용.
- test, train 데이터로 나눌때 비율이 잘 나뉘었는지 그래프로 보여줌.
- update_layout 에 barmode 옵션으로 overlay 를 정의하면, 겹쳐서 그려준다.
```
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Histogram(x=X_train['quality'], name='Train'))
fig.add_trace(go.Histogram(x=X_test['quality'], name='Test'))

fig.update_layout(barmode='overlay')
fig.update_trace(opacity=0.75)
fig.show()
```
# 2. 모듈
-훈련,테스트 데이터 분류 모듈
```
from sklearn.model_selection import train_test_split   
```
- 디시전트리 모델 모듈
```
from sklearn.tree import DecisionTreeClassifier     
```
- accracy score 모듈
```
from sklearn.metrics import accuracy_score
```
- 디시젼트리 시각화용 그래프 모듈
```
from graphviz import Source 
from sklearn.tree import export_graphviz
```
- 디시젼트리의 결정경계 그래프 모듈
```
from mlxtend.plotting import plot_decision_regions    
```
- 그래프 패키지
```
import matplotlib.pyplot as plt
```
- 그래프 패키지 (box plot, pair plot, joint pair)
```
import seaborn as sns    
```
- go 그래프 패키지 (확대, 요약 정보 보기 가능)
```
import plotly.graph_objects as go   
```
- 데이터 전처리를 위한 스케일러 모듈
```
from sklearn.preprocessing import MinMaxScaler(), StandardScaler()
```
- numpy 패키지
```
import numpy as np
```

# 3. 데이터 전처리
- Scaler : 어떤것이 더 좋은지는 따져봐야 한다.
- 디시젼트리에서는 스케일러가 결과에 큰 영향을 끼치진 않는다.
- box 그래프로 데이터의 상태를 확인한다.

```
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Box(y=X_mms_pd['pH'], name='pH')
fig.show()
```
- MMS : 최대값 1, 최소값 0 으로 맞춘다.
- SS : 평균 0, 표준편차 1 로 맞춘다.

```
form sklearn.preprocessing import MinMaxScaler(), StandardScaler()

MMS = MinMaxScaler()
SS = StandardScaler()

MMS.fit(X)
SS.fit(X)

wine_mms = MMS.trasform(X)
wine_ss = SS.transform(X)

wine_mms_pd = pd.DataFrame(wine_mms, columns=X.columns)
wine_ss_pd = pd.DataFrame(wine_ss, columns=X.columns)
```

# 4. 모듈 설치
- 디시젼트리 분석을 좀더 유용하게 사용하기 위한 모듈 설치

## 로컬 상태 확인
- !pip freeze : 설치된 전체 모듈 리스트 확인
- !pip -V : pip 의 버전확인
- !python -V : python 의 버전확인

## graphviz
- http://www.graphviz.org/
- Graphviz는 오픈소스 시각화 소프트웨어로 구조화된 정보를 추상화된 그래프나 네트워크의 형태로 제시
- pip install graphviz

## mlxtend
- 결정나무의 결정경계, 기존데이터에 예측값을 얹어서 보여주기 등이 가능.
- sklearn 에는 없는 몇몇 유용한 기능이 있음
- pip install mlxtend

# 5. 중요 이론

## 디시젼트리 이론
- https://ratsgo.github.io/machine%20learning/2017/03/26/tree/
- https://kolikim.tistory.com/22
