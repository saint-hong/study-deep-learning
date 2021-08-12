# DecisionTreeClassifier()

## Summary

### 정의
- 의사결정나무는 여러가지 규칙을 순차적으로 적용하면서 독립변수 공간을 분할하는 분류 모형
-  CART (classification and regression tree) : 분류 classification 와 회귀 분석 regression 에 모두 사용가능
    - 데이터를 분석하여 데이터 사이에 존재하는 패턴을 예측 가능한 규칙들의 조합으로 나타내는 모델
    - 규칙의 조합을 나타내는 방식이 나무와 같다고 하여 의사결정나무라고 함
    - 순차적으로 질문을 던져서 답을 고르게 하는 방식 ex) 스무고개 놀이

### committess 와 boosting
- 분류 classification 과 회귀 regression 문제를 풀기 위하여 다양한 종류의 머신러닝 모델이 있는데 여러 모델을 조합하면 성능이 더 나아질 수 있다.
    - 위원회방식 committees : L개의 서로 다른 모델을 훈련해서 각 모델이 내는 예측값의 평균을 통해 예측을 하는 방식
    - 부스팅방식 boosting : 여러모델을 순차적으로 훈련하며, 각 모델을 훈련하기 위한 오류함수는 이전 모델의 결과에 의해 조절됨
    - 여러 모델 중 하나의 모델을 선택해서 예측을 시행하는 방법
    - 의사결정나무는 마지막 방식에 해당함

### 의사결정나무를 이용한 분류법
- 여러가지 독립변수 중 하나의 독립변수를 선택, 그 독립변수에 대한 기준값(threshold)를 정한다. 이를 분류규칙이라고 한다.
- 전체 학습데이터 집합(부모노드)을 해당 독립변수의 값이 기준값보다 작은 데이터 그룹(자식노드1)과 해당 독립변수의 값이 기준값보다 큰 데이터 그룹(자식노드2)로 나뉜다.
- 각각의 자식노드에 대해 1~2의 단계를 반복하여 하위의 자식 노드를 만든다. 단, 자식노드에 한가지 클래스의 데이터만 존재한다면 더이상 자식노드를 나누지 않고 중지한다. 


### 특징
- 이진분류 binary classification 문제와 같다
    - 연속형 수치 데이터를 예측 할 수 있다
- 분류 classification 와 회귀 regression 문제에 모두 적용 가능
    - 어떤 D 데이터가 주어졌을 때 D 집합을 대표할수 있는 값을 반환한다. 
    - 분류 방식 : 예측값의 최빈값을 사용하여 예측
    - 회귀 방식 : 예측값의 평균값을 사용하여 예측
- 나무의 구조
    - 나무의 구성요소는 트리 구조와 같다
        - root node - branch - node(parents or child) - leaf node 
        - root node : 정보획득량 값이 가장 큰 feature
        - node : input data의 분류 특성 attribute, feature
        - branch : 특성이 가질 수 있는 값 value
        - leaf node : 분류결과 out put
    - 데이터를 구분짓는 어떤 특성(feature)을 node에 배치하느냐에 따라서 여러가지 tree를 만들 수 있다.
    - 루트노드에서 분화하여 노드들이 생기며 이 노드들이 새로운 노드로 분화하면 부모-자식 노드관계가 성립된다. 이렇게 분화하는 방식은 최초의 데이터 집합이 노드 상호간의 교집합이 없는 부분집합으로 나누어지는 것과 같다.
    - 한 번 분기할 때마다 두 개의 변수영역으로 구분한다. yes or no

### 이론
- 분류 규칙을 정하는 방법 
    - 부모노드와 자식노드간의 엔트로피를 가장 낮게 만드는 최상의 독립변수와 기준값을 찾는것
    - 이것을 정량화한 것이 정보획득량 information gain 이다. 
    - 기본적으로 모든 독립변수와 모든 가능한 기준값에 대해 정보획득량을 구하여 가장 정복획득량이 큰 독립 변수와 기준값을 선택
- 정보획득량 information gain 
    -  X라는 조건에 의해 확률변수 Y의 엔트로피가 얼마나 감소하였는가를 나타내는 값
    - <src img="https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20%5Cfn_cm%20%5Clarge%20IG%5BY%2C%20X%5D%3DH%5BY%5D%20-%20H%5BY%7CX%5D">
    - Y의 엔트로피에서 X에 대한 Y의 조건부 엔트로피를 뺀 값
    - tree가 자라면서 각 분기점에서 어떤 feature를 선택할지 정할 때 사용된다.
        - A, B, C 각 feature 를 기준으로 정보획득량을 구한뒤 값을 비교 가장 큰 값이 상위의 root node의 특성이된다.
    - 최초 데이터의 불순도율에서 각 분기지점들의 불순도율의 합을 뺀 값
    - 루트노드의 불순도율 - 하위 노드의 불순도율의 합
    - 분할된 노드들의 불순도가 작을 수록 정보획득량이 증가한다.
- 엔트로피 entropy
    - 정보획득량
    - 정보이론에서 엔트로피는 우리가 가지고 있지 않은 정보의 양을 의미
        - 순도가 높은 데이터는 정보를 많이 가지고 있다는 의미, 즉 예측이 가능하며 새롭게 얻을 정보량이 적다. 엔트로피가 낮은 상태이다.
        - 순도가 낮은 데이터는 정보가 부족하다는 의미, 즉 예측이 불확실하며 새롭게 얻을 정보량이 많다. 엔트로피가 높은 상태이다.
    - 데이터를 구분하기 전 정보획득량
        - $Entropy(A)=-\sum_{k=1}^{m}{p_klog_2}{(p_k)}$
    - 데이터를 구분하고 난 후의 정보획득량
        - $Entropy(A)=\sum_{i=1}^{d}R_i(-\sum_{k=1}^{m}p_klog2(p_k))$
- 지니계수 Gini Index 
    - $G.I(A) = \sum_{i=1}^{d}(R_i(1-\sum_{k=1}^{m}(p^2_{ik})))$
    - 엔트로피와 유사한 불순도 지표
    - 그래프의 면적이 엔트로피보다 약간 좁다
    - 데이터를 절반으로 나누었을 때 불순도율이 최대
- 불순도와 불확실성
    - 데이터간의 영역을 나누는 기준 : 각 영역의 순도/확실성이 증가하고 불순도/불확실성이 최대한 감소하도록 만드는 기준을 찾는다
    - 순도 homogeneity : 구분 된 영역의 유사성이 높은 상태
    - 불순도 impurity, 불확실성 uncertainty : 구분 된 영역의 유사성이 낮은 상태 
    - 정보 획득 information gain : 순도 증가, 불순도 감소하는 방식        
- 불순도의 지표
    - 물리학의 엔트로피 개념을 정보이론에 도입하여 사용함
    - 데이터 집합에서 특정한 데이터끼리 구분지을 때 어떻게 나눌지에 관한 이론
    - 데이터 집합에서 구분하기 전과 구분한 이후의 엔트로피 값을 비교함
    - 엔트로피 값 감소 = 불확실성 감소 = 순도 증가
    - 엔트로피 0 : 순도 최대, 불순도 최소
    - 엔트로피 1 : 순도 최소, 불순도 최소
- 가중치 weighted information gain
    - 여기에서 불순도율에 대한 비교가 필요하게 된다.
    - 같은 값이더라도 어떻게 나누었는지에 따라서 의미의 중요도가 다르기 때문.
    - 데이터 개수가 충분히 많을 수록 가중치를 높게 부여한다.
    - 가중치 : 분할전 데이터 크기에 대한 분할 후 데이터 크기의 비율
- 오분류오차 misclassification error 
    - 불순도 지표, 잘 사용하지 않음
    
### 모델의 학습 방법
- 재귀적 분기 recursive partitioning : 변수 영역을 두 개로 구분
    - 데이터를 특정 변수 기준으로 정렬한 뒤 가능한 모든 분기점(특성을 구분짓는 모든 지점)에 대해여 엔트로피/지니계수를 구해 분기전과 분기후의 정보획득을 조사한다.
    - 1회 분기를 계산하는 경우의 수는 개체수 n, 변수 d 일 때 d(n-1) 개
- 가지 치기 prunning : 분기지점이 많아졌을 때 과적합을 방지하기 위해 특정 분기지점을 잘라내어 상위분기지점에 합치는 과정, 분할/분기와 반대개념
    - 데이터를 구분짓기 위해 모든 분기지점의 순도가 100%가 되면 오히려 학습데이터에 과적합 현상이 발생함. 
    - 오분류율이 더 높아지는 경향이 발생하는데 이를 방지하기 위해 구분지점들을 통합하여 적절한 불순도율을 찾는 과정.
- 가지치기를 결정하는 기준 : 가지를 친다=분할/분기하지 않는다
    - test 데이터를 넣어 성능 검증 후, 가지치기 결정
    - 통계적 중요성 statistical significance 에 기반하여 분할 후 두 지점간의 차이에 따라서 가지치기 결정 (chi-square). information gain 값이 크더라도 chi-square 값이 작으면 분할하지 않는다. 가지를 친다. merge 한다.  
    - 비용함수(cost function)이 최소인 분기지점을 찾도록 학습된다. 
        - $CC(T) = Err(T) + \alpha + L(T)$ 
        - CC(T) : 오류가(Err) 적으면서 분기지점의 수가(L) 적은 단순한 모델일 수록 작은 값
        - Err(T) : 검증데이터에 대한 오분류율
        - L(T) : 분기지점의 수 (구조의 복잡도)
        - alpha : Err 과 L의 결합 가중치 (사용자가 부여함 보통 0.01~0.1)

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
# 2. 관련 모듈
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
## graphviz
- http://www.graphviz.org/
- Graphviz는 오픈소스 시각화 소프트웨어로 구조화된 정보를 추상화된 그래프나 네트워크의 형태로 제시
- pip install graphviz

## mlxtend
- 결정나무의 결정경계, 기존데이터에 예측값을 얹어서 보여주기 등이 가능.
- sklearn 에는 없는 몇몇 유용한 기능이 있음
- pip install mlxtend

