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

# 2. 의사결정나무 분석을 위한 모듈

- from sklearn.model_selection import train_test_split   -> 훈련,테스트 데이터 분류 모듈
- from sklearn.tree import DecisionTreeClassifier     -> 결정나무 모델 모듈
- from sklearn.metrics import accuracy_score      -> acc 스코어 모듈

- from graphviz import Source     -> 결정나무 시각화용 그래프 모듈 
- from sklearn.tree import export_graphviz     -> 결정나무 시각화용 그래프 모듈

- from mlxtend.plotting import plot_decision_regions     -> 결정나무 모델의 결정경계 그래프 모듈

- import numpy as np     -> numpy 패키지
- import matplotlib.pyplot as plt     -> 그래프 패키지 (셋팅)
- import seaborn as sns     -> 그래프 패키지 (box plot, pair plot, joint pair)

