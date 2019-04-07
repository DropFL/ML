# HW1: Linear / Logistic Regression

첫 번째 과제는 선형 / 로지스틱 회귀를 [NumPy](http://www.numpy.org/)로 구현하는 것입니다. 전체적인 틀은 제공되며, 여기에는 [utils 모듈](utils.py)과 data-set 등이 포함됩니다. 이 파일들은 과제 수행에 있어 **수정해선 안되는 파일**입니다.

## utils

utils는 data-set의 로드, 프로그램의 전체적인 초기화, 결과 분석 등을 담당하는 모듈입니다. 이 모듈은 import되어 실행될 수도 있지만, 추가적으로 main으로서 실행될 수도 있습니다. 이 경우, 이 모듈의 역할들을 정상적으로 수행해 내는지 확인하는 작업을 수행합니다.

## [linear_main](linear_main.py), [logistic_main](logistic_main.py)

각각 선형 / 로지스틱 회귀를 실행하는 main이 되는 모듈입니다. 여기에서 Hyper-parameter와 최적화 기법이 결정되며 학습 결과, 추정값과 실제값의 오차 등이 출력됩니다.

## [LinearRegression](models/LinearRegression.py), [LogisticRegression](models/LogisticRegression.py)

각각 선형 / 로지스틱 회귀 알고리즘의 구현체가 있습니다. 각각은 그 이름에 해당되는 클래스를 갖고 있으며, `train` 함수와 `eval` 함수를 메서드로 보유하고 있습니다.

`train(x, y, epochs, batch_size, lr, optim)` 함수는 `(x, y)`로 구성된 데이터를 `batch_size` 단위로 쪼개 `optim` 전략에 따라 `lr`의 비율로 학습하는 방식으로 전체 데이터를 `epochs` 번 반복 학습합니다. 데이터는 `x`에 따른 종속변수 `y`로 구성됩니다. 함수의 리턴값은 최종 epoch에서 mini batch들을 학습하는 데에 발생한 loss들의 평균입니다.

`eval(x)` 함수는 지금까지 학습하여 도출한 상관관계에 기반하여 주어진 `x`에 따른 추정값 `ŷ`을 계산해 반환합니다.

## [Optimizer](optim/Optimizer.py)

학습하는 과정에서 해에 도달하기 위해 취하는 전략들이 정의되어 있습니다. 이 모듈는 `SGD`, `Momentum`, `RMSProp`이 있습니다.

각각은 `update(w, grad, lr)` 메서드를 갖고 있는데, 이 메서드는 현재 가중치인 `w`, 해당 지점에서의 gradient인 `grad`, 학습 비율인 `lr`을 전달받아 데이터를 학습한 후 수정된 가중치를 반환합니다.