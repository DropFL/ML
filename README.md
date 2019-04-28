# HW1: Linear / Logistic Regression

두 번째 과제는 Softmax 기반 분류(Classification) 학습 모델을 [NumPy](http://www.numpy.org/)로 구현하는 것입니다. 전체적인 틀은 이전의 HW1과 같은 형태로 제공됩니다. 그에 따라 본 README에서는 기존 모델과의 차이점 만을 설명합니다.

## [main](main.py)

Softmax 기반 학습 모델을 실행하는 main이 되는 모듈입니다. HW2에서의 학습 모델은 단 하나이기 때문에, HW1와 달리 `main`도 하나만 존재합니다. 그러나 이를 제외한 모듈의 역할 및 수행 작업은 동일합니다.

## [SoftmaxClassifier](models/SoftmaxClassifier.py)

Softmax 알고리즘의 구현체가 있습니다. 이전의 `*Regression` 클래스와 동일하게 `train`, `eval` 함수를 메서드로 보유하고 있으며, 추가로 `softmax_loss`와 `compute_grad` 함수를 구현합니다. 추가된 함수는 `main`에서 직접 호출되지 않는, 현재로써는 오로지 재사용성을 위한 함수입니다.

`softmax_loss(prob, label)`는 주어진 확률 분포(`prob`)와 실제 분류(`label`)에 따라 softmax 모델의 로스 함수를 계산합니다.

`compute_grad(x, weight, prob, label)`는 인자로 전달받은 정보들을 바탕으로 `weight`의 각 feature에 따른 그래디언트 값을 계산합니다. `prob`와 `label`은 `softmax_loss`와 같으며, `x`는 입력된 데이터, `weight`는 현재 가중치를 의미합니다.

위 함수들은 주석으로 문서화되어 있습니다. 각 인자들의 차원과 같은 보다 상세한 정보는 이를 참조하시기 바랍니다.