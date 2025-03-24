# 🌟 Boosting 완전 정복 가이드

> 이 문서는 머신러닝에서 **Boosting**이 무엇인지,  
> 그리고 **AdaBoost**, **Gradient Boosting**이 어떻게 작동하는지를  
> 초심자의 눈높이에서 수학적 구조까지 포함하여 정리한 자료입니다.

---

## ✅ 1. Boosting이란?

Boosting은 **약한 모델(weak learner)** 여러 개를 순차적으로 학습시켜  
**점점 더 정확한 예측 모델을 만드는 앙상블 학습 방법**입니다.

- 약한 모델이란? → 단독으로는 성능이 낮지만, 조금은 정답을 맞추는 모델 (ex. 깊이가 낮은 결정 트리)
- 여러 모델을 반복적으로 연결해서 오차를 줄여나감
- 핵심 아이디어는:
  > "이전 모델이 틀린 걸 다음 모델이 잘 맞추도록 하자"

---

## ⚡ Boosting vs Bagging

| 항목 | Bagging | Boosting |
|------|--------|----------|
| 학습 방식 | 병렬 | 순차 |
| 목적 | Variance 감소 (과적합 방지) | Bias 감소 (과소적합 개선) |
| 예시 | Random Forest | AdaBoost, Gradient Boosting |
| 데이터 샘플링 | 부트스트랩 | 전 데이터 사용, 가중치 조정 |
| 모델 간 의존성 | 독립 | 순차적 의존성 |

---

## 🔶 2. AdaBoost (Adaptive Boosting)

### 🔍 핵심 아이디어:

- 모델이 **틀린 샘플에 가중치를 더 부여**
- 다음 모델이 **틀린 애들을 더 잘 맞추도록 유도**
- 각 모델은 자신이 잘 맞춘 정도에 따라 **가중치**를 받아 투표함

### ⚙️ 작동 절차:

1. 모든 샘플에 **동일한 가중치**로 시작
2. 약한 모델 학습 → 예측
3. **틀린 샘플의 가중치 증가**, 맞춘 샘플은 감소
4. 다음 모델은 업데이트된 가중치로 학습
5. 최종 예측은 모든 모델의 **가중 평균**

### 📐 수식 요약:

- 오차율:
  \[
  \varepsilon_t = \sum w_i \cdot \mathbb{1}[h_t(x_i) \ne y_i]
  \]
- 모델 가중치:
  \[
  \alpha_t = \frac{1}{2} \log\left( \frac{1 - \varepsilon_t}{\varepsilon_t} \right)
  \]
- 샘플 가중치 업데이트:
  \[
  w_i \leftarrow w_i \cdot \exp(\alpha_t \cdot \mathbb{1}[h_t(x_i) \ne y_i])
  \]

### 🧠 직관:

> “틀린 문제는 빨간펜으로 표시하고, 다음 모델이 그 부분을 더 집중적으로 학습하게 한다.”

---

## 🌊 3. Gradient Boosting

### 🔍 핵심 아이디어:

- **잔차(오차)**를 예측하는 모델을 반복적으로 학습
- 각 단계는 손실 함수의 **기울기(gradient)**를 따라 **예측 함수를 개선**

### 🧱 작동 구조:

1. 초기 모델 $F_0(x)$ 설정 (보통 평균값)
2. 반복적으로:
   - 현재 예측 $F_{t-1}(x)$의 **기울기(gradient)** 계산
   - 그 값을 **예측하는 트리 $h_t(x)$ 학습**
   - 예측 함수 업데이트:
     \[
     F_t(x) = F_{t-1}(x) - \eta \cdot h_t(x)
     \]

> $\eta$는 학습률 (learning rate)

---

## 🧮 4. 손실 함수에 따른 gradient 예시

| 손실 함수 | 정의 | Gradient (기울기) |
|-----------|------|-------------------|
| MSE (회귀) | $\frac{1}{2}(y - F(x))^2$ | $F(x) - y$ |
| Cross Entropy (분류) | $- y \log \sigma(F(x)) - (1 - y) \log(1 - \sigma(F(x)))$ | $\sigma(F(x)) - y$ |

- $\sigma(F(x))$는 시그모이드 함수로 예측 확률을 의미
- Gradient Boosting에서는 항상 이 기울기를 **예측하는 모델을 학습**함

---

## 🎯 5. Gradient Boosting이 경사하강법인 이유

Gradient Boosting은 **벡터 공간이 아닌 함수 공간(function space)**에서의 **경사 하강법(gradient descent)**입니다.

- 우리가 최적화하려는 대상은 수치 벡터가 아닌 함수 $F(x)$ 자체
- 손실 함수의 **기울기를 계산하고**, 그 기울기를 따라 $F(x)$를 업데이트
- 이 과정을 반복하며 손실 함수를 최소화

---

## 💡 핵심 요약

- Boosting은 이전 모델의 실수를 **보완**하는 방식
- AdaBoost는 **가중치 조정**으로 틀린 샘플에 집중
- Gradient Boosting은 **기울기를 예측**해서 모델을 점진적으로 보정
- 최종 목표는 **손실 함수의 값을 줄이는 것**

---

## 📌 기억해야 할 직관

> **AdaBoost**  
> “틀린 문제에 빨간펜 → 다음 모델이 더 집중해서 학습”

> **Gradient Boosting**  
> “오차 방향(기울기)을 따라 모델을 조금씩 개선”

---

## ✨ 확장 개념 (추후 학습 추천)

- XGBoost, LightGBM, CatBoost 등 → Gradient Boosting을 최적화한 버전들
- Regularization (규제), Early Stopping, Tree 구조 튜닝 등 고급 테크닉

---
