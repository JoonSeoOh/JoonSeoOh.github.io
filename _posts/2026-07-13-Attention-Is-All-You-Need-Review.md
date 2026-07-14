---
title: "Attention Is All You Need 논문리뷰"
date: 2026-07-13 18:00:00 +0900
categories: [논문리뷰, Attention Is All You Need, 리뷰]
tags: [deep-learning, transformer, attention, nlp]
math: true
---

## 1. Abstract

### 📌 기존 모델의 한계 제시
그동안 Sequence Transduction(시퀀스 변환) 모델들의 주류는 인코더와 디코더를 포함한 복잡한 RNN이나 CNN 기반 아키텍처였습니다. 당시 최고 성능을 내던 모델들 역시 대개 RNN이나 CNN 구조에 Attention 메커니즘을 결합한 형태를 취하고 있었습니다.

### ✔️ Transformer의 제안
이 논문에서는 기존의 순환 구조(Recurrent)나 합성곱(Convolution)을 완전히 걷어내고, **오직 Attention 메커니즘만을 기반으로 작동하는 새로운 아키텍처인 Transformer**를 제안합니다.

### 👆 실험적 성과
* **병렬화 및 학습 시간:** 순환 구조를 완전히 없앰으로써 연산의 병렬 처리가 가능해졌고, 이로 인해 학습 시간을 획기적으로 단축했습니다.
* **번역 성능 향상:** 기존 최고 성능 모델들을 능가하는 번역 퀄리티를 달성했습니다.
* **일반화 능력 확보:** 다른 대규모 데이터셋이나 다양한 작업에서도 훌륭한 일반화 능력을 보여주었습니다.

---

## 2. Introduction

### ⚠️ RNN의 고질적인 문제점: 순차적 계산
LSTM이나 GRU 같은 순환 모델들은 입력 및 출력 시퀀스의 Symbol Position(기호 위치)을 따라 한 걸음씩 순차적으로 계산을 진행합니다. 즉, 현재 상태의 은닉 벡터인 $h_t$를 계산하려면, 반드시 바로 직전 상태의 은닉 벡터인 $h_{t-1}$과 현재 위치 $t$의 입력값이 필요합니다.

* **병렬화의 부재와 메모리 제약:** 이러한 Sequential Nature(순차적 성격)은 학습 시에 **연산의 병렬화**를 근본적으로 방해합니다. 문장의 길이가 길어질수록 모든 계산을 동시에 처리할 수 없고 하나씩 밀고 나가야 하므로, Batch 크기가 크게 제한되고 메모리 제약이 극도로 심해집니다.
* 기존 연구들이 연산 효율이나 모델 성능을 높이기 위해 여러 트릭(Factorization 트릭, Conditional Computation 등)을 도입해보았지만, **순차적 계산**이라는 구조적 제약은 여전히 해결되지 않은 숙제로 남아있었습니다.

### ✔️ Attention의 역할과 한계
Attention은 입출력 시퀀스 내의 거리에 상관없이 단어 간의 의존성을 모델링할 수 있게 해주는 핵심 장치로 자리 잡았습니다. 그러나, 여전히 이전 연구에서의 Attention 메커니즘은 RNN과 결합하여 사용되는 한계가 있었습니다.

* **Transformer:** 해당 논문에서 제시하는 방법론입니다. 순환 구조 (recurrence)를 과감히 버리고, 입출력 사이의 Global Dependency(전역적 의존성)을 오로지 Attention에만 의존하여 뽑아내는 구조를 제안합니다. 이를 통해 병렬화가 가능해지고, 시간적인 효율을 증가시킬 수 있습니다. 

### ✔️ 전역적 의존성 (Global Dependency)이란 무엇인가?
기계번역이나 언어 모델링에서 아주 중요하게 다루어지는 개념이 바로 '전역적 의존성'입니다. 

> 💡 **말이 멀리 떨어져 있어도 관계를 맺는 현상: **
> 예를 들어, "과거에 수많은 실패와 고난, 역경을 겪으며 힘든 시간을 보냈던 **그 청년은**, 끊임없는 노력 끝에 마침내 거대한 글로벌 기업의 **CEO가 되었다**."라는 문장이 있습니다. 
> 
> 여기서 **'그 청년은'**과 **'CEO가 되었다'**는 문장 내 위치상으로는 매우 멀리 떨어져 있습니다. 하지만 문맥을 정확히 파악하기 위해서는 이 두 단어의 관계를 반드시 하나로 연결 지어야 합니다. 

이처럼 **문장 전체(Global)를 관통하며 멀리 떨어진 요소들이 서로 의존하고 있는 관계를 전역적 의존성**이라고 부릅니다. 이전 연구의 RNN, LSTM, CNN 구조에서는 입력 사이의 거리가 멀어질수록 정보가 소실되거나 희석되는 경사 소실(Gradient Vanishing) 문제 등으로 이를 다루기가 매우 힘들었습니다. Transformer는 바로 이 문제를 완벽하게 극복하기 위해 설계되었습니다.

---

## 3. Background

### ✔️ CNN 기반 모델과의 비교
순차적 계산을 줄이기 위해 CNN을 기본 빌딩 블록으로 삼아 모든 입출력 위치의 은닉 표현을 병렬로 계산하려는 시도(ConvS2S, ByteNet 등)들이 존재했습니다. 

하지만 이러한 CNN 기반 모델들은 임의의 두 입출력 위치 사이의 신호를 연관 짓는 데 필요한 연산 수가 두 위치 사이의 거리에 따라 증가한다는 한계가 있습니다.
* **ConvS2S:** 거리에 따라 선형적(Linear)으로 증가
* **ByteNet:** 거리에 따라 로그 스케일(Logarithmic)로 증가

결과적으로 두 단어 사이의 거리가 멀어지면 멀어질수록 **서로 멀리 떨어진 위치 간의 의존성을 학습하기가 매우 어려워집니다.**

### 👆 Transformer가 보완하는 점
Transformer는 이를 단 **상수 번(Constant, $O(1)$)의 연산**으로 줄여버립니다. 

⚠️ 다만, Attention 가중치가 적용된 위치들을 가중 평균 내는 과정에서 Effective Resolution(유효 해상도)이 다소 감소하는 대가가 따르는데, 논문에서는 뒤이어 소개할 **Multi-Head Attention** 구조를 통해 이 해상도 감소 문제를 멋지게 해결해 냅니다.

---

> **현재까지 내용만 들어도 머리가 복잡할 수 있습니다.**
> 하지만 지금까지, 해당 논문이 이야기하고자 하는 바는 매우 단순합니다.
> 
> "기존의 RNN은 문장이 주어지면, 한 글자 한 글자 순서대로 계산해야 해서 너무 느리고 병렬적인 처리가 불가능하다는 단점이 존재했습니다. 따라서, 이전 연구에서는 이를 CNN으로 해결하고자 하였지만, 멀리 떨어진 단어 조합을 잘 매칭(Global Dependency)하지 못한다는 문제가 있었습니다. 따라서, 해당 논문에서는 RNN, CNN 구조를 탈피하고, 오직 단어들끼리 서로의 연관성을 한 번에 파악하는 'Self-Attention'만으로 완전히 새로운 최고 성능의 모델인 **Transformer**를 만들었습니다."

---

## 4. Model Architecture

대부분의 경쟁력 있는 Sequence Transduction 모델들은 인코더-디코더(Encoder-Decoder) 구조를 가지고 있습니다. 여기서 인코더는 기호 표현으로 이루어진 입력 시퀀스 $(x_1, \dots, x_n)$을 연속적인 표현 시퀀스 $\mathbf{z} = (z_1, \dots, z_n)$으로 매핑합니다. 

$\mathbf{z}$가 주어지면, 디코더는 출력 시퀀스 $(y_1, \dots, y_m)$의 기호 표현들을 한 번에 하나씩 생성해 냅니다. 각 단계에서 모델은 Auto-regressive(자기회귀) 성격을 띠며, 다음 기호를 생성할 때 이전에 생성된 기호들을 추가적인 입력으로 사용합니다.

Transformer는 인코더와 디코더 모두에 대해 Stacked Self-Attention과 Point-wise, Fully Connected Layer를 사용하며 전체적인 아키텍처를 구성합니다.

---

### 4.1. Encoder and Decoder Stacks

#### 인코더 (Encoder)
인코더는 $N = 6$개의 동일한 레이어 스택으로 구성되어 있습니다. 각 레이어는 두 개의 서브 레이어를 가집니다.

* **첫 번째 서브 레이어:** Multi-Head Self-Attention 메커니즘
* **두 번째 서브 레이어:** 단순한 Position-wise Fully Connected Feed-Forward 네트워크

✔️ **Residual Connection & Layer Normalization**
각각의 두 서브 레이어 주위에는 Residual Connection(잔차 연결)을 적용한 뒤, 이어서 Layer Normalization(레이어 정규화)을 수행합니다. 즉, 각 서브 레이어의 최종 출력은 다음과 같습니다.

$$\mathrm{LayerNorm}(x + \mathrm{Sublayer}(x))$$

여기서 $\mathrm{Sublayer}(x)$는 서브 레이어 자체에 의해 구현된 함수입니다. 이러한 Residual Connection을 편리하게 적용하기 위해, 모델 내의 모든 서브 레이어와 Embedding 레이어는 $d_{\mathrm{model}} = 512$ 차원의 출력을 생성하도록 통일되어 있습니다.

#### 디코더 (Decoder)
디코더 또한 $N = 6$개의 동일한 레이어 스택으로 구성됩니다. 인코더 레이어에 있는 두 개의 서브 레이어 외에도, 디코더는 세 번째 서브 레이어를 추가로 도입합니다.

* **추가된 서브 레이어:** 인코더 스택의 출력을 대상으로 Multi-Head Attention을 수행하는 레이어

인코더와 마찬가지로 각 서브 레이어 주위에 Residual Connection을 적용하고 Layer Normalization을 수행합니다. 

⚠️ **Masking 메커니즘**
디코더 스택 내의 Self-Attention 서브 레이어는 **Masking(마스킹)**을 적용하여 변형됩니다. 이 마스킹 설정은 Position $i$에 대한 예측이 오직 $i$보다 작은 위치에 있는 알고 있는 출력들에만 의존할 수 있도록 보장합니다. 즉, 미래의 시점을 미리 보고(Cheating) 예측하는 것을 물리적으로 차단합니다.

---

### 4.2. Attention

Attention 기능은 Query와 Key-Value 쌍의 집합을 출력에 매핑하는 것으로 설명할 수 있습니다. 여기서 Query, Keys, Values 및 출력은 모두 벡터입니다. 출력은 갹 Value의 가중 합으로 계산되며, 각 Value에 할당된 가중치는 Query와 해당 Key의 호환성 함수(Compatibility Function)에 의해 계산됩니다.

Query, Keys, Values 
           ↓
 [ Compatibility Function ] → 가중치(Weight) 계산
           ↓
  [ 가중치 × Values ] → 최종 출력(Weighted Sum)

#### 4.2.1. Scaled Dot-Product Attention
이 논문에서 사용하는 특정한 Attention을 **Scaled Dot-Product Attention**이라고 부릅니다. 입력은 $d_k$ 차원의 Query 및 Key들과 $d_v$ 차원의 Value들로 구성됩니다.

Query와 모든 Key들의 Dot Product(내적)를 계산한 뒤, 각각을 $\sqrt{d_k}$로 나눕니다. 그리고 Value들에 대한 가중치를 얻기 위해 Softmax 함수를 적용합니다.

실제 계산 시에는 여러 Query를 하나의 행렬 $Q$로 묶어 동시에 Attention 함수를 계산합니다. Key와 Value들 역시 행렬 $K$와 $V$로 packed 됩니다. 최종 출력 행렬은 다음과 같은 수식으로 나타납니다.

$$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V$$

---

### 💡 왜 $\sqrt{d_k}$로 스케일링(Scale) 해줄까?

가장 흔하게 쓰이는 Attention 메커니즘은 크게 두 가지가 있습니다.
1. **Additive Attention (부가적인 어텐션):** 단일 은닉 레이어를 가진 Feed-forward 네트워크를 사용하여 호환성 함수를 계산합니다.
2. **Dot-Product Attention (내적 어텐션):** 논문에서 채택한 방식으로, 스케일링 인자인 $\sqrt{d_k}$를 제외하면 구조가 동일합니다.

Dot-product attention은 고도로 최적화된 행렬 곱셈 코드를 사용하여 구현할 수 있기 때문에, **실제 연산 속도가 훨씬 빠르고 공간 효율적**이라는 강력한 장점이 있습니다.

$d_k$ 값이 작을 때는 두 메커니즘이 비슷한 성능을 보이지만, **$d_k$ 값이 커질수록 스케일링을 하지 않은 내적 어텐션은 부가적인 어텐션보다 성능이 떨어지는 현상**이 발생합니다.

⚠️ **성능 저하의 원인과 해결책**
$d_k$ 값이 매우 커지면, 내적(Dot product) 결과값의 크기 역시 전반적으로 커지게 됩니다. 이는 Softmax 함수를 통과할 때 극단적으로 작은 경사(Gradient)를 가진 영역으로 밀어 넣는 결과를 초래합니다. 즉, 소프트맥스 출력값이 특정 원소에만 1에 가깝게 몰리고 나머지는 0에 수렴하여 **경사 소실(Gradient Vanishing)** 문제가 발생하는 것입니다.

이러한 부작용을 상쇄하기 위해, 내적 결과값에 **$\sqrt{d_k}$로 나누어 주는 스케일링 과정**을 도입하여 제어합니다.

---

#### 4.2.2. Multi-Head Attention

$d_{\mathrm{model}}$ 차원의 Query, Key, Value들로 단 한 번의 단일 Attention을 수행하는 대신, Query, Key, Value들을 각각 서로 다르게 학습된 선형 투영(Linear Projection)을 통해 각각 $d_k$, $d_k$, $d_v$ 차원으로 $h$번 선형 투영하는 것이 훨씬 효과적이라는 것을 발견했습니다.

이렇게 투영된 Query, Key, Value들의 각 버전에 대해 Attention 함수를 병렬로 수행하여 $d_v$ 차원의 출력값들을 얻어냅니다. 이들을 하나로 Concat(연결)한 뒤 다시 한번 투영하여 최종 출력값을 도출합니다.

Multi-Head Attention 구조는 모델이 서로 다른 위치에 있는 **다양한 표현 서브스페이스(Representation Subspaces)의 정보에 동시에 공동으로 주목(Attend)**할 수 있도록 합니다. 단일 어텐션 헤드 구조에서는 가중 평균 연산으로 인해 이러한 정보 조합이 희석되어 불가능합니다.

$$\mathrm{MultiHead}(Q, K, V) = \mathrm{Concat}(\mathrm{head}_1, \dots, \mathrm{head}_h)W^O$$

$$\text{where } \mathrm{head}_i = \mathrm{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

이때 사용되는 학습 가능한 파라미터 행렬들의 차원은 다음과 같습니다.
* $W_i^Q \in \mathbb{R}^{d_{\mathrm{model}} \times d_k}$
* $W_i^K \in \mathbb{R}^{d_{\mathrm{model}} \times d_k}$
* $W_i^V \in \mathbb{R}^{d_{\mathrm{model}} \times d_v}$
* $W^O \in \mathbb{R}^{hd_v \times d_{\mathrm{model}}}$

논문에서는 $h = 8$개의 병렬 어텐션 헤드를 사용합니다. 각 헤드에 대해 차원을 다음과 같이 줄여서 할당합니다.

$$d_k = d_v = d_{\mathrm{model}} / h = 512 / 8 = 64$$

각 헤드의 차원이 줄어들었기 때문에, 전체 연산 비용은 전체 해상도를 가진 단일 헤드 어텐션의 연산 비용과 거의 유사합니다.

---

### 4.3. Applications of Attention in our Model

Transformer는 Multi-Head Attention을 다음 세 가지 서로 다른 방식으로 활용합니다.

1. **Encoder-Decoder Attention 레이어 (디코더에 위치):**
   * Query는 직전 디코더 레이어에서 오고, Key와 Value는 인코더의 최종 출력값에서 가져옵니다. 
   * 이를 통해 디코더의 모든 위치가 입력 시퀀스의 모든 위치에 골고루 주목할 수 있게 됩니다. (기존 RNN 기반 시퀀스 투 시퀀스 모델의 전형적인 어텐션 방식 메커니즘 모방)

2. **인코더의 Self-Attention 레이어:**
   * Query, Key, Value가 모두 인코더의 직전 레이어 출력이라는 동일한 소스에서 나옵니다.
   * 인코더의 각 위치는 인코더 직전 레이어의 모든 위치에 주목할 수 있습니다.

3. **디코더의 Self-Attention 레이어:**
   * 디코더의 각 위치가 해당 위치를 포함하여 그 이전의 모든 위치에 주목할 수 있도록 합니다.
   * ⚠️ **Auto-regressive 성격 보존:** 디코더에서 정보가 왼쪽 방향으로 흘러가 역방향성을 해치는 것을 방지해야 하므로, Scaled Dot-Product Attention 내부에서 무효한 위치의 모든 값을 Softmax의 입력 직전에 $-\infty$(음의 무한대)로 설정하는 마스킹을 구현합니다.

---

### 4.4. Position-wise Feed-Forward Networks

인코더와 디코더의 각 레이어에는 Attention 서브 레이어 외에도 Fully Connected Feed-Forward 네트워크가 포함되어 있습니다. 이는 각 위치에 별도로, 그리고 동일하게 적용되므로 **Position-wise**라고 부릅니다. 

이 구조는 두 번의 선형 변환과 그 사이에 ReLU 활성화 함수로 구성됩니다.

$$\mathrm{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

선형 변환은 서로 다른 위치에서 동일하게 적용되지만, 레이어와 레이어 사이에는 서로 다른 파라미터를 사용합니다. 이는 커널 크기가 1인 합성곱(Convolution)을 두 번 수행하는 것과 동일한 효과를 냅니다.

입력과 출력의 차원은 $d_{\mathrm{model}} = 512$이며, 내부 은닉 레이어의 차원은 $d_{ff} = 2048$로 확장되었다가 다시 축소됩니다.

---

### 💡 Embedding 벡터의 내적과 선형대수학적 의미

여기서 잠깐 리마인드를 하자면, 컴퓨터는 단어 자체를 이해하지 못하므로 고차원 공간 상의 좌표(벡터)로 변환하는 Embedding 과정을 거칩니다. 

이 벡터 공간 상에서 두 단어 벡터를 **내적(Dot Product)** 한다는 것은, 두 벡터가 가리키는 방향이 얼마나 일치하는지(유사한지) 측정하는 연산입니다. 문맥상 자주 함께 쓰이거나 의미가 유사한 단어들은 임베딩 벡터의 방향성이 비슷해져 내적값이 커지게 됩니다. 

즉, "모델이 단어의 의미와 관계를 깨우치며 데이터에 맞게 최적화되는 임베딩 벡터를 만들어 낸다"는 선형대수학적 배경이 이 아키텍처의 밑바탕에 깔려 있습니다.

---

### 4.5. Positional Encoding

Transformer는 RNN이나 CNN을 전혀 사용하지 않기 때문에, 문장 안에 단어들이 들어오는 **'순서나 위치 정보'가 모델 자체에는 전혀 입력되지 않습니다.** 모델 입장에서는 단어들이 순서 없이 사방으로 흩어진 주머니(Bag of Words)처럼 보일 수 있는 치명적인 문제가 존재합니다.

👆 **해결책: 위치 정보 주입**
이를 해결하기 위해 시퀀스 내 토큰의 상대적 또는 절대적 위치 정보를 주입하는 **Positional Encoding**을 인코더와 디코더의 가장 밑단(Input Embedding 직후)에 더해줍니다. Positional Encoding 벡터의 차원도 $d_{\mathrm{model}} = 512$로 똑같기 때문에, 기존 임베딩 벡터와 그대로 **더하기(Addition) 연산**을 수행할 수 있습니다.

본 논문에서는 다양한 주기를 가진 사인(Sine) 함수와 코사인(Cosine) 함수를 이용합니다.

$$
\begin{aligned} 
PE_{(\mathrm{pos}, 2i)} &= \sin\left(\frac{\mathrm{pos}}{10000^{2i / d_{\mathrm{model}}}}\right) \\ 
PE_{(\mathrm{pos}, 2i+1)} &= \cos\left(\frac{\mathrm{pos}}{10000^{2i / d_{\mathrm{model}}}}\right) 
\end{aligned}
$$

* **$\mathrm{pos}$:** 단어의 절대적인 위치 고유 번호
* **$i$:** 차원의 인덱스

이 함수를 선택한 이유는 선형 변환을 통해 상대적인 위치 차이를 쉽게 학습할 수 있고, 모델이 학습 중에 보지 못했던 더 긴 문장이 들어와도 안정적으로 확장(Extrapolate)할 수 있을 것이라 판단했기 때문입니다.

---