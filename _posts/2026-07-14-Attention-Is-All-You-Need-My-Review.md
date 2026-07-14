---
title: "총정리 및 어려웠던 부분"
date: 2026-07-14 15:00:00 +0900
categories: [논문리뷰, Attention Is All You Need]
tags: [deep-learning, transformer, attention, nlp]
math: true
---

---
title: "[Paper Review] Attention Is All You Need: Transformer 완벽 해부 및 심화 분석"
date: 2026-07-14 18:20:00 +0900
categories: [AI Study, Paper Review]
tags: [deep learning, nlp, transformer, attention, paper review, machine learning]
---

안녕하세요. 오늘은 자연어 처리(NLP)를 넘어 최신 인공지능 트렌드의 근간이 된 기념비적인 논문, **"Attention Is All You Need"**를 심층 리뷰해 보겠습니다. 

머신러닝과 딥러닝 모델링을 공부하다 보면 반드시 마주치게 되는 이 논문은, 기존의 순환(Recurrence) 구조를 완전히 배제하고 오직 Attention 메커니즘만으로 최고 수준의 성능을 이끌어낸 아키텍처, **Transformer**를 제안합니다.

---

## 1. Introduction: 기존 모델의 근본적 한계와 Global Dependency

기존 Sequence Transduction 모델들의 주류는 인코더와 디코더를 결합한 복잡한 RNN(LSTM, GRU 등)이나 CNN이었습니다. 

* **RNN 계열의 순차적 계산(Sequential Nature) 문제:** 현재 상태의 은닉 벡터 h_t를 계산하려면 반드시 이전 상태의 은닉 벡터 h_t-1이 필요합니다. 이는 메모리 제약을 발생시키고, 무엇보다 **연산의 병렬화(Parallelization)**를 근본적으로 불가능하게 만듭니다. 문장이 길어질수록 병목 현상은 심해집니다.
* **CNN 계열(ConvS2S, ByteNet)의 한계:** 병렬 처리를 위해 CNN을 도입한 시도도 있었지만, 이 모델들은 멀리 떨어진 두 단어의 신호를 연결하기 위해 수많은 레이어를 거쳐야 했습니다(선형 또는 로그 스케일로 연산량 증가). 

**Transformer의 핵심 아이디어**는 순차 연산을 과감히 버리고, **Global Dependency(전역적 의존성)**를 오로지 Attention만으로 잡아내자는 것입니다. 
예를 들어, *"과거에 수많은 실패를 겪었던 그 청년은, 마침내 거대한 글로벌 기업의 CEO가 되었다."* 라는 문장에서 '청년'과 'CEO'는 물리적 거리가 멀지만 강하게 연결되어야 합니다. Transformer는 문장 전체를 한 번에 조망하여 이러한 장거리 의존성을 효과적으로 포착합니다.

---

## 2. Model Architecture: 거대한 인코더-디코더의 재구성

Transformer는 큰 틀에서 인코더-디코더 구조를 유지하지만, 내부의 RNN 셀들을 들어내고 그 자리를 **Self-Attention**과 **Point-wise Feed-Forward Network**로 채웠습니다.

### 2.1. Encoder Stack
총 6개(N=6)의 레이어를 층층이 쌓아 올립니다. 각 레이어는 다음 2개의 Sub-Layer로 구성됩니다.
1.  **Multi-Head Self-Attention**
2.  **Position-wise Feed-Forward Network (FFN)**

**💡 아키텍처 디테일: 잔차 연결과 레이어 정규화**
각 서브 레이어를 통과할 때마다 모델은 다음과 같은 연산을 수행합니다.
* **LayerNorm(x + Sublayer(x))**

여기서 `x + Sublayer(x)`는 **잔차 연결(Residual Connection)**을 의미합니다. 6층이나 되는 깊은 신경망에서 앞쪽의 중요한 정보가 흐려지거나 미분값이 0으로 수렴하는 **경사 소실(Gradient Vanishing)**을 막기 위해 원래의 입력값 x를 그대로 더해줍니다. 이후 **레이어 정규화(Layer Normalization)**를 통해 입력 분포를 일정하게 유지시켜 학습 속도와 안정성을 극대화합니다. (모든 출력 차원은 d_model = 512로 통일됩니다.)

### 2.2. Decoder Stack
디코더 역시 6개의 레이어를 쌓지만, 인코더의 결과를 받아오기 위해 총 3개의 Sub-Layer를 가집니다.
1.  **Masked Multi-Head Self-Attention:** 디코더는 단어를 자기회귀적(Auto-regressive)으로 순차 생성해야 합니다. 뒤에 나올 미래의 단어를 미리 '커닝'하는 것을 막기 위해 마스킹(Masking)을 적용합니다.
2.  **Encoder-Decoder Attention:** 디코더의 위치가 입력 문장(인코더 내용) 중 어디에 집중해야 할지 결정합니다.
3.  **Position-wise Feed-Forward Network**

---

## 3. Attention: 모델의 심장

이 논문에서 Attention은 **"Query(질문)를 Key(식별자)-Value(값) 쌍과 연관 지어 가중합(Weighted sum)을 구하는 과정"**으로 정의됩니다.

### 3.1. Scaled Dot-Product Attention
Transformer는 가장 연산이 빠르고 공간 효율적인 Dot-Product(내적) 방식을 사용합니다.

* **Attention(Q, K, V) = softmax(QK^T / √d_k) * V**

**❓ 왜 √d_k로 스케일링을 할까요?**
Key 벡터의 차원 수(d_k)가 커지면 내적(QK^T)의 결괏값 자체가 지나치게 커집니다. 이 큰 값이 softmax 함수를 그대로 통과하면, 함수의 양극단으로 값이 쏠리며 기울기가 거의 0에 가까워지는 **경사 소실 구간**에 빠지게 됩니다. 이를 방지하기 위해 차원의 제곱근(√d_k)으로 값을 나누어 안정적인 기울기를 확보하는 수학적 테크닉입니다.

### 3.2. Multi-Head Attention
512차원의 큰 벡터를 한 번에 연산하는 대신, 선형 사영(Linear Projection)을 통해 작은 차원으로 8번(h=8) 쪼개어 병렬로 어텐션을 수행합니다. (d_k = d_v = 512 / 8 = 64차원)

* **MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W^O**

이 방식의 강력함은 모델이 **"다양한 표현 부분 공간(Representation Subspaces)"**에 동시에 집중할 수 있다는 점입니다. 하나의 헤드는 주어-동사의 문법적 관계를 추적하고, 다른 헤드는 지시대명사가 가리키는 대상을 추적하는 식으로 훨씬 입체적인 문맥 파악이 가능해집니다.

---

## 4. 모델의 약점 극복: Positional Encoding

Transformer에는 RNN이나 CNN이 없으므로 단어의 '위치나 순서'를 알 수 없습니다. 컴퓨터 입장에서는 문장이 그저 단어들이 마구잡이로 섞인 주머니(Bag of Words)처럼 보일 수 있습니다. 이를 해결하기 위해 입력 임베딩에 위치 정보를 더해주는 **Positional Encoding**을 사용합니다.

저자들은 주기가 다른 사인(sin), 코사인(cos) 함수를 사용했습니다.
* **PE(pos, 2i) = sin(pos / 10000^(2i / d_model))**
* **PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))**

*(pos는 단어의 절대적 위치, i는 차원의 인덱스)*

**❓ 왜 하필 삼각함수인가요?**
고정된 오프셋 k에 대해, PE(pos+k)를 PE(pos)의 선형 결합으로 표현할 수 있기 때문입니다. 즉, 모델이 단어들 사이의 '상대적인 거리'를 규칙적으로 아주 쉽게 학습할 수 있는 환경을 제공합니다.

---

## 5. Conclusion: 왜 Self-Attention이어야 하는가?

논문은 RNN, CNN 대비 Self-Attention의 당위성을 세 가지로 증명합니다.

1.  **계산 복잡도 (Computational Complexity):** 문장 길이(n)가 차원 수(d)보다 짧은 대부분의 기계 번역 환경에서, Self-Attention (O(n^2 * d))은 RNN (O(n * d^2))보다 레이어당 총 연산량이 적습니다.
2.  **병렬화 (Parallelization):** 순차적인 연산이 필요 없어 O(1)의 횟수로 병렬 처리가 가능합니다. (RNN은 O(n) 강제)
3.  **장거리 의존성 (Long-Range Dependencies):** 네트워크 내에서 어떤 두 단어가 신호를 주고받는 최대 경로 길이가 O(1)로 짧아져, 정보의 손실 없이 문맥을 완벽히 파악합니다.

Transformer는 딥러닝 아키텍처의 패러다임을 순환 연산에서 **병렬 어텐션**으로 완벽히 전환시킨 기념비적인 연구이며, 오늘날 BERT, GPT 등 대규모 언어 모델(LLM)의 튼튼한 뼈대가 되었습니다.