알겠습니다. 우리가 논의 중인 연구의 맥락을 다음과 같이 명확하게 정의할 수 있습니다.

---

### 1. 문제 정의: 선형 모델의 한계

차량 제어, 특히 MPC와 같은 모델 기반 예측 제어(Model-Based Predictive Control)의 성능은 플랜트 모델의 정확성에 의해 결정됩니다.

전통적인 차량 동역학 제어는 계산 비용이 저렴한 선형 타이어 모델(e.g., $F_y = C_{\alpha} \alpha$)에 기반한 바이시클 모델(Bicycle Model)을 $f_{\text{nominal}}$로 사용합니다. 이 모델은 일상적인 주행(small $\alpha$ 영역)에서는 유효하지만, 자율주행의 핵심인 **극한 거동(Limit-handling) 시나리오**에서는 치명적인 한계를 가집니다.

$\alpha$가 커지는 비선형 영역에 진입하면, 선형 모델은 타이어 횡력의 **포화(Saturation)**, **복합 슬립(Combined-slip) 효과**, **수직력($F_z$)에 의한 특성 변화** 등을 전혀 예측하지 못합니다. 이 모델 불일치(Model Mismatch)는 MPC의 예측을 오염시켜, 제어 성능 저하, 제약 조건(Constraint) 위반, 심각한 경우 제어기 불안정성(Instability)을 초래합니다.

---

### 2. 핵심 방법론: GP-Augmented MPC

본 연구는 이 문제를 $f_{\text{nominal}}$을 폐기하는 것이 아니라, 데이터 기반 모델로 **보강(Augment)**하는 접근을 취합니다.

- **Residual Modeling:** 실제 차량 동역학 $f_{\text{true}}$와 $f_{\text{nominal}}$ 사이의 차이, 즉 **Residual($d = f_{\text{true}} - f_{\text{nominal}}$)**을 모델링합니다. 이 $d$는 우리가 무시했던 모든 비선형 물리 현상(포화, 커플링 등)을 포함합니다.
- **Gaussian Process (GP) 적용:** $d$를 모델링하기 위한 비모수적(Non-parametric) 회귀 방법으로 GP를 선택합니다. GP는 복잡한 비선형 함수를 학습할 수 있을 뿐만 아니라, 예측의 **불확실성(Uncertainty, $\sigma^2_{GP}$)**을 정량적으로 제공한다는 핵심 이점을 가집니다.
- **실시간 구현 (Sparse GP):** 표준 GP는 MPC의 실시간 요구사항(e.g., 20ms)을 만족할 수 없으므로, Inducing points를 사용하는 **Sparse GP (SGP)**를 오프라인에서 학습시켜, 온라인에서는 $O(M^2)$의 빠른 예측($\mu_{GP}, \sigma^2_{GP}$)이 가능하도록 합니다. 이것은 모든 것을 구현한 뒤 추후에 고려한다.

---

### 3. 기대 기여: 불확실성 기반 강건 제어

단순히 GP의 평균값($\mu_{GP}$)을 $f_{\text{nominal}}$에 더해 예측 정확도를 높이는 것($\dot{x} = f_{\text{nominal}} + \mu_{GP}$)은 첫 번째 단계에 불과합니다.

본 연구의 핵심 기여는 GP가 제공하는 **불확실성($\sigma^2_{GP}$)**을 MPC의 최적화 문제에 명시적으로 통합하는 것입니다.

- **Uncertainty Propagation:** $\sigma^2_{GP}$를 예측 호라이즌(Prediction Horizon) 동안 전파(Propagate)하여, MPC가 각 예측 상태($x_k$)의 신뢰도를 인지하도록 합니다.
- **Chance-Constrained MPC:** 이 불확실성을 사용하여 "95%의 확률로 $F_y$가 $\mu F_z$를 넘지 않는다"와 같은 확률적 제약(Chance-Constraint)을 수립합니다.
- **결과 (Robust Control):** GP가 학습 데이터가 부족하여 높은 $\sigma^2_{GP}$를 보고하는 영역(e.g., Post-peak 영역, 미경험 노면)에 진입하려 하면, MPC는 Chance-Constraint를 만족시키기 위해 **자동으로 더 보수적인(Conservative) 제어** 입력을 선택합니다.

요약하자면, 본 연구의 맥락은 **"Sparse GP를 통해 선형 타이어 모델의 비선형성을 보강하고, GP가 제공하는 불확실성을 Chance-Constraint MPC에 통합함으로써, 극한의 주행 상황에서도 안전성과 강건성(Robustness)을 보장하는 고성능 자율주행 제어기를 개발하는 것"**입니다.
