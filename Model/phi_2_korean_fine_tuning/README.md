# 일단 이 친구는 fine-tuning에 실패했습니다.
- 원인에 대해서는 loss가 4점대까지만 떨어지지 않은 것이 원인일 수도 있습니다.
- test set을 정확히 사용하지 않은 것이 문제일 수도 있습니다.
- loss 값을 적절히 사용하지 않은 것이 문제일 수도 있습니다.
- 하이퍼파라미터가 문제일 수도 있습니다.

## 이 디렉토리를 사용하는 방법은 다음과 같습니다.
### 모델 학습
- 모델 학습에 대한 코드는 KoAlpaca_dataset_fine_tuning_study.ipynb와 KoAlpaca_dataset_fine_tuniing_phi_2.py입니다.
    - 둘의 차이는 공부하면서 하나씩 돌린 ipynb file과 돌리고 기다린 py file의 차이일 뿐 차이는 존재하지 않습니다.
### 모델 로드 방법
- model을 load하는 방법은 load_model.ipynb에 작성했습니다.
- 이 LLM이 얼마나 부족한지는 코드 결과를 보면 알 수 있습니다.