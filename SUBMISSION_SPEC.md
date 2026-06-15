# Submission Specification

이 문서는 슈퍼마리오 강화학습 프로젝트의 최종 제출 규격입니다.

## 1. 최종 제출 형식

최종 제출은 zip 파일 하나입니다.

제출 zip의 최상위에는 최소 다음 세 파일이 있어야 합니다.

```text
agent.py
model.pt
train.py
```

권장 예:

```text
team03_submission.zip
├─ agent.py
├─ model.pt
├─ train.py
└─ submission_meta.json
```

`agent.py`가 추가 파이썬 파일을 import한다면 해당 파일도 zip에 함께 포함해야 합니다.

`train.py`는 평가 서버가 직접 실행하지 않습니다. 다만 제출 모델을 어떻게 학습했는지 확인하기 위한 필수 제출 파일입니다.

예:

```python
from my_network import MyPolicy
```

위와 같은 코드가 있다면 제출 zip 안에 `my_network.py`도 있어야 합니다.

## 2. 필수 Agent 인터페이스

평가 서버는 학생의 학습 코드를 실행하지 않습니다. 평가 서버는 제출된 `agent.py`에서 `Agent` 클래스를 import하고 다음 두 메서드만 사용합니다.

```python
agent = Agent.load("model.pt", observation_space, action_space, device)
action = agent.act(observation)
```

따라서 `agent.py`는 반드시 다음 구조를 만족해야 합니다.

```python
class Agent:
    @classmethod
    def load(cls, path, observation_space, action_space, device="cpu"):
        ...

    def act(self, observation):
        ...
```

`load()`는 `model.pt`를 읽어서 평가 가능한 agent 객체를 반환해야 합니다.

`act()`는 관측 하나를 받아 `0`부터 `11` 사이의 정수 action 하나를 반환해야 합니다.

## 3. 평가 환경

최종 평가는 32개 고정 스테이지 전체에서 진행됩니다.

```text
Environment: SuperMarioBros-1-1-v0 ~ SuperMarioBros-8-4-v0
Number of stages: 32
Observation shape: (4, 84, 84)
Observation dtype: uint8
Preprocessing: grayscale, 84x84 resize, frame stack 4, frame skip 4
Action space: Discrete(12)
```

행동 번호는 `gym_super_mario_bros.actions.COMPLEX_MOVEMENT`와 같은 순서를 따릅니다.

```text
0: NOOP
1: RIGHT
2: RIGHT_JUMP
3: RIGHT_RUN
4: RIGHT_RUN_JUMP
5: JUMP
6: LEFT
7: LEFT_JUMP
8: LEFT_RUN
9: LEFT_RUN_JUMP
10: DOWN
11: UP
```

학생이 학습할 때도 이 관측/행동 규격을 맞추는 것을 권장합니다. `mario_rl/env.py`와 `mario_rl/actions.py`를 수정해 학습하면 평가 서버의 환경과 달라져 성능이 나오지 않거나 실행이 실패할 수 있습니다.

주의: 이전 7개 행동(`Discrete(7)`)으로 학습한 모델은 현재 평가 환경의 12개 행동 정책과 호환되지 않습니다.

## 4. 학습 환경

제공된 `student_template/train.py`의 기본 학습 환경은 다음과 같습니다.

```text
SuperMarioBrosRandomStages-v0
```

이 환경은 매 에피소드 무작위 스테이지를 사용하므로 32개 전체 스테이지 평가에 맞춘 일반화 학습에 더 적합합니다.

## 5. model.pt 규칙

`model.pt`의 내부 형식은 자유입니다. 단, 제출한 `agent.py`의 `Agent.load()`가 그 파일을 읽을 수 있어야 합니다.

즉 다음 한 쌍이 서로 맞아야 합니다.

```text
agent.py
model.pt
```

예를 들어 PyTorch `state_dict`를 저장했다면 `Agent.load()`도 같은 구조의 네트워크를 만들고 `load_state_dict()`를 호출해야 합니다.

## 6. 라이브러리 규칙

평가 서버는 제출 zip 안의 `requirements.txt`를 자동 설치하지 않습니다.

기본적으로 학생은 제공된 `project_student_kit/requirements.txt` 안의 라이브러리만 사용해야 합니다.

기본 허용 라이브러리:

```text
numpy
gymnasium
torch
gym-super-mario-bros
nes-py
imageio
opencv-python
```

추가 라이브러리를 사용하려면 사전에 허용 여부를 확인해야 합니다. 평가 서버에 설치되지 않은 라이브러리를 import하면 제출물은 실행 실패 처리될 수 있습니다.

## 7. 금지 사항

평가 중 `act()` 안에서 학습을 새로 시작하면 안 됩니다. `act()`는 이미 학습된 모델로 행동만 선택해야 합니다.

금지되는 예:

```text
act() 안에서 환경 생성
act() 안에서 모델 재학습
외부 인터넷 접속
절대경로 의존
교수자 PC의 특정 파일 경로 의존
무한 루프 또는 과도한 파일 쓰기
평가 서버 환경 변경 시도
```

## 8. 점수 계산

평가 점수는 게임 내 score가 아니라 스테이지 클리어 수와 진행도로 계산합니다.

```text
raw_score = clear_count * 100000 + mean_progress
```

의미:

```text
clear_count: 32개 스테이지 중 클리어한 스테이지 수
mean_progress: 각 스테이지의 최대 도달거리 기반 평균 진행도
```

클리어 수가 최우선 기준이고, 클리어 수가 같을 때 평균 진행도가 순위를 가릅니다.
