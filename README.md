# Super Mario RL Student Kit

이 폴더는 학생에게 배포하는 슈퍼마리오 강화학습 프로젝트 기본 패키지입니다.
목표는 `SuperMarioBros-1-1-v0`부터 `SuperMarioBros-8-4-v0`까지 전체 32개 스테이지를 최대한 많이 클리어하는 것입니다.

학습 기본 환경은 매 에피소드 무작위 스테이지가 나오는 `SuperMarioBrosRandomStages-v0`이고, 최종 평가는 32개 고정 스테이지 전체에서 진행됩니다.
학생은 최종적으로 `agent.py`, `model.pt`, `train.py`를 제출합니다.

## 1. 설치

Python 3.10, 3.11 또는 3.12 사용을 권장합니다.

Windows PowerShell 기준:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Windows에서 프로젝트 경로에 대괄호, 괄호, 특수문자가 포함되어 `Activate.ps1` 실행이 실패하면 가상환경을 `C:\venvs\mario_rl`처럼 단순한 경로에 만든 뒤 사용하세요.

```powershell
python -m venv C:\venvs\mario_rl
C:\venvs\mario_rl\Scripts\Activate.ps1
cd path\to\project_student_kit
pip install -r requirements.txt
```

Windows에서 `nes-py` 설치가 실패하면 Microsoft C++ Build Tools의 `Desktop development with C++` 설치가 필요할 수 있습니다.

## 2. 폴더 구성

```text
project_student_kit/
├─ README.md
├─ SUBMISSION_SPEC.md
├─ requirements.txt
├─ mario_rl/
│  ├─ __init__.py
│  ├─ actions.py
│  ├─ config.py
│  ├─ env.py
│  └─ interface.py
└─ student_template/
   ├─ agent.py
   ├─ train.py
   ├─ make_submission.ps1
   └─ submission_meta.json
```

`mario_rl/`은 공통 환경 코드입니다. 학생은 이 폴더의 파일을 수정하지 않는 것을 원칙으로 합니다. 평가 서버도 동일한 관측/행동 규격을 사용합니다.

## 3. 평가 규격

최종 평가 기준:

```text
Environment: SuperMarioBros-1-1-v0 ~ SuperMarioBros-8-4-v0 전체 32개 스테이지
Observation shape: (4, 84, 84)
Observation dtype: uint8
Preprocessing: grayscale, 84x84 resize, frame stack 4, frame skip 4
Action space: Discrete(12)
```

행동 번호는 `gym_super_mario_bros.actions.COMPLEX_MOVEMENT`와 같은 순서입니다.

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

## 4. 학습

기본 PPO 예시를 실행하려면 다음과 같이 합니다. `--env`를 생략하면 랜덤 스테이지 환경인 `SuperMarioBrosRandomStages-v0`를 사용합니다.

```powershell
cd student_template
python train.py --total-steps 1000000 --out model.pt
```

제공된 `train.py`는 예시 코드입니다. 학생은 새로운 학습 파일을 만들거나, 네트워크 구조와 학습 알고리즘을 자유롭게 바꿀 수 있습니다. 단, 최종 제출물은 반드시 `agent.py`의 `Agent.load()`와 `Agent.act()` 인터페이스로 동작해야 합니다.

## 5. 제출 파일 만들기

학습 후 `student_template` 폴더 안에 최소 다음 세 파일이 있어야 합니다.

```text
agent.py
model.pt
train.py
```

제출 zip은 `make_submission.ps1`로 만들 수 있습니다.

```powershell
cd student_template
.\make_submission.ps1 -TeamId team03
```

생성되는 파일 예:

```text
team03_submission.zip
```

zip을 열었을 때 최상위에 `agent.py`와 `model.pt`가 바로 보여야 합니다. 추가 `.py` 파일을 import한다면 그 파일도 zip에 함께 들어가야 합니다.

## 6. 리더보드 제출

생성한 zip 파일을 리더보드 제출 페이지에 업로드합니다.

제출 후 리더보드 서버는 다음 과정을 자동으로 수행합니다.

```text
1. zip 저장
2. agent.py와 model.pt 존재 확인
3. Agent.load()로 model.pt 로드
4. 전체 32개 스테이지에서 Agent.act(obs)를 반복 호출
5. 클리어 수와 평균 진행도로 점수 계산
```

같은 `team_id`로 다시 제출하면 기존 제출은 새 제출로 교체됩니다. 단, 기존 제출이 평가 중인 경우에는 완료 후 다시 제출해야 합니다.

자세한 제출 규격은 `SUBMISSION_SPEC.md`를 따르세요.
