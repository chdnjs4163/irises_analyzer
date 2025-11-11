🌸 대화형 붓꽃 품종 분석기 (Interactive Iris Species Classifier)

머신러닝을 이용해 실시간으로 붓꽃 품종을 예측하는 풀스택(Full-Stack) 웹 애플리케이션입니다.

Flask 백엔드가 Scikit-learn 모델의 학습 및 예측을 담당하고, 프론트엔드에서는 사용자의 입력을 받아 Plotly.js로 결과를 동적으로 시각화합니다. 사용자는 직접 붓꽃의 특징을 입력하여 품종을 예측해볼 수 있으며, 다양한 ML 모델의 학습 과정을 시뮬레이션할 수도 있습니다.

🖥️ 실행 화면 (Screenshot)
<img width="502" height="760" alt="image" src="https://github.com/user-attachments/assets/15644478-94ee-44ed-9dc0-90d39f48c988" />


✨ 주요 기능

* 🌸 **실시간 품종 예측**: 4가지 붓꽃 특징(꽃받침/꽃잎의 길이와 너비)을 입력하면, 학습된 모델이 즉시 품종(Setosa, Versicolor, Virginica)을 예측하여 결과와 이미지를 보여줍니다.
* 📊 **인터랙티브 시각화**: Plotly를 사용하여 모델의 테스트 데이터셋 분포(실제값 vs 예측값)를 인터랙티브 산점도로 시각화합니다.
* 📌 **사용자 입력값 표시**: 예측 시, 사용자가 입력한 값이 산점도 그래프 위에 '빨간색 별'로 표시되어 데이터 분포 내 위치를 직관적으로 확인할 수 있습니다.
* 🤖 **ML 모델 학습 시뮬레이터**: 결정 트리(Decision Tree), K-최근접 이웃(KNN), 로지스틱 회귀(Logistic Regression) 3가지 모델 중 하나를 선택하여 즉시 학습시키고, 그 결과(정확도, 시각화 그래프)를 비교할 수 있습니다.

🛠️ 아키텍처 구조

<img width="424" height="803" alt="image" src="https://github.com/user-attachments/assets/54a6869f-31c0-436a-84b7-8db3329edb95" />

⚙️ 기술 스택 (Tech Stack)

🖥️ Frontend
* HTML5 / CSS3
* JavaScript (ES6+)
* Fetch API (비동기 통신)
* Plotly.js (데이터 시각화)

⚙️ Backend
* Python 3
* Flask (웹 프레임워크 및 API 서버)
* Scikit-learn (머신러닝 모델 학습/예측)
* Pandas / NumPy (데이터 처리)
* Plotly (Python Lib, 그래프 객체 생성)

🚀 설치 및 실행 방법

 1. 저장소 클론 (Clone)
```bash
git clone [https://github.com/](https://github.com/)[Your-Username]/[Your-Repository-Name].git
cd [Your-Repository-Name]
2. 가상 환경 생성 및 활성화 (권장)
Bash

Python 가상 환경 생성
python -m venv venv

가상 환경 활성화 (Windows)
.\venv\Scripts\activate

가상 환경 활성화 (macOS/Linux)
source venv/bin/activate
3. 필요 라이브러리 설치
Bash

pip install flask pandas scikit-learn plotly numpy
4. 붓꽃 이미지 준비
이 프로젝트는 예측 결과에 맞는 이미지를 동적으로 보여줍니다. static/images 폴더를 생성하고, 그 안에 아래 3개의 붓꽃 이미지를 넣어주세요.

static/
└── images/
    ├── setosa.jpg
    ├── versicolor.jpg
    └── virginica.jpg
5. Flask 서버 실행
Bash

python app.py
6. 웹 브라우저 접속
서버가 실행되면 웹 브라우저를 열고 다음 주소로 접속합니다.

https://www.google.com/search?q=http://127.0.0.1:5000

💡 향후 개선 사항
모델 로딩 최적화: 현재 API가 호출될 때마다 모델을 실시간으로 재학습합니다. pickle 등을 이용해 미리 학습된 모델을 파일로 저장하고, 서버 시작 시 메모리에 한 번만 로드하는 방식으로 리팩토링하여 서버 응답 속도를 개선할 수 있습니다.
