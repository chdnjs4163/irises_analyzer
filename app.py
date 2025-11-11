from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import plotly.express as px
import json
import plotly
import numpy as np

app = Flask(__name__)

# --- 데이터 준비 ---
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='species')
target_names = iris.target_names

model = None

@app.route('/')
def index():
    """메인 페이지를 렌더링합니다."""
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    """(시뮬레이터 전용) 이제 입력값이 있으면 그래프에 함께 표시합니다."""
    global model
    params = request.json
    algorithm = params.get('algorithm', 'dt')
    test_size = params.get('test_size', 0.3)
    
    # 프론트엔드에서 보낸 features(입력값) 데이터를 받습니다.
    features = params.get('features', None)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    if algorithm == 'knn':
        model = KNeighborsClassifier(n_neighbors=5)
    elif algorithm == 'lr':
        model = LogisticRegression(max_iter=200, random_state=42)
    else:
        model = DecisionTreeClassifier(random_state=42)
        
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    test_df = X_test.copy()
    test_df['실제 품종'] = y_test.map({i: name for i, name in enumerate(target_names)})
    test_df['예측 품종'] = pd.Series(y_pred, index=test_df.index).map({i: name for i, name in enumerate(target_names)})
    
    fig = px.scatter(
        test_df, x='petal length (cm)', y='petal width (cm)',
        color='실제 품종', symbol='예측 품종',
        title=f"'{algorithm.upper()}' 모델 예측 결과 (정확도: {accuracy:.2f})"
    )
    
    # 만약 features 데이터가 함께 전송되었다면, 그래프에 빨간 별을 추가합니다.
    if features:
        try:
            numeric_features = [float(f) for f in features]
            fig.add_scatter(
                x=[numeric_features[2]], y=[numeric_features[3]],
                mode='markers', marker=dict(color='red', size=15, symbol='star'),
                name='입력값'
            )
        except (ValueError, IndexError):
            pass

    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return jsonify({'accuracy': f"{accuracy * 100:.2f}%", 'graph_json': graph_json})

@app.route('/predict', methods=['POST'])
def predict():
    """(핵심 기능) 예측과 모델 시각화를 동시에 처리하여 모든 결과를 반환합니다."""
    global model
    try:
        features = [float(request.form[f]) for f in ['sl', 'sw', 'pl', 'pw']]
        features_arr = np.array([features])
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        prediction_idx = model.predict(features_arr)[0]
        prediction_name = target_names[prediction_idx]
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        test_df = X_test.copy()
        test_df['실제 품종'] = y_test.map({i: name for i, name in enumerate(target_names)})
        test_df['예측 품종'] = pd.Series(y_pred, index=test_df.index).map({i: name for i, name in enumerate(target_names)})
        
        fig = px.scatter(
            test_df, x='petal length (cm)', y='petal width (cm)',
            color='실제 품종', symbol='예측 품종',
            title=f"'결정 트리' 모델 예측 결과 (정확도: {accuracy:.2f})"
        )
        
        fig.add_scatter(
            x=[features[2]], y=[features[3]],
            mode='markers', marker=dict(color='red', size=15, symbol='star'),
            name='입력값'
        )
        
        graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        return jsonify({
            'prediction': prediction_name,
            'image_url': f"/static/images/{prediction_name}.jpg",
            'accuracy': f"{accuracy * 100:.2f}%",
            'graph_json': graph_json
        })
    except Exception as e:
        return jsonify({'error': f'오류 발생: {e}'})

if __name__ == '__main__':
   app.run(host='0.0.0.0', port=5000, debug=True)