
#=================flask code starts here
from flask import Flask, render_template, request, session
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import html
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from keras.layers import MaxPooling2D, Convolution2D, Dense, Flatten, RepeatVector
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from sklearn.ensemble import RandomForestClassifier
from attention import attention

app = Flask(__name__)
app.secret_key = 'welcome'

# ================= DATA PREPROCESS =================
file_path = "Dataset/click_fraud_dataset.csv"

if not os.path.exists(file_path):
    file_path = "click_fraud_dataset.csv"

dataset = pd.read_csv(file_path)
labels = ['Human Clicked', 'Bot Clicked']

label_encoder = []
columns = dataset.columns
types = dataset.dtypes.values

for j in range(len(types)):
    if types[j] == 'object':
        le = LabelEncoder()
        dataset[columns[j]] = le.fit_transform(dataset[columns[j]].astype(str))
        label_encoder.append([columns[j], le])

dataset.fillna(dataset.mean(), inplace=True)

Y = dataset['is_fraudulent'].astype(int)
dataset.drop(['is_fraudulent'], axis=1, inplace=True)

X = dataset.values

selector = RFE(estimator=RandomForestClassifier(), n_features_to_select=16)
selector.fit(X, Y)
X = selector.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

data = np.load("model/data.npy", allow_pickle=True)
X_train, X_test, y_train, y_test = data

X_train1 = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1, 1))
y_train1 = to_categorical(y_train)

# ================= MODEL =================
def getModel():
    ext_model = Sequential()
    ext_model.add(Convolution2D(32, (1, 1), input_shape=(X_train1.shape[1], 1, 1), activation='relu'))
    ext_model.add(MaxPooling2D(pool_size=(1, 1)))
    ext_model.add(Convolution2D(32, (1, 1), activation='relu'))
    ext_model.add(MaxPooling2D(pool_size=(1, 1)))
    ext_model.add(Flatten())
    ext_model.add(RepeatVector(3))
    ext_model.add(attention(return_sequences=True))
    ext_model.add(Flatten())
    ext_model.add(Dense(256, activation='relu'))
    ext_model.add(Dense(y_train1.shape[1], activation='softmax'))
    ext_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    ext_model.load_weights("model/extension_weights.hdf5")
    return ext_model

# ================= ROUTES =================

# ================= graph =================
@app.route('/AllGraph')
def all_graph():

    algorithms = [
        'Logistic Regression','Decision Tree','Random Forest','KNN','ANN',
        'Gradient Boosting','LightGBM','XGBoost','Naive Bayes','SVM','CNN','DNN','RNN'
    ]

    accuracy = [97.267,98.933,77.467,73.200,72.400,98.000,98.533,98.667,89.333,73.800,95.667,73.800,98.733]
    precision = [96.573,98.044,76.877,48.649,60.286,96.454,97.349,97.579,85.492,36.900,93.212,36.900,97.825]
    recall = [96.343,99.277,59.049,49.922,55.698,98.645,99.006,99.097,92.527,50.000,96.326,50.000,98.978]
    fscore = [96.457,98.639,59.310,43.215,55.523,97.475,98.137,98.304,87.606,42.463,94.599,42.463,98.382]

    return render_template('all_graph.html',
                           labels=algorithms,
                           accuracy=accuracy,
                           precision=precision,
                           recall=recall,
                           fscore=fscore)


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/UserLogin')
def UserLogin():
    return render_template('UserLogin.html')

@app.route('/UserLoginAction', methods=['POST'])
def UserLoginAction():
    user = request.form['t1']
    password = request.form['t2']

    if user == "chetan" and password == "12345":
        return render_template('Dashboard.html', user=user)
    else:
        return render_template('UserLogin.html', msg="Invalid login")
    
    

# DASHBOARD PAGE
@app.route('/Dashboard')
def Dashboard():
    return render_template('Dashboard.html')

# PREDICTION PAGE
@app.route('/Predict')
def Predict():
    return render_template('Predict.html')

# RESULT GRAPH PAGE
@app.route('/Graph')
def graph():
    bot_percent = session.get('bot_percent', 0)
    return render_template('graph.html', prediction=bot_percent)
# # ALL GRAPH PAGE
# @app.route('/AllGraph')
# def all_graph():
#     generate_all_algorithm_graph()   
#     return render_template('all_graph.html')

# LOGOUT
@app.route('/Logout')
def Logout():
    return render_template('index.html')

# ================= PREDICTION =================
@app.route('/PredictAction', methods=['POST'])
def PredictAction():

    ext_model = getModel()

    # Read uploaded file
    if 'file' not in request.files or request.files['file'].filename == '':
        return render_template('Predict.html', msg="Please choose a CSV file before running prediction.")

    file = request.files['file']
    testData = pd.read_csv(file)

    # Match dataset structure
    testData = testData.reindex(columns=dataset.columns, fill_value=0)
    data = testData.copy()

    # Encoding
    for col_name, le in label_encoder:
        if col_name in testData.columns:
            try:
                testData[col_name] = le.transform(testData[col_name].astype(str))
            except:
                testData[col_name] = 0

    # Fill missing
    testData.fillna(dataset.mean(), inplace=True)

    # Feature selection
    testData_values = selector.transform(testData.values)

    # Reshape
    testData_values = np.reshape(testData_values, (testData_values.shape[0], testData_values.shape[1], 1, 1))

    # Prediction
    predict = ext_model.predict(testData_values)
    predict = np.argmax(predict, axis=1)

    # ================= STORE PERCENTAGE =================
    human = list(predict).count(0)
    bot = list(predict).count(1)

    total = human + bot
    if total > 0:
        bot_percent = round((bot / total) * 100, 2)
    else:
        bot_percent = 0

    session['bot_percent'] = bot_percent

    # ================= SAFE GRAPH =================
    try:
        human = list(predict).count(0)
        fraud = list(predict).count(1)

        labels_graph = ['Human', 'Bot ']
        values = [human, bot]

        if not os.path.exists("static"):
            os.makedirs("static")

        graph_path = os.path.join("static", "result.png")

        colors = ["#10b981", "#dc2626"]
        plt.figure(figsize=(6, 4))
        plt.bar(labels_graph, values, color=colors)
        plt.title("Bot Click Detection Results")
        plt.ylabel("Clicks")
        plt.tight_layout()
        plt.savefig(graph_path)
        plt.close()

        graph_html = "<img class='result-image' src='/static/result.png' alt='Prediction result graph'>"

    except Exception as e:
        print("Graph Error:", e)
        graph_html = "<p class='alert'>Graph could not be generated</p>"

    # ================= UI OUTPUT =================
    output = f"""
    <section class="prediction-card">
        <div class="prediction-header">
            <div>
                <p class="eyebrow">Prediction Dashboard</p>
                <h1>Click classification results</h1>
                <p class="muted">Each uploaded row is classified using the trained deep learning model.</p>
            </div>
            <div class="result-summary">
                <span class="summary-pill">{human} human</span>
                <span class="summary-pill">{bot} fraud</span>
                <span class="summary-pill">{bot_percent}% bot clicks</span>
            </div>
        </div>
        {graph_html}
        <div class="table-wrap">
            <table class="results-table">
                <thead>
                    <tr><th>Data</th><th>Prediction</th><th>Reason</th></tr>
                </thead>
                <tbody>
    """

    for i in range(len(predict)):
        output += "<tr>"
        row_data = html.escape(str(list(data.iloc[i])))
        output += f"<td>{row_data}</td>"

        if predict[i] == 0:
            output += "<td><span class='badge badge-human'>Human</span></td>"
        else:
            output += "<td><span class='badge badge-fraud'>Fraud</span></td>"

        output += "<td>ML-based prediction</td>"
        output += "</tr>"

    output += """
                </tbody>
            </table>
        </div>
    </section>
    """

    return render_template('UserScreen.html', msg=output)

# ================= RUN =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

