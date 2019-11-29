from flask import Flask, send_from_directory, request
import pandas as pd
import pickle
import json

app = Flask(__name__)

@app.route('/predict', methods=['POST','GET'])
@app.route('/api/predict', methods=['POST','GET'])
def predict():
    json_ = request.get_json()
    print(json_)

    predict_df = pd.DataFrame(columns=field_names) # empty df for req data
    predict_df = predict_df.append(pd.Series(), ignore_index=True)

    if json_:
        predict_data = json_
    else:
        with open("sample_item.json", "rb") as file:
            predict_data = json.loads(file.read())
    for item in predict_data.items():
        if type(item[1]) != str:
            predict_df[item[0]][0] = item[1]
        else: # it's a string - onehot encoded
            for col_name, col_val in predict_df.iteritems():
                if(col_name.startswith(item[0])):
                    if col_name == item[0] + "-" + item[1]:
                        predict_df[col_name][0] = 1
                    else:
                        predict_df[col_name][0] = 0

    prediction = model.predict(predict_df)
    return_dict = {"prediction": prediction[0]}
    return(json.dumps(return_dict))

@app.route('/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route('/')
def send_root():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    field_names = pickle.load(open("column_names.pkl", "rb"))
    model = pickle.load(open("model.pkl", "rb"))
    app.run(port=8000)
