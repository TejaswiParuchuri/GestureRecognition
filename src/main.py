from flask import Flask, request, jsonify
import joblib
import numpy as np
from collections import Counter


app = Flask(__name__)


@app.route('/predictgesture', methods=['POST'])
def hello():
    # get json from the incoming request
    input_json = request.get_json(force=True)
    scaling = joblib.load('scaling.pkl')
    knn = joblib.load('knn.pkl')
    decision_tree = joblib.load('decision_tree.pkl')
    random_forest = joblib.load('random_forest.pkl')
    mlp_classifier = joblib.load('mlp_classifier.pkl')
    prediction = dict()
    data_set = []
    # for each image in the json extract the feature points
    for image in input_json:
        feature_points = []
        wrist_right = np.array([image['keypoints'][10]['position']['x'], image['keypoints'][10]['position']['y']])
        wrist_left = np.array([image['keypoints'][9]['position']['x'], image['keypoints'][9]['position']['y']])
        elbow_right = np.array([image['keypoints'][8]['position']['x'], image['keypoints'][8]['position']['y']])
        elbow_left = np.array([image['keypoints'][7]['position']['x'], image['keypoints'][7]['position']['y']])
        nose = np.array([image['keypoints'][0]['position']['x'], image['keypoints'][0]['position']['y']])
        left_ear = np.array([image['keypoints'][1]['position']['x'], image['keypoints'][1]['position']['y']])
        right_ear = np.array([image['keypoints'][2]['position']['x'], image['keypoints'][2]['position']['y']])
        left_eye = np.array([image['keypoints'][3]['position']['x'], image['keypoints'][3]['position']['y']])
        right_eye = np.array([image['keypoints'][4]['position']['x'], image['keypoints'][4]['position']['y']])
        left_shoulder = np.array([image['keypoints'][5]['position']['x'], image['keypoints'][5]['position']['y']])
        right_shoulder = np.array([image['keypoints'][6]['position']['x'], image['keypoints'][6]['position']['y']])
        # find the difference between wrists and nose,ears,eyes,shoulders,elbows
        feature_points.append(np.linalg.norm(wrist_left-wrist_right))
        feature_points.append(np.linalg.norm(nose-wrist_left))
        feature_points.append(np.linalg.norm(nose-wrist_right))
        feature_points.append(np.linalg.norm(left_ear-wrist_left))
        feature_points.append(np.linalg.norm(left_ear-wrist_right))
        feature_points.append(np.linalg.norm(right_ear-wrist_left))
        feature_points.append(np.linalg.norm(right_ear-wrist_right))
        feature_points.append(np.linalg.norm(left_eye-wrist_left))
        feature_points.append(np.linalg.norm(left_eye-wrist_right))
        feature_points.append(np.linalg.norm(right_eye-wrist_left))
        feature_points.append(np.linalg.norm(right_eye-wrist_right))
        feature_points.append(np.linalg.norm(left_shoulder-wrist_left))
        feature_points.append(np.linalg.norm(left_shoulder-wrist_right))
        feature_points.append(np.linalg.norm(right_shoulder-wrist_left))
        feature_points.append(np.linalg.norm(right_shoulder-wrist_right))
        feature_points.append(np.linalg.norm(elbow_left-wrist_left))
        feature_points.append(np.linalg.norm(elbow_left-wrist_right))
        feature_points.append(np.linalg.norm(elbow_right-wrist_left))
        feature_points.append(np.linalg.norm(elbow_right-wrist_right))
        # find the difference between elbows and nose,ears,eyes,shoulders,wrists
        feature_points.append(np.linalg.norm(elbow_left-elbow_right))
        feature_points.append(np.linalg.norm(nose-elbow_left))
        feature_points.append(np.linalg.norm(nose-elbow_right))
        feature_points.append(np.linalg.norm(left_ear-elbow_left))
        feature_points.append(np.linalg.norm(left_ear-elbow_right))
        feature_points.append(np.linalg.norm(right_ear-elbow_left))
        feature_points.append(np.linalg.norm(right_ear-elbow_right))
        feature_points.append(np.linalg.norm(left_eye-elbow_left))
        feature_points.append(np.linalg.norm(left_eye-elbow_right))
        feature_points.append(np.linalg.norm(right_eye-elbow_left))
        feature_points.append(np.linalg.norm(right_eye-elbow_right))
        feature_points.append(np.linalg.norm(left_shoulder-elbow_left))
        feature_points.append(np.linalg.norm(left_shoulder-elbow_right))
        feature_points.append(np.linalg.norm(right_shoulder-elbow_left))
        feature_points.append(np.linalg.norm(right_shoulder-elbow_right))
        feature_points.append(np.linalg.norm(wrist_left-elbow_left))
        feature_points.append(np.linalg.norm(wrist_left-elbow_right))
        feature_points.append(np.linalg.norm(wrist_right-elbow_left))
        feature_points.append(np.linalg.norm(wrist_right-elbow_right))
        data_set.append(feature_points)
    # predict the gesture
    test_x = scaling.transform(np.array(data_set))[:, 0:38]
    knn_predict = knn.predict(test_x)
    decision_tree_predict = decision_tree.predict(test_x)
    random_forest_predict = random_forest.predict(test_x)
    mlp_classifier_predict = mlp_classifier.predict(test_x)
    # add it to dictionary to return
    prediction[1] = Counter(knn_predict.tolist()).most_common(1)[0][0]
    prediction[2] = Counter(decision_tree_predict.tolist()).most_common(1)[0][0]
    prediction[3] = Counter(random_forest_predict.tolist()).most_common(1)[0][0]
    prediction[4] = Counter(mlp_classifier_predict.tolist()).most_common(1)[0][0]
    # convert dictionary to json and return the result of prediction
    return jsonify(prediction)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)