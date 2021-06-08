import os
import joblib
import json
import numpy as np
from collections import Counter



def extract_points(common_left,common_right,nose,left_ear,right_ear,left_eye,right_eye,left_shoulder,right_shoulder,other_left,other_right): #for extracting feature points
    
    feature_points=[]
    
    feature_points.append(np.linalg.norm(nose-common_left))
    feature_points.append(np.linalg.norm(nose-common_right))
    feature_points.append(np.linalg.norm(left_ear-common_left))
    feature_points.append(np.linalg.norm(left_ear-common_right))
    feature_points.append(np.linalg.norm(right_ear-common_left))
    feature_points.append(np.linalg.norm(right_ear-common_right))
    feature_points.append(np.linalg.norm(left_eye-common_left))
    feature_points.append(np.linalg.norm(left_eye-common_right))
    feature_points.append(np.linalg.norm(right_eye-common_left))
    feature_points.append(np.linalg.norm(right_eye-common_right))
    feature_points.append(np.linalg.norm(left_shoulder-common_left))
    feature_points.append(np.linalg.norm(left_shoulder-common_right))
    feature_points.append(np.linalg.norm(right_shoulder-common_left))
    feature_points.append(np.linalg.norm(right_shoulder-common_right))
    feature_points.append(np.linalg.norm(other_left-common_left))
    feature_points.append(np.linalg.norm(other_left-common_right))
    feature_points.append(np.linalg.norm(other_right-common_left))
    feature_points.append(np.linalg.norm(other_right-common_right))  
    
    return feature_points
    

gestures_count=0.0
knn_count=0.0
decision_tree_count=0.0
random_forest_count=0.0
mlp_classifier_count=0.0
#path of the folder where json files are there

jsons_path="E:/MC/uploadJSONS/uploadJSONS/"


scaling=joblib.load('scaling.pkl')
knn=joblib.load('knn.pkl')
decision_tree=joblib.load('decision_tree.pkl')
random_forest=joblib.load('random_forest.pkl')
mlp_classifier=joblib.load('mlp_classifier.pkl')
#go through each and every json file in the given location
for root,directories,files in os.walk(jsons_path):
    for file in files:
        if file.endswith('.json'):
            with open(os.path.join(root,file)) as input_file:
                file_data=json.load(input_file)
                data_set=[]
                #for each image in the json extract the feature points
                for image in file_data:
                    feature_points=[]
                    wrist_right=np.array([image['keypoints'][10]['position']['x'],image['keypoints'][10]['position']['y']])
                    wrist_left=np.array([image['keypoints'][9]['position']['x'],image['keypoints'][9]['position']['y']])
                    elbow_right=np.array([image['keypoints'][8]['position']['x'],image['keypoints'][8]['position']['y']])
                    elbow_left=np.array([image['keypoints'][7]['position']['x'],image['keypoints'][7]['position']['y']])
                    nose=np.array([image['keypoints'][0]['position']['x'],image['keypoints'][0]['position']['y']])
                    left_ear=np.array([image['keypoints'][1]['position']['x'],image['keypoints'][1]['position']['y']])
                    right_ear=np.array([image['keypoints'][2]['position']['x'],image['keypoints'][2]['position']['y']])
                    left_eye=np.array([image['keypoints'][3]['position']['x'],image['keypoints'][3]['position']['y']])
                    right_eye=np.array([image['keypoints'][4]['position']['x'],image['keypoints'][4]['position']['y']])
                    left_shoulder=np.array([image['keypoints'][5]['position']['x'],image['keypoints'][5]['position']['y']])
                    right_shoulder=np.array([image['keypoints'][6]['position']['x'],image['keypoints'][6]['position']['y']])
                    
                    #find the difference between wrists and nose,ears,eyes,shoulders,elbows
                    feature_points.append(np.linalg.norm(wrist_left-wrist_right))
                    feature_points=feature_points+extract_points(wrist_left,wrist_right,nose,left_ear,right_ear,left_eye,right_eye,left_shoulder,right_shoulder,elbow_left,elbow_right)
                    
                    #find the difference between elbows and nose,ears,eyes,shoulders,wrists
                    feature_points.append(np.linalg.norm(elbow_left-elbow_right))
                    feature_points=feature_points+extract_points(elbow_left,elbow_right,nose,left_ear,right_ear,left_eye,right_eye,left_shoulder,right_shoulder,wrist_left,wrist_right)
                    
                    data_set.append(feature_points)
                #predict the gesture
                test_x=scaling.transform(np.array(data_set))[:,0:38]
                knn_predict=knn.predict(test_x)
                decision_tree_predict=decision_tree.predict(test_x)
                random_forest_predict=random_forest.predict(test_x)
                mlp_classifier_predict=mlp_classifier.predict(test_x)
                
                gestures_count+=1
                #check gesture predicted by each model is correct or not
                if Counter(knn_predict.tolist()).most_common(1)[0][0] in root.lower() or Counter(knn_predict.tolist()).most_common(1)[0][0] in file.lower():
                    knn_count+=1
                if Counter(decision_tree_predict.tolist()).most_common(1)[0][0] in root.lower() or Counter(decision_tree_predict.tolist()).most_common(1)[0][0] in file.lower():
                    decision_tree_count+=1
                if Counter(random_forest_predict.tolist()).most_common(1)[0][0] in root.lower() or Counter(random_forest_predict.tolist()).most_common(1)[0][0] in file.lower():
                    random_forest_count+=1
                if Counter(mlp_classifier_predict.tolist()).most_common(1)[0][0] in root.lower() or Counter(mlp_classifier_predict.tolist()).most_common(1)[0][0] in file.lower():
                    mlp_classifier_count+=1
                    
#calculate the accuracy of gestures correctly predicted by the models
print("KNN accuracy is : ",knn_count/gestures_count)
print("DecisionTree accuracy is : ",decision_tree_count/gestures_count)
print("RandomForest accuracy is : ",random_forest_count/gestures_count)
print("NeuralNetwork accuracy is : ",mlp_classifier_count/gestures_count)                  