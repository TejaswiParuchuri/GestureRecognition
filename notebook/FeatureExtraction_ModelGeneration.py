import os
import numpy as np
import pandas as pd
import json
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import joblib
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("-choice", metavar=3, dest='choice', default='3', help="1 - for feature extraction 2 - for training model 3 - for both 1 & 2 ", type=int )
parser.add_argument('-fileName', help="enter file name to save features", metavar='feature_points.csv', action='store', dest='file_name', default='feature_points.csv', type=str)

#path of the folder where json files are there
jsons_path="E:/MC/Tuesday_Assignment_2_json (1)/"

def extract_features(file_name):
    #delete if there is already a file existing with that name in the location
    delete_if_exists(file_name)
    #go through each and every json file in the given location
    for root,directories,files in os.walk(jsons_path):
        for file in files:
            if file.endswith('.json'):
                with open(os.path.join(root,file)) as input_file:
                    file_data=json.load(input_file)
                    #for each image in the json extract the feature points
                    for image in file_data:
                        
                        feature_points=[]
                        #get all the points in the json for the image
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
                                    
                        #add label to the features extracted
                        if 'buy' in root.lower() or 'buy' in file.lower():
                            feature_points.append('buy') 
                        elif 'communicate' in root.lower() or 'communicate' in file.lower():
                            feature_points.append('communicate')
                        elif 'fun' in root.lower() or 'fun' in file.lower():
                            feature_points.append('fun')
                        elif 'hope' in root.lower() or 'hope' in file.lower():
                            feature_points.append('hope')
                        elif 'mother' in root.lower() or 'mother' in file.lower():
                            feature_points.append('mother')
                        elif 'really' in root.lower() or 'really' in file.lower():
                            feature_points.append('really')
                        #append it to the file
                        with open(jsons_path+file_name,'a') as output_file:
                            pd.DataFrame([feature_points]).to_csv(output_file,index=False, header=False)


def delete_if_exists(file_name): #to delete file
    if os.path.exists(os.path.join(jsons_path,file_name)):
        os.remove(os.path.join(jsons_path,file_name))
    return 

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
    
def train_model(file_name): #for training the model
    if (os.path.exists(os.path.join(jsons_path,file_name))!=True):
        print (file_name,' does not exist in given path')
        return 
    features=pd.read_csv(jsons_path+file_name)
    columns=[]
    for column in features.columns:
        columns.append(column)
    columns[-1]='label'
    features.columns=columns 
    features_array=features.values
    
    #split features and labels
    X= features_array[:,0:38]
    Y=features_array[:,38]
    
    #scale features
    scaling=preprocessing.MinMaxScaler()
    
    X_fitting=scaling.fit_transform(X)

    
    #split features ad lables into training and testing data
    train_x,test_x,train_y,test_y= model_selection.train_test_split(X_fitting,Y,test_size = 0.30,random_state=7)
    

    #K NearestNeightbors
    KNN(train_x,train_y,test_x,test_y)
    

    #Decision Tree
    DecisionTree(train_x,train_y,test_x,test_y)
    
    train_x,test_x,train_y,test_y= model_selection.train_test_split(X_fitting,Y,test_size = 0.30,random_state=40)
    #Random Forest
    RandomForest(train_x,train_y,test_x,test_y)
    
    
    #Neural Network classifier
    NeuralNetworks(train_x,train_y,test_x,test_y)
    
    joblib.dump(scaling,'scaling.pkl')
    
    
def KNN(train_x,train_y,test_x,test_y):
    #Knearest Neighbors
    knn=KNeighborsClassifier(n_neighbors=99,metric="euclidean")
    knn.fit(train_x,train_y)
    test=knn.predict(test_x)

    print('K Nearest Neighbors accuracy on testing data',accuracy_score(test_y,test))
    
    joblib.dump(knn,'knn.pkl')
    
    
    return

def DecisionTree(train_x,train_y,test_x,test_y):
    #Decision Tree
    decision_tree=DecisionTreeClassifier()
    decision_tree.fit(train_x,train_y)
    test=decision_tree.predict(test_x)
    
    print('Decision Tree accuracy on testing data:',accuracy_score(test_y,test))
    
    joblib.dump(decision_tree,'decision_tree.pkl')
    
    return
    
def RandomForest(train_x,train_y,test_x,test_y):
    #Raondom Forest
    random_forest=RandomForestClassifier()
    random_forest.fit(train_x,train_y)
    test=random_forest.predict(test_x)
    
    print('Random Forest accuracy on testing data:',accuracy_score(test_y,test))
    
    joblib.dump(random_forest,'random_forest.pkl')
    
    return

def NeuralNetworks(train_x,train_y,test_x,test_y):
    #Neural Network
    print('Neural Networks training will take some time. Please wait .......')
    
    mlp=MLPClassifier(hidden_layer_sizes=(30,30), max_iter=10000, solver='adam', random_state=21,tol=0.000000001)
    mlp.fit(train_x,train_y)
    test=mlp.predict(test_x)
    
    print('Neural Network classifier accuracy on testing data:',accuracy_score(test_y,test))
    
    joblib.dump(mlp,'mlp_classifier.pkl')
    
    return

if __name__ == "__main__":
    # This is the first function that will execute on executing the program by taking the command line parameters if there are any
    args = parser.parse_args()
    #print(args.choice,args.file_name)
    if args.choice==3:
        extract_features(args.file_name) 
        train_model(args.file_name)
    elif args.choice==2:
        train_model(args.file_name)
    elif args.choice==1:
        extract_features(args.file_name) 
    else:
        print('Enter correct choice')
    pass                    
               