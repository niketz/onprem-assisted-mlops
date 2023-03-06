import os
import argparse
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, accuracy_score
import pandas as pd
import numpy as np
import re
import seaborn as sn
import matplotlib.pyplot as plt
import joblib
import mlflow
from urllib.parse import urlparse
import yaml
class Classification():

    def __init__(self, args):
        '''
        Initialize Steps
        ----------------
            1. Initalize Azure ML Run Object
            2. Create directories
        '''
        self.args = args
        os.makedirs('./model_metas', exist_ok=True)
        self.random_state = 1984

    def get_data(self, file_name):
        '''
        Get the input CSV file from workspace's default data store
        Args :
            container_name : name of the container to look for input CSV
            file_name : input CSV file name inside the container
        Returns :
            data_ds : Azure ML Dataset object
        '''
        print("DEBUG ---------------------------------------------------------")
        print(file_name)
        data_ds=pd.read_csv(file_name)
        return data_ds

    def read_params(self,config_path):
        with open (config_path) as yaml_file:
            config = yaml.safe_load(yaml_file)
        return config

    def create_confusion_matrix(self, y_true, y_pred, csv, png):
        '''
        Create confusion matrix
        '''

        try:
            confm = confusion_matrix(y_true, y_pred, labels=np.unique(y_pred))
            print("Shape : ", confm.shape)

            df_cm = pd.DataFrame(confm, columns=np.unique(y_true), index=np.unique(y_true))
            df_cm.index.name = 'Actual'
            df_cm.columns.name = 'Predicted'
            df_cm.to_csv(csv, index=False)
            # self.run.upload_file(name="./outputs/"+name+".csv",path_or_stream=name+".csv")

            plt.figure(figsize = (120,120))
            sn.set(font_scale=1.4)
            c_plot = sn.heatmap(df_cm, fmt="d", linewidths=.2, linecolor='black',cmap="Oranges", annot=True,annot_kws={"size": 16})
            plt.savefig(png)
            # self.run.log_image(name=name, plot=plt)

        except Exception as e:
            #traceback.print_exc()
            print(e)
            logging.error("Create consufion matrix Exception")

    def create_outputs(self, y_true, y_pred, X_test, name):
        '''
        Create prediction results as a CSV
        '''
        pred_output = {"Actual "+self.args.target_column : y_true[self.args.target_column].values, "Predicted "+self.args.target_column: y_pred[self.args.target_column].values}
        pred_df = pd.DataFrame(pred_output)
        pred_df = pred_df.reset_index()
        X_test = X_test.reset_index()
        final_df = pd.concat([X_test, pred_df], axis=1)
        final_df.to_csv(name, index=False)
        # self.run.upload_file(name="./outputs/"+name+".csv",path_or_stream=name+".csv")

    def validate(self, y_true, y_pred, X_test, config):
        Precision=round(precision_score(y_true, y_pred, average='weighted'), 2)
        Recall=round(recall_score(y_true, y_pred, average='weighted'), 2)
        Accuracy=round(accuracy_score(y_true, y_pred), 2)
        self.create_confusion_matrix(y_true, y_pred, config['reports']['confusion_matrix_csv'], config['reports']['confusion_matrix_png'])
        y_pred_df = pd.DataFrame(y_pred, columns = [self.args.target_column])
        self.create_outputs(y_true, y_pred_df,X_test, config['reports']['predictions'])
        # self.run.tag(self.args.tag_name)
        return Precision, Recall, Accuracy

    def lr_training(self):

        #Read the processed CSV file (Not actual CSV)
        act_filename =  self.args.input_csv
        temp = act_filename.split(".")
        temp[0] = temp[0]+"_train"
        train_filename = ".".join(temp)
        train_filename=act_filename

        self.final_df = self.get_data(file_name=train_filename)

        self.X = self.final_df[self.args.training_columns.split(",")]
        self.y = self.final_df[[self.args.target_column]]
        self.penalty='none'

        config = self.read_params(self.args.config)
        # X_train,X_test,y_train,y_test=train_test_split(self.X,self.y,test_size=1-self.args.train_size,random_state=self.random_state)
        if 'train_size' in config['split_data']:
            self.args.train_size=config['split_data']['train_size']
        if 'random_state' in config['estimators']['lr_training']['params']:
            self.random_state=config['estimators']['lr_training']['params']['random_state']
        if 'penalty' in config['estimators']['lr_training']['params']:
            self.penalty=config['estimators']['lr_training']['params']['penalty']
        X_train,X_test,y_train,y_test=train_test_split(self.X,self.y,test_size=1-self.args.train_size,random_state=self.random_state)

        mlflow_config = config["mlflow_config"]
        run_name = mlflow_config["run_name"]
        mlflow.set_tracking_uri(mlflow_config["remote_server_uri"])
        mlflow.set_experiment(mlflow_config["experiment_name"])

        with mlflow.start_run(run_name=run_name) as mlops_run:
            model = LogisticRegression(penalty=self.penalty, random_state=self.random_state)
            model.fit(X_train,y_train)

            # 5 Cross-validation
            CV = 5
            accuracies = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=CV)
            acc = np.mean(accuracies)
            print("Cross Validation accuracy mean: ", acc)

            y_pred = model.predict(X_test)
            print("Test Accuracy Score : ", accuracy_score(y_test, y_pred))

            joblib.dump(model, self.args.model_path)

            (Precision, Recall, Accuracy) = self.validate(y_test, y_pred, X_test, config)
            mlflow.log_metric("Precision", Precision)
            mlflow.log_metric("Recall", Recall)
            mlflow.log_metric("Accuracy", Accuracy)

            for k,v in config["estimators"]["lr_training"]["params"].items():
                mlflow.log_param(k,v)

            tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme

            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(
                    model,
                    "model",
                    registered_model_name=model.__class__.__name__)
            else:
                mlflow.sklearn.load_model(model, "model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Azure DevOps Pipeline')
    parser.add_argument('--config', type=str, help='Config file')
    parser.add_argument('--input_csv', type=str, help='Input CSV file')
    parser.add_argument('--dataset_name', type=str, help='Dataset name to store in workspace')
    parser.add_argument('--dataset_desc', type=str, help='Dataset description')
    parser.add_argument('--model_path', type=str, help='Path to store the model')
    parser.add_argument('--artifact_loc', type=str,help='DevOps artifact location to store the model', default='')
    parser.add_argument('--training_columns', type=str, help='model training columns comma separated')
    parser.add_argument('--target_column', type=str, help='target_column of model prediction')
    parser.add_argument('--train_size', type=float, help='train data size percentage. Valid values can be 0.01 to 0.99')
    parser.add_argument('--tag_name', type=str, help='Model Tag name')
    parser.add_argument('--processed_file_path', default = '/', type=str, help='processed dataset storage location path')
    args = parser.parse_args()
    classifier = Classification(args)
    classifier.__init__(args)
    classifier.lr_training()
