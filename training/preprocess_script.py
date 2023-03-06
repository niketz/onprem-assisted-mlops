import os
import argparse
import numpy as np
import pandas as pd
import yaml
import re


class Preprocessing():

    def __init__(self, args):
        '''
        Initialize Steps
        ----------------
            1. Initalize Azure ML Run Object
            2. Load Workspace
        '''
        self.args = args
        # self.workspace = Workspace.from_config()
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

    def remove_outlier_treatment(self):
        from scipy import stats
        # self.final_df = self.get_data(str(self.args.input_csv).split("/")[-1])
        config=self.read_params(self.args.config)
        input_csv=config['data_source']['source']
        self.final_df=self.get_data(input_csv)
        print("Input DF Info",self.final_df.info())
        print("Input DF Head",self.final_df.head())
        num_features = self.final_df.select_dtypes(include=np.number).columns.tolist()
        self.final_df[num_features] = self.final_df[num_features][(np.abs(stats.zscore(self.final_df[num_features])) < 3).all(axis=1)]
        # for feature in self.final_df.columns:
        #     if feature in self.final_df.select_dtypes(include=np.number).columns.tolist():
        #         mean = np.mean(self.final_df[feature])
        #         std = np.std(self.final_df[feature])
        #         threshold = 3
        #         outlier = []
        #         for i in self.final_df[feature]:
        #             z = (i-mean)/std
        #             if z > threshold:
        #                 outlier.append(i)
        #         for i in outlier:
        #             self.final_df[feature] = np.delete(self.final_df[feature], np.where(self.final_df[feature]==i))
        # self.final_df.to_csv(str(self.args.input_csv).split("/")[-1].split(".")[0]+"_train.csv", index=False)
        self.final_df.to_csv(config['processed_data']['dataset_csv'], index=False)

    def missing_value_treatment(self):
        '''
        Missing Value Treatment
        '''
        config=self.read_params(self.args.config)
        input_csv=config['processed_data']['dataset_csv']
        self.final_df = self.get_data(input_csv)
        print("Input DF Info",self.final_df.info())
        print("Input DF Head",self.final_df.head())
        for feature in self.final_df.columns:
            if feature in self.final_df.select_dtypes(include=np.number).columns.tolist():
                self.final_df[feature].fillna(self.final_df[feature].mean(), inplace=True)
            else:
                self.final_df[feature].fillna(self.final_df[feature].mode()[0], inplace=True)

        # self.final_df.to_csv(str(self.args.input_csv).split("/")[-1].split(".")[0]+"_train.csv", index=False)
        self.final_df.to_csv(config['processed_data']['dataset_csv'], index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Azure DevOps Pipeline')
    parser.add_argument('--config', type=str, help='Config file')
    parser.add_argument('--input_csv', default='Iris.csv', type=str, help='Input CSV file')
    parser.add_argument('--dataset_name', default = 'iris_ds', type=str, help='Dataset name to store in workspace')
    parser.add_argument('--dataset_desc', default = 'IRIS_DataSet_Description', type=str, help='Dataset description')
    parser.add_argument('--training_columns', default = 'SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm', type=str, help='model training columns comma separated')
    parser.add_argument('--target_column', default = 'Species', type=str, help='target_column of model prediction')
    parser.add_argument('--processed_file_path', default = '/', type=str, help='processed dataset storage location path')
    args = parser.parse_args()
    preprocessor = Preprocessing(args)
    preprocessor.__init__(args)
    preprocessor.remove_outlier_treatment()
    preprocessor.missing_value_treatment()
