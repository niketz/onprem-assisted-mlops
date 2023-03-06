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

    def nlp_preprocess(self):
        from tqdm import tqdm
        from bs4 import BeautifulSoup
        config=self.read_params(self.args.config)
        self.df = pd.read_csv(config["data_source"]["source"])
        self.df = self.df.dropna(subset=[config['base']['train_cols'],config['base']['target_col']])

        # https://gist.github.com/sebleier/554280
        # we are removing the words from the stop words list: 'no', 'nor', 'not'
        # <br /><br /> ==> after the above steps, we are getting "br br"
        # we are including them into stop words list
        # instead of <br /> if we have <br/> these tags would have revmoved in the 1st step

        stopwords= set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've","you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself','she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their','theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those','am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does','did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of','at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after','above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further','then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more','most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very','s', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're','ve', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',"hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',"mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't",'won', "won't", 'wouldn', "wouldn't"])
        print("Data Pre-Processing")
        preprocessed_descriptions = []
        # tqdm is for printing the status bar
        for sentance in tqdm(self.df[config['base']['train_cols']].values):
            sentance = re.sub("(\\W)+"," ",sentance)
            sentance = re.sub(r"http\S+", "", sentance)
            sentance = BeautifulSoup(sentance, 'lxml').get_text()
            sentance = re.sub(r"won't", "will not", sentance)
            sentance = re.sub(r"can\'t", "can not", sentance)
            sentance = re.sub(r"n\'t", " not", sentance)
            sentance = re.sub(r"\'re", " are", sentance)
            sentance = re.sub(r"\'s", " is", sentance)
            sentance = re.sub(r"\'d", " would", sentance)
            sentance = re.sub(r"\'ll", " will", sentance)
            sentance = re.sub(r"\'t", " not", sentance)
            sentance = re.sub(r"\'ve", " have", sentance)
            sentance = re.sub(r"\'m", " am", sentance)
            sentance = re.sub("\S*\d\S*", "", sentance).strip()
            sentance = re.sub('[^A-Za-z]+', ' ', sentance)
            # https://gist.github.com/sebleier/554280
            sentance = ' '.join(e.lower() for e in sentance.split() if e.lower() not in stopwords)
            preprocessed_descriptions.append(sentance.strip())
        self.df[config['base']['train_cols']] = preprocessed_descriptions
        self.df[config['base']['train_cols']].replace('', np.nan, inplace=True)
        self.df.dropna(subset=[config['base']['train_cols']], inplace=True)

        self.df.to_csv(config['processed_data']['dataset_csv'], index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Azure DevOps Pipeline')
    parser.add_argument('--config', type=str, help='Config file')
    parser.add_argument('--input_csv', default='ticket_type.csv', type=str, help='Input CSV file')
    parser.add_argument('--dataset_name', default = 'ticket_ds', type=str, help='Dataset name to store in workspace')
    parser.add_argument('--dataset_desc', default = 'TICKET_DataSet_Description', type=str, help='Dataset description')
    parser.add_argument('--training_columns', default = 'short_desc', type=str, help='model training columns comma separated')
    parser.add_argument('--target_column', default = 'ticket_type', type=str, help='target_column of model prediction')
    parser.add_argument('--processed_file_path', default = '/', type=str, help='processed dataset storage location path')
    args = parser.parse_args()
    preprocessor = Preprocessing(args)
    preprocessor.__init__(args)
    preprocessor.nlp_preprocess()
