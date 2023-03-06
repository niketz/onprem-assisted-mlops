import subprocess
# from onprem_assisted_mlops.settings import URI
import os
from azureml.core.dataset import Dataset
import azureml.core
from azureml.core import Workspace, Datastore
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
import os
import string
import random
from datetime import datetime, timedelta
from urllib.parse import quote
import json
import ruamel.yaml
from onprem_assisted_mlops.settings import BASE_DIR
import shutil
from msrest.authentication import BasicAuthentication
from azure.devops.v6_0.pipelines.pipelines_client import PipelinesClient
import shutil
from azure.devops.v6_0.pipelines.pipelines_client import PipelinesClient
from azure.devops.v6_0.build.build_client import BuildClient
from msrest.authentication import BasicAuthentication
import yaml
import requests
import base64
import json
import pandas as pd
from datetime import datetime as dt
import textwrap
from ruamel.yaml.scalarstring import PreservedScalarString
import yaml
from inspect import getmembers, isfunction, getsourcelines, getsource
from mlops_apis import functions
import imp
import io
from github import Github
from git import Repo
import stat

def script_builder(input_params):
    
    user_input_preprocess = input_params['preprocess_config']['missing_value_treatment']
    user_input_training = input_params['user_input_training']
    
    ##Code of preprocess builder
    if user_input_preprocess == True:
        config = input_params['preprocess_config']
        imp.reload(functions)
        
        class CodeGenerator:
            def __init__(self, indentation='    '):
                self.indentation = indentation
                self.level = 0
                self.code = ''

            def indent(self):
                self.level += 1

            def dedent(self):
                if self.level > 0:
                    self.level -= 1

            def __add__(self, value):
                temp = CodeGenerator(indentation=self.indentation)
                temp.level = self.level
                temp.code = str(self) + ''.join([self.indentation for i in range(0, self.level)]) + str(value)
                return temp
                
            def __str__(self):
                return str(self.code)
            
        imports = ["os",
                "azureml.core[Workspace, Datastore, Dataset",
                "argparse",
                "numpy as np",
                ]
        L = []
        
        for importlib in imports:
            if importlib.find("[") != -1:
                L.append("from "+importlib.split("[")[0]+ " import "+importlib.split("[")[1]+"\n")
            else:
                L.append("import "+importlib+"\n")
                
        # L.append("os.environ['AZURE_DEVOPS_EXT_GITHUB_PAT'] = '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'\n")
        # L.append("os.environ['AZURE_DEVOPS_EXT_PAT'] = 'l2ckn6zjch4hsfjcjtwczznuoozrqafgyaff7cya5m5ru3nwmfuq'\n")

        classification_functions = ["__init__","get_files_from_datastore","upload_processed_file_to_datastore"]
        # regression_functions = ["__init__","get_files_from_datastore","create_regression_pipeline","create_confusion_matrix","create_outputs","regression_validate"]
        
        for key,value in config.items():
            if value:
                classification_functions.append(key)
        
        funcs = getmembers(functions.PreprocessFunctions, isfunction)
        # print(funcs)
        ind = CodeGenerator()
        ind += "".join(L)
        ind += "\n\n"
        ind+="class AzurePreprocessing():\n"
        functions_list = classification_functions

        for each in functions_list:
            for func in funcs:
                if func[0]==each:
                    ind+="\n"
                    # ind.indent()
                    flines = [line.rstrip()+"\n" for line in getsourcelines(func[1])[0]]
                    for fline in flines:
                        ind+= fline
                    # ind.dedent()
                    
        from mlops_apis.arguments import arguments_function
        argument_list = arguments_function(input_params)
        argument_list['arguments_preprocess']
        ind.dedent()
        ind+='\nif __name__ == "__main__":\n'
        ind.indent()
        
        # Data balancing technique,preprocess_remove_whitespace applicable only when univariate and text classification is True. 
        argparams = argument_list['arguments_preprocess']
        
        for line in argparams:
            ind+=line+"\n"
        ind += "preprocessor.__init__(args)\n"
        
        for key,value in config.items():
            if value:
                ind += "preprocessor."+key+"()\n"        

        # writing to file
        file1 = open(input_params['preprocess_file_location'], 'w')
        file1.writelines(str(ind))
        file1.close()
        print("Generated preprocess_script.py file on the go.")
        
    if user_input_training == True:
        ##Code of training builder
        
        config = input_params['model_config']
        print(config)

        imp.reload(functions)

        class CodeGenerator:
            def __init__(self, indentation='    '):
                self.indentation = indentation
                self.level = 0
                self.code = ''

            def indent(self):
                self.level += 1

            def dedent(self):
                if self.level > 0:
                    self.level -= 1

            def __add__(self, value):
                temp = CodeGenerator(indentation=self.indentation)
                temp.level = self.level
                temp.code = str(self) + ''.join([self.indentation for i in range(0, self.level)]) + str(value)
                return temp
                
            def __str__(self):
                return str(self.code)
            
        imports = ["os",
                "argparse",
                "logging",
                "sklearn.model_selection[train_test_split",
                "sklearn.ensemble[RandomForestClassifier,ExtraTreesClassifier",
                "sklearn.svm[LinearSVC",
                "sklearn.naive_bayes[MultinomialNB",
                "sklearn.linear_model[LogisticRegression",
                "sklearn.tree[DecisionTreeClassifier",
                "sklearn.model_selection[cross_val_score",
                "sklearn.metrics[classification_report, confusion_matrix, precision_score, recall_score, accuracy_score",
                "pandas as pd",
                "numpy as np",
                "re",
                "seaborn as sn",
                "matplotlib.pyplot as plt",
                "joblib",
                ]
        L = []
        
        for importlib in imports:
            if importlib.find("[") != -1:
                L.append("from "+importlib.split("[")[0]+ " import "+importlib.split("[")[1]+"\n")
            else:
                L.append("import "+importlib+"\n")
                
        classification_functions = ["__init__","get_files_from_datastore","create_confusion_matrix","create_outputs","validate"]
        # regression_functions = ["__init__","get_files_from_datastore","create_regression_pipeline","create_confusion_matrix","create_outputs","regression_validate"]
        
        for key,value in config.items():
            if value:
                classification_functions.append(key)
        print("classification_functions: ",classification_functions)
                

        funcs = getmembers(functions.TrainingFunctions, isfunction)
        # print(funcs)
        ind = CodeGenerator()
        ind += "".join(L)
        ind+="class AzureClassification():\n"
        functions_list = classification_functions
        print("functions_list: ",functions_list)

        for each in functions_list:
            for func in funcs:
                if func[0]==each:
                    ind+="\n"
                    # ind.indent()
                    flines = [line.rstrip()+"\n" for line in getsourcelines(func[1])[0]]
                    for fline in flines:
                        ind+= fline
                    # ind.dedent()
                    
        from mlops_apis.arguments import arguments_function
        argument_list = arguments_function(input_params)
        ind.dedent()
        ind+='\nif __name__ == "__main__":\n'
        ind.indent()
        
        # Data balancing technique,preprocess_remove_whitespace applicable only when univariate and text classification is True. 
        argparams = argument_list['arguments_training']
        
        for line in argparams:
            ind+=line+"\n"
        ind += "classifier.__init__(args)\n"
        
        for key,value in config.items():
            if value:
                ind += "classifier."+key+"()\n" 
                

        # writing to file
        file1 = open(input_params['training_file_location'], 'w')
        file1.writelines(str(ind))
        file1.close()
        print("Generated training.py file on the go.")

def new_script_builder(input_params):
    
    user_input_preprocess = input_params['preprocess_config'] #['missing_value_treatment']
    user_input_training = input_params['user_input_training']
    print("user_input_preprocess: ",user_input_preprocess)
    ##Code of preprocess builder
    if user_input_preprocess:
        config = input_params['preprocess_config']
        print("preprocess_config: ",config)
        imp.reload(functions)
        
        class CodeGenerator:
            def __init__(self, indentation='    '):
                self.indentation = indentation
                self.level = 0
                self.code = ''

            def indent(self):
                self.level += 1

            def dedent(self):
                if self.level > 0:
                    self.level -= 1

            def __add__(self, value):
                temp = CodeGenerator(indentation=self.indentation)
                temp.level = self.level
                temp.code = str(self) + ''.join([self.indentation for i in range(0, self.level)]) + str(value)
                return temp
                
            def __str__(self):
                return str(self.code)
            
        imports = ["os",
                # "azureml.core[Workspace, Datastore, Dataset",
                "argparse",
                "numpy as np",
                "pandas as pd",
                "yaml",
                "re"
                ]
        L = []
        
        for importlib in imports:
            if importlib.find("[") != -1:
                L.append("from "+importlib.split("[")[0]+ " import "+importlib.split("[")[1]+"\n")
            else:
                L.append("import "+importlib+"\n")
                
        # L.append("os.environ['AZURE_DEVOPS_EXT_GITHUB_PAT'] = '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'\n")
        # L.append("os.environ['AZURE_DEVOPS_EXT_PAT'] = 'l2ckn6zjch4hsfjcjtwczznuoozrqafgyaff7cya5m5ru3nwmfuq'\n")

        # classification_functions = ["__init__","get_files_from_datastore","upload_processed_file_to_datastore"]
        # regression_functions = ["__init__","get_files_from_datastore","create_regression_pipeline","create_confusion_matrix","create_outputs","regression_validate"]
        classification_functions = ["__init__","get_data","read_params"]
        
        for key,value in config.items():
            if value:
                classification_functions.append(key)
        print("functions_list: ",classification_functions)
        
        funcs = getmembers(functions.PreprocessFunctions, isfunction)
        # print(funcs)
        ind = CodeGenerator()
        ind += "".join(L)
        ind += "\n\n"
        ind+="class Preprocessing():\n"
        functions_list = classification_functions

        for each in functions_list:
            for func in funcs:
                if func[0]==each:
                    ind+="\n"
                    # ind.indent()
                    flines = [line.rstrip()+"\n" for line in getsourcelines(func[1])[0]]
                    for fline in flines:
                        ind+= fline
                    # ind.dedent()
                    
        from mlops_apis.arguments import arguments_function
        argument_list = arguments_function(input_params)
        argument_list['arguments_preprocess']
        ind.dedent()
        ind+='\nif __name__ == "__main__":\n'
        ind.indent()
        
        # Data balancing technique,preprocess_remove_whitespace applicable only when univariate and text classification is True. 
        argparams = argument_list['arguments_preprocess']
        
        for line in argparams:
            ind+=line+"\n"
        ind += "preprocessor.__init__(args)\n"
        
        for key,value in config.items():
            if value:
                ind += "preprocessor."+key+"()\n"        

        # writing to file
        file1 = open(input_params['preprocess_file_location'], 'w')
        file1.writelines(str(ind))
        file1.close()
        print("Generated preprocess_script.py file on the go.")
        
    if user_input_training == True:
        ##Code of training builder
        
        config = input_params['model_config']
        print(config)

        imp.reload(functions)

        class CodeGenerator:
            def __init__(self, indentation='    '):
                self.indentation = indentation
                self.level = 0
                self.code = ''

            def indent(self):
                self.level += 1

            def dedent(self):
                if self.level > 0:
                    self.level -= 1

            def __add__(self, value):
                temp = CodeGenerator(indentation=self.indentation)
                temp.level = self.level
                temp.code = str(self) + ''.join([self.indentation for i in range(0, self.level)]) + str(value)
                return temp
                
            def __str__(self):
                return str(self.code)
            
        imports = ["os",
                "argparse",
                "logging",
                "sklearn.model_selection[train_test_split",
                "sklearn.ensemble[RandomForestClassifier,ExtraTreesClassifier",
                "sklearn.svm[LinearSVC",
                "sklearn.naive_bayes[MultinomialNB",
                "sklearn.linear_model[LogisticRegression",
                "sklearn.tree[DecisionTreeClassifier",
                "sklearn.model_selection[cross_val_score",
                "sklearn.metrics[classification_report, confusion_matrix, precision_score, recall_score, accuracy_score",
                "pandas as pd",
                "numpy as np",
                "re",
                "seaborn as sn",
                "matplotlib.pyplot as plt",
                "joblib",
                "mlflow",
                "urllib.parse[urlparse",
                "yaml",
                ]
        L = []
        
        for importlib in imports:
            if importlib.find("[") != -1:
                L.append("from "+importlib.split("[")[0]+ " import "+importlib.split("[")[1]+"\n")
            else:
                L.append("import "+importlib+"\n")
                
        # classification_functions = ["__init__","get_files_from_datastore","create_confusion_matrix","create_outputs","validate"]
        # regression_functions = ["__init__","get_files_from_datastore","create_regression_pipeline","create_confusion_matrix","create_outputs","regression_validate"]
        classification_functions = ["__init__","get_data","read_params","create_confusion_matrix","create_outputs","validate"]
        
        for key,value in config.items():
            if value:
                classification_functions.append(key)
                
        funcs = getmembers(functions.TrainingFunctions, isfunction)
        # print(funcs)
        ind = CodeGenerator()
        ind += "".join(L)
        ind+="class Classification():\n"
        functions_list = classification_functions

        for each in functions_list:
            for func in funcs:
                if func[0]==each:
                    ind+="\n"
                    # ind.indent()
                    flines = [line.rstrip()+"\n" for line in getsourcelines(func[1])[0]]
                    for fline in flines:
                        ind+= fline
                    # ind.dedent()
                    
        from mlops_apis.arguments import arguments_function
        argument_list = arguments_function(input_params)
        ind.dedent()
        ind+='\nif __name__ == "__main__":\n'
        ind.indent()
        
        # Data balancing technique,preprocess_remove_whitespace applicable only when univariate and text classification is True. 
        argparams = argument_list['arguments_training']
        
        for line in argparams:
            ind+=line+"\n"
        ind += "classifier.__init__(args)\n"
        
        for key,value in config.items():
            if value:
                ind += "classifier."+key+"()\n"                 

        # writing to file
        file1 = open(input_params['training_file_location'], 'w')
        file1.writelines(str(ind))
        file1.close()
        print("Generated training.py file on the go.")

def generate_params_file(input_params):
    yaml = ruamel.yaml.YAML()
    yaml.preserve_quotes = True
    with open(input_params['train_config_location']) as fp:
        config_file = yaml.load(fp)

    config_file['split_data']['train_size']=input_params['train_size']
    for key,value in input_params['model_config'].items():
        if value:
            config_file['estimators']=dict()
            config_file['estimators'][key]=dict()
            config_file['estimators'][key]['params']=input_params['model_params']
    config_file['mlflow_config']['experiment_name']=input_params['EXPERIMENT_NAME']
    config_file['mlflow_config']['run_name']="mlops"
    config_file['base']['train_cols']=input_params['train_columns']
    config_file['base']['target_col']=input_params['target_columns']
    config_file['data_source']['source']=os.path.join("data/source/",input_params['dataset_name'])
    config_file['processed_data']['dataset_csv']=os.path.join("data/processed/","{}_train.csv".format(input_params['base_data_name']))
    
    save_file_location = os.path.join(input_params['train_config_save_location'], "params.yml")
           
    with open(save_file_location, 'w') as fp:
        yaml.dump(config_file, fp)

    print("Runconfig Generated")
        
    return save_file_location, config_file

def generate_yaml_file(input_params):
    #Read the YAML
    yaml = ruamel.yaml.YAML()
    yaml.preserve_quotes = True
    with open(input_params['yaml_template_location']) as fp:
        pipeline_yaml = yaml.load(fp)
    for idx, step in enumerate(pipeline_yaml['jobs']['build']['steps']):
        if 'name' in step and step['name']=="Pre processing":
            step['run']="python src/preprocess_script.py --config={} --input_csv={}  --training_columns={} --target_column={}".format("params.yml","data/source/"+input_params['train_stage_input_csv'], input_params['train_columns'], input_params['target_columns'])
        if 'name' in step and step['name']=="Train and Evaluate":
            step['run']="mlflow db upgrade sqlite:///mlflow.db\nmlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 127.0.0.1 -p 1234 & disown\npython src/training.py --config={} --input_csv={}  --training_columns={} --target_column={} --model_path={} --train_size={}\nfuser -k 1234/tcp".format("params.yml","data/source/"+input_params['train_stage_input_csv'], input_params['train_columns'], input_params['target_columns'],"saved_models/model.joblib",input_params['train_size'])

    #Save the YAML output in a location
    file_name = os.path.basename(input_params['yaml_template_location'])
    save_file_location = os.path.join(input_params['yaml_output_folder_location'] ,"mlops_pipeline.yml")
    with open(save_file_location, 'w') as fp:
        yaml.dump(pipeline_yaml, fp)
        
    print("YAML file generated!")

    return save_file_location, pipeline_yaml

def generate_mlops_pipeline_files(input_params):
    try:
        print("Generating the MLOPs Pipeline Files..................")

        folder_name_for_repo = input_params['repo_name']
        save_file_name = os.path.basename(input_params['save_file_location'])

        # if(os.path.isdir(os.path.join(BASE_DIR, folder_name_for_repo))):
        #     os.remove(os.path.join(BASE_DIR, folder_name_for_repo))
        #     print("Existing repo folder deleted")

        if os.path.isfile(os.path.join(BASE_DIR,folder_name_for_repo,'.github', 'workflows', save_file_name)):  
            os.remove(os.path.join(BASE_DIR,folder_name_for_repo, '.github', 'workflows', save_file_name))
            print("Deleteing the YAML file from repo")

        if os.path.isfile(os.path.join(BASE_DIR,folder_name_for_repo, 'src', os.path.basename(input_params['training_file_location']))):  
            os.remove(os.path.join(BASE_DIR,folder_name_for_repo, 'src', os.path.basename(input_params['training_file_location'])))
            print("Deleteing the training script file from repo")

        if os.path.isfile(os.path.join(BASE_DIR,folder_name_for_repo, 'src', os.path.basename(input_params['preprocess_file_location']))):  
            os.remove(os.path.join(BASE_DIR,folder_name_for_repo, 'src', os.path.basename(input_params['preprocess_file_location'])))
            print("Deleteing the preprocess script file from repo")

        if os.path.isfile(os.path.join(BASE_DIR,folder_name_for_repo, os.path.basename(input_params['train_config_file']))):  
            os.remove(os.path.join(BASE_DIR,folder_name_for_repo, os.path.basename(input_params['train_config_file'])))
            print("Deleteing the config json file from repo")

        shutil.copy(input_params['save_file_location'], folder_name_for_repo+"/.github/workflows")
        shutil.copy(input_params['preprocess_file_location'], folder_name_for_repo+"/src")
        shutil.copy(input_params['training_file_location'], folder_name_for_repo+"/src")
        shutil.copy(input_params['train_config_save_location'], folder_name_for_repo)
        shutil.copy(input_params['dataset'], folder_name_for_repo+"/data/source")
        # shutil.copy(input_params['score_file_save_location'], os.path.join(folder_name_for_repo, "inference"))
        # shutil.copy(input_params['smoke_test_file_save_location'], os.path.join(folder_name_for_repo, "tests", "smoke"))
        # shutil.copy(input_params['conda_config_save_location'], os.path.join(folder_name_for_repo, "deployment"))

        return True

    except:
        return False

def create_new_repository_and_clone(input_params):
    # using an access token
    clone_directory = os.path.join(BASE_DIR, input_params['repo_name'])

    if(os.path.isdir(os.path.join(BASE_DIR, clone_directory))):
        os.remove(os.path.join(BASE_DIR, clone_directory))
        print("Existing repo folder deleted")

    g = Github("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")
    user = g.get_user()
    try:
        repo = user.create_repo(input_params['repo_name'])
        result_clone_repo = Repo.clone_from(input_params["base_pipeline_repo"],clone_directory)
    except:
        print("Repo already exists, cloning from existing repo")
        result_clone_repo = Repo.clone_from(input_params["new_pipeline_commit_repo"],clone_directory)
    return result_clone_repo

def commit_and_push_changes_to_new_repo(input_params):
    print("Generating the MLOPs Pipeline Files..................")

    folder_name_for_repo = input_params['repo_name']
    save_file_name = os.path.basename(input_params['save_file_location'])

    # if(os.path.isdir(os.path.join(BASE_DIR, folder_name_for_repo))):
    #     os.remove(os.path.join(BASE_DIR, folder_name_for_repo))
    #     print("Existing repo folder deleted")

    if os.path.isfile(os.path.join(BASE_DIR,folder_name_for_repo,'.github', 'workflows', save_file_name)):  
        os.remove(os.path.join(BASE_DIR,folder_name_for_repo, '.github', 'workflows', save_file_name))
        print("Deleteing the YAML file from repo")

    if os.path.isfile(os.path.join(BASE_DIR,folder_name_for_repo, 'src', os.path.basename(input_params['training_file_location']))):  
        os.remove(os.path.join(BASE_DIR,folder_name_for_repo, 'src', os.path.basename(input_params['training_file_location'])))
        print("Deleteing the training script file from repo")

    if os.path.isfile(os.path.join(BASE_DIR,folder_name_for_repo, 'src', os.path.basename(input_params['preprocess_file_location']))):  
        os.remove(os.path.join(BASE_DIR,folder_name_for_repo, 'src', os.path.basename(input_params['preprocess_file_location'])))
        print("Deleteing the preprocess script file from repo")

    if os.path.isfile(os.path.join(BASE_DIR,folder_name_for_repo, os.path.basename(input_params['train_config_file']))):  
        os.remove(os.path.join(BASE_DIR,folder_name_for_repo, os.path.basename(input_params['train_config_file'])))
        print("Deleteing the config json file from repo")

    shutil.copy(input_params['save_file_location'], os.path.join(BASE_DIR,folder_name_for_repo,".github","workflows"))
    shutil.copy(input_params['preprocess_file_location'], os.path.join(BASE_DIR,folder_name_for_repo,"src"))
    shutil.copy(input_params['training_file_location'], os.path.join(BASE_DIR,folder_name_for_repo,"src"))
    shutil.copy(input_params['train_config_save_location'], os.path.join(BASE_DIR,folder_name_for_repo))
    shutil.copy(input_params['dataset'], os.path.join(BASE_DIR,folder_name_for_repo,"data","source"))
    # shutil.copy(input_params['score_file_save_location'], os.path.join(folder_name_for_repo, "inference"))
    # shutil.copy(input_params['smoke_test_file_save_location'], os.path.join(folder_name_for_repo, "tests", "smoke"))
    # shutil.copy(input_params['conda_config_save_location'], os.path.join(folder_name_for_repo, "deployment"))
    
    import time
    time.sleep(20)

    repo_directory = os.path.join(BASE_DIR, input_params['repo_name'])
    repo=Repo(repo_directory)
    commit_message = 'Automated MLOPs Pipeline Commit'
    files = repo.git.diff(None, name_only=True)
    repo.git.add('.')
    repo.index.commit(commit_message)
    # remote = repo.create_remote("origin", url=input_params["new_pipeline_commit_repo"])
    repo.remotes["origin"].set_url(input_params["new_pipeline_commit_repo"])
    remote = repo.remote("origin")
    remote.push(refspec='{}:{}'.format("HEAD", "main"))
    
    # shutil.rmtree(repo_directory, ignore_errors=True)
    # print("------------------------ Repository Deleted from Project Directory ------------------------")
    return remote