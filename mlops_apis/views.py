from django.shortcuts import render
from mlops_apis import api_methods
import os
import requests
from mlops_apis import functions
from rest_framework.views import APIView
import random
import string
from onprem_assisted_mlops.settings import BASE_DIR
from django.http import HttpResponse, JsonResponse
import json

# Create your views here.

class CreateOnpremRepo(APIView):

    # @csrf_exempt
    def post(self, request):
        input_params=dict()

        #Variables comming from frontend
        dataset_name = str(request.data['dataset_name']).split("/")[-1]
        base_data_name = str(dataset_name.split(".")[0].split("_")[0]).lower()
        print(request.data['dataset_name'])
        input_params['dataset'] = request.data['dataset_name']
        input_params['dataset_name'] = dataset_name
        input_params['base_data_name'] = base_data_name
        input_params['repo_name'] = str(request.data['pipeline_name']).capitalize() + "-MLOps-Repository"
        input_params['pipeline_name'] = request.data['pipeline_name']
        input_params['pipeline_description'] = "{} Pipeline Creation".format(request.data['experiment_name'])
        input_params['EXPERIMENT_NAME'] = request.data['experiment_name']
        input_params['train_columns'] = ",".join(request.data['train_columns'])
        input_params['target_columns'] = request.data['target_columns']
        input_params['train_size'] = request.data['train_size']
        input_params['computeVMSize'] = request.data['ml-compute_v_m_size']
        input_params['missing_value'] = request.data['missing_value']
        input_params['remove_outlier'] = request.data['remove_outlier']
        input_params['nlp_preprocess'] = request.data['nlp_preprocess']
        
        #Model Selection
        # input_params['model_selection'] = request.data['model_selection']
        # input_params['run_traininfrasetup'] = request.data["tasks"]['run_traininfrasetup']
        # input_params['run_preprocess'] = request.data["tasks"]['run_preprocess']
        # input_params['run_train'] = request.data["tasks"]['run_train']
        # input_params['run_deployinfrasetup'] = request.data["tasks"]['run_deployinfrasetup']
        # input_params['run_deploytoaks'] = request.data["tasks"]['run_deploytoaks']
        # input_params['run_publishendpoint'] = request.data["tasks"]['run_publishendpoint']
        # input_params['run_deltraincluster'] = False
        # input_params['run_delinfcluster'] = False

        #Variables which are formed dynamically using inputs from UI
        # input_params['train_stage_input_csv'] = dataset_name       
        input_params['train_stage_input_csv'] = str(request.data['dataset_name']).split("/")[-1]                #filename
        input_params['train_stage_model_path'] = "./models/{}_model.pkl".format(base_data_name)      
        input_params['train_stage_dataset_desc'] = "{}_DataSet_Description".format(base_data_name.upper())         
        input_params['train_stage_dataset_name'] = '{}_ds'.format(base_data_name) 
        input_params['train_stage_dataset_container_name'] = str(request.data['dataset_name']).split("/")[1]   #upload_id
        # input_params['train_stage_dataset_container_name'] = str(request.data['dataset_name']).split("/")[1]   #upload_id
        # input_params['train_stage_run_configuration_name'] = "{}_training".format(base_data_name)
        # input_params['register_model_name'] = "{}".format(base_data_name.upper())
        # input_params['register_model_algo_description'] = "{}_Decision_Tree_Classifier".format(base_data_name.upper()) 
        # input_params['publish_model_artifactName'] = "{}TrainingArtifacts".format(base_data_name.upper())
        # input_params['tagname'] = '{}_classification_tag'.format(base_data_name)


        input_params['yaml_template_location']=os.path.join(BASE_DIR, 'training/base_mlops_pipeline.yml')
        input_params['yaml_output_folder_location']=os.path.join(BASE_DIR, 'training/')
        save_file_location, azure_yaml_file = api_methods.generate_yaml_file(input_params)
        input_params['save_file_location'] = save_file_location
        # input_params['azure_yaml_file'] = azure_yaml_file

        #Params needed for generation of training files.
        preprocess_config_dict = dict()
        preprocess_config_dict["remove_outlier_treatment"] = input_params["remove_outlier"]
        preprocess_config_dict["missing_value_treatment"] = input_params["missing_value"]
        preprocess_config_dict["nlp_preprocess"] = input_params["nlp_preprocess"]
        input_params['preprocess_config'] = preprocess_config_dict
        input_params['user_input_training'] = request.data["tasks"]['run_traininfrasetup']

        input_params['model_config'] = {'rf_model_training': False, 'lr_model_training': False, 'lr_training': False, 'nlp_lr_training':False, 'xtc_model_training': False, 'svc_model_training': False}

        if(request.data["model_selection"]["model"] == "Random Forest"):
            input_params['model_config']['rf_model_training'] = True
        elif(request.data["model_selection"]["model"] == "Logistic Regression"):
            input_params['model_config']['lr_model_training'] = True
        elif(request.data["model_selection"]["model"] == "XTC"):
            input_params['model_config']['xtc_model_training'] = True
        elif(request.data["model_selection"]["model"] == "SVC"):
            input_params['model_config']['svc_model_training'] = True
        elif(request.data["model_selection"]["model"] == "New Logistic Regression"):
            input_params['model_config']['lr_training'] = True
        elif(request.data["model_selection"]["model"] == "NLP Logistic Regression"):
            input_params['model_config']['nlp_lr_training'] = True
        
        input_params['model_params']=request.data['model_selection']['params']

        #Build Training Script
        input_params['preprocess_file_location'] = os.path.join(BASE_DIR, 'training/preprocess_script.py')
        input_params['training_file_location'] = os.path.join(BASE_DIR, 'training/training.py')
        input_params['train_config_file'] = os.path.join(BASE_DIR, 'training/params.yml')
        print(input_params)
        api_methods.new_script_builder(input_params)

        respDict={"message":"success"}

        #Build Train Config Script
        input_params['train_config_save_location'] = os.path.join(BASE_DIR, 'training/')
        input_params['train_config_location'] = os.path.join(BASE_DIR, 'training/base_params.yml')
        save_file_location_config, config_file = api_methods.generate_params_file(input_params)
        input_params['train_config_save_location'] = save_file_location_config
        # input_params['train_config_file'] = config_file
        print(save_file_location_config)
        
        data = dict()
        input_params['PROJECT_NAME'] = "A715809"
        # input_params['PROJECT_NAME'] = "IN-ATOS-AARA"
        input_params['base_repo_name']="MLOps-Repository"
        input_params["new_pipeline_commit_repo"] = "https://github.com/{}/{}.git".format(input_params['PROJECT_NAME'],input_params['repo_name'])
        input_params["base_pipeline_repo"] = "https://github.com/{}/{}.git".format(input_params['PROJECT_NAME'],input_params['base_repo_name'])
        input_params['created_repo_details'] = api_methods.create_new_repository_and_clone(input_params)
        # flag=api_methods.generate_mlops_pipeline_files(input_params)
        input_params['created_repo_import_details'] = api_methods.commit_and_push_changes_to_new_repo(input_params)
        # task_id = tasks.commit_file_in_yaml_to_azure_repo_task.delay(input_params)
        # pipeline_creation_result_dic = api_methods.create_new_pipeline(input_params)

        return HttpResponse(json.dumps(respDict))

        #Generate Conda Deployment Config File for each dataset
        # input_params['conda_config_save_location'] = os.path.join(BASE_DIR, 'yaml_outputs/training/')
        # conda_deployment_config_save_file_location, conda_config_file = api_methods.generate_conda_config_file(input_params)
        # input_params['conda_config_save_location'] = conda_deployment_config_save_file_location
        # input_params['conda_config_file'] = conda_config_file  

        #Build TEST & Deployment Config JSON, save it in the templates folder
        # api_methods.create_deployment_config_json(input_params)

        #Generate Score.py file and smoke_test.py file dynamically based on amlops_config.json
        # api_methods.generate_deployment_files(input_params)


        #Check if pipeline is already present or not.
        # pipeline_exists, pipeline_dict = api_methods.check_if_pipeline_exists(input_params)


        # print(input_params)
        # #Save dictionary --> Use this to debu Update YAML function.
        # with open('/home/saugata/Desktop/input_params.pickle', 'wb') as handle:
        #     pickle.dump(input_params, handle)

        # if(pipeline_exists):
        #     data = dict()
            
        #     task_id = tasks.commit_file_in_yaml_to_azure_repo_task.delay(input_params)
        #     input_params['pipeline_creation_status'] = False
        #     data['pipeline_details'] = {'pipeline_name': input_params['pipeline_name'],
        #                                 'definition_id': pipeline_dict[input_params['pipeline_name']],
        #                                 'process_name': "commit_file_in_yaml_to_azure_repo_task",
        #                                 'task_id': str(task_id)}  

        #     input_params['pipeline_name'] = input_params['pipeline_name']
        #     input_params['definition_id'] = pipeline_dict[input_params['pipeline_name']]
        #     input_params['PIPELINE_ID'] = input_params['definition_id']
        #     data['execution_parameters'] =  input_params  
        # else:
        #     #Clone the earlier repo from previous branch to new branch
        #     #Build pipeline uisng the new branch
        #     data = dict()
        #     input_params["new_pipeline_commit_repo"] = "https://SyntbotsAI-RnD@dev.azure.com/SyntbotsAI-RnD/{}/_git/{}".format(input_params['PROJECT_NAME'],input_params['repo_name'])
        #     input_params['created_repo_details'] = api_methods.create_new_repository(input_params)
        #     input_params['created_repo_import_details'] = api_methods.clone_existing_repo_to_new_repo(input_params)
        #     task_id = tasks.commit_file_in_yaml_to_azure_repo_task.delay(input_params)
        #     pipeline_creation_result_dic = api_methods.create_new_pipeline(input_params)
        #     input_params['pipeline_creation_status'] = pipeline_creation_result_dic['pipeline_creation_status']
        #     data['pipeline_details'] = {'pipeline_name': pipeline_creation_result_dic['name'],
        #                                 'definition_id': pipeline_creation_result_dic['id'],
        #                                 'process_name': "commit_file_in_yaml_to_azure_repo_task",
        #                                 'task_id': str(task_id)}  

        #     input_params['pipeline_name'] = pipeline_creation_result_dic['name']
        #     input_params['definition_id'] = pipeline_creation_result_dic['id']
        #     input_params['PIPELINE_ID'] = pipeline_creation_result_dic['id']
        #     data['execution_parameters'] =  input_params          


        # if(task_id):
        #     return StandardResponse.Response(True, "Success. ", data)
        # else:
        #     return StandardResponse.Response(False, "Error. ", "Pipeline is not created.")