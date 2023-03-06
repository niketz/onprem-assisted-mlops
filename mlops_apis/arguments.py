def arguments_function(input_params):

        argument_list = {
        'arguments_preprocess': ["parser = argparse.ArgumentParser(description='Azure DevOps Pipeline')",
                # "parser.add_argument('--parent_bucket_name', default='{}', type=str, help='Name of container where data is present')".format(input_params['CONTAINER_NAME']),
                # "parser.add_argument('--container_name', default='{}', type=str, help='Name of folder where data is present')".format(input_params['train_stage_dataset_container_name']),
                "parser.add_argument('--config', type=str, help='Config file')",
                "parser.add_argument('--input_csv', default='{}', type=str, help='Input CSV file')".format(input_params['train_stage_input_csv']),
                "parser.add_argument('--dataset_name', default = '{}', type=str, help='Dataset name to store in workspace')".format(input_params['train_stage_dataset_name']),
                "parser.add_argument('--dataset_desc', default = '{}', type=str, help='Dataset description')".format(input_params['train_stage_dataset_desc']),
                "parser.add_argument('--training_columns', default = '{}', type=str, help='model training columns comma separated')".format(input_params['train_columns']),
                "parser.add_argument('--target_column', default = '{}', type=str, help='target_column of model prediction')".format(input_params['target_columns']), 
                "parser.add_argument('--processed_file_path', default = '/', type=str, help='processed dataset storage location path')",
                "args = parser.parse_args()",
                "preprocessor = Preprocessing(args)",
                ],
        'arguments_training': ["parser = argparse.ArgumentParser(description='Azure DevOps Pipeline')",
                # "parser.add_argument('--container_name', type=str, help='Path to default datastore container')",
                "parser.add_argument('--config', type=str, help='Config file')",
                "parser.add_argument('--input_csv', type=str, help='Input CSV file')",
                "parser.add_argument('--dataset_name', type=str, help='Dataset name to store in workspace')",
                "parser.add_argument('--dataset_desc', type=str, help='Dataset description')",
                "parser.add_argument('--model_path', type=str, help='Path to store the model')",
                "parser.add_argument('--artifact_loc', type=str,help='DevOps artifact location to store the model', default='')",
                "parser.add_argument('--training_columns', type=str, help='model training columns comma separated')",
                "parser.add_argument('--target_column', type=str, help='target_column of model prediction')",
                "parser.add_argument('--train_size', type=float, help='train data size percentage. Valid values can be 0.01 to 0.99')",
                "parser.add_argument('--tag_name', type=str, help='Model Tag name')", 
                "parser.add_argument('--processed_file_path', default = '/', type=str, help='processed dataset storage location path')",
                "args = parser.parse_args()",
                "classifier = Classification(args)",
                ]
        }
        return argument_list