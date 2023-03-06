from scipy import stats
import numpy as np
import joblib
import re
import os
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
# from xgboost import XGBClassifier
from sklearn.svm import LinearSVC
import mlflow

class oldPreprocessFunctions:
    
    def __init__(self, args): 
        '''
        Initialize Steps
        ----------------
            1. Initalize Azure ML Run Object
            2. Load Workspace
        '''
        self.args = args
        self.workspace = Workspace.from_config()
        self.random_state = 1984
    ####endinit

    ####getfile
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
    ####endgetfile

    ####getfile
    def get_files_from_datastore(self, container_name, file_name, workspace, datastore): 
        '''
        Get the input CSV file from workspace's default data store
        Args :
            container_name : name of the container to look for input CSV
            file_name : input CSV file name inside the container
        Returns :
            data_ds : Azure ML Dataset object
        '''
        print("DEBUG ---------------------------------------------------------")
        print(container_name)
        print(file_name)

        datastore_paths = [(datastore, os.path.join(container_name,file_name))]
        data_ds = Dataset.Tabular.from_delimited_files(path=datastore_paths)
        dataset_name = self.args.dataset_name
        if dataset_name not in workspace.datasets:
            print("Registering Model to Workspace")
            data_ds = data_ds.register(workspace=workspace,
                        name=dataset_name,
                        description=self.args.dataset_desc,
                        tags={'format': 'CSV'},
                        create_new_version=True)
        else:
            print('Dataset {} already in workspace '.format(dataset_name))
        return data_ds      
    ####endgetfile

    def upload_processed_file_to_datastore(self):
        
        '''
        Stores the processed file in intermediate host directory, 
        uploads the file to azure datastorage,
        deleted the file from host directory after upload is complete.
        '''     
        act_filename =  self.args.input_csv
        temp = act_filename.split(".")
        temp[0] = temp[0]+"_PreProcess"
        upload_filename = ".".join(temp)
        processed_file_temp_path = self.args.processed_file_path
        if not os.path.exists(processed_file_temp_path):
            os.makedirs(processed_file_temp_path)
        self.final_df.to_csv(os.path.join(processed_file_temp_path, upload_filename), index=None)
        self.datastore.upload(src_dir=processed_file_temp_path, target_path=self.args.container_name)
        print("Processed file uploaded to Azure Datastorage")

        #Remove Temporary Files
        os.remove(os.path.join(processed_file_temp_path, upload_filename))
        print("Processed file deleted from Host Agent")

    ####create_pipeline
    def missing_value_treatment(self):
        '''
        Data training and Validation
        '''  
        self.datastore = Datastore.get(self.workspace, self.workspace.get_default_datastore().name)
        print("Received datastore")
        input_ds = self.get_files_from_datastore(self.args.container_name, 
                                                self.args.input_csv,
                                                self.workspace, 
                                                self.datastore)
        self.final_df = input_ds.to_pandas_dataframe()
        print("Input DF Info",self.final_df.info())
        print("Input DF Head",self.final_df.head())
        for feature in self.final_df.columns:
            if feature in self.final_df.select_dtypes(include=np.number).columns.tolist():
                self.final_df[feature].fillna(self.final_df[feature].mean(), inplace=True)
            else:
                self.final_df[feature].fillna(self.final_df[feature].mode()[0], inplace=True)
                
        self.upload_processed_file_to_datastore()

    def remove_outlier_treatment(self):
        from scipy import stats
        self.datastore = Datastore.get(self.workspace, self.workspace.get_default_datastore().name)
        print("Received datastore")
        input_ds = self.get_files_from_datastore(self.args.container_name, 
                                                    self.args.input_csv,
                                                    self.workspace, 
                                                    self.datastore)
        self.final_df = input_ds.to_pandas_dataframe()
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
        self.upload_processed_file_to_datastore()


    def create_classification_text_pipeline(self): 
        '''
        Data training and Validation
        '''        
        self.datastore = Datastore.get(self.workspace, self.workspace.get_default_datastore().name)
        print("Received datastore")
        input_ds = self.get_files_from_datastore(self.args.container_name,self.args.input_csv)
        final_df = input_ds.to_pandas_dataframe()
        print("Input DF Info",final_df.info())
        print("Input DF Head",final_df.head())
        final_df = final_df.dropna(subset=[self.args.training_columns,self.args.target_column])
        
        X = final_df[self.args.training_columns]
        y = final_df[[self.args.target_column]]

        # X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1-self.args.train_size,random_state=1984)
        
        self.create_distribution_plot(final_df,os.path.splitext(os.path.basename(self.args.input_csv))[0])

        final_df["preprocessed_"+self.args.target_column] = final_df[self.args.training_columns].apply(lambda x:self.preprocess(x))
        
        if self.args.self.args.balancing_technique_technique=="SMOTE" or self.args.self.args.balancing_technique_technique=="RUS":
            X_train, X_test, y_train, y_test = train_test_split(final_df, 
                                                        final_df[self.args.target_column], 
                                                        test_size=1-self.args.train_size, 
                                                            random_state=self.random_state)
        if self.args.self.args.balancing_technique_technique=="ROS":
            X_train, X_test, y_train, y_test = train_test_split(final_df["preprocessed_"+self.args.training_columns], 
                                                            final_df[self.args.target_column], 
                                                            test_size=1-self.args.train_size, 
                                                            random_state=self.random_state)

        if self.args.self.args.balancing_technique_technique=="ROS":
            oversample = RandomOverSampler()
            tf_idf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1,3), stop_words=self.stopwords)
            X_train1 = tf_idf.fit_transform(X_train)
            X, y = oversample.fit_resample(X_train1, y_train.ravel())

        if self.args.self.args.balancing_technique_technique=="SMOTE":
            oversample = SMOTE()
            tf_idf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1,3), stop_words='english')
            X_train1 = tf_idf.fit_transform(X_train["preprocessed_"+self.args.training_columns])
            X, y = oversample.fit_resample(X_train1, y_train.ravel())

        if self.args.self.args.balancing_technique_technique=="RUS":
            rus = RandomUnderSampler()
            tf_idf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1,3), stop_words=self.stopwords)
            X_train1 = tf_idf.fit_transform(X_train["preprocessed_"+self.args.training_columns]).toarray()
            X, y = rus.fit_resample(X_train1, y_train.ravel())

        models = [
            RandomForestClassifier(n_estimators=100, max_depth=5, random_state=self.random_state,n_jobs=-1),
            LinearSVC(),
            MultinomialNB(),
            LogisticRegression(random_state=self.random_state,n_jobs=-1),
            ExtraTreesClassifier(n_estimators=100, random_state=self.random_state,n_jobs=-1),
            DecisionTreeClassifier()
        ]

        # 5 Cross-validation
        CV = 5
        cv_df = pd.DataFrame(index=range(CV * len(models)))

        entries = []
        model_names=[]
        for model in models:
            model_name = model.__class__.__name__
            model_names.append(model_name)
            accuracies = cross_val_score(model, X, y, scoring='accuracy', cv=CV)
            for fold_idx, accuracy in enumerate(accuracies):
                entries.append((model_name, fold_idx, accuracy))

        cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
        mean_accuracy = cv_df.groupby('model_name').accuracy.mean()
        std_accuracy = cv_df.groupby('model_name').accuracy.std()

        acc = pd.concat([mean_accuracy, std_accuracy], axis= 1, 
                ignore_index=True)
        acc.columns = ['Mean Accuracy', 'Standard deviation']
        print("Best Model Selected {}".format(acc['Mean Accuracy'].idxmax()))
        model = models[model_names.index(acc['Mean Accuracy'].idxmax())]
        # model = DecisionTreeClassifier()
        model.fit(X,y)
        y_pred = model.predict(X_test)
        print("Model Score : ", model.score(X_test,y_test))

        joblib.dump(model, self.args.model_path)

        self.validate(y_test, y_pred, X_test)

        match = re.search('([^\/]*)$', self.args.model_path)
        # Upload Model to Run artifacts
        self.run.upload_file(name=self.args.artifact_loc + match.group(1),
                                path_or_stream=self.args.model_path)

        print("Run Files : ", self.run.get_file_names())
        self.run.complete()

    def create_regression_pipeline(self): 
        '''
        Data training and Validation
        '''        
        self.datastore = Datastore.get(self.workspace, self.workspace.get_default_datastore().name)
        print("Received datastore")
        input_ds = self.get_files_from_datastore(self.args.container_name,self.args.input_csv)
        final_df = input_ds.to_pandas_dataframe()
        print("Input DF Info",final_df.info())
        print("Input DF Head",final_df.head())

        X = final_df[self.args.training_columns.split(",")]
        y = final_df[[self.args.target_column]]

        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1-self.args.train_size,random_state=1984)
        
        model = LinearRegression()
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        print("Model Score : ", model.score(X_test,y_test))

        joblib.dump(model, self.args.model_path)

        self.regression_validate(y_test, y_pred, X_test)

        match = re.search('([^\/]*)$', self.args.model_path)
        # Upload Model to Run artifacts
        self.run.upload_file(name=self.args.artifact_loc + match.group(1),
                                path_or_stream=self.args.model_path)

        print("Run Files : ", self.run.get_file_names())
        self.run.complete()    
    ####endcreate_pipeline

    def decontracted(self,phrase):
        # specific
        phrase = re.sub(r"won't", "will not", phrase)
        phrase = re.sub(r"can\'t", "can not", phrase)

        # general
        phrase = re.sub(r"n\'t", " not", phrase)
        phrase = re.sub(r"\'re", " are", phrase)
        phrase = re.sub(r"\'s", " is", phrase)
        phrase = re.sub(r"\'d", " would", phrase)
        phrase = re.sub(r"\'ll", " will", phrase)
        phrase = re.sub(r"\'t", " not", phrase)
        phrase = re.sub(r"\'ve", " have", phrase)
        phrase = re.sub(r"\'m", " am", phrase)
        return phrase

    def preprocess(self,sentence):
        if self.args.preprocess_remove_whitespace:
            sentence = re.sub("(\\W)+"," ",sentence)
        if self.args.preprocess_remove_hyperlinks:
            sentence = re.sub(r"http\S+", "", sentence)
        if self.args.preprocess_remove_htmltags:
            sentence = BeautifulSoup(sentence, 'lxml').get_text()
        if self.args.preprocess_expand_words:
            sentence = self.decontracted(sentence)
        if self.args.preprocess_remove_numericdata:
            sentence = re.sub("\S*\d\S*", "", sentence).strip()
        if self.args.preprocess_remove_anyspecialchars:
            sentence = re.sub('[^A-Za-z]+', ' ', sentence)
        if self.args.preprocess_remove_stopwords:
            sentence = ' '.join(e.lower() for e in sentence.split() if e.lower() not in self.stopwords)
        if self.args.preprocess_strip:
            sentence = sentence.strip()
        return sentence

    ####confusion_matrix
    def create_confusion_matrix(self, y_true, y_pred, name):
        '''
        Create confusion matrix
        '''
        try:
            confm = confusion_matrix(y_true, y_pred, labels=np.unique(y_pred))
            print("Shape : ", confm.shape)

            df_cm = pd.DataFrame(confm, columns=np.unique(y_true), index=np.unique(y_true))
            df_cm.index.name = 'Actual'
            df_cm.columns.name = 'Predicted'
            df_cm.to_csv(name+".csv", index=False)
            self.run.upload_file(name="./outputs/"+name+".csv",path_or_stream=name+".csv")

            plt.figure(figsize = (120,120))
            sn.set(font_scale=1.4)
            c_plot = sn.heatmap(df_cm, fmt="d", linewidths=.2, linecolor='black',cmap="Oranges", annot=True,annot_kws={"size": 16})
            plt.savefig("./outputs/"+name+".png")
            self.run.log_image(name=name, plot=plt)
            
        except Exception as e:
            #traceback.print_exc()
            print(e)
            logging.error("Create consufion matrix Exception")
    ####endconfusion_matrix

    def create_distribution_plot(self, final_df, name):
        '''
        Create distribution plot
        '''
        try:
            ax = final_df[self.args.target_column].value_counts().plot.bar(x=self.args.training_columns)
            ax.set_xticklabels(ax.get_xticklabels())
            ax.figure.savefig("./outputs/"+name+"_distribution_plot.png")
            self.run.log_image(name=name, plot=plt)
        except Exception as e:
            #traceback.print_exc()    
            logging.error("Create distribution plot Exception")

    ####outputs
    def create_outputs(self, y_true, y_pred, X_test, name):
        '''
        Create prediction results as a CSV
        '''
        pred_output = {"Actual "+self.args.target_column : y_true[self.args.target_column].values, "Predicted "+self.args.target_column: y_pred[self.args.target_column].values}
        pred_df = pd.DataFrame(pred_output)
        pred_df = pred_df.reset_index()
        X_test = X_test.reset_index()
        final_df = pd.concat([X_test, pred_df], axis=1)
        final_df.to_csv(name+".csv", index=False)
        self.run.upload_file(name="./outputs/"+name+".csv",path_or_stream=name+".csv")
    ####endoutputs

    ####validate
    def validate(self, y_true, y_pred, X_test):
        self.run.log(name="Precision", value=round(precision_score(y_true, y_pred, average='weighted'), 2))
        self.run.log(name="Recall", value=round(recall_score(y_true, y_pred, average='weighted'), 2))
        self.run.log(name="Accuracy", value=round(accuracy_score(y_true, y_pred), 2))

        self.create_confusion_matrix(y_true, y_pred, "confusion_matrix")

        y_pred_df = pd.DataFrame(y_pred, columns = [self.args.target_column])
        self.create_outputs(y_true, y_pred_df,X_test, "predictions")
        self.run.tag(self.args.tag_name)


    def regression_validate(self, y_true, y_pred, X_test):
        # Evaluating Model's Performance
        self.run.log(name="Mean Absolute Error:", value=mean_absolute_error(y_true, y_pred))
        self.run.log(name="Mean Squared Error:", value=mean_squared_error(y_true, y_pred))
        self.run.log(name="Mean Root Squared Error:", value=np.sqrt(mean_squared_error(y_true, y_pred)))

        self.create_confusion_matrix(y_true, y_pred, "confusion_matrix")

        y_pred_df = pd.DataFrame(y_pred, columns = [self.args.target_column])
        self.create_outputs(y_true, y_pred_df,X_test, "predictions")
        self.run.tag(self.args.tag_name)            
    ####endvalidate


class oldTrainingFunctions:

    ####init
    def __init__(self, args):
        '''
        Initialize Steps
        ----------------
            1. Initalize Azure ML Run Object
            2. Create directories
        '''
        self.args = args
        # self.run = Run.get_context()
        # self.workspace = self.run.experiment.workspace
        os.makedirs('./model_metas', exist_ok=True)
        self.random_state = 1984
    ####endinit

    ####getfile
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
    ####endgetfile


    def get_files_from_datastore(self, container_name, file_name):
        '''
        Get the input CSV file from workspace's default data store
        Args :
            container_name : name of the container to look for input CSV
            file_name : input CSV file name inside the container
        Returns :
            data_ds : Azure ML Dataset object
        '''
        datastore_paths = [(self.datastore, os.path.join(container_name,file_name))]
        data_ds = Dataset.Tabular.from_delimited_files(path=datastore_paths)
        dataset_name = self.args.dataset_name     
        if dataset_name not in self.workspace.datasets:
            data_ds = data_ds.register(workspace=self.workspace,
                        name=dataset_name,
                        description=self.args.dataset_desc,
                        tags={'format': 'CSV'},
                        create_new_version=True)
        else:
            print('Dataset {} already in workspace '.format(dataset_name))
        return data_ds  


    def rf_model_training(self):
        
        self.datastore = Datastore.get(self.workspace, self.workspace.get_default_datastore().name)
        print("Received datastore")
        
        #Read the processed CSV file (Not actual CSV)
        act_filename =  self.args.input_csv
        temp = act_filename.split(".")
        temp[0] = temp[0]+"_PreProcess"
        upload_filename = ".".join(temp)

        input_ds = self.get_files_from_datastore(self.args.container_name, file_name=upload_filename)
        self.final_df = input_ds.to_pandas_dataframe()

        # self.final_df = self.get_processed_files_from_swd()
        
        self.X = self.final_df[self.args.training_columns.split(",")]
        self.y = self.final_df[[self.args.target_column]]

        X_train,X_test,y_train,y_test=train_test_split(self.X,self.y,test_size=1-self.args.train_size,random_state=self.random_state)
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=self.random_state,n_jobs=-1)
        model.fit(X_train,y_train)

        # 5 Cross-validation
        CV = 5
        accuracies = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=CV)
        acc = np.mean(accuracies)
        print("Cross Validation accuracy mean: ", acc)

        y_pred = model.predict(X_test)
        print("Test Accuracy Score : ", accuracy_score(y_test, y_pred))

        joblib.dump(model, self.args.model_path)

        self.validate(y_test, y_pred, X_test)

        match = re.search('([^\/]*)$', self.args.model_path)
        # Upload Model to Run artifacts
        self.run.upload_file(name=self.args.artifact_loc + match.group(1),
                                path_or_stream=self.args.model_path)

        print("Run Files : ", self.run.get_file_names())
        self.run.complete()

    def lr_model_training(self):
        
        self.datastore = Datastore.get(self.workspace, self.workspace.get_default_datastore().name)
        print("Received datastore")
        
        #Read the processed CSV file (Not actual CSV)
        act_filename =  self.args.input_csv
        temp = act_filename.split(".")
        temp[0] = temp[0]+"_PreProcess"
        upload_filename = ".".join(temp)

        input_ds = self.get_files_from_datastore(self.args.container_name, file_name=upload_filename)
        self.final_df = input_ds.to_pandas_dataframe()

        # self.final_df = self.get_processed_files_from_swd()
        
        self.X = self.final_df[self.args.training_columns.split(",")]
        self.y = self.final_df[[self.args.target_column]]

        X_train,X_test,y_train,y_test=train_test_split(self.X,self.y,test_size=1-self.args.train_size,random_state=self.random_state)
        model = LogisticRegression(n_estimators=100, max_depth=5, random_state=self.random_state,n_jobs=-1)
        model.fit(X_train,y_train)

        # 5 Cross-validation
        CV = 5
        accuracies = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=CV)
        acc = np.mean(accuracies)
        print("Cross Validation accuracy mean: ", acc)

        y_pred = model.predict(X_test)
        print("Test Accuracy Score : ", accuracy_score(y_test, y_pred))

        joblib.dump(model, self.args.model_path)

        self.validate(y_test, y_pred, X_test)

        match = re.search('([^\/]*)$', self.args.model_path)
        # Upload Model to Run artifacts
        self.run.upload_file(name=self.args.artifact_loc + match.group(1),
                                path_or_stream=self.args.model_path)

        print("Run Files : ", self.run.get_file_names())
        self.run.complete()

    def new_lr_model_training(self):
        
        # self.datastore = Datastore.get(self.workspace, self.workspace.get_default_datastore().name)
        # print("Received datastore")
        
        #Read the processed CSV file (Not actual CSV)
        act_filename =  self.args.input_csv
        temp = act_filename.split(".")
        temp[0] = temp[0]+"_train"
        train_filename = ".".join(temp)
        train_filename=act_filename

        self.final_df = self.get_data(file_name=train_filename)
        # self.final_df = input_ds.to_pandas_dataframe()

        # self.final_df = self.get_processed_files_from_swd()
        
        self.X = self.final_df[self.args.training_columns.split(",")]
        self.y = self.final_df[[self.args.target_column]]

        X_train,X_test,y_train,y_test=train_test_split(self.X,self.y,test_size=1-self.args.train_size,random_state=self.random_state)
        
        remote_server_uri = "http://127.0.0.1:1234"
        run_name = "mlops"
        mlflow.set_tracking_uri(remote_server_uri)
        mlflow.set_experiment("LogisticRegression")

        with mlflow.start_run(run_name=run_name) as mlops_run:
            model = LogisticRegression(penalty='l2', random_state=self.random_state)
            model.fit(X_train,y_train)

            # 5 Cross-validation
            CV = 5
            accuracies = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=CV)
            acc = np.mean(accuracies)
            print("Cross Validation accuracy mean: ", acc)

            y_pred = model.predict(X_test)
            print("Test Accuracy Score : ", accuracy_score(y_test, y_pred))

            joblib.dump(model, self.args.model_path)

            (Precision, Recall, Accuracy) = self.new_validate(y_test, y_pred, X_test)
            mlflow.log_metric("Precision", Precision)
            mlflow.log_metric("Recall", Recall)
            mlflow.log_metric("Accuracy", Accuracy)

            tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme

            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(
                    model, 
                    "model", 
                    registered_model_name="LogisticRegression")
            else:
                mlflow.sklearn.load_model(model, "model")
        # Upload Model to Run artifacts
        # self.run.upload_file(name=self.args.artifact_loc + match.group(1),
        #                         path_or_stream=self.args.model_path)

        # print("Run Files : ", self.run.get_file_names())
        # self.run.complete()


    def xtc_model_training(self):
        
        self.datastore = Datastore.get(self.workspace, self.workspace.get_default_datastore().name)
        print("Received datastore")
        
        #Read the processed CSV file (Not actual CSV)
        act_filename =  self.args.input_csv
        temp = act_filename.split(".")
        temp[0] = temp[0]+"_PreProcess"
        upload_filename = ".".join(temp)

        input_ds = self.get_files_from_datastore(self.args.container_name, file_name=upload_filename)
        self.final_df = input_ds.to_pandas_dataframe()

        # self.final_df = self.get_processed_files_from_swd()
        
        self.X = self.final_df[self.args.training_columns.split(",")]
        self.y = self.final_df[[self.args.target_column]]

        X_train,X_test,y_train,y_test=train_test_split(self.X,self.y,test_size=1-self.args.train_size,random_state=self.random_state)
        model = ExtraTreesClassifier(n_estimators=100, max_depth=5, random_state=self.random_state,n_jobs=-1)
        model.fit(X_train,y_train)

        # 5 Cross-validation
        CV = 5
        accuracies = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=CV)
        acc = np.mean(accuracies)
        print("Cross Validation accuracy mean: ", acc)

        y_pred = model.predict(X_test)
        print("Test Accuracy Score : ", accuracy_score(y_test, y_pred))

        joblib.dump(model, self.args.model_path)

        self.validate(y_test, y_pred, X_test)

        match = re.search('([^\/]*)$', self.args.model_path)
        # Upload Model to Run artifacts
        self.run.upload_file(name=self.args.artifact_loc + match.group(1),
                                path_or_stream=self.args.model_path)

        print("Run Files : ", self.run.get_file_names())
        self.run.complete()


    def svc_model_training(self):
        
        self.datastore = Datastore.get(self.workspace, self.workspace.get_default_datastore().name)
        print("Received datastore")
        
        #Read the processed CSV file (Not actual CSV)
        act_filename =  self.args.input_csv
        temp = act_filename.split(".")
        temp[0] = temp[0]+"_PreProcess"
        upload_filename = ".".join(temp)

        input_ds = self.get_files_from_datastore(self.args.container_name, file_name=upload_filename)
        self.final_df = input_ds.to_pandas_dataframe()

        # self.final_df = self.get_processed_files_from_swd()
        
        self.X = self.final_df[self.args.training_columns.split(",")]
        self.y = self.final_df[[self.args.target_column]]

        X_train,X_test,y_train,y_test=train_test_split(self.X,self.y,test_size=1-self.args.train_size,random_state=self.random_state)
        model = LinearSVC(n_estimators=100, max_depth=5, random_state=self.random_state,n_jobs=-1)
        model.fit(X_train,y_train)

        # 5 Cross-validation
        CV = 5
        accuracies = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=CV)
        acc = np.mean(accuracies)
        print("Cross Validation accuracy mean: ", acc)

        y_pred = model.predict(X_test)
        print("Test Accuracy Score : ", accuracy_score(y_test, y_pred))

        joblib.dump(model, self.args.model_path)

        self.validate(y_test, y_pred, X_test)

        match = re.search('([^\/]*)$', self.args.model_path)
        # Upload Model to Run artifacts
        self.run.upload_file(name=self.args.artifact_loc + match.group(1),
                                path_or_stream=self.args.model_path)

        print("Run Files : ", self.run.get_file_names())
        self.run.complete()

    def create_confusion_matrix(self, y_true, y_pred, name):
        '''
        Create confusion matrix
        '''
        
        try:
            confm = confusion_matrix(y_true, y_pred, labels=np.unique(y_pred))
            print("Shape : ", confm.shape)

            df_cm = pd.DataFrame(confm, columns=np.unique(y_true), index=np.unique(y_true))
            df_cm.index.name = 'Actual'
            df_cm.columns.name = 'Predicted'
            df_cm.to_csv(name+".csv", index=False)
            self.run.upload_file(name="./outputs/"+name+".csv",path_or_stream=name+".csv")

            plt.figure(figsize = (120,120))
            sn.set(font_scale=1.4)
            c_plot = sn.heatmap(df_cm, fmt="d", linewidths=.2, linecolor='black',cmap="Oranges", annot=True,annot_kws={"size": 16})
            plt.savefig("./outputs/"+name+".png")
            self.run.log_image(name=name, plot=plt)
            
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
        final_df.to_csv(name+".csv", index=False)
        self.run.upload_file(name="./outputs/"+name+".csv",path_or_stream=name+".csv")


    def validate(self, y_true, y_pred, X_test):
        self.run.log(name="Precision", value=round(precision_score(y_true, y_pred, average='weighted'), 2))
        self.run.log(name="Recall", value=round(recall_score(y_true, y_pred, average='weighted'), 2))
        self.run.log(name="Accuracy", value=round(accuracy_score(y_true, y_pred), 2))
        self.create_confusion_matrix(y_true, y_pred, "confusion_matrix")
        y_pred_df = pd.DataFrame(y_pred, columns = [self.args.target_column])
        self.create_outputs(y_true, y_pred_df,X_test, "predictions")
        self.run.tag(self.args.tag_name)
    
    def new_validate(self, y_true, y_pred, X_test):
        Precision=round(precision_score(y_true, y_pred, average='weighted'), 2)
        Recall=round(recall_score(y_true, y_pred, average='weighted'), 2)
        Accuracy=round(accuracy_score(y_true, y_pred), 2)
        # self.create_confusion_matrix(y_true, y_pred, "confusion_matrix")
        # y_pred_df = pd.DataFrame(y_pred, columns = [self.args.target_column])
        # self.create_outputs(y_true, y_pred_df,X_test, "predictions")
        # self.run.tag(self.args.tag_name)
        return Precision, Recall, Accuracy
    
    def eval_metrics(self, y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return rmse, mae, r2

    def create_regression_pipeline(self): 
        '''
        Data training and Validation
        '''        
        self.datastore = Datastore.get(self.workspace, self.workspace.get_default_datastore().name)
        print("Received datastore")
        input_ds = self.get_files_from_datastore(self.args.container_name,self.args.input_csv)
        final_df = input_ds.to_pandas_dataframe()
        print("Input DF Info",final_df.info())
        print("Input DF Head",final_df.head())

        X = final_df[self.args.training_columns.split(",")]
        y = final_df[[self.args.target_column]]

        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1-self.args.train_size,random_state=1984)
        
        model = LinearRegression()
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        print("Model Score : ", model.score(X_test,y_test))

        joblib.dump(model, self.args.model_path)

        self.regression_validate(y_test, y_pred, X_test)

        match = re.search('([^\/]*)$', self.args.model_path)
        # Upload Model to Run artifacts
        self.run.upload_file(name=self.args.artifact_loc + match.group(1),
                                path_or_stream=self.args.model_path)

        print("Run Files : ", self.run.get_file_names())
        self.run.complete()    
    ####endcreate_pipeline

    def decontracted(self,phrase):
        # specific
        phrase = re.sub(r"won't", "will not", phrase)
        phrase = re.sub(r"can\'t", "can not", phrase)

        # general
        phrase = re.sub(r"n\'t", " not", phrase)
        phrase = re.sub(r"\'re", " are", phrase)
        phrase = re.sub(r"\'s", " is", phrase)
        phrase = re.sub(r"\'d", " would", phrase)
        phrase = re.sub(r"\'ll", " will", phrase)
        phrase = re.sub(r"\'t", " not", phrase)
        phrase = re.sub(r"\'ve", " have", phrase)
        phrase = re.sub(r"\'m", " am", phrase)
        return phrase

    def preprocess(self,sentence):
        if self.args.preprocess_remove_whitespace:
            sentence = re.sub("(\\W)+"," ",sentence)
        if self.args.preprocess_remove_hyperlinks:
            sentence = re.sub(r"http\S+", "", sentence)
        if self.args.preprocess_remove_htmltags:
            sentence = BeautifulSoup(sentence, 'lxml').get_text()
        if self.args.preprocess_expand_words:
            sentence = self.decontracted(sentence)
        if self.args.preprocess_remove_numericdata:
            sentence = re.sub("\S*\d\S*", "", sentence).strip()
        if self.args.preprocess_remove_anyspecialchars:
            sentence = re.sub('[^A-Za-z]+', ' ', sentence)
        if self.args.preprocess_remove_stopwords:
            sentence = ' '.join(e.lower() for e in sentence.split() if e.lower() not in self.stopwords)
        if self.args.preprocess_strip:
            sentence = sentence.strip()
        return sentence


    def regression_validate(self, y_true, y_pred, X_test):
        # Evaluating Model's Performance
        self.run.log(name="Mean Absolute Error:", value=mean_absolute_error(y_true, y_pred))
        self.run.log(name="Mean Squared Error:", value=mean_squared_error(y_true, y_pred))
        self.run.log(name="Mean Root Squared Error:", value=np.sqrt(mean_squared_error(y_true, y_pred)))

        self.create_confusion_matrix(y_true, y_pred, "confusion_matrix")

        y_pred_df = pd.DataFrame(y_pred, columns = [self.args.target_column])
        self.create_outputs(y_true, y_pred_df,X_test, "predictions")
        self.run.tag(self.args.tag_name)            
    ####endvalidate

class PreprocessFunctions:
    
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
    ####endinit

    ####getfile
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
    ####endgetfile

    def read_params(self,config_path):
        with open (config_path) as yaml_file:
            config = yaml.safe_load(yaml_file)
        return config

    ####create_pipeline
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

class TrainingFunctions:

    ####init
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
    ####endinit

    ####getfile
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
    ####endgetfile

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

    def nlp_lr_training(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        #Read the processed CSV file (Not actual CSV)
        config = self.read_params(self.args.config)
        train_filename=config['processed_data']['dataset_csv']

        self.final_df = self.get_data(file_name=train_filename)

        self.X = self.final_df[self.args.training_columns.split(",")]
        self.y = self.final_df[[self.args.target_column]]
        self.penalty='none'

        # X_train,X_test,y_train,y_test=train_test_split(self.X,self.y,test_size=1-self.args.train_size,random_state=self.random_state)
        if 'train_size' in config['split_data']:
            self.args.train_size=config['split_data']['train_size']
        if 'random_state' in config['estimators']['nlp_lr_training']['params']:
            self.random_state=config['estimators']['nlp_lr_training']['params']['random_state']
        if 'penalty' in config['estimators']['nlp_lr_training']['params']:
            self.penalty=config['estimators']['nlp_lr_training']['params']['penalty']
        
        X_train,X_test,y_train,y_test=train_test_split(self.X,self.y,test_size=1-self.args.train_size,random_state=self.random_state)

        tf_idf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 3), stop_words="english")
        X_train = tf_idf.fit_transform(X_train[config['base']['train_cols']]).toarray()
        
        mlflow_config = config["mlflow_config"]
        run_name = mlflow_config["run_name"]
        mlflow.set_tracking_uri(mlflow_config["remote_server_uri"])
        mlflow.set_experiment(mlflow_config["experiment_name"])

        with mlflow.start_run(run_name=run_name) as mlops_run:
            model = LogisticRegression(penalty=self.penalty, random_state=self.random_state)
            model.fit(X_train,y_train.values.ravel())

            # 5 Cross-validation
            CV = 5
            accuracies = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=CV)
            acc = np.mean(accuracies)
            print("Cross Validation accuracy mean: ", acc)

            test_features = tf_idf.transform(X_test[config['base']['train_cols']]).toarray()
            y_pred = model.predict(test_features)
            print("Test Accuracy Score : ", accuracy_score(y_test, y_pred))

            joblib.dump(model, self.args.model_path)

            (Precision, Recall, Accuracy) = self.validate(y_test, y_pred, X_test, config)
            mlflow.log_metric("Precision", Precision)
            mlflow.log_metric("Recall", Recall)
            mlflow.log_metric("Accuracy", Accuracy)

            for k,v in config["estimators"]["nlp_lr_training"]["params"].items():
                mlflow.log_param(k,v)

            tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme

            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(
                    model,
                    "model",
                    registered_model_name=model.__class__.__name__)
            else:
                mlflow.sklearn.load_model(model, "model")
        