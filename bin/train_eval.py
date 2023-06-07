# train_eval.py

import os
import wandb
import pandas as pd
import numpy as np
import time

import plotly.express as px
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from utils import review_subseries, get_longest_subseries_idx, ts_list_concat, make_index_same

import darts
from darts import TimeSeries
from darts.utils.missing_values import extract_subseries
from darts.dataprocessing.transformers.boxcox import BoxCox
from darts.dataprocessing.transformers.scaler import Scaler
from darts.dataprocessing import Pipeline
from darts.metrics import rmse
from darts.models import (
                            BlockRNNModel, NBEATSModel, RandomForest, 
                            LightGBMModel, XGBModel, LinearRegressionModel
                            )



from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau



dir_path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'clean_data')




class Config:
    '''
    Class to store config parameters, to circumvent the wandb.config when combining multiple models.
    '''

    def __init__(self):
        self.data = {}

    def __getattr__(self, key):
        if key in self.data:
            return self.data[key]
        else:
            raise AttributeError(f"'Config' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        if key == 'data':
            # Allow normal assignment for the 'data' attribute
            super().__setattr__(key, value)
        else:
            self.data[key] = value

    def __delattr__(self, key):
        if key in self.data:
            del self.data[key]
        else:
            raise AttributeError(f"'Config' object has no attribute '{key}'")

    def __len__(self):
        return len(self.data)

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    @classmethod
    def from_dict(cls, data):
        config = cls()
        for key, value in data.items():
            config[key] = value  # Preserve nested dictionaries without converting
        return config




def get_model_instances(tuned_models, config_per_model):

    '''Returns a list of model instances for the models that were tuned and appends a linear regression model.'''

    
    config = Config().from_dict(config_per_model[tuned_models[0]][0])
    model_instances = {}
    for model in tuned_models:
        print('getting model instance for ' + model)
        config = Config().from_dict(config_per_model[model][0])
        print(config)
        model_instances[model] = get_model(config)

    # since we did not optimize the hyperparameters for the linear regression model, we need to create a new instance
    print('getting model instance for linear regression')
    lr_model = LinearRegressionModel(
    lags = config.n_lags,
    lags_future_covariates=[0],
    output_chunk_length= config.n_ahead,
    add_encoders=config.datetime_encoders,
    random_state=42)

    model_instances['lr'] = lr_model
    return model_instances





def get_model(config):

    '''Returns model instance, based on the config.'''

    model = config.model


    # for torch models

    optimizer_kwargs = {}
    try:
        optimizer_kwargs['lr'] = config.lr
    except:
        optimizer_kwargs['lr'] = 1e-3
    
    pl_trainer_kwargs = {
    'max_epochs': 20,
    'accelerator': 'gpu',
    'devices': [0],
    'callbacks': [EarlyStopping(monitor='val_loss', patience=5, mode='min')],
    #'logger': WandbLogger(log_model='all'),
    }

    schedule_kwargs = {
        'patience': 2,
        'factor': 0.5,
        'min_lr': 1e-5,
        'verbose': True
        }


    if model == 'xgb':

        try:
            xgb_kwargs = {
            'n_estimators': config.n_estimators,
            'max_depth': config.max_depth,
            'learning_rate': config.learning_rate,
            'min_child_weight': config.min_child_weight,
            'objective': config.objective,
            'reg_lambda': config.reg_lambda,
            'early_stopping_rounds': 10
            }
        except:
            xgb_kwargs ={}

        model = XGBModel(lags=config.n_lags,
                            lags_future_covariates=[0],
                            add_encoders=config.datetime_encoders, 
                            output_chunk_length=config.n_ahead, 
                            likelihood=config.liklihood,
                            random_state=42,
                            **xgb_kwargs
                            )
    
    elif model == 'lgbm':

        
        try:
            lightgbm_kwargs = {
            'n_estimators': config.n_estimators,
            'max_depth': config.max_depth,
            'learning_rate': config.learning_rate,
            'min_child_weight': config.min_child_weight,
            'num_leaves': config.num_leaves,
            'objective': config.objective,
            'min_child_samples': config.min_child_samples}

        except:
            lightgbm_kwargs = {}
        
        model = LightGBMModel(lags=config.n_lags,
                            lags_future_covariates=[0],
                            add_encoders=config.datetime_encoders,
                            output_chunk_length=config.n_ahead,
                            likelihood=config.liklihood,
                            random_state=42,
                            **lightgbm_kwargs
                            )

    elif model == 'rf':

        rf_kwargs = {
            'n_estimators': config.n_estimators,
            'max_depth': config.max_depth,
            'min_samples_split': config.min_samples_split,
            'min_samples_leaf': config.min_samples_leaf,
            }

        
        
        model = RandomForest(lags=config.n_lags,
                            lags_future_covariates=[0],
                            add_encoders=config.datetime_encoders,
                            output_chunk_length=config.n_ahead,
                            random_state=42,
                            **rf_kwargs
                            )


    elif model == 'nbeats':

        try:
            nbeats_kwargs = {
                'batch_size': config.batch_size,
            }

        except:
            nbeats_kwargs = {} 

        model = NBEATSModel(
                        input_chunk_length=config.n_lags,
                        output_chunk_length=config.n_ahead,
                        add_encoders=config.datetime_encoders,
                        likelihood=config.liklihood,
                        pl_trainer_kwargs=pl_trainer_kwargs,
                        optimizer_kwargs=optimizer_kwargs,
                        lr_scheduler_cls=ReduceLROnPlateau,
                        lr_scheduler_kwargs=schedule_kwargs,
                        random_state=42,
                        **nbeats_kwargs
                    )

    
    elif model == 'gru':


        try:
            rnn_kwargs = {
                'hidden_dim': config.hidden_dim,
                'n_rnn_layers': config.n_rnn_layers,
                'batch_size': config.batch_size,
                'dropout': config.dropout,
            }
        except:
            rnn_kwargs = {}

        model = BlockRNNModel(  
                        model = 'GRU',
                        input_chunk_length=config.n_lags,
                        output_chunk_length=config.n_ahead,
                        add_encoders=config.datetime_encoders,
                        likelihood=config.liklihood,
                        pl_trainer_kwargs=pl_trainer_kwargs,
                        optimizer_kwargs=optimizer_kwargs,
                        lr_scheduler_cls=ReduceLROnPlateau,
                        lr_scheduler_kwargs=schedule_kwargs,
                        random_state=42,
                        **rnn_kwargs
                    )

    return model



def get_best_run_config(project_name, metric, model, scale):

    '''
    
    Returns the config of the best run of a sweep for a given model and location.
    
    '''

    sweeps = []
    config = None
    name = None

    api = wandb.Api()
    for project in api.projects():
        if project_name == project.name:
            sweeps = project.sweeps()

    for sweep in sweeps:
        if model in sweep.name and scale in sweep.name:
            best_run = sweep.best_run(order=metric)
            config = best_run.config
            name = best_run.name

    if config == None:
        print(f"Could not find a sweep for model {model} and scale {scale} in project {project_name}.")
    
    return config, name


def data_pipeline(config):

    if config.temp_resolution == 60:
        timestep_encoding = ["hour"] 
    elif config.temp_resolution == 15:
        timestep_encoding = ['quarter']
    else:
        timestep_encoding = ["hour", "minute"]


    datetime_encoders =  {
                        "cyclic": {"future": timestep_encoding}, 
                        "position": {"future": ["relative",]},
                        "datetime_attribute": {"future": ["dayofweek", "week"]},
                        'position': {'past': ['relative'], 'future': ['relative']},
                }

    datetime_encoders = datetime_encoders if config.datetime_encodings else None

    config['datetime_encoders'] = datetime_encoders


    config.timesteps_per_hour = int(60 / config.temp_resolution)
    config.n_lags = config.lookback_in_hours * config.timesteps_per_hour
    config.n_ahead = config.horizon_in_hours * config.timesteps_per_hour
    config.eval_stride = int(np.sqrt(config.n_ahead)) # evaluation stride, how often to evaluate the model, in this case we evaluate every n_ahead steps

    # Loading Data
    df_train = pd.read_hdf(os.path.join(dir_path, f'{config.spatial_scale}.h5'), key=f'{config.location}/{config.temp_resolution}min/train_target')
    df_val = pd.read_hdf(os.path.join(dir_path, f'{config.spatial_scale}.h5'), key=f'{config.location}/{config.temp_resolution}min/val_target')
    df_test = pd.read_hdf(os.path.join(dir_path, f'{config.spatial_scale}.h5'),key=f'{config.location}/{config.temp_resolution}min/test_target')

    df_cov_train = pd.read_hdf(os.path.join(dir_path, f'{config.spatial_scale}.h5'), key=f'{config.location}/{config.temp_resolution}min/train_cov')
    df_cov_val = pd.read_hdf(os.path.join(dir_path, f'{config.spatial_scale}.h5'), key=f'{config.location}/{config.temp_resolution}min/val_cov')
    df_cov_test = pd.read_hdf(os.path.join(dir_path,f'{config.spatial_scale}.h5'), key=f'{config.location}/{config.temp_resolution}min/test_cov')

    # Heat wave covariatem, categorical variable
    df_cov_train['heat_wave'] =  df_cov_train[df_cov_train.columns[0]] > df_cov_train[df_cov_train.columns[0]].quantile(0.95)
    df_cov_val['heat_wave'] =  df_cov_val[df_cov_val.columns[0]] > df_cov_val[df_cov_val.columns[0]].quantile(0.95)
    df_cov_test['heat_wave'] =  df_cov_test[df_cov_test.columns[0]] > df_cov_test[df_cov_test.columns[0]].quantile(0.95)

    # into darts format
    ts_train = darts.TimeSeries.from_dataframe(df_train, freq=str(config.temp_resolution) + 'min')
    ts_train = extract_subseries(ts_train)
    ts_val = darts.TimeSeries.from_dataframe(df_val, freq=str(config.temp_resolution) + 'min')
    ts_val = extract_subseries(ts_val)
    ts_test = darts.TimeSeries.from_dataframe(df_test, freq=str(config.temp_resolution) + 'min')
    ts_test = extract_subseries(ts_test)

    # Covariates
    if config.weather:
        ts_cov_train = darts.TimeSeries.from_dataframe(df_cov_train, freq=str(config.temp_resolution) + 'min')
        ts_cov_val = darts.TimeSeries.from_dataframe(df_cov_val, freq=str(config.temp_resolution) + 'min')
        ts_cov_test = darts.TimeSeries.from_dataframe(df_cov_test, freq=str(config.temp_resolution) + 'min')
    else:
        ts_cov_train = None
        ts_cov_val = None
        ts_cov_test = None

    # Reviewing subseries to make sure they are long enough
    ts_train, ts_cov_train = review_subseries(ts_train, config.n_lags + config.n_ahead, ts_cov_train)
    ts_val, ts_cov_val = review_subseries(ts_val, config.n_lags + config.n_ahead, ts_cov_val)
    ts_test, ts_cov_test = review_subseries(ts_test, config.n_lags +config.n_ahead, ts_cov_test)

    # getting the index of the longest subseries, to be used for evaluation later
    config.longest_ts_val_idx = get_longest_subseries_idx(ts_val)
    config.longest_ts_test_idx = get_longest_subseries_idx(ts_test)

    # Preprocessing Pipeline
    pipeline = Pipeline( # missing values have been filled in the 'data_prep.ipynb'
                    [
                    BoxCox() if config.boxcox else None,
                    Scaler(MinMaxScaler()),
                    ]
                    )
    ts_train_piped = pipeline.fit_transform(ts_train)
    ts_val_piped = pipeline.transform(ts_val)
    ts_test_piped = pipeline.transform(ts_test)

    # Weather Pipeline
    if config.weather:
        pipeline_weather = Pipeline([Scaler(RobustScaler())])
        ts_train_weather_piped = pipeline_weather.fit_transform(ts_cov_train)
        ts_val_weather_piped = pipeline_weather.transform(ts_cov_val)
        ts_test_weather_piped = pipeline_weather.transform(ts_cov_test)
    else:
        ts_train_weather_piped = None
        ts_val_weather_piped = None
        ts_test_weather_piped = None

    trg_train_inversed = pipeline.inverse_transform(ts_train_piped, partial=True) 
    trg_val_inversed = pipeline.inverse_transform(ts_val_piped, partial=True)[config.longest_ts_val_idx] 
    trg_test_inversed = pipeline.inverse_transform(ts_test_piped, partial=True)[config.longest_ts_test_idx]

    return pipeline, ts_train_piped, ts_val_piped, ts_test_piped, ts_train_weather_piped, ts_val_weather_piped, ts_test_weather_piped, trg_train_inversed, trg_val_inversed, trg_test_inversed



def train_models(models:list, ts_train_piped, ts_train_weather_piped=None, ts_val_piped=None, ts_val_weather_piped=None, use_cov_as_past=False):
    '''This function trains a list of models on the training data and validates them on the validation data if it is possible.
    
    
    '''

    run_times = {}
    
    for model in models:
        start_time = time.time()
        print(f'Training {model.__class__.__name__}')
        if model.supports_future_covariates:
            try:
                model.fit(ts_train_piped, future_covariates=ts_train_weather_piped, val_series=ts_val_piped, val_future_covariates=ts_val_weather_piped)
            except:
                model.fit(ts_train_piped, future_covariates=ts_train_weather_piped)
        elif use_cov_as_past and not model.supports_future_covariates:
            try:
                model.fit(ts_train_piped, past_covariates=ts_train_weather_piped, val_series=ts_val_piped, val_past_covariates=ts_val_weather_piped)
            except:
                model.fit(ts_train_piped, past_covariates=ts_train_weather_piped)
        else:
            try:
                model.fit(ts_train_piped, val_series=ts_val_piped)
            except:
                model.fit(ts_train_piped)
        
        end_time = time.time()
        run_times[model.__class__.__name__] = end_time - start_time
    return models, run_times





def predict_testset(model, ts, ts_covs, n_lags, n_ahead, eval_stride, pipeline):
    '''
    This function predicts the test set using a model and returns the predictions as a dataframe. Used in hyperparameter tuning.
    '''

    print('Predicting test set...')
    

    historics = model.historical_forecasts(ts, 
                                        future_covariates= ts_covs if model.supports_future_covariates else None,
                                        start=ts.get_index_at_point(n_lags),
                                        verbose=False,
                                        stride=eval_stride, 
                                        forecast_horizon=n_ahead, 
                                        retrain=False, 
                                        last_points_only=False, # leave this as False unless you want the output to be one series, the rest will not work with this however
                                        )
    
    

    historics_gt = [ts.slice_intersect(historic) for historic in historics]
    score = np.array(rmse(historics_gt, historics)).mean()

    ts_predictions = ts_list_concat(historics, eval_stride) # concatenating the batches into a single time series for plot 1, this keeps the n_ahead
    ts_predictions_inverse = pipeline.inverse_transform(ts_predictions) # inverse transform the predictions, we need the original values for the evaluation
    
    return ts_predictions_inverse.pd_series().to_frame('prediction'), score



def calc_error_scores(metrics, ts_predictions_inverse, trg_inversed):
    metrics_scores = {}
    for metric in metrics:
        score = metric(ts_predictions_inverse, trg_inversed)
        metrics_scores[metric.__name__] = score
    return metrics_scores

def get_error_metric_table(metrics, ts_predictions_per_model, trg_test_inversed):

    error_metric_table = {}
    for model_name, ts_predictions_inverse in ts_predictions_per_model.items():
        ts_predictions_inverse, trg_inversed = make_index_same(ts_predictions_inverse, trg_test_inversed)
        metrics_scores = calc_error_scores(metrics, ts_predictions_inverse, trg_inversed)
        error_metric_table[model_name] = metrics_scores
    
    df_metrics  = pd.DataFrame(error_metric_table).T
    df_metrics.index.name = 'model'
    return df_metrics


def calc_metrics(df_compare, metrics):
    "calculates metrics for a dataframe with a ground truth column and predictions, ground truth column must be the first column"
    metric_series_list = {}
    for metric in metrics:
        metric_name = metric.__name__
        metric_result = df_compare.apply(lambda x: metric(x, df_compare.iloc[:,0]), axis=0)
        if metric.__name__ == 'mean_squared_error':
            metric_result = np.sqrt(metric_result)
            metric_name = 'root_mean_squared_error'
        elif metric.__name__ == 'r2_score':
            metric_result = 1 - metric_result

        metric_series_list[metric_name] = metric_result

    df_metrics = pd.DataFrame(metric_series_list).iloc[1:,:]
    return df_metrics
