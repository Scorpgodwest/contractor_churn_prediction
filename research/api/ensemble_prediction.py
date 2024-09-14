import joblib
import os
import numpy as np
import pandas as pd
import json
from sklearn.metrics import roc_auc_score
class EnsemblePredictionAPI:
    def __init__(self, folder_path='api/models/'):
        """ Инициализирую папку с моделями, их число и уникальные имена """
        self.folder_path = folder_path
        self.models, self.model_names, self.n_folds = self.load_all_models()
        
    def hill_climbing(self, x, y, x_test):
        """ Фунция  для подбора моделей для теста """
        scores = {}
        for col in x.columns:
            scores[col] = roc_auc_score(y, x[col])
    
        scores = {k: v for k, v in sorted(scores.items(), key = lambda item: item[1], reverse = True)}
    
        x = x[list(scores.keys())]
        x_test = x_test[list(scores.keys())]
    
        STOP = False
        current_best_ensemble = x.iloc[:,0]
        current_best_test_preds = x_test.iloc[:,0]
        MODELS = x.iloc[:,1:]
        weight_range = np.arange(-0.5, 0.51, 0.01) 
        history = [roc_auc_score(y, current_best_ensemble)]
        j = 0
    
        while not STOP:
            j += 1
            potential_new_best_cv_score = roc_auc_score(y, current_best_ensemble)
            k_best, wgt_best = None, None
            for k in MODELS:
                for wgt in weight_range:
                    potential_ensemble = (1 - wgt) * current_best_ensemble + wgt * MODELS[k]
                    cv_score = roc_auc_score(y, potential_ensemble)
                    if cv_score > potential_new_best_cv_score:
                        potential_new_best_cv_score = cv_score
                        k_best, wgt_best = k, wgt
    
            if k_best is not None:
                current_best_ensemble = (1 - wgt_best) * current_best_ensemble + wgt_best * MODELS[k_best]
                current_best_test_preds = (1 - wgt_best) * current_best_test_preds + wgt_best * x_test[k_best]
                MODELS.drop(k_best, axis = 1, inplace = True)
                if MODELS.shape[1] == 0:
                    STOP = True
                history.append(potential_new_best_cv_score)
            else:
                STOP = True
            
        hill_ens_pred_1 = current_best_ensemble
        hill_ens_pred_2 = current_best_test_preds
        
        return [hill_ens_pred_1, hill_ens_pred_2]
        

    def load_all_models(self):
        """ Загрузка моделей """
        models = []
        model_names = set()
        
        for file_name in os.listdir(self.folder_path):
            if file_name.endswith('.joblib'):
                parts = file_name.split('_fold_')
                model_name = parts[0]
                fold_idx = int(parts[1].replace('.joblib', ''))
    
                while len(models) <= fold_idx:
                    models.append({})
    
                models[fold_idx][model_name] = joblib.load(os.path.join(self.folder_path, file_name))
                model_names.add(model_name)
        
        n_folds = len(models)
        return models, list(model_names), n_folds

    
    def ensemble_predict(self, test_data, strategy='average'):
        """ Вектор предсказаний """
        predictions = np.zeros((test_data.shape[0], len(self.model_names), self.n_folds))
        # models_preds = {}
        for i in range(self.n_folds):
            for j, model_name in enumerate(self.model_names):
                predictions[:, j, i] = self.models[i][model_name].predict_proba(test_data)[:, 1]
                # models_preds[f'{j}_{i}'] = predictions[:, j, i]

        if strategy == 'average':
            avg_predictions = np.mean(np.mean(predictions, axis=2), axis=1)
        elif strategy == 'median':
            avg_predictions = np.median(np.mean(predictions, axis=2), axis=1)
        else:
            raise ValueError(f"Ты шо ввел?: {strategy}")
        
        # result = {"predictions": avg_predictions.tolist()}
        result = avg_predictions.tolist()
        # return result, pd.DataFrame(models_preds)
        return result

    def process(self, test_data):
        """ Дропаем ненужное """
        self.cols_to_concat = test_data[['contract_id', 'report_date']]
        return test_data.fillna(0).drop(['report_date', 'contract_date', 'contract_id'], axis=1)

    def predict(self, test_data, strategy='average', use_hill_climbing=False, X_train_data=None, y_train_data=None):
 
        test_data = self.process(test_data)
        # if use_hill_climbing:
        #     X_train_data = self.process(X_train_data)
        #     _, models_preds_train = self.ensemble_predict(X_train_data)
        #     _, models_preds_test = self.ensemble_predict(test_data)
        #     hill_ens_preds = self.hill_climbing(models_preds_train, y_train_data, models_preds_test)[1]
        #     return hill_ens_preds
            # return models_preds_train
        # predictions, all_models_output = self.ensemble_predict(test_data, strategy)    
        predictions = self.ensemble_predict(test_data, strategy)
        
            
        
        result = pd.concat([self.cols_to_concat, pd.Series(predictions, name='score')], axis=1)
        
        return result, json.loads(result.groupby('contract_id').apply(lambda x: x.to_json(orient='values')).to_json())



if __name__ == "__main__":
    print('Это импортить надо')