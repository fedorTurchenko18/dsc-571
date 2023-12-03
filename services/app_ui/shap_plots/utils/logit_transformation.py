import numpy as np
import shap

from scipy.special import expit
from typing import Literal

def shap_transform_scale(original_shap_values: shap.Explainer, pred_proba: Literal['XGBClassifier.fit(X, y).predict_proba(X)']) -> shap.Explainer:
    
    # untransformed_base_value = original_shap_values.base_values[-1]
    untransformed_base_value = original_shap_values.base_values
   
    #Computing the original_explanation_distance to construct the distance_coefficient later on
    # original_explanation_distance = np.sum(original_shap_values.values, axis=1)
    original_explanation_distance = np.sum(original_shap_values.values)
    
    base_value = expit(untransformed_base_value) # = 1 / (1+ np.exp(-untransformed_base_value))

    #Computing the distance between the model_prediction and the transformed base_value
    distance_to_explain = np.abs(pred_proba - base_value)

    #The distance_coefficient is the ratio between both distances which will be used later on
    distance_coefficient = np.abs(original_explanation_distance / distance_to_explain)

    #Transforming the original shapley values to the new scale
    shap_values_transformed = original_shap_values / distance_coefficient

    #Finally resetting the base_value as it does not need to be transformed
    shap_values_transformed.base_values = base_value
    shap_values_transformed.data = original_shap_values.data
    
    #Now returning the transformed array
    return shap_values_transformed