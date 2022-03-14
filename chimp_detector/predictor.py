from chimp_detector.config import Hyperparams

from tensorflow import keras
import numpy as np
import pandas as pd

def predict_featuremap(featuremap):
    model = keras.models.load_model(Hyperparams.PATH_TO_MODEL)
    predictions_probs = model.predict(featuremap).flatten()
    prdicitions_binary = predictions_probs > 0.5
    return predictions_probs,prdicitions_binary

def produce_final_output_csv(predictions_probs,predictions_binary,timepoints_of_fmap_frames_in_s ):

    timepoints_of_fmap_frames_in_s = timepoints_of_fmap_frames_in_s.flatten()

    df =  pd.DataFrame({"timepoint_in_seconds": timepoints_of_fmap_frames_in_s, "drumming_probability": predictions_probs,
                                   "drumming_binarized": predictions_binary})
    df = df.drop_duplicates(subset="timepoint_in_seconds", ignore_index=True)
    df["drumming_probability"] = df["drumming_probability"].astype(float).round(3)

    return df