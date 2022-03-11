from chimp_detector.config import Hyperparams

from tensorflow import keras
import numpy as np
import pandas as pd

def predict_featuremap(featuremap):
    model = keras.models.load_model(Hyperparams.PATH_TO_MODEL)
    predictions_probs = model.predict(featuremap).flatten()
    prdicitions_binary = predictions_probs > 0.5
    return predictions_probs,prdicitions_binary

def produce_final_output_csv(predictions_probs,predictions_binary ):
    hopsize_between_frames_in_s = Hyperparams.WIN_LENGTH_MS * 0.001 * Hyperparams.STFT_WIN_OVERLAP_PERCENT
    timepoints_in_s = np.asarray([i * hopsize_between_frames_in_s for i in range(predictions_probs.shape[0])])

    return  pd.DataFrame({"timepoint_in_seconds": timepoints_in_s, "drumming_probability": predictions_probs,
                                   "drumming_binarized": predictions_binary})