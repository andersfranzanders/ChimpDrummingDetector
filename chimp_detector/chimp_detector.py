from chimp_detector import audio_processor, featuremap_processor, predictor





def detect_chimpz(path):

   signal = audio_processor.read_in_audio(path)
   featuremap = audio_processor.extract_spectrogram(signal)
   featuremap = featuremap_processor.denoise_featuremap(featuremap)
   featuremap = featuremap_processor.standartize_featuremap(featuremap)
   featuremap = featuremap_processor.segment_featuremap(featuremap)
   predictions_probs, predictions_binary = predictor.predict_featuremap(featuremap)

   output_dataframe = predictor.produce_final_output_csv(predictions_probs,predictions_binary)

   return (output_dataframe)
