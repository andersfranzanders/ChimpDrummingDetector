from chimp_detector import audio_processor, featuremap_processor





def detect_chimpz(path):

   signal = audio_processor.read_in_audio(path)
   featuremap = audio_processor.extract_spectrogram(signal)
   featuremap = featuremap_processor.denoise_featuremap(featuremap)
   featuremap = featuremap_processor.standartize_featuremap(featuremap)
   featuremap = featuremap_processor.segment_featuremap(featuremap)

   return (0)
