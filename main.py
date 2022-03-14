import chimp_detector

path = "/home/anders/labpdgx/labp/3 Forschung/Remote Detection/Species Detection Audio/Audio_Ammie_Kalan/ARUdata_targetedchimpexcerpts/ARU4_20101009_114554_Jul01chimps.wav"

output_df = chimp_detector.detect_drumming(path)


print(output_df)