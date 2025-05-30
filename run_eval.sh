# !bin/bash
# for iemocap 4 way
python eval.py --dataset="iemocap_4" --modalities="atv" --from_begin --epochs=55

# for iemocap 6 way
# python eval.py --dataset="iemocap" --modalities="atv"
python preprocess.py --dataset="iemocap"
python train.py --dataset="iemocap" --modalities="atv" --from_begin --epochs=55
python eval.py --dataset="iemocap" --modalities="atv"


