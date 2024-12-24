Dural-Domain Multimodal Fuion Network(DDMF-Net)

**Dataset**: download [MIMIC-IV](https://physionet.org/content/mimiciv/3.0/), [MIMIC-IV-ED](https://physionet.org/content/mimic-iv-ed/2.2/), [MIMIC-IV-ECG](https://physionet.org/content/mimic-iv-ecg/1.0/), [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.1.0/)

**Dataset-Construction**: run dataset_cons.py

**Dataset-preprocessing**: run train_test_split.py----> run resize.py

**Train**: terminal: python main.py --fusion_model DDMF_Net --fusion_type deeper_frequency_fusion_mod --lr 0.0001 --name ccloss_jsdweight --batch_size 128

**Test**: terminal: python main.py --fusion_model DDMF_Net --fusion_type deeper_frequency_fusion_mod_test --lr 0.0001 --name ccloss_jsdweight --batch_size 128
