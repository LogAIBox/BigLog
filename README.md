<p align="center">
    <img src="/BIGLOG.png" alt="logo" width=300 height=300 />
</p>

## 📃 Note

You should download **pytorch_model.bin** file from https://huggingface.co/thatbbob/BigLog/tree/main and replace the **pretrained/pytorch_model.bin** file.

## 📣 Introduction
Biglog is a unified log analysis framework that utilizes a pre-trained language model as the encoder to produce a domain-agnostic representations of input logs and integrates downstream modules to perform specific log tasks. This is the official repository for paper 《Biglog: Unsupervised Large-scale Pre-training for a Unified Log Representation》, published at IWQoS 2023.
## ✨ Implementation Details
In pre-trained phase, 8 V100 GPUs (32GB memory) is used. The model is pre-trained with 10K steps of parameters updating and a learning rate 2e-4. Batch size is 512 and the MLM probability is 0.15. Warm-up ratio is 0.05 and weight_decay is 0.01. For the fine-tuning of specific analysis tasks, since only a few parameters in classification heads or pooling layers need updating, only 1 or 2 V100 GPUs (16GB memory) are utilized and the fine-tuning process is within 2 epochs of traversing the dataset. Depending on the complexity of different tasks, learning rate is set between 0.5e-4 and 10e-4 and the batch size is 8 or 32. 
## 🔰 Installation

**pip install**
```
$ pip install transformers
```
## 📝 Usage
### Pre-training with your own logs
```
$ python /scripts/pretraining_mlm.py [NUM_OF_PROC] [TRAIN_DATA_PATH] [EVAL_DATA_PATH] [TOKENIZER_DATA_PATH] [INITIAL_CHECK_POINT] [OUTPUT_PATH] [BATCH_SIZE] [LEARNING_RATE] [WEIGHT_DECAY] [EPOCH] [WARM_UP_RATIO] [SAVE_STEPS] [SAVE_TOTAL_LIMIT] [MLM_PROBABILITY] [GRADIENT_ACC]
```
NUM_OF_PROC: number of process used in data loading  
TRAIN_DATA_PATH: train data path  
EVAL_DATA_PATH: evaluate data path  
TOKENIZER_DATA_PATH: biglog tokenizer path  
INITIAL_CHECK_POINT: initial checkpoint  
OUTPUT_PATH: model save path  
BATCH_SIZE: batch size  
LEARNING_RATE: lr  
WEIGHT_DECAY: weight decay  
EPOCH: total epoch  
WARM_UP_RATIO: ratio of warm up for pre-training  
SAVE_STEPS: model save frequency  
SAVE_TOTAL_LIMIT: limitation of saved models  
MLM_PROBABILITY:  mask probability  
GRADIENT_ACC: gradient accumulation steps  

### Use Biglog to acquire log embeddings
(1) load Biglog tokenizer
```
from transformers import AutoTokenizer
biglog_tokenizer = AutoTokenizer.from_pretrained('/pretrained')
```
(2) load pre-trained Biglog
```
from transformers import BertModel
model = BertModel.from_pretrained('/pretrained')
```
(3) tokenize log data
```
tokenized_data=biglog_tokenizer(YOUR_DATA,padding = "longest",truncation=True,max_length=150)
```
(4) get embeddings
```
out=model(torch.tensor(tokenized_data['input_ids']))
log_embedding=out[0]
```
## Citation
```
@INPROCEEDINGS{10188759,
  author={Tao, Shimin and Liu, Yilun and Meng, Weibin and Ren, Zuomin and Yang, Hao and Chen, Xun and Zhang, Liang and Xie, Yuming and Su, Chang and Oiao, Xiaosong and Tian, Weinan and Zhu, Yichen and Han, Tao and Qin, Ying and Li, Yun},
  booktitle={2023 IEEE/ACM 31st International Symposium on Quality of Service (IWQoS)}, 
  title={Biglog: Unsupervised Large-scale Pre-training for a Unified Log Representation}, 
  year={2023},
  volume={},
  number={},
  pages={1-11},
  doi={10.1109/IWQoS57198.2023.10188759}}
```
