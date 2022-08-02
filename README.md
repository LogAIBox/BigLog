<p align="center">
    <img src="/BIGLOG.png" alt="logo" width=300 height=300 />
</p>

## 📃 Note

You should download **pytorch_model.bin** file from  https://drive.google.com/file/d/12phqzt2QiAHjaO-BUDSyzNcmdg835rJG/view?usp=sharing  and replace the **pretrained/pytorch_model.bin** file.

## 📣 Introduction
Biglog is a unified log analysis framework that utilizes a pre-trained language model as the encoder to produce a domain-agnostic representations of input logs and integrates downstream modules to perform specific log tasks.
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
$ python pretraining_mlm.py [NUM_OF_PROC] [TRAIN_DATA_PATH] [EVAL_DATA_PATH] [TOKENIZER_DATA_PATH] [INITIAL_CHECK_POINT] [OUTPUT_PATH] [BATCH_SIZE] [LEARNING_RATE] [WEIGHT_DECAY] [EPOCH] [WARM_UP_RATIO] [SAVE_STEPS] [SAVE_TOTAL_LIMIT] [MLM_PROBABILITY] [GRADIENT_ACC]
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
## ⛏ Software development

### Unit tests

```shell
$ pip install -r test/requirements.txt
$ make
```

### Team development

[Travis CI](https://travis-ci.org/) and [AppVeyor](https://ci.appveyor.com/) is place for continuous integration.

### Coding styles

[flake8](http://flake8.pycqa.org/en/latest/index.html), [Codecov](https://codecov.io/) and [pylint](https://www.pylint.org/) are used

## 😉 Author

pyecharts are co-maintained by:

* [@chenjiandongx](https://github.com/chenjiandongx)
* [@chfw](https://github.com/chfw)
* [@kinegratii](https://github.com/kinegratii)
* [@sunhailin-Leo](https://github.com/sunhailin-Leo)

For more contributors, please visit [pyecharts/graphs/contributors](https://github.com/pyecharts/pyecharts/graphs/contributors)

## 💌 Donation

To develop and maintain pyecharts, it took me a lot of overnights. If you think pyecharts has helped you, please consider buying me a coffee:

<img src="https://user-images.githubusercontent.com/19553554/35425853-500d6b5c-0299-11e8-80a1-ebb6629b497e.png" width="19.8%" alt="Alipay">　　　<img src="https://user-images.githubusercontent.com/19553554/35425854-504e716a-0299-11e8-81fc-4a511f1c47e8.png" width="20%" alt="Wechat">


Please also buy the other maintainer a coffee if you think their work helped you too [donation details](http://pyecharts.org/#/zh-cn/donate)

## 📃 License

MIT [©chenjiandongx](https://github.com/chenjiandongx)



