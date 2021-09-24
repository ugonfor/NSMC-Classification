# NSMC-Classification
NSMC Text Classification using BERT

## Directory tree

```shell
ugonfor@gongon:~/NSMC-Classification$ tree
.
├── README.md
├── data_analysis.ipynb
├── log
│   ├── logMon Sep 20 11:24:58 2021
│   └── logSun Sep 19 22:14:11 2021
├── main.py
├── model.py
├── model_pt
│   └── model4Sun Sep 19 22_50_40 2021.pth
├── nsmc
│   ├── ratings.txt
│   ├── ratings_test.txt
│   └── ratings_train.txt
├── preprocess.py
└── utils.py

3 directories, 12 files
```

* `data_analysis.ipynb` : NSMC 데이터셋의 길이, tokenize 시 토큰 수를 알아보기 위한 notebook

* `log/*` : train시 로그파일과, load시 로그 파일 두 개 로그파일 존재.

* `nsnc/*` : nsmc 데이터셋

* `model_pt/*` : 모델 저장 되는 폴더

* `main.py` : Bert Model을 Train하는 main python code.

* `model.py` : BertClassifier 모델을 정의한 code

* `preprocess.py` : NSMCDataset을 load하기 위해 Dataset을 정의한 code

* `utils.py` : 재현을 위한 시드 값 고정, 모델 저장, 모델 불러오기 함수 구현

## How to Use
### configuration

main.py 내에 아래와 같이 설정 들을 조절하는 부분이 존재.

* `device`의 경우 args 인자로 전달할 수 있다.

* `MULTIGPU`를 사용여부 : True/False

* `Bert Tokenizer의 maximum length`

* `batch size` : 배치사이즈

* `train mode / load mode` 설정:
train mode로 할 시 Bert모델을 train하지만, load mode로 할 시, model_pt/{model_name} 의 파일로부터 model state 정보를 가져와서 ratings_test.txt에 대해서 test를 진행한다.

* `model_path` : 모델을 저장/ 저장되어있는 주소

* `EPOCHS` : 훈련 epochs 수

위 값들에 대해서 main.py을 수정한 후, 실행하면 된다.
```py

# configure
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if args.device:
    device = args.device

MULTIGPU = False 
MAX_LEN = 30
BATCH_SIZE = 32
MODE = 'train' # 'load'
model_name = 'model4Sun Sep 19 22:50:40 2021.pth' # if load
model_path = './model_pt'
EPOCHS = 4

os.makedirs("./log",exist_ok=True)
logfile = f"./log/log{time.ctime()}" 

log = open(logfile, "w")

set_seed(42)

```

실행은 다음과 같이 --device 인자로 실행할 device를 설정해준다.

```shell
$ python main.py --device cuda:0
```


---
## Reference
https://www.analyticsvidhya.com/blog/2020/07/transfer-learning-for-nlp-fine-tuning-bert-for-text-classification/
https://www.kaggle.com/atulanandjha/bert-testing-on-imdb-dataset-extensive-tutorial
https://github.com/e9t/nsmc
https://huggingface.co/kykim/bert-kor-base/tree/main
https://tutorials.pytorch.kr/beginner/basics/data_tutorial.html
