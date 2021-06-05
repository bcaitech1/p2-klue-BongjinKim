# 문장 내 개체간 관계 추출
관계 추출(Relation Extraction)은 문장의 단어(Entity)에 대한 속성과 관계를 예측하는 문제입니다.

- `input` : sentence, entity1, entity2 의 정보를 입력으로 사용 합니다.


- `output` : relation 42개 classes 중 1개의 class를 예측한 값입니다.


- `metric` : accuracy

## Dataset

- `train.tsv`: 총 9000개


- `test.tsv`: 총 1000개 (정답 라벨 blind)


- `answer`: 정답 라벨 (비공개)


- `label_type.pkl` : 총 42개 classes

## Code
### Files
- `train.py`

    baseline code를 학습시키기 위한 파일입니다. 저장된 model관련 파일은 results 폴더에 있습니다.


- `inference.py`

    학습된 model을 통해 prediction하며, 예측한 결과를 csv 파일로 저장해줍니다. 저장된 파일은 prediction 폴더에 있습니다.


- `load_data.py` 
    
    baseline code의 전처리와 데이터셋 구성을 위한 함수들이 있는 코드입니다.


- `loss.py` 
    
    loss 함수를 따로 구현해 놓은 코드입니다.

### Install Libraries
```python
pip install -r requirements.txt
```
### Trainng
```
python train.py
```
- 기본으로 설정된 hyperparameter로 train.py 실행합니다.

- baseline code에서는 500 step마다 logs 폴더와 results 폴더에 각각 텐서보드 기록과 model이 저장됩니다.
### Inference
```python
python inference.py --model_dir=./results/checkpoint-500
```
- 학습된 모델을 추론합니다.

- 제출을 위한 csv 파일을 만들고 싶은 model의 경로를 model_dir에 입력해 줍니다.

- 오류 없이 진행 되었다면, ./prediction/pred_answer.csv 파일이 생성 됩니다.

## Competition
### Final Score
![image](https://media.vlpt.us/images/loulench/post/77f75ac3-1b67-4d2d-98cf-2672179d931e/image.png)
### Work flow
|Model|Tecnics|Accuracy|
|---|---|---|
|multi lingual based bert|params not tuned|52.1|
|kobert|params not tuned|55.0|
|koelectra|params not tuned|67.7|
|bert-kor-base|params not tuned|71.8|
|bert-kor-base|params tuned with BackTranslation|72.4|
|bert-kor-base|params tuned|74.8|
|bert-kor-base|params tuned, with NER|75.4|
|xml-roberta-large|params not tuned|78|
|xml-roberta-large|params tuned, with NER|79.4|

### References
- BERT - https://arxiv.org/abs/1810.04805
- Generating Datasets with Pretrained Language Models - https://arxiv.org/abs/2104.07540
- ZmBART - https://arxiv.org/abs/2106.01597
- An Improved Baseline for Sentence-level Relation Extraction - https://arxiv.org/pdf/2102.01373.pdf

