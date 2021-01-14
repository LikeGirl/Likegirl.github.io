---
layout: post
title: (추천시스템) 협업 필터링 -메모리 기반 추천
tags: [추천시스템]
math: true
date: 2021-01-13 22:55 
comments : true
---

해당 글은 마이크로소프트가 공개한 협업 필터링 jupyter notebook의 참고하여 작성한 글입니다. 


마이크로소프트가 공개한 코드를 보고 싶다면 --> [여기](https://github.com/microsoft/recommenders)

오늘 정리할 내용은 다음과 같습니다. 
- 협업 필터링의 개념 
- 협업 필터링 (코드) : microsoft의 reco_util 모듈 활용 

---
## 협업 필터링(Collaborative Filtering) 의 개요 

✔ 협업 필터링이란, 사용자와 제품 간의 `상호작용` 데이터를 바탕으로, 유저에게 이전에 좋아했던 제품과 유사한 제품을 추천하거나(item-Based Collaborative Filtering)과 유저의 취향과 유사한 취향을 가진 고객이 좋아하는 제품을 추천하는 방식(User-Based Collaborative Filtering)을 말합니다. 

- User-Based CF : 누가, 무엇을 얼마나 좋아하는지 표현 (사용자ID,아이템ID,선호값)
- Item-Based CF : 아이템과 아이템이 얼마나 연관이 있는지 표현 (아이템ID,아이템ID,선호값)

사용자는 과거에 상호작용한(구매)한 적 있는 아이템과 유사한 아이템을 ***선호***합니다. 

### 협업 필터링 유의사항
- 사용자/아이템의 피쳐(feature) 정보를 활용하지 않기 때문에 다른 알고리즘(예:Neural CF 등)에 비해 성능이 떨어질 수 있음
- 메모리 부족 현상 : 사용자/아이템의 최소 기준 선정 후 적용 필요
- `평점과 같이 직접적인 선호도 관련 데이터가 없을 경우에는, 선호값에 대한 정의 필요 ` <br>
   예) 구매수, 클릭수, 장바구니 횟수 등 


<협업 필터링 알고리즘 개념도(***simple***)>

<img src="https://recodatasets.blob.core.windows.net/images/sar_schema.svg?sanitize=true">
---

## Code
- microsoft 의 reco_util 모듈 활용
- `아래의 코드를 실행하고 싶으시면, microsoft recommender git clone 이후 실행해주세요`😀
>  git clone https://github.com/Microsoft/Recommenders   

```python
# 파이썬 코드를 실행하기 전에 항상 모든 모듈을 Reload
import autoreload
%load_ext autoreload
%autoreload 2

# 경로 지정 
import sys
sys.path.append("../../")

os.chdir('e:/github/recommender_github/Recommenders')

import logging
import numpy as np
import pandas as pd

import scrapbook as sb # notebook에서 실행한 데이터의 값이나 시각화 컨텐츠를 기록 
# https://nteract-scrapbook.readthedocs.io/en/latest/

from sklearn.preprocessing import minmax_scale

from reco_utils.common.python_utils import binarize
from reco_utils.common.timer import Timer
from reco_utils.dataset import movielens
from reco_utils.dataset.python_splitters import python_stratified_split
from reco_utils.evaluation.python_evaluation import (
    map_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    rmse,
    mae,
    logloss,
    rsquared,
    exp_var
)
from reco_utils.recommender.sar import SAR

print("System version: {}".format(sys.version))
print("Pandas version: {}".format(pd.__version__))
```

### 1 데이터 불러오기

협업 필터링에 필요한 데이터
-  `<User ID>, <Item ID>,<Time>,[<Event Type>], [<Event Weight>]`

각 행은 사용자와 아이템 간의 하나의 상호작용을 의미한다. 이 상호작용은 e-commerce 분야에서 상품을 클릭하거나, 장바구니에 추가하거나, 추천 링크를 연결하거나, 기타 등등의 행동을 의미한다. 각각의 event_type은 각기 다른 가중치를 할당할 수 있다. 예를 들어, 구매 이벤트를 10, viewing은 1로 가중치를 설정할 수 있다.

MovieLens 데이터셋을 이용해 실습해보자. 

```python
# 추천 아이템의 개수를 선정 
TOP_K = 10 

# MovieLens 데이터셋의 size 선택 : 100k, 1m, 10m, or 20m
MOVIELENS_DATA_SIZE = '100k'

# MovieLens Dataset 다운로드 --> movielens.py 

data = movielens.load_pandas_df(
    size = MOVIELENS_DATA_SIZE
)

# 메모리 사용량을 줄이기 위해 float를 32bit로 변경
data['rating'] = data['rating'].astype(np.float32)

# data 확인해보기
print(data.shape) # (100000, 4)
print('userID의 unique한 개수는:{}'.format(data['userID'].nunique()))
print('itemID unique한 개수는:{}'.format(data['itemID'].nunique()))

```
### 2 데이터 분할 (train/test)
train/test 분할 시, test셋에 존재하는 user_id는 training 셋에서도 존재해야합니다

```python
# python_splitter.py 함수 이용해서 데이터 분할
train,test = python_stratified_split(data,
ratio=0.75,
col_user = 'userID',
col_item = 'itemID',
seed = 42
)
```
### 3 모델 학습

```python
# 파이썬 로깅(logging)
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s %(levelname)-8s %(message)s')

# 모델 학습
model = SAR(
    col_user = "userID",
    col_item="itemID",
    col_rating="rating",
    col_timestamp="timestamp",
    similarity_type="jaccard", 
    time_decay_coefficient=30, 
    timedecay_formula=True,
    normalize=True
)
```

### 4 모델 학습 및 TEST 셋 예측 (TOP-K 아이템 추천)


- 아이템 간의 동시 발생 매트릭스를 계산
- `Co-occurence`는 사용자별 두 아이템이 함께 등장한 횟수를 의미
- `Co-occurence` 매트릭스를 만들면, 아이템 간의 `similarity matrix`를 만든다. (선택한 유사도 방법에 따라)
- 유저와 아이템 간의 강도(strength)를 파악하기 위해 `affinity matrix`를 계산한다. (선호도는 평점 또는 영화를 본 횟수, 이벤트의 수 등)
- 추천은 affinity matrix $A$와 similarity matrix $S$를 곱하여 수행된다.
- 결과는 추천 점수 매트릭스 $R$
- `recommend_k_items`함수를 이용하여 top-k 결과를 계산한다.

```python
# train 학습
with Timer() as train_time:
    model.fit(train)

print("Took {} seconds for training.".format(train_time.interval))
```

```python
# test 예측
with Timer() as test_time:
    top_k = model.recommend_k_items(test,remove_seen=True)

print("Took {} seconds for prediction.".format(test_time.interval))
```

### 5 추천 성능 평가 
- `python_evaluation`모듈을 이용하여 ranking 메트릭을 통해 성능평가
- MAP(Mean Average Precision), NDCG(Normalized Discounted Cumulative Gain), precision, and top-k 아이템에 대한 recall을 이용

```python
# MAP
eval_map = map_at_k(test,top_k,col_user='userID',col_item='itemID', col_rating='rating', k=TOP_K)

# NDCG 
eval_ndcg = ndcg_at_k(test,top_k,col_user='userID',col_item='itemID', col_rating='rating', k=TOP_K)

# precision
eval_precision = precision_at_k(test, top_k, col_user='userID', col_item='itemID', col_rating='rating', k=TOP_K)

# recall
eval_recall = recall_at_k(test, top_k, col_user='userID', col_item='itemID', col_rating='rating', k=TOP_K)

# rmse 
eval_rmse = rmse(test, top_k, col_user='userID', col_item='itemID', col_rating='rating')

# mae 
eval_mae = mae(test, top_k, col_user='userID', col_item='itemID', col_rating='rating')

# rsqueared
eval_rsquared = rsquared(test, top_k, col_user='userID', col_item='itemID', col_rating='rating')

# exp_var 
eval_exp_var = exp_var(test, top_k, col_user='userID', col_item='itemID', col_rating='rating')

```
```python
# rating 값을 positivity_threshold 겂울 지정해 0과 1값으로 변환
positivity_threshold = 2 # 2 초과는 1, 2 이하는 0 |

test_bin = test.copy()
test_bin['rating'] = binarize(test_bin['rating'],positivity_threshold)
```

```python
# 추천 점수(예측 값)을 min-max 스케일링
top_k_prob = top_k.copy()

top_k_prob['prediction'] = minmax_scale(
    top_k_prob['prediction'].astype(float)

)

eval_logloss = logloss(test_bin,top_k_prob,col_user='userID',col_item='itemID',
col_rating='rating'
)
```

```python 
# 추천 성능 평가 결과 (offline-test)
print("Model:\t",
      "Top K:\t%d" % TOP_K,
      "MAP:\t%f" % eval_map,
      "NDCG:\t%f" % eval_ndcg,
      "Precision@K:\t%f" % eval_precision,
      "Recall@K:\t%f" % eval_recall,
      "RMSE:\t%f" % eval_rmse,
      "MAE:\t%f" % eval_mae,
      "R2:\t%f" % eval_rsquared,
      "Exp var:\t%f" % eval_exp_var,
      "Logloss:\t%f" % eval_logloss,
      sep='\n')

# 특정 유저에 대한 추천 결과 
user_id = 876

ground_truth = test[test['userID'] == user_id].sort_values(by='rating',
ascending=False)[:TOP_K]

prediction = model.recommend_k_items(pd.DataFrame(dict(userID=[user_id])),
remove_seen=True)

pd.merge(ground_truth,prediction,on=['userID', 'itemID'],how='left').sort_values(by='prediction',ascending=False)

```
### 결론
- 협업 필터링의 개념을 이해하고, 추천 시스템을 빠르게 돌려보고 싶으신 분은 추천 👍
- 특히, 추천 성능 평가 결과 모듈은 추천 알고리즘 비교할 때 사용하면 좋을 듯 👍
- 협업 필터링이 뭐죠? 기본 개념을 이해하지 못한 사람에게는 비추 😣 
  (차근차근 사용자-아이템 매트릭스 만들고, 유사도 매트릭스 등등 한땀한땀 코드 작성하면서 이해하는 것을 추천)
