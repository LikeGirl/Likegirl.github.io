---
layout: post
title: (추천시스템) Matrix Factorization - SVD 
tags: [추천시스템]
math: true
date: 2021-01-14 22:31
comments : true
---

Matrix Factorization 기법 중에 SVD 기반 추천의 개념을 이해하고, 파이썬 코드로 구현해 본 포스팅입니다.  

오늘 정리할 내용은 다음과 같습니다. 
- Matrix Factorization의 개념 
- SVD를 이용한 MF
- 코드 구현 

---
## Matrix Factorization(MF)의 개요

<img src="https://miro.medium.com/max/5130/1*b4M7o7W8bfRRxdMxtFoVBQ.png">

✔ Matrix Factorization(MF)은 잠재요인 협업 필터링이라고 표현하며, (사용자X 아이템)으로 구성된 하나의 행렬을 2개의 행렬로 분해하는 방법을 말한다. 
- 사용자/아이템의 특성을 벡터로 간략히 요약 하는 모델링
- 사용자/아이템의 잠재요인(latent factor)를 찾는 것! 
- 사용자와 아이템을 같은 vector 공간에 표현
- 같은 vector 공간에서 사용자와 아이템이 가까우면 유사, 멀리 떨어져 있으면 유사하지 않다. 

<img src="https://csdl-images.computer.org/mags/co/2009/08/figures/mco20090800302.gif">

#### 영화 추천에서 잠재요인 2개인 경우의 사용자/영화(아이템)의 특성을 두개의 요인에 따라 같은 2차원 공간에 표현한 그림입니다. 
#### 그림을 보면, 어떤 사용자가 어떤 영화를 좋아할지를 알 수 있다. 라이온킹은 dave의 취향에 잘 맞을 것이고, Gus는 덤앤더머를 선호할 것이라는 것을 알 수 있다. 즉 영화의 특성과 사용자의 특성이 각각 2개의 잠재요인으로 분해되었고, 이 잠재요인을 보면 어떤 영화가 어떤 사용자의 취향에 맞을지를 예상해볼 수 있다.
---

## SVD를 이용한 MF
- 차원 축소기법 중의 하나
- 데이터(사용자X아이템)를 `3개의 행렬로 분해해서` 이를 학습시키고, 이 3개의 행렬로 원래의 행렬로 재현하는 기법
- 3개의 행렬로 분해: $U (사용자의 latent factor) ,\sum(latent factor의 중요도,고유값 분해값),V (아이템의 latent factor)$
- Latent Factor는 user와 item이 공통으로 갖는 특징

<img src="https://blog.kakaocdn.net/dn/QRIy9/btqzIy9Ey5Z/AQWbBJ2tUNwmARie0bWQ80/img.png">

## SVD를 이용한 MF
MovieLens 데이터셋을 활용하여 , SVD를 직접 구현해보고, 적절한 K의 값을 찾아보자.

(1) MovieLens 데이터셋 불러오기

```python
# 데이터셋 불러오기 : microsoft reco_util 모듈을 이용하여 MovieLens 데이터 불러오기 

# reco_util 모듈이 있는 경로 지정 
import sys
os.chdir('e:/github/recommender_github/Recommenders')

import numpy as np
import pandas as pd

import math

from reco_utils.dataset import movielens

MOVIELENS_DATA_SIZE = '100k'

# movielens.py 
data = movielens.load_pandas_df(
    size = MOVIELENS_DATA_SIZE
)

# 메모리 사용량을 줄이기 위해 float를 32bit로 변경
data['rating'] = data['rating'].astype(np.float32)

```
(2) train/test 분할

```python
# train/test split 
train_df, test_df = train_test_split(data, test_size=0.2, random_state=1234)
```
(3) Sparse Matirx 만들기

```python
sparse_matrix = train_df.groupby('movieId').apply(lambda x: pd.Series(x['rating'].values, index=x['userId'])).unstack()
sparse_matrix.index.name = 'movieId'
```

(4) 결측값 채우기 (user 기준 평점 평균값, item 기준 평점 평균값으로 채우기) 
```python
# fill sparse matrix with average of movie ratings
sparse_matrix_withmovie = sparse_matrix.apply(lambda x: x.fillna(x.mean()), axis=1)

# fill sparse matrix with average of user ratings
sparse_matrix_withuser = sparse_matrix.apply(lambda x: x.fillna(x.mean()), axis=0)
```

(5) svd 함수를 이용해 사용자/아이템 잠재요인행렬 추출

```python
def get_svd(s_matrix, k=300): # 잠재요인의 수를 결정
  u, s, vh = np.linalg.svd(s_matrix.transpose()) # 3개의 행렬 분해 
  S = s[:k] * np.identity(k, np.float)
  T = u[:,:k]
  Dt = vh[:k,:]

  item_factors = np.transpose(np.matmul(S, Dt)) # 아이템의 latent factor
  user_factors = np.transpose(T) # 사용자의 latent factor

  return item_factors, user_factors

```

```python

# 아이템 기준 latent factor 
item_factors, user_factors = get_svd(sparse_matrix_withmovie)
prediction_result_df = pd.DataFrame(np.matmul(item_factors, user_factors),
                                    columns=sparse_matrix_withmovie.columns.values, index=sparse_matrix_withmovie.index.values)

movie_prediction_result_df = prediction_result_df.transpose()\

# 사용자 기준 latent factor 
item_factors, user_factors = get_svd(sparse_matrix_withuser)
prediction_result_df = pd.DataFrame(np.matmul(item_factors, user_factors),
                                    columns=sparse_matrix_withuser.columns.values, index=sparse_matrix_withuser.index.values)

user_prediction_result_df = prediction_result_df.transpose()
```

(6) 평가 및 예측 평점 구하기 

```python
def evaluate(test_df, prediction_result_df):
  groups_with_movie_ids = test_df.groupby(by='movieId')
  groups_with_user_ids = test_df.groupby(by='userId')
  intersection_movie_ids = sorted(list(set(list(prediction_result_df.columns)).intersection(set(list(groups_with_movie_ids.indices.keys())))))
  intersection_user_ids = sorted(list(set(list(prediction_result_df.index)).intersection(set(groups_with_user_ids.indices.keys()))))

  print(len(intersection_movie_ids))
  print(len(intersection_user_ids))

  compressed_prediction_df = prediction_result_df.loc[intersection_user_ids][intersection_movie_ids]

  # test_df에 대해서 RMSE 계산
  grouped = test_df.groupby(by='userId')
  rmse_df = pd.DataFrame(columns=['rmse'])
  for userId, group in tqdm(grouped):
      if userId in intersection_user_ids:
          pred_ratings = compressed_prediction_df.loc[userId][compressed_prediction_df.loc[userId].index.intersection(list(group['movieId'].values))]
          pred_ratings = pred_ratings.to_frame(name='rating').reset_index().rename(columns={'index':'movieId','rating':'pred_rating'})
          actual_ratings = group[['rating', 'movieId']].rename(columns={'rating':'actual_rating'})

          final_df = pd.merge(actual_ratings, pred_ratings, how='inner', on=['movieId'])
          final_df = final_df.round(4) # 반올림
          
          if not final_df.empty:
            rmse = sqrt(mean_squared_error(final_df['actual_rating'], final_df['pred_rating']))
            rmse_df.loc[userId] = rmse

  return final_df, rmse_df

result_df, _ = evaluate(test_df, user_prediction_result_df)
print(result_df)
print("For user matrix")
print(f"RMSE: {sqrt(mean_squared_error(result_df['actual_rating'].values, result_df['pred_rating'].values))}")
```

(7) 잠재요인수(K) 결정
K의 수를 50부터 200까지 10씩 증가하면서 RMSE 값을 비교하여 최적의 K값을 찾아보자.

```python
def find_best_k(sparse_matrix, maximum_k=100):
    print("\nFind best optimized k for Matrix Factorization")
    k_candidates = np.arange(50, maximum_k, 10)
    final_df = pd.DataFrame(columns=['rmse'], index=k_candidates)
    for k in tqdm(k_candidates):
        item_factors, user_factors = get_svd(sparse_matrix, k)
        each_results_df = pd.DataFrame(np.matmul(item_factors, user_factors),
                                    columns=sparse_matrix.columns.values, index=sparse_matrix.index.values)
        each_results_df = each_results_df.transpose()
        
        result_df, _ = evaluate(test_df, each_results_df)
        each_rmse = sqrt(mean_squared_error(result_df['actual_rating'].values, result_df['pred_rating'].values))

        final_df.loc[k]['rmse'] = each_rmse

    return final_df

res = find_best_k(sparse_matrix_withmovie, 200)
```

K 값의 변화에 따른 RMSE 변화 그래프 
```python
plt.plot(res.index, res.rmse)
plt.title("Find best optimized k for Matrix Factorization", fontsize=20)
plt.xlabel('number of k', fontsize=15)
plt.ylabel('rmse', fontsize=15)
plt.show()
```

```
