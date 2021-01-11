---
layout: post
title: (data anaylsis_basis) 데이터분석기초
tags: [데이터분석기초]
math: true
date: 2021-01-11 23:08 
comments : true
---
---
정형 데이터(Tabular Data)를 불러오는 방법에 대해 작성한 글입니다. 

- 판다스 라이브러리를 이용해 데이터를 불러오느 방법을 정리하였습니다 🙂


오늘 정리할 내용은 다음과 같습니다. (링크) 

- pandas 라이브러리에서 제공하는 데이터 포맷은 ?
- Text (CSV, JSON, MS Excel 데이터 불러오기)
- binary (HDF5 Format, SAS 파일)
- SQL(쿼리)로 데이터 불러오기


pandas 라이브러리를 이용한 데이터 불러오기 

- pandas 라이브러리는 DataFrame, Series 등의 데이터 객체를 이용해서 데이터를 쉽게 가공할 수 있는 오픈소스 라이브러리입니다.

[참고링크]( https://pandas.pydata.org/docs/user_guide/io.html)

- pandas 라이브리를 이용해 불러올 수 있는 데이터 유형은 다음 <표>와 같습니다
- text (CSV, JSON,MS excel 등), binary(HDF5 Format,SAS,SPSS), SQL(SQL,Google Bigquery) 등을 제공
---


##  pandas 라이브러리에서 제공하는 load data type 

|Format Type|Data Description|Reader|Writer|  
|-----------|---|---|---|---|---|
|text|CSV|read_csv|to_csv|
|text|Fixed-Width Text File|read_fwf|
|text|JSON|read_json|to_json|
|text|HTML|read_html|to_html|
|text|Local clipboard|read_clipboard|to_clipboard|
|binary|MS Excel|read_excel|to_excel|
|binary|HDF5 Format|read_hdf|to_hdf|
|binary|Feather Format|read_feather|to_feather|
|binary|Parquet Format|read_parquet|to_parquet|
|binary|ORC Format|read_orc||
|binary|Msgpack|read_msgpack|to_msgpack|
|binary|Stata|read_stata|to_stata|
|binary|SAS|read_sas|to_feather|
|binary|SPSS|read_spss|to_parquet|
|binary|Python Pickle Format|read_pickle|to_pickle|
|SQL|SQL|read_sql|to_sql|
|SQL|Google BigQuery|read_gbq|to_gbq|

---
##  Code 

- CSV 데이터 불러오기 

```python
## CSV 데이터 불러오기
import pandas as pd 
data = pd.read_csv('uber_css_data.csv')
```

- JSON 데이터 불러오기  

```python
# JSON 데이터 생성 (dictionary)
import json
patients = {
         "Name":{"0":"John","1":"Nick","2":"Ali","3":"Joseph"},
         "Gender":{"0":"Male","1":"Male","2":"Female","3":"Male"},
         "Nationality":{"0":"UK","1":"French","2":"USA","3":"Brazil"},
         "Age" :{"0":10,"1":25,"2":35,"3":29}
}

# # JSON 데이터 저장
import json
with open('patients.json','w') as fp:
    json.dump(patients,fp)


# pandas 라이브러리 이용해 json 데이터 불러오기 
import pandas as pd
patients_df = pd.read_json('patients.json')
patients_df.head()
```

- MS Excel 데이터 불러오기

```python
# 엑셀 데이터 불러오기
df_xls = pd.read_excel("data.xlsx", sheet_name="Sheet1") # sheet 명 지정 필요
df_xls.head(1)
```

- SAS 파일 불러오기 

```python
# SAS 파일 물러오기 
df = pd.read_sas("sas_data.sas7bdat")


def do_something(chunk):
    pass

# 10,000개 라인 chunk_size 마다 데이터 불러오기
with pd.read_sas("sas_xport.xpt", chunk=100000) as rdr:
    for chunk in rdr:
        do_something(chunk)
```

DB연동 후 SQL로 불러오는 방법은 다음 포스팅에서 정리할 예정입니다. 

