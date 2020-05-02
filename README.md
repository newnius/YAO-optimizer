

## Feed

/feed?job=lstm&seq=1&value=3

## train
/train?job=lstm

## predict
/predict?job=lstm&seq=2


## Make Data

```bash
cat data/stock_data.csv | awk -F',' '{print NR "," $5 }' > data/stock.csv
```