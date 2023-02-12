# QuerySuggestions
Final project on the "Ranking and Matching" block

As a final project, it is necessary to implement a system of prompts for similar questions on Quora data. The search is performed exclusively on the main question (title) without clarifying details.

The system is represented by a microservice based on Flask. The top-level pipeline and criteria can be represented as follows:
- First, the request is filtered by language (using the LangDetect library) - all requests for which a certain language is not equal to "en" are excluded.
- Then, candidate questions are searched using FAISS (by vector similarity)
- These candidates are re-ranked by the KNRM model, after which up to 10 candidates are issued as a response.

Data is stored with git lfs

# how to run
```bash
docker-compose up --build
```


# tests
```bash
curl -X GET \
  'http://127.0.0.1:11000/ping' \
  --header 'Accept: */*' \
  --header 'Content-Type: application/json' \


curl -X POST \
  'http://127.0.0.1:11000/update_index' \
  --header 'Accept: */*' \
  --header 'Content-Type: application/json' \
  --data-raw '{
  "documents":{
    "1": "mama1 mama2 mama3"
  }
}'


curl -X POST \
  'http://127.0.0.1:11000/query' \
  --header 'Accept: */*' \
  --header 'Content-Type: application/json' \
  --data-raw '{
  "queries":["Абракадарбра","qwerty"]
}'
```

