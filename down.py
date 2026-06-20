# from huggingface_hub import snapshot_download

# # repo_id는 레포지토리 주소의 '작성자/이름' 부분입니다.
# snapshot_download(repo_id="BAAI/bge-m3", local_dir="./다운로드경로")

# from transformers import pipeline

# pipe = pipeline("fill-mask", model="kakaobank/kf-deberta-base")

import torch
from transformers import pipeline
from pprint import pprint

pipe = pipeline(
    "fill-mask",
    model="skt/A.X-Encoder-base",
    torch_dtype=torch.bfloat16,
)

input_text = "한국의 수도는 <mask>."
results = pipe(input_text)
pprint(results)
# [{'score': 0.07568359375,
#  'sequence': '한국의 수도는 서울.',
#  'token': 31430,
#  'token_str': '서울'}, ...
