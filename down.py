from huggingface_hub import snapshot_download

# repo_id는 레포지토리 주소의 '작성자/이름' 부분입니다.
snapshot_download(repo_id="BAAI/bge-m3", local_dir="./다운로드경로")
