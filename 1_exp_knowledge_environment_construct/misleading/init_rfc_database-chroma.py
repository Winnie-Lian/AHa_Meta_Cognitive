import requests
import re
from sentence_transformers import SentenceTransformer
import chromadb
import torch
import os

# 检测GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
def download_rfc(rfc_number, save_dir="rfc_docs"):
    """
    下载 RFC 文档，如果文档已存在则跳过下载。
    :param rfc_number: RFC 文档编号
    :param save_dir: 保存文档的目录
    :return: 文档的文件路径
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    file_path = os.path.join(save_dir, f"rfc{rfc_number}.txt")
    
    # 检查文件是否已存在
    if os.path.exists(file_path):
        print(f"RFC {rfc_number} already exists at {file_path}, skipping download.")
        return file_path
    
    # 下载文档
    url = f"https://www.rfc-editor.org/rfc/rfc{rfc_number}.txt"
    response = requests.get(url)
    
    if response.status_code == 200:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(response.text)
        print(f"Downloaded RFC {rfc_number} to {file_path}")
        return file_path
    else:
        print(f"Failed to download RFC {rfc_number}")
        return None# 下载RFC文档
def preprocess_rfc(file_path, chunk_size=5, min_length=50):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # 清理文档
    text = re.sub(r"\n\s*\n", "\n", text)
    text = re.sub(r"RFC \d+.*\n", "", text)
    text = re.sub(r"\[Page \d+\].*", "", text)
    text = re.sub(r"-{3,}", "", text)
    text = text.strip()

    # 按句子分割
    sentences = re.split(r"(?<=[.!?])\s+", text)
    segments = []
    
    # 合并句子为更大的片段
    for i in range(0, len(sentences), chunk_size):
        chunk = " ".join(sentences[i:i + chunk_size]).strip()
        if len(chunk) >= min_length:
            segments.append(chunk)
    
    print(f"Processed {len(segments)} segments from {file_path}")
    return segments

def build_and_save_chroma_database(rfc_numbers, model_name='/home/xjx/hallucination/exp/misleading/generate/rag/all-mpnet-base-v2', collection_name="rfc_collection"):
    client = chromadb.PersistentClient(path="./rfc_chroma_db")
    collection = client.get_or_create_collection(name=collection_name)
    
    all_segments = []
    for rfc in rfc_numbers:
        file_path = download_rfc(rfc)
        if file_path:
            segments = preprocess_rfc(file_path, chunk_size=5, min_length=50)
            all_segments.extend(segments)
    
    embedder = SentenceTransformer(model_name, device=device)
    embeddings = embedder.encode(all_segments, convert_to_numpy=True, show_progress_bar=True, batch_size=32)
    
    collection.add(
        embeddings=embeddings.tolist(),
        documents=all_segments,
        ids=[f"seg_{i}" for i in range(len(all_segments))]
    )
    
    print(f"Saved {len(all_segments)} segments to Chroma collection '{collection_name}'")


def main():
    rfc_numbers = ["9001", "9000", "9030", "9026", "9005", 
                    "9014", "9363", "9334", "9439", "9114",
                    "9204", "9287", "9220", "9147", "8888",
                    "9191", "8949", "9200", "9272", "8784",
                    "8966", "9002", "9473", "9449", "9421",
                    "9221", "8879", "8484", "8555", "8961",
                    "9019", "8812", "9257", "9139", "9076",
                    "9417", "9290", "9113", "8881", "9360",
                    "9485", "9297", "9458", "9178", "9457",
                    "9453", "9497", "9382", "9501", "9374",
                    ]
    build_and_save_chroma_database(rfc_numbers)

if __name__ == "__main__":
    main()
