import re
from langchain_chroma import Chroma
from langchain_core.documents import Document
from utils.my_llm import embeddings
from tools.get_projet_root import PROJECT_ROOT
import time
import json
from pathlib import Path
import hashlib
import argparse
def load_manifest(manifest_path: Path):
    items = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            abs_path = PROJECT_ROOT / obj["path"]
            items.append({
                "file_path": abs_path,
                "law_source": obj["source"],
                "law_year": obj["year"],
            })
    return items

STATE_FILE = PROJECT_ROOT / "db" / "index_state.json"

def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def load_index_state():
    if not STATE_FILE.exists():
        return {}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return {}
            return json.loads(content)
    except (json.JSONDecodeError, OSError):
        return {}
        
def save_index_state(state):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

def split_text(file_path, law_source, law_year):
    
    article_pattern = r'^第[一二三四五六七八九十百千]+条'
    part_pattern = r"第[一二三四五六七八九十百千]+编"
    chapter_pattern = r"第[一二三四五六七八九十百千]+章"

    chunks = []
    current_part = "无"
    current_chapter = "无"
    current_article_num = ""
    current_text_buffer = ""

    with open(file_path, "r" ,encoding = "utf-8") as f:
        print(f"正在处理文件: {file_path}")
        for line in f:
            line = line.strip()
            if not line:
                continue

            if re.match(part_pattern, line):
                current_part = line
                current_chapter = "无"
            elif re.match(chapter_pattern, line):
                current_chapter = line
            elif re.match(article_pattern, line):
                if current_text_buffer != "":   
                    chunk = {
                        "page_content": current_text_buffer,
                        "metadata": {
                            "source": law_source,
                            "year": law_year,
                            "part": current_part,
                            "chapter": current_chapter,
                            "article_num": current_article_num,
                        }
                    }
                    chunks.append(chunk)
                match_obj = re.match(article_pattern, line)
                current_article_num = match_obj.group(0)
                current_text_buffer = line + "\n"
            else:
                if current_text_buffer != "":
                    current_text_buffer += line + "\n"
            
        if current_text_buffer != "":
            chunk = {
                "page_content": current_text_buffer,
                "metadata": {
                    "source": law_source,
                    "year": law_year,
                    "part": current_part,
                    "chapter": current_chapter,
                    "article_num": current_article_num,
                }
            }
            chunks.append(chunk)
        print(f"文件处理完成: {file_path}")
    return chunks

CHROMA_DB_PATH = PROJECT_ROOT / "db" / "chroma_law"
CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)

def create_vector_store(manifest_path: Path):
    docs = []
    entries = load_manifest(manifest_path)
    old_state = load_index_state()
    new_state = dict(old_state)
    print("开始创建向量数据库...")
    for e in entries:
        file_path = e["file_path"]
        rel_path = str(file_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        digest = file_sha256(file_path)

        if old_state.get(rel_path) == digest:
            print(f"跳过未变化文件: {rel_path}")
            continue

        print(f"处理新增/变更文件: {rel_path}")
        chunks = split_text(e["file_path"], e["law_source"], e["law_year"])
        for chunk in chunks:
            doc = Document(page_content=chunk["page_content"], metadata=chunk["metadata"])
            docs.append(doc)
        
        new_state[rel_path] = digest

    if not docs:
        print("没有新增/变更文件，无需创建向量数据库")
        return True
    print(f"本次更新{len(docs)} 条法条数据")
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DB_PATH),
    )
    batch_size = 100
    for i in range(0, len(docs), batch_size):
        end_idx = min(i+batch_size, len(docs))
        print(f"正在处理第{i+1}条到第{end_idx}条数据...")
        vector_store.add_documents(docs[i:i+batch_size])
        time.sleep(0.5)
    save_index_state(new_state)
    print("向量数据库创建完成")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="documents/manifest.jsonl", help="manifest 文件路径（相对项目根目录）")
    args = parser.parse_args()

    manifest_path = PROJECT_ROOT / args.manifest
    create_vector_store(manifest_path)
