import os

DATA_PATH = "data/processed"


def load_txt_files(data_path=DATA_PATH):
    """
    Load all txt files from processed data 
    """
    documents = []

    for file_name in os.listdir(data_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(data_path, file_name)

            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

                documents.append({
                    "file_name": file_name,
                    "content": text
                })

    return documents


if __name__ == "__main__":
    docs = load_txt_files()

    print(f"Loaded {len(docs)} documents\n")

    for doc in docs[:2]:
        print("FILE:", doc["file_name"])
        print("CONTENT PREVIEW:", doc["content"][:300])
        print("-" * 50)