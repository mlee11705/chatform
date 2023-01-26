import faiss
import pickle
import json
import argparse

def chain(question):
    index = faiss.read_index("docs.index")

    with open("faiss_store.pkl", "rb") as f:
        store = pickle.load(f)

    store.index = index
    chain = VectorDBQAWithSourcesChain.from_llm(llm=OpenAI(temperature=0), vectorstore=store)
    result = chain({"question": question})
    return {"answer": result['answer'], "sources": result['sources']}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("question", type=str, help="question to be answered")
    args = parser.parse_args()
    answer = chain(args.question)
    print(json.dumps(answer))
