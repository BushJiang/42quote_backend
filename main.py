# 文件：service.py

from fastapi import FastAPI

# That is the file where NeuralSearcher is stored
from neural_searcher import NeuralSearcher

app = FastAPI()

# Create an instance of the neural searcher
# collection_name = "chinese_poet"
collection_name = "golden"

neural_searcher = NeuralSearcher(collection_name=collection_name)

@app.get("/api/search")
def search_startup(q: str):
	return {
		"result": neural_searcher.search(text=q)
	}

@app.get("/api/search_filter")
def search_filter_startup(q: str, author: str):
	return {
		"result": neural_searcher.search_with_author_filter(text=q, author_of_interest=author)
	}


if __name__ == "__main__":
	import uvicorn
	uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)



