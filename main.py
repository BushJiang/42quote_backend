# 文件：service.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# That is the file where NeuralSearcher is stored
from neural_searcher import NeuralSearcher


app = FastAPI()



# 实现跨域连接
# 设置允许的来源
origins = [
    "https://four2quote-frontend.onrender.com",  # Render 前端服务的 URL
    "http://localhost",  # 本地开发
    "http://localhost:8080",  # 本地开发，Vue 默认端口
]

# 设置中间件，
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)





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
	uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
