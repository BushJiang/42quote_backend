# 功能：神经网络搜索器类

# 导入必要的库
from qdrant_client import QdrantClient
# from sentence_transformers import SentenceTransformer
from fastembed.embedding import FlagEmbedding as Embedding

from qdrant_client.http.models import Filter




# 定义一个神经网络搜索类
class NeuralSearcher:
    # 构造函数，初始化类实例时调用
    def __init__(self, collection_name): 
        # 将集合名称保存为类属性
        self.collection_name = collection_name
        # 初始化句子编码模型，用于将文本转换为向量
        # 支持多语言的模型：paraphrase-multilingual-MiniLM-L12-v2
        self.model = Embedding(model_name="BAAI/bge-small-zh-v1.5", max_length=512)
        
        
        # 初始化 Qdrant 客户端，用于与 Qdrant 服务进行通信
        cluster_url = "https://f16bb39d-a7c0-4979-88f9-6f7c39d58812.us-east4-0.gcp.cloud.qdrant.io"
        qdrant_apikey = "NRwk2vo5HRHtu-k6rnbKWnUy6sl61psQHUUCv7RkclNs_T4r3B7pzg"


        # 使用本地向量数据库
        # self.qdrant_client = QdrantClient(host='localhost', port=6335)

        # 使用qdrant cloud向量数据库
        self.qdrant_client = QdrantClient(
            url=cluster_url, 
            api_key=qdrant_apikey,
        )

    
    # 定义一个搜索函数，用于执行基于文本的搜索
    def search(self, text: str):
        # 使用模型将文本转换为向量
        # self.model.embed(text)是一个生成器，可以生成一个numpy。vector需要转换成列表
        vector =next(self.model.embed(text)).tolist()
        # 使用 Qdrant 客户端在指定集合中进行搜索
        search_result = self.qdrant_client.search(
	        # 指定搜索的集合
            collection_name=self.collection_name,  
            # 使用转换后的向量作为查询
            query_vector=vector, 
            # 不使用过滤条件                
            query_filter=None,  
            # 返回最相似的前5个结果                   
            limit=5                                  
        )
        # 提取搜索结果中的有效载荷（payload）
        payloads = [hit.payload for hit in search_result]
        return payloads

    # 定义一个带有城市过滤条件的搜索函数
    def search_with_author_filter(self, text: str, author_of_interest: str):
        # 同样使用模型将文本转换为向量
        # self.model.embed(text)是一个生成器，可以生成一个numpy。vector需要转换成列表
        vector =next(self.model.embed(text)).tolist()
        
        # 定义一个过滤条件，仅返回特定城市的结果
        city_filter = Filter(**{
            "must": [{
	            # 指定要匹配的字段为"city"
                "key": "author",               
                "match": {
	                # 设置匹配条件，选择"city"字段值为指定城市的数据
                    "value": author_of_interest  
                }
            }]
        })
        # 使用带有过滤条件的搜索查询
        search_result = self.qdrant_client.search(
	        # 指定搜索的集合
            collection_name=self.collection_name,  
            # 使用转换后的向量作为查询
            query_vector=vector,    
            # 应用城市过滤条件               
            query_filter=city_filter,  
            # 返回最相似的前5个结果            
            limit=5                                  
        )
        # 提取搜索结果中的有效载荷（payload）
        payloads = [hit.payload for hit in search_result]
        return payloads


