# 一. RAG基础入门

## 1. 解锁RAG

### 1.1 什么是RAG?

#### 1.1.1 核心定义

**RAG核心：**将模型内部学到的”**参数化知识**“(模型权重中固化的、模糊的”记忆“)，与外部知识库的”**非参数化知识**“(精准、可随时更新的外部数据)相结合。

**运作逻辑：**LLM生成文本之前，先通过检索机制从外部知识库中动态获取相关信息，并将这些参考资料融入生成过程，从而提升输出的准确性和时效性。

#### 1.1.2 技术原理

<img src="[https://github.com/DorriG/all-in-rag-notes/edit/main/Chapter1/pic/rag.png](https://github.com/DorriG/all-in-rag-notes/blob/main/Chapter1/pic/rag.png)" style="zoom:50%;" />

1.  **检索阶段：寻找”非参数化知识“**
   - **知识向量化：** **Embedding Model**将外部知识库编码为向量索引(Index)，存入**向量数据库**。
   - **语义召回：** 用户查询，检索模块利用Embedding Model将问题向量化，通过相似度检索，从向量数据库中检索与问题相关的多个文档片段。
2. **生成阶段：融合两种知识**
   - **上下文整合：** 生成模块接受检索阶段送来的相关片段及用户的原始问题。
   - **指令引导生成：** 该模块遵循预设的Prompt指令，将上下文与问题有效整合，并引导LLM进行可控的、有理有据的文本生成。

#### 1.1.3 技术演进分类

<img src="C:\Users\tingting\AppData\Roaming\Typora\typora-user-images\image-20251215144827510.png" alt="image-20251215144827510" style="zoom:50%;" />

| 初级 RAG（Naive RAG） | 高级 RAG（Advanced RAG）                 | 模块化 RAG（Modular RAG）                                    |                                                              |
| --------------------- | ---------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **流程**              | **离线:** `索引` **在线:** `检索 → 生成` | **离线:** `索引` **在线:** `...→ 检索前 → ... → 检索后 → ...` | 积木式可编排流程                                             |
| **特点**              | 基础线性流程                             | 增加**检索前后**的优化步骤                                   | 模块化、可组合、可动态调整                                   |
| **关键技术**          | 基础向量检索                             | **查询重写（Query Rewrite）** **结果重排（Rerank）**         | **动态路由（Routing）** **查询转换（Query Transformation）** **多路融合（Fusion）** |
| **局限性**            | 效果不稳定，难以优化                     | 流程相对固定，优化点有限                                     | 系统复杂性高                                                 |

> **离线：**提前完成数据预处理工作。
>
> **在线：**用户发起请求后的实时处理流程。

### 1.2 为什么要使用RAG？

#### 1.2.1 技术选型 RAG vs. 微调

优先选择对模型改动最小、成本最低的方案，技术路径选择遵循的顺序：**提示词工程--->检索增强生成--->微调**。

<img src="C:\Users\tingting\AppData\Roaming\Typora\typora-user-images\image-20251215145754256.png" alt="image-20251215145754256" style="zoom:50%;" />

**LLM Optimization：** 对模型本身进行多大程度的修改，从左到右，优化的程度越来越深，提示词工程完全不改变模型权重，微调会直接修改模型参数。

**Context Optimization：**对输入给模型的信息进行多大程度的增强，从下到上，增强的程度越来越高，其中提示工程知识优化提问方式，RAG通过引入外部知识库，丰富上下文信息。

**选择路径**

- **提示词工程：**精心设计提示词来引导模型，适用于任务简单、模型已有相关知识的场景。
- **RAG：**模型缺乏特定或实时知识而无法回答，就使用RAG，通过外挂知识库为其提供上下文信息。
- **微调：**目标是改变模型”如何做？“。

**RAG优点：**

| 问题                      | RAG的解决方案                      |
| ------------------------- | ---------------------------------- |
| **静态知识局限**          | 实时检索外部知识库，支持动态更新   |
| **幻觉（Hallucination）** | 基于检索内容生成，错误率降低       |
| **领域专业性不足**        | 引入领域特定知识库（如医疗/法律）  |
| **数据隐私风险**          | 本地化部署知识库，避免敏感数据泄露 |

### 1.3 如何上手RAG？

#### 1.3.1 基础工具选择

**开发模式：** **LangChain** 或 **LlamaIndex** 等成熟框架快速集成，**也可以选择不依赖框架的原生开发**，以获得对系统流程更精细的控制力。

**记忆载体**（向量数据库）：既有 **Milvus**、**Pinecone** 等适合大规模数据的方案，也有 **FAISS**、**Chroma** 等轻量级或本地化的选择，需根据具体业务规模灵活决定。

**评估工具：** **RAGAS** 、**TruLens** 。

#### 1.3.2 四步构建最小可行系统-MVP

1. **数据准备与清洗：**将PDF、Word等多源异构数据标准化，并采用合理的分块策略。
2. **索引构建：**将切分好的文本通过**Embedding Model**转为向量，存入向量数据库。
3. **检索策略优化：**不依赖单一的向量搜索。采用**混合检索**（向量+关键词）等方式来提升召回率，引入重排序模型对检索结果进行二次精选。
4. **生成与提示工程：**设计一套清晰的**Prompt模板**，引导LLM基于检索到的上下文回答用户问题，来防止幻觉。

#### 1.3.3 进阶与评估

1. **评估维度与挑战**

   几个量化评估的维度：

   - **检索相关性：**找到的内容是否包含答案
   - **生成质量**
   - **语义准确性：**回答的意思是否正确
   - **词汇匹配度：**专业术语是否使用得当

   在不同的任务当中还有**检索依赖性**、**多维推理能力**。

2. **优化方向与架构演进**

   - **性能层面：**通过索引分层（对高频数据启用缓存）和多模态扩展（图像/表格）。
   - **架构层面：**简单的线性流程正在被更复杂的设计模式所取代。
   - 通过分支模型进行处理多路检索、通过循环模式进行自我修正。

### 1.4 一个demo

```python
import os
# hugging face镜像设置，如果国内环境无法使用启用该设置
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek

load_dotenv()

markdown_path = r"/home/dorri/.ssh/05_LLMs_STUDY/all-in-rag-main/data/C1/markdown/easy-rl-chapter1.md"

# 加载本地markdown文件
loader = UnstructuredMarkdownLoader(markdown_path)
docs = loader.load()

# 文本分块
text_splitter = RecursiveCharacterTextSplitter()
chunks = text_splitter.split_documents(docs)

# 中文嵌入模型
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
  
# 构建向量存储
vectorstore = InMemoryVectorStore(embeddings)
vectorstore.add_documents(chunks)

# 提示词模板
prompt = ChatPromptTemplate.from_template("""请根据下面提供的上下文信息来回答问题。
请确保你的回答完全基于这些上下文。
如果上下文中没有足够的信息来回答问题，请直接告知：“抱歉，我无法根据提供的上下文找到相关信息来回答此问题。”

上下文:
{context}

问题: {question}

回答:"""
                                          )

# 配置大语言模型
llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0.7,
    max_tokens=4096,
    api_key=os.getenv("DEEPSEEK_API_KEY")
)

# 用户查询
question = "文中举了哪些例子？"

# 在向量存储中查询相关文档
retrieved_docs = vectorstore.similarity_search(question, k=3)
docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

answer = llm.invoke(prompt.format(question=question, context=docs_content))
print(answer)

```

**Output：**

```python
content='根据上下文信息，文中举了以下例子：\n\n1.  
**自然界中的羚羊**：羚羊出生后通过试错学习站立和奔跑，以适应环境。\n2.  
**股票交易**：通过不断买卖股票并根据市场反馈来学习如何最大化奖励。\n3.  
**玩雅达利游戏（如Breakout、Pong）**：通过不断试错来学习如何通关。\n4.  
**选择餐馆**：利用是去已知喜欢的餐馆；探索是尝试新的餐馆。\n5.  
**做广告**：利用是采取已知最优的广告策略；探索是尝试新的广告策略。\n6.  
**挖油**：利用是在已知有油的地方挖；探索是在新的地方挖油。\n7.  
**玩游戏（如《街头霸王》）**：利用是总是采取某种特定策略（如蹲角落出脚）；探索是尝试新的招式。' 
additional_kwargs={'refusal': None} 
response_metadata={
    'token_usage': {
        'completion_tokens': 185, 
        'prompt_tokens': 5550, 
        'total_tokens': 5735, 
        'completion_tokens_details': None, 
        'prompt_tokens_details': {
            'audio_tokens': None, 
            'cached_tokens': 0}, 
        'prompt_cache_hit_tokens': 0, 
        'prompt_cache_miss_tokens': 5550}, 
    'model_name': 'deepseek-chat', 
    'system_fingerprint': 'fp_eaab8d114b_prod0820_fp8_kvcache', 
    'id': 'c17c369f-ef01-48bb-953f-39e40da0e622', 
    'service_tier': None, 
    'finish_reason': 'stop', 
    'logprobs': None} 
id='run--7d3a1dfb-5665-4a37-b7b7-f1c5fd5fbbc2-0' 
usage_metadata={
    'input_tokens': 5550, 
    'output_tokens': 185, 
    'total_tokens': 5735, 
    'input_token_details': {'cache_read': 0}, 
    'output_token_details': {}}
```

### 1.5 作业

1. LangChain代码最终得到的输出携带了各种参数，查询相关资料尝试把这些参数过滤掉得到`content`里的具体回答。

   ```python
   只要print(answer.content)即可
   ```

2. 修改Langchain代码中`RecursiveCharacterTextSplitter()`的参数`chunk_size`和`chunk_overlap`，观察输出结果有什么变化。

   ```python
   chunk_size=1000, chunk_overlap=200 （答案更范，且全）
   结果：
   根据上下文，文中举了以下例子：
   1. 选择餐馆的例子：利用是指去最喜欢的餐馆，探索是指尝试新的餐馆。
   2. 做广告的例子：利用是指采取最优广告策略，探索是指尝试新的广告策略。
   3. 挖油的例子：利用是指在已知地方挖油，探索是指在新地方挖油。
   4. 玩游戏的例子：利用是指总是采取某一种策略（如玩《街头霸王》时蹲在角落一直出脚），探索是指尝试新的招式（如放出“大招”）。
   5. 象棋选手的例子：奖励是赢棋（正奖励）或输棋（负奖励）。
   6. 股票管理的例子：奖励由股票获取的奖励与损失决定。
   7. 玩雅达利游戏的例子：奖励是增加或减少的游戏分数。
   8. 小车上山（MountainCar-v0）的例子：用于说明与Gym库的交互，包括观测空间和动作空间。
   
   chunk_size=500, chunk_overlap=100  （答案更加精准）
   根据上下文，文中举了以下例子：
   1. **强化学习有意思的例子**：
      - DeepMind 研发的走路的智能体（学习在曲折道路上保持平衡）。
      - 机械臂抓取（使用强化学习训练统一的抓取算法适应不同物体）。
   2. **探索和利用的例子**：
      - 选择餐馆（利用：去最喜欢的餐馆；探索：尝试新餐馆）。
      - 做广告（利用：采取最优广告策略；探索：尝试新广告策略）。
      - 挖油（利用：在已知地方挖油；探索：在新地方挖油）。
      - 玩游戏（如《街头霸王》，利用：一直采取出脚策略；探索：尝试新招式）。
   3. **奖励的例子**：
      - 象棋选手（赢棋得正奖励，输棋得负奖励）。
       
   chunk_size=200, chunk_overlap=50 （相对于第一个没那么全）
   根据上下文，文中举的例子包括：
   1. 选择餐馆的例子，用来说明探索（尝试新餐馆）和利用（去最喜欢的餐馆）的概念。
   2. 做广告的例子，同样用来说明探索（尝试新广告策略）和利用（采取最优广告策略）的概念。
   3. 自然界中羚羊学习站立和奔跑的例子，作为强化学习在现实生活中的应用。
   4. 股票交易的例子，说明通过买卖股票并根据市场反馈学习以使奖励最大化的过程。
   ```

3. 给LlamaIndex代码添加代码注释。

   ```python
   import os
   # os.environ['HF_ENDPOINT']='https://hf-mirror.com'
   from dotenv import load_dotenv
   from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings 
   from llama_index.llms.deepseek import DeepSeek
   from llama_index.embeddings.huggingface import HuggingFaceEmbedding
   
   load_dotenv()
   
   # 设置LLM和嵌入模型
   Settings.llm = DeepSeek(model="deepseek-chat", api_key=os.getenv("DEEPSEEK_API_KEY"))
   Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")
   
   # 读取文档
   docs = SimpleDirectoryReader(input_files=[r"/home/dorri/.ssh/05_LLMs_STUDY/all-in-rag-main/data/C1/markdown/easy-rl-chapter1.md"]).load_data()
   
   # 构建索引
   index = VectorStoreIndex.from_documents(docs)
   
   # 构建查询引擎
   query_engine = index.as_query_engine()
   
   # 查看提示词模板
   print(query_engine.get_prompts())
   
   # 查询
   print(query_engine.query("文中举了哪些例子?"))

   ```

