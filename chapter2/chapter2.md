- ## 2. 数据准备

  ### 2.1 数据加载

  #### 2.1.1 文档加载器

  负责将各种格式的非结构化文档（PDF、Word、Markdown、HTML）转换为程序可以处理的结构化数据。数据加载的质量会直接影响后续的索引构建、检索效果和最终的生成质量。

  **1. 主要功能**

  1.  **文档格式解析：** 将不同格式的文档解析为文本内容。
  2.  **元数据提取：** 在解析文档内容的同时，提取相关的元数据信息，如文档来源、页码等。
  3.  **统一数据格式：** 将解析后的内容转换为统一的数据格式，便于后续处理。

  **2. 当前主流RAG文档加载器**

  | 工具名称            | 特点                           | 适用场景           | 性能表现              |
  | ------------------- | ------------------------------ | ------------------ | --------------------- |
  | **PyMuPDF4LLM**     | PDF→Markdown转换，OCR+表格识别 | 科研文献、技术手册 | 开源免费，GPU加速     |
  | **TextLoader**      | 基础文本文件加载               | 纯文本处理         | 轻量高效              |
  | **DirectoryLoader** | 批量目录文件处理               | 混合格式文档库     | 支持多格式扩展        |
  | **Unstructured**    | 多格式文档解析                 | PDF、Word、HTML等  | 统一接口，智能解析    |
  | **FireCrawlLoader** | 网页内容抓取                   | 在线文档、新闻     | 实时内容获取          |
  | **LlamaParse**      | 深度PDF结构解析                | 法律合同、学术论文 | 解析精度高，商业API   |
  | **Docling**         | 模块化企业级解析               | 企业合同、报告     | IBM生态兼容           |
  | **Marker**          | PDF→Markdown，GPU加速          | 科研文献、书籍     | 专注PDF转换           |
  | **MinerU**          | 多模态集成解析                 | 学术文献、财务报表 | 集成LayoutLMv3+YOLOv8 |

  #### 2.1.2 Unstructured文档处理库

  一个专业的文档处理库，专门设计用于RAG和AI微调场景的非结构化数据预处理。

  **1. 核心优势 **

  - **格式支持广泛：** PDF、Word、Excel、HTML、Markdown等，统一的API接口，不需要为不同格式编写不同代码。
  - **智能内容解析：** 自动识别文档结构，标题、段落、表格、列表等，保留文档元数据信息。

  **2. 支持的文档元素类型**

  | 元素类型            | 描述                                                         |
  | ------------------- | ------------------------------------------------------------ |
  | `Title`             | 文档标题                                                     |
  | `NarrativeText`     | 由多个完整句子组成的正文文本，不包括标题、页眉、页脚和说明文字 |
  | `ListItem`          | 列表项，属于列表的正文文本元素                               |
  | `Table`             | 表格                                                         |
  | `Image`             | 图像元数据                                                   |
  | `Formula`           | 公式                                                         |
  | `Address`           | 物理地址                                                     |
  | `EmailAddress`      | 邮箱地址                                                     |
  | `FigureCaption`     | 图片标题/说明文字                                            |
  | `Header`            | 文档页眉                                                     |
  | `Footer`            | 文档页脚                                                     |
  | `CodeSnippet`       | 代码片段                                                     |
  | `PageBreak`         | 页面分隔符                                                   |
  | `PageNumber`        | 页码                                                         |
  | `UncategorizedText` | 未分类的自由文本                                             |
  | `CompositeElement`  | 分块处理时产生的复合元素*                                    |

  > `CompositeElement` 是通过分块处理产生的特殊元素类型，由一个或多个连续的文本元素组合而成。

  #### 2.1.4 一个demo

  【Unstructured】https://docs.unstructured.io/open-source/core-functionality/partitioning

  ```python
  from unstructured.partition.auto import partition
  
  # PDF文件路径
  pdf_path = r"/home/dorri/.ssh/05_LLMs_STUDY/all-in-rag-main/data/C2/pdf/rag.pdf"
  
  # 使用Unstructured加载并解析PDF文档
  elements = partition(
      filename=pdf_path,
      content_type="application/pdf"
  )
  
  # 打印解析结果
  print(f"解析完成: {len(elements)} 个元素, {sum(len(str(e)) for e in elements)} 字符")
  
  # 统计元素类型
  from collections import Counter
  types = Counter(e.category for e in elements)
  print(f"元素类型: {dict(types)}")
  
  # 显示所有元素
  print("\n所有元素:")
  for i, element in enumerate(elements, 1):
      print(f"Element {i} ({element.category}):")
      print(element)
      print("=" * 60)
  ```

  #### 2.1.5 作业

  使用`partition_pdf`替换当前`partition`函数并分别尝试用`hi_res`和``进行解析，观察输出结果有何变化。

  > hi_res 模式在解析 PDF 时不仅关注文本内容，还保留了文档的版式与语义结构信息，这使得后续的文本分块、检索和生成过程能够以更合理的语义单元为基础。而 ocr_only 模式仅关注字符级识别，适合扫描文档，但不利于构建高质量的 RAG 上下文。

  ### 2.2 文本分块

  #### 2.2.1 理解文本分块

  **文本分块：** 将加载后的长篇文档，切分成更小、更易于处理的单元。chunk是后续向量检索和模型处理的基本单位。

  #### 2.2.2 文本分块重要性

  **两大考量：** 模型的**上下文限制**&检索生成的**性能需求**。

  1.  **满足模型上下文限制**

     - **Embedding Model：** 将文本块转换为向量，这类模型有严格的输入长度上限。超出限制的文本块在输入时会被截断，导致信息丢失，生成的向量无法完整代表原文的语义。
     - **LLM：** 根据检索到的上下文生成答案。LLM也有上下文窗口限制。检索到的所有文本块，以及用户问题和提示词，都必须放入这个窗口。

  2. **为什么“块”不是越大越好？**

     - **嵌入过程中的信息损失：** 

       - **分词：** 将输入的文本块分解成一个个token。
       - **向量化：** 将每个token生成一个高维向量表示。
       - **池化：** 通过某种方法，将所有token的向量压缩成一个单一的向量，这个向量代表整个文本块的语义。

       在压缩的过程中，信息损失无法避免。文本块越长，包含的语义信息越多，单一向量所承载的信息就越稀释，导致表示变得笼统，关键细节被模糊化，降低检索的精度。

     - **生成过程的不可控**

       即使检索初多个文本块给到LLM，也会出现关键信息被大量无关信息淹没的情况。研究表明，当LLM处理非常大信息量的上下文时，通常倾向于记住开头和结尾的信息。

     - **主题稀释导致检索失败**

       一个好的文本块应该聚焦在一个明确、单一的主题。如果一个块包含太多不相关的主题，其语义就会被稀释，导致检索的精度损失。

  #### 2.2.3 分块策略 

  1.  **固定大小分块**

     - **按段落分割**：`CharacterTextSplitter` 采用默认分隔符 `"\n\n"`，使用正则表达式将文本按段落进行分割，通过 `_split_text_with_regex` 函数处理。
     - **智能合并**：调用继承自父类的 `_merge_splits` 方法，将分割后的段落依次合并。该方法会监控累积长度，当超过 `chunk_size` 时形成新块，并通过重叠机制（`chunk_overlap`）保持上下文连续性，同时在必要时发出超长块的警告。 

     `CharacterTextSplitter` 实际实现的并非严格的固定大小分块。根据 `_merge_splits` 源码逻辑，这种方法会：

     - **优先保持段落完整性**：只有当添加新段落会导致总长度超过 `chunk_size` 时，才会结束当前块
     - **处理超长段落**：如果单个段落超过 `chunk_size`，系统会发出警告但仍将其作为完整块保留
     - **应用重叠机制**：通过 `chunk_overlap` 参数在块之间保持内容重叠，确保上下文连续性

     性能

     - **优点：** 实现简单、处理速度块、计算开销小。
     - **缺点：** 可能会在语义边界处切断文本，影响内容的完整性和连贯性。

  ```python
  from langchain.text_splitter import CharacterTextSplitter
  from langchain_community.document_loaders import TextLoader
  
  loader = TextLoader("../../data/C2/txt/蜂医.txt")
  docs = loader.load()
  
  text_splitter = CharacterTextSplitter(
      chunk_size=200,    # 每个块的目标大小为100个字符
      chunk_overlap=10   # 每个块之间重叠10个字符，以缓解语义割裂
  )
  
  chunks = text_splitter.split_documents(docs)
  
  print(f"文本被切分为 {len(chunks)} 个块。\n")
  print("--- 前5个块内容示例 ---")
  for i, chunk in enumerate(chunks[:5]):
      print("=" * 60)
      # chunk 是一个 Document 对象，需要访问它的 .page_content 属性来获取文本
      print(f'块 {i+1} (长度: {len(chunk.page_content)}): "{chunk.page_content}"')
  ```

  2. **递归字符分块**

     分块器通过分隔符层级递归处理，相对与固定大小分块，改善了超长文本的处理效果。采用一组有层次结构的分隔符（如段落、句子、单词）进行递归分割，旨在有效平衡语义完整性与块大小控制。这种分层处理机制，能够在尽可能保持高级语义结构完整性的同时，有效控制大小。

     算法流程：

     - **寻找有效分隔符：** 从分隔符列表中从前到后遍历，找到第一个在当前文本中**存在**的分隔符。如果都不存在，使用最后一个分隔符（通常是空字符串 `""`）。
     - **切分与分类处理：** 使用选定的分隔符切分文本，然后遍历所有片段
       - **如果片段不超过块大小：** 暂存到`_good_splits`中，准备合并
       - **如果片段超过块大小**
         - 先将暂存的合格片段通过`_merge_splits`合并成块
         - 再检查是都还有剩余分隔符
           - **有剩余分隔符：** 递归调用`_split_text`继续分割
           - **无剩余分隔符：** 直接保留为超长块
     - **最终处理：** 将剩余的暂存片段合并成最后的块。

     **实现细节**：

     - **批处理机制**: 先收集所有合格片段（`_good_splits`），遇到超长片段时才触发合并操作。
     - **递归终止条件**: 关键在于 `if not new_separators` 判断。当分隔符用尽时（`new_separators` 为空），停止递归，直接保留超长片段。确保算法不会无限递归。

     **与固定大小分块的关键差异**：

     - 固定大小分块遇到超长段落时只能发出警告并保留。
     - 递归分块会继续使用更细粒度的分隔符（句子→单词→字符）直到满足大小要求。

  ```python 
  from langchain.text_splitter import RecursiveCharacterTextSplitter
  from langchain_community.document_loaders import TextLoader
  
  loader = TextLoader(r"/home/dorri/.ssh/05_LLMs_STUDY/all-in-rag-main/data/C2/txt/蜂医.txt", encoding="utf-8")
  docs = loader.load()
  
  text_splitter = RecursiveCharacterTextSplitter(
      # 针对中英文混合文本，定义一个更全面的分隔符列表
      separators=["\n\n", "\n", "。", "，", " ", ""], # 按顺序尝试分割
      chunk_size=200,
      chunk_overlap=10
  )
  
  chunks = text_splitter.split_documents(docs)
  
  print(f"文本被切分为 {len(chunks)} 个块。\n")
  print("--- 前5个块内容示例 ---")
  for i, chunk in enumerate(chunks[:5]):
      print("=" * 60)
      print(f'块 {i+1} (长度: {len(chunk.page_content)}): "{chunk.page_content}"')
  ```

  3. **语义分块**

     在语义主题发生显著变化的地方进行切分。

     **实现原理**

     - **句子分割 (Sentence Splitting)**: 用标准的句子分割规则（例如，基于句号、问号、感叹号）将输入文本拆分成一个句子列表。
     - **上下文感知嵌入 (Context-Aware Embedding)**: 这是 `SemanticChunker` 的一个关键设计。该分块器不是对每个句子独立进行嵌入，而是通过 `buffer_size` 参数（默认为1）来捕捉上下文信息。对于列表中的每一个句子，这种方法会将其与前后各 `buffer_size` 个句子组合起来，然后对这个临时的、更长的组合文本进行嵌入。这样，每个句子最终得到的嵌入向量就融入了其上下文的语义。
     - **计算语义距离 (Distance Calculation)**: 计算每对**相邻**句子的嵌入向量之间的余弦距离。这个距离值量化了两个句子之间的语义差异——距离越大，表示语义关联越弱，跳跃越明显。
     - **识别断点 (Breakpoint Identification)**: `SemanticChunker` 会分析所有计算出的距离值，并根据一个统计方法（默认为 `percentile`）来确定一个动态阈值。例如，它可能会将所有距离中第95百分位的值作为切分阈值。所有距离大于此阈值的点，都被识别为语义上的“断点”。
     - **合并成块 (Merging into Chunks)**: 最后，根据识别出的所有断点位置，将原始的句子序列进行切分，并将每个切分后的部分内的所有句子合并起来，形成一个最终的、语义连贯的文本块。

     **断点识别方法 (`breakpoint_threshold_type`)**

     如何定义“显著的语义跳跃”是语义分块的关键。`SemanticChunker` 提供了几种基于统计的方法来识别断点：

     - `percentile` (百分位法 - **默认方法**):
       - **逻辑**: 计算所有相邻句子的语义差异值，并将这些差异值进行排序。当一个差异值超过某个百分位阈值时，就认为该差异值是一个断点。
       - **参数**: `breakpoint_threshold_amount` (默认为 `95`)，表示使用第95个百分位作为阈值。这意味着，只有最显著的5%的语义差异点会被选为切分点。
     - `standard_deviation` (标准差法):
       - **逻辑**: 计算所有差异值的平均值和标准差。当一个差异值超过“平均值 + N * 标准差”时，被视为异常高的跳跃，即断点。
       - **参数**: `breakpoint_threshold_amount` (默认为 `3`)，表示使用3倍标准差作为阈值。
     - `interquartile` (四分位距法):
       - **逻辑**: 使用统计学中的四分位距（IQR）来识别异常值。当一个差异值超过 `Q3 + N * IQR` 时，被视为断点。
       - **参数**: `breakpoint_threshold_amount` (默认为 `1.5`)，表示使用1.5倍的IQR。
     - `gradient` (梯度法):
       - **逻辑**: 这是一种更复杂的方法。它首先计算差异值的变化率（梯度），然后对梯度应用百分位法。对于那些句子间语义联系紧密、差异值普遍较低的文本（如法律、医疗文档）特别有效，因为这种方法能更好地捕捉到语义变化的“拐点”。
       - **参数**: `breakpoint_threshold_amount` (默认为 `95`)。

  ```python
  from langchain_experimental.text_splitter import SemanticChunker
  from langchain_community.embeddings import HuggingFaceEmbeddings
  from langchain_community.document_loaders import TextLoader
  
  embeddings = HuggingFaceEmbeddings(
      model_name="BAAI/bge-small-zh-v1.5",
      model_kwargs={'device': 'cpu'},
      encode_kwargs={'normalize_embeddings': True}
  )
  
  # 初始化 SemanticChunker
  text_splitter = SemanticChunker(
      embeddings,
      breakpoint_threshold_type="percentile" # 也可以是 "standard_deviation", "interquartile", "gradient"
  )
  
  loader = TextLoader(r"/home/dorri/.ssh/05_LLMs_STUDY/all-in-rag-main/data/C2/txt/蜂医.txt", encoding="utf-8")
  documents = loader.load()
  
  docs = text_splitter.split_documents(documents)
  
  print(f"文本被切分为 {len(docs)} 个块。\n")
  print("--- 前2个块内容示例 ---")
  for i, chunk in enumerate(docs[:2]):
      print("=" * 60)
      print(f'块 {i+1} (长度: {len(chunk.page_content)}):\n"{chunk.page_content}"')
  ```

  4. **基于文档结构的分块**

     以Markdown文件为例：

     LangChain 提供了 `MarkdownHeaderTextSplitter` 来处理。

     - **实现原理**: 该分块器的主要逻辑是“先按标题分组，再按需细分”。
       1. **定义分割规则**: 用户首先需要提供一个标题层级的映射关系，例如 `[ ("#", "Header 1"), ("##", "Header 2") ]`，告诉分块器 `#` 是一级标题，`##` 是二级标题。
       2. **内容聚合**: 分块器会遍历整个文档，将每个标题下的所有内容（直到下一个同级或更高级别的标题出现前）聚合在一起。每个聚合后的内容块都会被赋予一个包含其完整标题路径的元数据。
     - **元数据注入的优势**: 经过分割后，这个段落形成的文本块，其元数据就会是 `{"Header 1": "第三章：模型评估", "Header 2": "3.2节：评估指标"}`。这种元数据为每个块提供了精确的“地址”，极大地增强了上下文的准确性，让大模型能更好地理解信息片段的来源和背景。
     - **局限性与组合使用**: 单纯按标题分割可能会导致一个问题：某个章节下的内容可能非常长，远超模型能处理的上下文窗口。为了解决这个问题，`MarkdownHeaderTextSplitter` 可以与其它分块器（如 `RecursiveCharacterTextSplitter`）**组合使用**。具体流程是：
       1. 第一步，使用 `MarkdownHeaderTextSplitter` 将文档按标题分割成若干个大的、带有元数据的逻辑块。
       2. 第二步，对这些逻辑块再应用 `RecursiveCharacterTextSplitter`，将其进一步切分为符合 `chunk_size` 要求的小块。由于这个过程是在第一步之后进行的，所有最终生成的小块都会**继承**来自第一步的标题元数据。
     - **RAG应用优势**: 这种两阶段的分块方法，既保留了文档的宏观逻辑结构（通过元数据），又确保了每个块的大小适中，是处理结构化文档进行RAG的理想方案。

  #### 2.2.4 其他开源框架中的分块策略

  1.  **Unstructured：基于文档元素的智能分块**

     - **分区 (Partitioning)**: 这是一个重要功能，负责将原始文档（如PDF、HTML）解析成一系列结构化的“元素”（Elements）。每个元素都带有语义标签，如 `Title` (标题)、`NarrativeText` (叙述文本)、`ListItem` (列表项) 等。这个过程本身就完成了对文档的深度理解和结构化。
     - **分块 (Chunking)**: 该功能建立在分区的结果之上。分块功能不是对纯文本进行操作，而是将分区产生的“元素”列表作为输入，进行智能组合。Unstructured 提供了两种主要的分块方法：
       - **`basic`**: 这是默认方法。这种方法会连续地组合文档元素（如段落、列表项），直到达到 `max_characters` 上限，尽可能地填满每个块。如果单个元素超过上限，则会对其进行文本分割。
       - **`by_title`**: 该方法在 `basic` 方法的基础上，增加了对“章节”的感知。该方法将 `Title` 元素视为一个新章节的开始，并强制在此处开始一个新的块，确保同一个块内不会包含来自不同章节的内容。这在**处理报告、书籍等结构化文档**时非常有用，效果类似于 LangChain 的 `MarkdownHeaderTextSplitter`，但适用范围更广。

     Unstructured 允许将分块作为分区的一个参数在单次调用中完成，也支持在分区之后作为一个独立的步骤来执行分块。这种“先理解、后分割”的策略，使得 Unstructured 能在最大程度上保留文档的原始语义结构，特别是在处理版式复杂的文档时，优势尤为明显。

  2.  **LlamaIndex：面向节点的解析与转换**

     [LlamaIndex](https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/modules/) 将数据处理流程抽象为对“**节点（Node）**”的操作。文档被加载后，首先会被解析成一系列的“节点”，分块只是节点转换（Transformation）中的一环。

     LlamaIndex 的分块体系有以下特点：

     -  **丰富的节点解析器 (Node Parser)**: LlamaIndex 提供了大量针对特定数据格式和方法的节点解析器，可以大致分为几类：
       - **结构感知型**: 如 `MarkdownNodeParser`, `JSONNodeParser`, `CodeSplitter` 等，能理解并根据源文件的结构（如Markdown标题、代码函数）进行切分。
       - **语义感知型**:
         - `SemanticSplitterNodeParser`: 与 LangChain 的 `SemanticChunker` 类似，这种解析器使用嵌入模型来检测句子之间的语义“断点”，从而在语义最连贯的地方进行切分。
         - `SentenceWindowNodeParser`: 这是一种巧妙的方法。该方法将文档切分成单个的句子，但在每个句子节点（Node）的元数据中，会存储其前后相邻的N个句子（即“窗口”）。这使得在检索时，可以先用单个句子的嵌入进行精确匹配，然后将包含上下文“窗口”的完整文本送给LLM，极大地提升了上下文的质量。
     - **灵活的转换流水线**: 用户可以构建一个灵活的流水线，例如先用 `MarkdownNodeParser` 按章节切分文档，再对每个章节节点应用 `SentenceSplitter` 进行更细粒度的句子级切分。每个节点都携带丰富的元数据，记录着其来源和上下文关系。
     - **良好的互操作性**: LlamaIndex 提供了 `LangchainNodeParser`，可以方便地将任何 LangChain 的 `TextSplitter` 封装成 LlamaIndex 的节点解析器，无缝集成到其处理流程中。

  3.  **ChunkViz：简易的可视化分块工具**

     在本文开头部分展示的分块图就是通过 [**ChunkViz**](https://chunkviz.up.railway.app/) 生成的。可以将你的文档、分块配置作为输入，用不同的颜色块展示每个 chunk 的边界和重叠部分，方便快速理解分块逻辑。