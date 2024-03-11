from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents.base import Document
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder

# 设置api key
OPENAI_API_KEY = "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"

#------------------ 设置好相应的模型---------------

# embedding 模型
embeddings_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
# llm
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
# prompt 模板
prompt = ChatPromptTemplate.from_messages([
    ("user","{chat_history}"),
    ("user","\n上面的文字是我们的对话记录,ai是你的回复,user是我之前的提问"),
    ("user","{input}"),
])

text_prompt = ChatPromptTemplate.from_template("""根据提供的文章片段来回答下面的问题：
    <context>
    {content}
    </context>
    问题：{input}""")


# splitter 对象
text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=0)

# chain = prompt | llm 组成一个链
# chain.invoke({"input":"input a question here"})

# ------------------存储状态的变量-----------------
chat_history = [] # 存放对话历史

# ----------------定义相关方法----------------------
def document_embedding(doc_route):
    """加载并且切分嵌入文档,返回一个Chroma对象"""
    raw_documents = TextLoader(doc_route, encoding='utf-8').load()
    documents = text_splitter.split_documents(raw_documents)
    return Chroma.from_documents(documents,embeddings_model)

def ChatWithText(db, query):
    """知识辅助大模型"""
    # 问答向量化并且相似度查询,返回一个chain对象
    query_vec = embeddings_model.embed_query(query)
    relative_info = db.similarity_search_by_vector(query_vec)
    # 相关段落结合大模型进行回复
    return llm.invoke(text_prompt.format(content=relative_info,input=query))

def ChatWithLLM(query):
    """仅仅和大模型对话"""
    if len(chat_history) == 0:
        response = llm.invoke(query)
        record_char_history(query, response.content)
    else:
        history = ""
        for i in chat_history:
            history += i

        new_query = prompt.format(chat_history=history,input=query)
        response = llm.invoke(new_query)
        record_char_history(query,response.content)

    print(response.content)


def record_char_history(user_content,ai_content):
    """记录对话记录"""
    chat_info = "user:"+user_content+"\nai:"+ai_content+"\n"
    chat_history.append(chat_info)

import time
from selenium.webdriver import Edge
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By

my_option = Options()
my_option.add_argument('headless')
web = Edge(options=my_option)

def get_search(query):
    """获取搜索引擎信息"""
    base_url = 'https://www.baidu.com/s?wd='
    final_url = base_url + query
    web.get(url=final_url)

    search_div = web.find_elements(By.CLASS_NAME,'result.c-container.xpath-log.new-pmd')

    link_list = []

    for item in search_div:
        h3 = item.find_element(By.CLASS_NAME,'c-title.t.t.tts-title')
        a = h3.find_element(By.TAG_NAME,'a')
        link = a.get_attribute('href')
        link_list.append(link)

    rawstring = ""
    for link in link_list[0:3]:
        web.get(url=link)
        time.sleep(0.5)
        p = web.find_elements(By.TAG_NAME, 'p')
        for item in p:
            if item.text == '\n': continue
            item.text.lstrip()
            item.text.rstrip()
            rawstring+=item.text
            if(len(rawstring) > 12000):break;
    rawdocuments = [Document(rawstring)]
    return rawdocuments;

def information_embedding(informations):
    raw_documents = informations
    documents = text_splitter.split_documents(raw_documents)
    return Chroma.from_documents(documents, embeddings_model)


if __name__ == '__main__':
    print("Welcome")
    choice = int(input("1.大模型问答 2.知识库问答 3.搜索引擎辅助问答:"))
    if choice == 1:
        while True:
            query = input("发送消息:")
            ChatWithLLM(query)

    elif choice == 2:
        route = input("输入文档路径:")
        db = document_embedding(route)
        print("文档处理完成")
        while True:
            query = input("发送消息")
            response = ChatWithText(db, query)
            print(response.content)
            record_char_history(query, response.content)

    elif choice == 3:
        query = input("需要的信息:")
        informations = get_search(query)
        db = information_embedding(informations)
        print("检索完成")
        while True:
            query = input("发送消息")
            response = ChatWithText(db, query)
            print(response.content)



