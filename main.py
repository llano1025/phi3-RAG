from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline, TextStreamer
)
import transformers
import torch
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
import langchain
import gc
import time
import textwrap
import glob
import os
import warnings
warnings.filterwarnings("ignore")

IS_LOAD_EMBED = True


class CFG:
    DEBUG = False

    # LLM
    model_name = 'microsoft/Phi-3-mini-128k-instruct'
    temperature = 0.4
    top_p = 0.90
    repetition_penalty = 1.15
    max_len = 8192
    max_new_tokens = 512

    # splitting
    split_chunk_size = 800
    split_overlap = 400

    # embeddings
    embeddings_model_repo = 'BAAI/bge-base-en-v1.5'

    # similar passages
    k = 6

    # paths
    PDFs_path = '/home/llanopi/dev/RAG/data/100-llm-papers-to-explore/'
    Embeddings_path = '/home/llanopi/dev/RAG/data/faiss-ml-papers-st'
    Output_folder = '/home/llanopi/dev/RAG/vectordb'


loader = DirectoryLoader(
    CFG.PDFs_path,
    glob="./*3215v3.pdf" if CFG.DEBUG else "./*.pdf",
    loader_cls=PyPDFLoader,
    show_progress=True,
    use_multithreading=True
)

documents = loader.load()
print(f'We have {len(documents)} pages in total')

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CFG.split_chunk_size,
    chunk_overlap=CFG.split_overlap
)

texts = text_splitter.split_documents(documents)

print(f'We have created {len(texts)} chunks from {len(documents)} pages')

if IS_LOAD_EMBED == False:
    # we create the embeddings if they do not already exist in the input folder
    if not os.path.exists(CFG.Embeddings_path + '/index.faiss'):

        print('Creating embeddings...\n\n')

        # download embeddings model
        embeddings = HuggingFaceInstructEmbeddings(
            model_name=CFG.embeddings_model_repo,
            model_kwargs={"device": "cuda"}
        )

        # create embeddings and DB
        vectordb = FAISS.from_documents(
            documents=texts,
            embedding=embeddings
        )

        # persist vector database
        # save in output folder
        vectordb.save_local(f"{CFG.Output_folder}/faiss_index_ml_papers")

else:
    # download embeddings model
    embeddings = HuggingFaceInstructEmbeddings(
        model_name=CFG.embeddings_model_repo,
        model_kwargs={"device": "cuda"}
    )

    # load vector DB embeddings
    vectordb = FAISS.load_local(
        CFG.Output_folder + '/faiss_index_ml_papers',  # from output folder
        embeddings,
        allow_dangerous_deserialization=True,
    )

# test if vector DB was loaded correctly
vectordb.similarity_search('scaling laws')


def build_model(model_repo=CFG.model_name):

    print('\nDownloading model: ', model_repo, '\n\n')

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_repo)

    # quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # model
    model = AutoModelForCausalLM.from_pretrained(
        model_repo,
        quantization_config=bnb_config,
        device_map='auto',
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    return tokenizer, model


tokenizer, model = build_model(model_repo=CFG.model_name)
streamer = TextStreamer(tokenizer)
gc.collect()
model.eval()
model.hf_device_map

terminators = [
    tokenizer.eos_token_id,
    tokenizer.bos_token_id
]


# hugging face pipeline
pipe = pipeline(
    task="text-generation",

    model=model,

    tokenizer=tokenizer,
    #     pad_token_id = tokenizer.eos_token_id,
    eos_token_id=terminators,

    do_sample=True,
    #     max_length = CFG.max_len,
    max_new_tokens=CFG.max_new_tokens,

    # Define your callbacks for handling streaming output
    streamer=streamer,

    temperature=CFG.temperature,
    top_p=CFG.top_p,
    repetition_penalty=CFG.repetition_penalty,
)

# langchain pipeline
llm = HuggingFacePipeline(pipeline=pipe)

prompt_template = """
<|system|>

You are an expert assistant that answers questions about machine learning and Large Language Models (LLMs).

You are given some extracted parts from machine learning papers along with a question.

If you don't know the answer, just say "I don't know." Don't try to make up an answer.

It is very important that you ALWAYS answer the question in the same language the question is in. Remember to always do that.

Use only the following pieces of context to answer the question at the end.

<|end|>

<|user|>

Context: {context}

Question is below. Remember to answer in the same language:

Question: {question}

<|end|>

<|assistant|>

"""


PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

retriever = vectordb.as_retriever(
    search_type="similarity",
    search_kwargs={"k": CFG.k}
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # map_reduce, map_rerank, stuff, refine
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True,
    verbose=False
)


def wrap_text_preserve_newlines(text, width=1500):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text


def process_llm_response(llm_response):
    ans = wrap_text_preserve_newlines(llm_response['result'])

    sources_used = ' \n'.join(
        [
            source.metadata['source'].split('/')[-1][:-4]
            + ' - page: '
            + str(source.metadata['page'])
            for source in llm_response['source_documents']
        ]
    )

    ans = ans + '\n\nSources: \n' + sources_used

    # return only the text after the pattern
    pattern = "<|assistant|>"
    index = ans.find(pattern)
    if index != -1:
        ans = ans[index + len(pattern):]

    return ans.strip()


def llm_ans(query):
    start = time.time()

    llm_response = qa_chain.invoke(query)
    ans = process_llm_response(llm_response)

    end = time.time()

    time_elapsed = int(round(end - start, 0))
    time_elapsed_str = f'\n\nTime elapsed: {time_elapsed} s'
    return ans + time_elapsed_str


while True:
    query = input("Please input query: ")
    if query == "exit":
        break
    else:
        result = llm_ans(query)
        print(result)
