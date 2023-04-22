from django.shortcuts import render
from django.http import HttpResponse
from transformers import GPT2LMHeadModel
from transformers import PreTrainedTokenizerFast
import torch
import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import DataLoader, Dataset
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
import re
import urllib.request

def index(request) :
    return render(request,
                  "GPTapp/index.html",
                  {})
def KGPT_input(request) :
    return render(request,
                  "GPTapp/KGPT_2.html",
                  {})

def generate_text(request):
  
    text = request.POST.get('prompt')
    
    tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2", bos_token='</s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>') 
    tokenizer.tokenize("안녕하세요. 한국어 GPT-2 입니다.😤:)l^o")
    model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
    input_ids = tokenizer.encode(text)
    # return HttpResponse(input_ids)   
    gen_ids = model.generate(torch.tensor([input_ids]),
                                max_length=128,
                                repetition_penalty=2.0,
                                pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=tokenizer.eos_token_id,
                                bos_token_id=tokenizer.bos_token_id,
                                use_cache=True)
    generated = tokenizer.decode(gen_ids[0,:].tolist())
    return render(request, 'GPTapp/KGPT_2.html', {'generated_text': generated ,"prompt" : text})

def chatBot(request):
    Q_TKN = "<usr>"   # 사용자(질문자)를 나타내는 스페셜 토큰입니다.
    A_TKN = "<sys>"   # 시스템(응답자)을 나타내는 스페셜 토큰입니다.
    BOS = '</s>'      # 문장의 시작을 나타내는 스페셜 토큰입니다.
    EOS = '</s>'      # 문장의 끝을 나타내는 스페셜 토큰입니다.
    MASK = '<unused0>'# 마스킹 처리를 위해 사용되는 스페셜 토큰입니다.
    SENT = '<unused1>'# 다중 문장 분류 등에 사용되는 스페셜 토큰입니다.
    PAD = '<pad>'     # 패딩 처리를 위해 사용되는 스페셜 토큰입니다.
    koGPT2_TOKENIZER = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
            bos_token=BOS, eos_token=EOS, unk_token='<unk>',
            pad_token=PAD, mask_token=MASK) 
    model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
    # 쳇봇 데이터 다운로드 
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData.csv",
        filename="ChatBotData.csv",
    )
    Chatbot_Data = pd.read_csv("ChatBotData.csv")
    # Test 용으로 300개 데이터만 처리한다.
    Chatbot_Data = Chatbot_Data[:300]
    Chatbot_Data.head()