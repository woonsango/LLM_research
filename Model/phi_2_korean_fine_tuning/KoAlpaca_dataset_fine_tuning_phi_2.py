from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    GenerationConfig
)

from tqdm import tqdm
from trl import SFTTrainer
import torch
import time
import pandas as pd
import numpy as np
from huggingface_hub import interpreter_login
from datasets import DatasetDict


from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from functools import partial


def dataset_load(huggingface_dataset_name):
    '''
        dataset을 load하는 함수
    '''
    #data set 다운
    dataset = load_dataset(huggingface_dataset_name)

    # 기존 DatasetDict 로드 또는 정의
    dataset_dict = dataset
    
    # 'train' 데이터셋을 'train'과 'temp'로 분할 (예: 70% 'train', 30% 'temp')
    train_testsplit = dataset_dict['train'].train_test_split(test_size=0.3)
    
    # 'temp'를 'validation'과 'test'로 분할 (예: 각각 'temp'의 50%, 총 데이터셋의 15%)
    val_testsplit = train_testsplit['test'].train_test_split(test_size=0.5)
    
    # 새로운 DatasetDict 생성
    dataset = DatasetDict({
        'train': train_testsplit['train'],  # 70% 데이터
        'validation': val_testsplit['train'],  # 15% 데이터
        'test': val_testsplit['test']  # 나머지 15% 데이터
    })

    return dataset

def tokenizer_fun(tokenizer_name):
    '''
        토크나이저를 설정하는 함수
    '''
    tokenizer_ko = AutoTokenizer.from_pretrained("EleutherAI/polyglot-ko-5.8b",trust_remote_code=True,padding_side="right", padding=True,add_eos_token=True,add_bos_token=True,use_fast=False)
    #tokenizer.pad_token = tokenizer.eos_token: 이 부분은 패딩 토큰을 EOS (End-Of-Sequence) 토큰으로 설정합니다. 이렇게 하면 모델이 패딩을 식별하는 데 사용되는 토큰을 EOS 토큰으로 사용하게 됩니다.
    tokenizer_ko.pad_token = tokenizer_ko.eos_token

    return tokenizer_ko

def gen_ko(model_name, prompt, max_length):
    # 입력 텍스트를 토크나이즈하고 모델에 입력할 형식으로 변환
    input_ids = tokenizer_ko.encode(prompt, max_length=max_length, padding=True, return_tensors="pt", truncation=True, return_attention_mask=True)
    
    # 모델에 입력하여 텍스트 생성
    outputs = model_name.generate(input_ids.to('cuda'), max_length=max_length,pad_token_id=tokenizer_ko.eos_token_id)

    # print(len(outputs))
    # print(outputs)
    
    
    # 생성된 텍스트 디코딩
    generated_text = tokenizer_ko.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text

def create_prompt_formats(sample):
    """
    Format various fields of the sample ('instruction','output')
    Then concatenate them using two newline characters 
    :param sample: Sample dictionnary
    """
    INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    INSTRUCTION_KEY = "### Instruct: Answer the questions below"
    RESPONSE_KEY = "### Output:"
    END_KEY = "### End"
    
    blurb = f"\n{INTRO_BLURB}"
    instruction = f"{INSTRUCTION_KEY}"
    input_context = f"{sample['instruction']}" if sample["instruction"] else None
    response = f"{RESPONSE_KEY}\n{sample['output']}"
    end = f"{END_KEY}"
    
    parts = [part for part in [blurb, instruction, input_context, response, end] if part]

    formatted_prompt = "\n\n".join(parts)
    sample["text"] = formatted_prompt

    return sample

## prompt를 모델 토크나이저를 사용하여 토큰화된 프롬프트로 처리
## 여기서의 목표는 일관된 길이의 입력 시퀀스를 생성하는 것
## 일관된 길이의 입력 시퀀스는 효율성을 최적화하고 계산 오버헤드를 최소화하여 언어 모델을 미세 조정하는 데 도움이 된다.
## 이러한 시퀀스가 모델의 최대 토큰 제한을 초과하지 않도록 확인하는 것이 중요하다

# SOURCE https://github.com/databrickslabs/dolly/blob/master/training/trainer.py
# 모델의 구성을 통해 최대 길이 설정을 가져오는 함수
# 모델의 구성 중에서 "n_positions", "max_position_embeddings", "seq_length"와 같은 설정을 확인하여 최대 길이를 찾습니다.
# 만약 최대 길이 설정이 없는 경우 기본값으로 1024를 사용합니다.
def get_max_length(model):
    conf = model.config
    max_length = None
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            print(f"Found max lenth: {max_length}")
            break
    if not max_length:
        max_length = 1024
        print(f"Using default max length: {max_length}")
    return max_length

# 주어진 배치를 토크나이징하여 전처리하는 함수입니다.
# 각 텍스트를 토크나이즈하고 최대 길이에 맞게 잘라내는 작업을 수행합니다.
def preprocess_batch(batch, tokenizer, max_length):
    """
    Tokenizing a batch
    """
    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
    )

# SOURCE https://github.com/databrickslabs/dolly/blob/master/training/trainer.py
#데이터셋을 전처리하여 모델 학습에 사용할 수 있는 형식으로 준비하는 함수입니다.
    #먼저 각 샘플에 프롬프트를 추가합니다.
    #다음으로 preprocess_batch 함수를 적용하여 각 배치를 전처리합니다. 이때 remove_columns 매개변수를 통해 불필요한 열을 제거합니다.
    #입력 시퀀스의 길이가 최대 길이를 초과하는 샘플을 제거합니다.
    #마지막으로 데이터셋을 섞습니다.
def preprocess_dataset(tokenizer: AutoTokenizer, max_length: int,seed, dataset):
    """Format & tokenize it so it is ready for training
    :param tokenizer (AutoTokenizer): Model Tokenizer
    :param max_length (int): Maximum number of tokens to emit from tokenizer
    """

    print(type(dataset))
    
    # Add prompt to each sample
    print("Preprocessing dataset...")
    dataset = dataset.map(create_prompt_formats)#, batched=True)
    
    # Apply preprocessing to each batch of the dataset & and remove 'instruction', 'context', 'response', 'category' fields
    # partial: 함수를 편하게 만듦
    # 여기서 데이터셋을 토큰화함
    _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
    # 이러면 dataset에 text만 남음
    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
        remove_columns=['url'],
    )

    # Filter out samples that have input_ids exceeding max_length
    # 입력 시퀀스의 길이가 max_length보다 큰 샘플을 필터링하여 제거
    dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < max_length)
    
    # Shuffle dataset
    dataset = dataset.shuffle(seed=seed)

    return dataset

def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"\ntrainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"



if __name__ == "__main__" :
    interpreter_login()

    dataset = dataset_load('beomi/KoAlpaca-v1.1a')

    tokenizer_ko = tokenizer_fun('EleutherAI/polyglot-ko-5.8b')

    compute_dtype = getattr(torch, "float16")
    #모델을 4bit 형식으로 로드(메모리 소비가 줄어듦)
    bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=False,
        )

    model_name='microsoft/phi-2'
    device_map = {"": 0}
    #모델을 양자화하여 다운(or load)
    original_model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                          device_map=device_map,
                                                          quantization_config=bnb_config,
                                                          trust_remote_code=True,
                                                          use_auth_token=True)
    
    ## Pre-process dataset
    max_length = get_max_length(original_model)
    print(max_length)
    
    print(dataset['train'][0])

    from transformers import set_seed
    seed = 42
    set_seed(seed)
    
    train_dataset = preprocess_dataset(tokenizer_ko, max_length,seed, dataset)

    original_model = prepare_model_for_kbit_training(original_model)

    config = LoraConfig(
        r=32, #Rank
        lora_alpha=32,
        target_modules=[
            'q_proj',
            'k_proj',
            'v_proj',
            'dense'
        ],
        bias="none",
        lora_dropout=0.05,  # Conventional
        task_type="CAUSAL_LM",
    )
    
    # 1 - Enabling gradient checkpointing to reduce memory usage during fine-tuning
    original_model.gradient_checkpointing_enable()
    
    peft_model = get_peft_model(original_model, config)

    output_dir = f'./model/peft-ko-training-{str(int(time.time()))}'
    import transformers
    
    peft_training_args = TrainingArguments(
        output_dir = output_dir,
        warmup_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=1000,
        learning_rate=2e-4,
        optim="paged_adamw_8bit",
        logging_steps=5,
        logging_dir="./logs",
        save_strategy="steps",
        save_steps=25,
        evaluation_strategy="steps",
        eval_steps=25,
        do_eval=True,
        gradient_checkpointing=True,
        report_to="none",
        overwrite_output_dir = True,
        group_by_length=True,
    )
    
    peft_model.config.use_cache = False
    
    peft_trainer = transformers.Trainer(
        model=peft_model,
        train_dataset=train_dataset['train'],
        eval_dataset=train_dataset['validation'],
        args=peft_training_args,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer_ko, mlm=False),
    )

    peft_trainer.train()

    