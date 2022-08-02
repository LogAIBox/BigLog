#-*- coding : utf-8-*-
from datasets import load_dataset
import os
from biglog_data_collator import DataCollatorForLanguageModeling_for_biglog,DataCollatorForTermMask
from transformers import (
    AutoTokenizer, AutoConfig, AutoModelForMaskedLM, BertForPreTraining,
    TextDatasetForNextSentencePrediction, TrainingArguments, Trainer, )
import pickle
import sys
from random import choice
def pretraining(num_proc, train_datasets_path,eval_datasets_path, bert_tokenizer_path, bert_model_path, output_path,
                per_device_train_batch_size, learning_rate, weight_decay,num_train_epochs,warmup_ratio,save_steps,
                save_total_limit, mlm_probability,args_local_rank,term_enhance_type,definitions_prompt_path,gradient_accumulation_steps,term_refs_path):
    # 1.tokenizer
    tokenizer_checkpoint = bert_tokenizer_path
    bert_tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)   
    # 2.pre-process
    datasets = load_dataset("text", data_files={"train": train_datasets_path})#, "validation": eval_datasets_path
    def tokenize_function(examples):
        seqs = []
        for seq in examples['text']:
            seq = seq.strip('\n')
            if len(seq) > 0 and not seq.isspace():
                seqs.append(seq)
        return bert_tokenizer(seqs,padding = "longest",truncation=True,max_length=150)#
    log_datasets = datasets.map(tokenize_function, batched=True, num_proc=num_proc, remove_columns=["text"])
    data_collator = DataCollatorForLanguageModeling_for_biglog(
        tokenizer=bert_tokenizer,
        mlm=True,
        mlm_probability=mlm_probability
    )
    # 3.model loading
    model_checkpoint = bert_model_path
    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

    # 4.pre-train
    training_args = TrainingArguments(
        output_dir=output_path,
        overwrite_output_dir=True,
        per_device_train_batch_size=per_device_train_batch_size,  # batch_size of every GPU
        learning_rate=learning_rate, 
        weight_decay=weight_decay, 
        num_train_epochs=num_train_epochs,
        warmup_ratio=warmup_ratio,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        local_rank=args_local_rank,
        gradient_accumulation_steps=gradient_accumulation_steps,
        remove_unused_columns=False 
        #no_cuda=True
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=log_datasets['train'],
        #eval_dataset = log_datasets["validation"],
        data_collator=data_collator,

    )
    trainer.train()
    trainer.save_model(output_path)


if __name__=='__main__':
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--local_rank", type=int)
    # args = parser.parse_args()
    num_proc = sys.argv[1]
    train_datasets_path = sys.argv[2]
    eval_datasets_path = sys.argv[3]
    bert_tokenizer_path = sys.argv[4]
    bert_model_path = sys.argv[5]
    output_path = sys.argv[6]
    per_device_train_batch_size = sys.argv[7]
    learning_rate = sys.argv[8]
    weight_decay = sys.argv[9]
    num_train_epochs = sys.argv[10]
    warmup_ratio = sys.argv[11]
    save_steps = sys.argv[12]
    save_total_limit = sys.argv[13]
    mlm_probability = sys.argv[14]
    term_enhance_type = sys.argv[15]
    args_local_rank = int(os.environ["LOCAL_RANK"])
    definitions_prompt_path=sys.argv[16]
    gradient_accumulation_steps=sys.argv[17]
    term_refs_path=sys.argv[18]
    pretraining(int(num_proc), train_datasets_path, eval_datasets_path, bert_tokenizer_path, bert_model_path, output_path,
                int(per_device_train_batch_size), float(learning_rate), float(weight_decay), int(num_train_epochs),
                float(warmup_ratio), int(save_steps),
                int(save_total_limit), float(mlm_probability),int(args_local_rank),str(term_enhance_type),str(definitions_prompt_path),int(gradient_accumulation_steps),str(term_refs_path))
