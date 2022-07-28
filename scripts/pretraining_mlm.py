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
    # 1、分词器tokenizer
    tokenizer_checkpoint = bert_tokenizer_path
    bert_tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
    # with open(os.path.join('/'.join(train_datasets_path.split('/')[:-1]),'matched_definitions_pruned_reverseIndex.pkl','rb')) as f:
    #     matched_definitions_pruned_reverseIndex=pickle.load(f)
    #术语增强: term mask add ref
    def term_mask_add_ref(examples,idxs):
        with open(term_refs_path,'rb') as f:
            index2term_refs=pickle.load(f)
        refs_batch = []
        for idx in idxs:
            if idx in index2term_refs:
                refs_batch.append(index2term_refs[idx])
            else:
                refs_batch.append([-1])
        examples['term_refs']=refs_batch  
        return examples
    # 术语增强: special token prompt
    def special_token_prompt(examples):#,idx2def=matched_definitions_pruned_reverseIndex
        templates=['[DP] What is %s ? %s'%('[TERM]','[DEFINITION]')
        ,'[DP] %s represents %s'%('[TERM]','[DEFINITION]'),
        '[DP] %s means %s'%('[TERM]','[DEFINITION]'),
        '[DP] %s is defined as %s'%('[TERM]','[DEFINITION]')]
        random_pool=list(range(100))
        special_tokens2defs={'[NUM]':'number','[TIMESTAMP]':'timestamp or date','[IP]':'IP address','[CODE]':'a segment of code','[FILE]':'file path or file name'}
        seqs = []
        for seq in examples['text']:
            seq = seq.strip('\n')
            if len(seq) > 0 and not seq.isspace():
                prompted_seq=[seq]
                for special_token,defs in special_tokens2defs.items():
                    if special_token in seq and choice(random_pool)<15:#按照15%的概率添加
                        prompted_seq.append(choice(templates).replace('[TERM]',special_token).replace('[DEFINITION]',defs))
                seqs.append(' '.join(prompted_seq))
        return bert_tokenizer(seqs,padding = "longest")   
    # 术语增强: definition prompt
    def definition_prompt(examples,idxs):#,idx2def=matched_definitions_pruned_reverseIndex
        templates=['[DP] What is %s ? %s'%('[TERM]','[DEFINITION]')
        ,'[DP] %s represents %s'%('[TERM]','[DEFINITION]'),
        '[DP] %s means %s'%('[TERM]','[DEFINITION]'),
        '[DP] %s is defined as %s'%('[TERM]','[DEFINITION]')]
        with open(definitions_prompt_path,'rb') as f:
            matched_definitions_pruned_reverseIndex=pickle.load(f)
        seqs = []
        for seq,idx in zip(examples['text'],idxs):
            seq = seq.strip('\n')
            if len(seq) > 0 and not seq.isspace():
                if idx in matched_definitions_pruned_reverseIndex:
                    prompted_seq=[seq]
                    for term,definition in matched_definitions_pruned_reverseIndex[idx]:
                        prompted_seq.append(choice(templates).replace('[TERM]',term).replace('[DEFINITION]',definition))
                    seqs.append(' '.join(prompted_seq))
                    #print(seqs[-1])
                else:
                    seqs.append(seq)
        return bert_tokenizer(seqs,padding = "longest")     
    # 2、预处理
    datasets = load_dataset("text", data_files={"train": train_datasets_path})#, "validation": eval_datasets_path
    def tokenize_function(examples):
        seqs = []
        for seq in examples['text']:
            seq = seq.strip('\n')
            if len(seq) > 0 and not seq.isspace():
                seqs.append(seq)
        return bert_tokenizer(seqs,padding = "longest",truncation=True,max_length=150)#
    if term_enhance_type=='none':
        log_datasets = datasets.map(tokenize_function, batched=True, num_proc=num_proc, remove_columns=["text"])
    elif term_enhance_type=='def_prompt':
        log_datasets = datasets.map(definition_prompt, batched=True, num_proc=num_proc, remove_columns=["text"],with_indices=True,batch_size=10000)
    elif term_enhance_type=='term_mask':
        log_datasets = datasets.map(tokenize_function, batched=True, num_proc=num_proc, remove_columns=["text"],keep_in_memory=True)\
            .map(term_mask_add_ref,batched=True, num_proc=num_proc,with_indices=True,batch_size=10000)
    elif term_enhance_type=='special_token_prompt':
        log_datasets = datasets.map(special_token_prompt, batched=True, num_proc=num_proc, remove_columns=["text"],batch_size=10000)
    if term_enhance_type=='term_mask':
        data_collator = DataCollatorForTermMask(
            tokenizer=bert_tokenizer,
            mlm=True,
            mlm_probability=mlm_probability    
        )    
    else:
        data_collator = DataCollatorForLanguageModeling_for_biglog(
            tokenizer=bert_tokenizer,
            mlm=True,
            mlm_probability=mlm_probability
        )
    # 3、模型加载
    model_checkpoint = bert_model_path
    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

    # 4、预训练
    training_args = TrainingArguments(
        output_dir=output_path,
        overwrite_output_dir=True,
        #evaluation_strategy="epoch",  # 每次epoch以后评估
        per_device_train_batch_size=per_device_train_batch_size,  # 每个GPU的batch_size
        learning_rate=learning_rate,  # 学习率
        weight_decay=weight_decay,  # 权重衰减
        num_train_epochs=num_train_epochs,  # 迭代次数
        warmup_ratio=warmup_ratio,  # 预热学习率比例
        save_steps=save_steps,  # 保存间隔数
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
        data_collator=data_collator,    # 数据预处理（mask）

    )
    trainer.train()
    trainer.save_model(output_path)

    # 5、评估
    # eval_results = trainer.evaluate()
    # print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")


if __name__=='__main__':
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--local_rank", type=int)
    # args = parser.parse_args()
    num_proc = sys.argv[1]
    # 运行参数
    train_datasets_path = sys.argv[2]
    eval_datasets_path = sys.argv[3]    # 数据路径
    bert_tokenizer_path = sys.argv[4]     # tokenizer路径
    bert_model_path = sys.argv[5]    # 模型架构路径
    output_path = sys.argv[6]    # 输出模型路径
    # 超参数
    per_device_train_batch_size = sys.argv[7]    # batch size
    learning_rate = sys.argv[8]    # 学习率
    weight_decay = sys.argv[9]     # 权重衰减
    num_train_epochs = sys.argv[10]     # epoch
    warmup_ratio = sys.argv[11]    # 预热学习比例
    save_steps = sys.argv[12]     # 保存间隔数步数
    save_total_limit = sys.argv[13]
    mlm_probability = sys.argv[14]     # 掩膜比例
    term_enhance_type = sys.argv[15]
    args_local_rank = int(os.environ["LOCAL_RANK"])#args.local_rank
    definitions_prompt_path=sys.argv[16]
    gradient_accumulation_steps=sys.argv[17]
    term_refs_path=sys.argv[18]
    pretraining(int(num_proc), train_datasets_path, eval_datasets_path, bert_tokenizer_path, bert_model_path, output_path,
                int(per_device_train_batch_size), float(learning_rate), float(weight_decay), int(num_train_epochs),
                float(warmup_ratio), int(save_steps),
                int(save_total_limit), float(mlm_probability),int(args_local_rank),str(term_enhance_type),str(definitions_prompt_path),int(gradient_accumulation_steps),str(term_refs_path))


    #测试
    #pretraining(4,'/home/l00650981/logModel/Pretrain/dataset/new_loghub_MLM_eval.log','/home/l00650981/logModel/Pretrain/dataset/new_loghub_MLM_eval.log','/home/l00650981/bert-base-uncased-biglog-tokenizer/','/home/l00650981/bert-base-uncased-biglog-initial/','/home/l00650981/logModel/',1,0.0001,0.01,16,0.05,10000,100,0.15,-1)
