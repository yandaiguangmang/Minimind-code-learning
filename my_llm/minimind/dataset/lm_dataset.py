from torch.utils.data import Dataset
import torch
import os
import random
from datasets import load_dataset
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def pre_processing_chat(conversations, add_system_ratio=0.2):#预处理对话
    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model."
    ]#预定义的系统提示词列表
    if conversations and conversations[0].get('role') != 'system':#如果对话列表不为空，且第一个消息的 role 不是 'system'（即没有系统提示）
        if random.random() < add_system_ratio: #以 add_system_ratio 的概率
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations#从预定义的 SYSTEM_PROMPTS 列表中随机选择一个系统提示，并将它作为 {'role': 'system', 'content': ...} 插入到对话最前面
    return conversations #返回对话列表

def post_processing_chat(prompt_content, empty_think_ratio=0.05):#后处理对话。prompt_content：经过模板渲染的 prompt 字符串。empty_think_ratio：保留空 <think> 标签的概率，默认为 0.05（即 5% 的概率保留，95% 的概率移除）。
    if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')#如果 prompt_content 中包含空标签且随机数大于 empty_think_ratio（即大概率移除），则将其替换为空字符串
    return prompt_content #返回处理后的prompt字符串

class PretrainDataset(Dataset):#预训练数据集类
    def __init__(self, data_path, tokenizer, max_length=512):#初始化函数，传入分词器、数据路径、最大长度
        super().__init__() #调用超类初始化方法
        self.tokenizer = tokenizer #分词器设置
        self.max_length = max_length #最大长度设置
        self.samples = load_dataset('json', data_files=data_path, split='train')# 使用 datasets 库加载 JSON 文件，指定数据文件路径和分割（train）

    def __len__(self): #实现继承的抽象类的方法，返回数据集样本数量
        return len(self.samples)#返回加载的数据集样本数量

    def __getitem__(self, index):#获取input ids和label
        sample = self.samples[index] #获取指定索引的样本
        tokens = self.tokenizer(str(sample['text']), add_special_tokens=False, max_length=self.max_length - 2, truncation=True).input_ids#先提取样本的原始文本text，用分词器将文本编码为token ids,设置不添加特殊token和限制最大长度（包含<BOS>、<EOS>）
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]#[BOS token]+tokens+[EOS token],添加开始和结束token id
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))#当长度不足最大长度时填充pad,用于批量处理对齐
        input_ids = torch.tensor(input_ids, dtype=torch.long)#将 input_ids 转换为64位整数类型张量
        labels = input_ids.clone()#复制input_ids为labels
        labels[input_ids == self.tokenizer.pad_token_id] = -100#进行mask那些被pad填充的位置，交叉熵损失函数规定了label值为-100的位置不参与loss计算
        return input_ids, labels #返回（input_ids,labels）


class SFTDataset(Dataset):#用于监督微调阶段的数据集类
    def __init__(self, jsonl_path, tokenizer, max_length=1024):#初始化函数，传入参数
        super().__init__() #调用父类初始化方法
        self.tokenizer = tokenizer#分词器赋值
        self.max_length = max_length#最大长度赋值
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')#使用 datasets 库加载 JSON 文件，指定数据文件路径和分割（train）
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids#预计算 bos_id：通过 tokenizer 对特定字符串进行编码，得到固定长度的 token 列表。这些字符串用于识别对话中助手的回答开始，bos_id：f'{tokenizer.bos_token}assistant\n'，表示助手开始回答的标记
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids#同理预计算eos_id,eos_id：f'{tokenizer.eos_token}\n'，表示回答结束的标记。

    def __len__(self):#返回样本数据的长度
        return len(self.samples)

    def create_chat_prompt(self, conversations):#将对话列表转换为模型所需的 prompt 字符串。conversations即对话列表，可能包含 system、user、assistant 等角色
        messages = conversations.copy() #复制对话列表
        tools = conversations[0]["functions"] if (conversations and conversations[0]["role"] == "system" and conversations[0].get("functions")) else None#如果第一个消息是 system 角色且包含 functions 字段，则将其作为工具（tools）传入模板。
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )#使用 tokenizer 的 apply_chat_template 方法返回对话列表转化为的字符串形式的prompt

    def generate_labels(self, input_ids):#根据输入的token ids生成对应的标签，仅对助手回答的部分进行预测即只计算助手回答部分的损失
        labels = [-100] * len(input_ids) #初始化一个与input_ids长度相同的全为-100的标签
        i = 0
        while i < len(input_ids): #遍历每个位置
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:#寻找bos位置
                start = i + len(self.bos_id) #记录助手回答开始的位置
                end = start #end赋值为开始的位置
                while end < len(input_ids):#end向后遍历
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:#寻找eos的位置
                        break #停止
                    end += 1 #加上eos占用的长度
                for j in range(start, min(end + len(self.eos_id), self.max_length)):#遍历 start 到 end + len(eos_id) 范围内的 token ID 复制到标签中
                    labels[j] = input_ids[j] #将范围内的 token ID 复制到标签中（end 是 eos_id 的起始位置）。这意味着标签包含助手回答的完整内容（包括结束标记）
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)#更新i,跳过已经处理的部分继续搜索
            else:
                i += 1 #i迭代，搜寻
        return labels#返回label

    def __getitem__(self, index): #获得input_ids和label
        sample = self.samples[index]#获取指定索引的样本
        conversations = pre_processing_chat(sample['conversations'])#调用 pre_processing_chat 对对话进行预处理（可能添加系统提示）。
        prompt = self.create_chat_prompt(conversations)#调用 create_chat_prompt 将对话转换为 prompt 字符串
        prompt = post_processing_chat(prompt)#调用 post_processing_chat 对 prompt 进行后处理（移除空思维链标签）
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]#使用 tokenizer 将 prompt 编码为 token IDs，并截断至 max_length。
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))#如果长度不足 max_length，用 PAD token 填充
        labels = self.generate_labels(input_ids)#调用 generate_labels 生成对应的标签。
        # # === 调试打印 ===
        # print(f"\n--- Sample {index} ---")
        # for i, (x, y) in enumerate(zip(input_ids[:-1], labels[1:])):
        #     print(f"{i:3d}: X={self.tokenizer.decode([x])!r:16s} ---> Y={self.tokenizer.decode([input_ids[i+1]])!r:16s} label={y}")
        # # ================
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)#返回 input_ids 和 labels 张量，数据格式为64为整数


class DPODataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
        self.samples = load_dataset('json', data_files=file_path, split='train')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        chosen = sample['chosen']  # 是一个 list，里面包含若干 {role, content}
        rejected = sample['rejected']  # 同上
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )
        chosen_prompt = post_processing_chat(chosen_prompt)

        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )
        rejected_prompt = post_processing_chat(rejected_prompt)
        chosen_encoding = self.tokenizer(
            chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )

        chosen_input_ids = chosen_encoding['input_ids']
        chosen_loss_mask = self.generate_loss_mask(chosen_input_ids)

        rejected_input_ids = rejected_encoding['input_ids']
        rejected_loss_mask = self.generate_loss_mask(rejected_input_ids)
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return {
            'x_chosen': x_chosen,
            'y_chosen': y_chosen,
            'mask_chosen': mask_chosen,
            'x_rejected': x_rejected,
            'y_rejected': y_rejected,
            'mask_rejected': mask_rejected
        }

    def generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask


class RLAIFDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        messages = []
        answer = ''
        for i, turn in enumerate(conversations):
            role = 'user' if i % 2 == 0 else 'assistant'
            messages.append({"role": role, "content": turn['content']})
            answer = turn['content']
        prompt = self.tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True  # 这里需要True
        )
        prompt = post_processing_chat(prompt)
        return prompt, answer

    def __getitem__(self, index):
        sample = self.samples[index]
        prompt, answer = self.create_chat_prompt(sample['conversations'])

        return {
            'prompt': prompt,
            'answer': answer
        }

if __name__ == "__main__":
    pass