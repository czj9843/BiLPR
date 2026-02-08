import asyncio
import random
import torch
from torch import nn
from config import DATA_PATH, RESULT_PATH
from stu_profile import Profile
from memory import Memory
from action import AgentAction
from utils import load_json, save_json
import numpy as np
import re
import os
import json
from agent_teacher2 import Teacher
from llm import LlamaLLM
from Train_KT_Agent import E_DKT
from transformers import BertModel, BertTokenizer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
class KTModel_AKT:
    def __init__(self, bert_path="/data2/Chenzejun/LLM/bert-master"):
        self.bert = BertModel.from_pretrained(bert_path).to(device)
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        
        # 加载数据映射文件
        with open("/data2/Chenzejun/RS_Agent_Code/Agent4Edu-main3/data/assist2009/keyid2idx.json", 'r') as f:
            self.keyid2idx = json.load(f)
        
        # 加载习题和知识点信息
        with open("/data2/Chenzejun/RS_Agent_Code/Agent4Edu-main3/data/assist2009/k2txt.json", 'r') as f:
            self.k2text = json.load(f)
        
        with open("/data2/Chenzejun/RS_Agent_Code/Agent4Edu-main3/data/assist2009/k2q.json", 'r') as f:
            self.k2q = json.load(f)
        
        with open("/data2/Chenzejun/RS_Agent_Code/Agent4Edu-main3/data/assist2009/q2k.json", 'r') as f:
            self.q2k = json.load(f)
        
        with open("/data2/Chenzejun/RS_Agent_Code/Agent4Edu-main3/data/assist2009/q2txt.json", 'r') as f:
            self.q2text = json.load(f)
        
        # 获取问题数量（从keyid2idx中）
        self.n_question = len(self.keyid2idx["questions"])
        print(f"问题数量: {self.n_question}")
        
        # 获取知识点数量
        with open("/data2/Chenzejun/RS_Agent_Code/Agent4Edu-main3/data/assist2009/k2txt.json", 'r') as f:
            k2txt_data = json.load(f)
            self.n_knowledge = len(k2txt_data)
        print(f"知识点数量: {self.n_knowledge}")
        
        # 加载AKT模型
        self.model = self.load_akt_model()
        self.model.eval()
        
        # 加载难度数据
        self.q_diff = torch.load("/data2/Chenzejun/RS_Agent_Code/Agent4Edu-main3/data/assist2009/assist2009_exer_diff.pt").to(device)
        self.kc_diff = torch.load("/data2/Chenzejun/RS_Agent_Code/Agent4Edu-main3/data/assist2009/assist2009_kc_diff.pt").to(device)
        self.out = (nn.Sequential(
            nn.Linear(256,1), nn.Sigmoid()
        ))
        print("AKT模型加载完成!")
    
    def load_akt_model(self):
        """加载预训练的AKT模型"""
        # 定义模型参数（需要与训练时的参数一致）
        model_params = {
            'n_question': self.n_knowledge,  # 问题数量
            'n_pid': self.n_question,        # pid数量，通常与问题数量相同
            'd_model': 256,                  # 模型维度
            'n_blocks': 1,                   # 块的数量
            'kq_same': 1,                    # 是否相同
            'dropout': 0.3,                  # dropout率
            'model_type': 'akt',             # 模型类型
            'final_fc_dim': 512,             # 最终全连接层维度
            'n_heads': 8,                    # 多头注意力头数
            'd_ff': 2048,                    # 前馈网络维度
            'l2': 1e-5,                      # L2正则化系数
            'separate_qa': False             # 是否分开qa
        }
        
        # 创建模型实例（需要先导入AKT类）
        # 注意：这里需要确保AKT类的定义在当前作用域中可用
        # 如果AKT类在另一个文件中，需要导入
        try:
            # 尝试从akt模块导入
            from akt import AKT
            model = AKT(**model_params).to(device)
        except ImportError:
            # 如果AKT类在当前文件中定义，直接使用
            # 这里需要确保AKT类的定义在KTModel_AKT类之前
            model = AKT(**model_params).to(device)
        
        # 加载模型权重
        model_path = "/data2/Chenzejun/model/akt_pid/assist2009_pid/_b24_nb1_gn-1_lr0.0002_s224_sl40_do0.3_dm256_ts1_kq1_l21e-05_8"
        
        # 检查模型文件是否存在

        if not os.path.exists(model_path):
            print(f"警告: 模型文件 {model_path} 不存在!")
            print(f"尝试查找模型文件...")
            # 尝试查找可能的模型文件
            import glob
            model_files = glob.glob("/data2/Chenzejun/model/akt_pid/assist2009_pid/*")
            if model_files:
                print(f"找到以下模型文件:")
                for f in model_files:
                    print(f"  {f}")
                # 使用第一个找到的模型文件
                model_path = model_files[0]
                print(f"使用模型文件: {model_path}")
            else:
                raise FileNotFoundError(f"找不到模型文件: {model_path}")
        
        # 加载模型权重
        try:
            # 尝试加载整个模型
            checkpoint = torch.load(model_path, map_location=device)
            
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    # 检查点包含模型状态字典
                    model.load_state_dict(checkpoint['model_state_dict'])
                elif 'model' in checkpoint:
                    # 检查点包含模型
                    model.load_state_dict(checkpoint['model'])
                else:
                    # 检查点就是模型状态字典
                    model.load_state_dict(checkpoint)
            else:
                # 检查点就是模型
                model = checkpoint.to(device)
            
            print(f"成功加载模型权重: {model_path}")
        except Exception as e:
            print(f"加载模型权重时出错: {e}")
            print("使用随机初始化的模型")
        
        return model
    
    def prepare_input_for_akt(self, exer_ids, kc_log, scores):#(exer_ids, kc_log, scores)
        """为AKT模型准备输入数据"""
        # 将习题ID转换为模型使用的ID
        # 注意：exer_ids可能是新ID（整数），需要确保它们与模型中的ID对应
             
        # 将exer_ids转换为模型输入格式
        q_data = []
        qa_data = []
        pid_data = []

        if isinstance(kc_log[0], str):
            q_data = [int(kc) for kc in kc_log]
        else:          
            q_data = kc_log

        pid_data = exer_ids  

        for i, kc_id in enumerate(q_data):
        # 构建qa_data：如果答对，则qa = kc + n_question；否则qa = kc
            if i < len(scores) and scores[i] == 1:
                qa_data.append(kc_id + self.n_knowledge)
            else:
                qa_data.append(kc_id)
        # target_length = 4
        # 转换为张量
        print('q_data',q_data)
        print('qa_data',qa_data)
        print('pid_data',pid_data)
           
        q_data = torch.tensor([q_data], dtype=torch.long).to(device)
        qa_data = torch.tensor([qa_data], dtype=torch.long).to(device)
        pid_data = torch.tensor([pid_data], dtype=torch.long).to(device)
        target = torch.tensor([scores], dtype=torch.float).to(device)
        
        return q_data, qa_data, pid_data, target
    
    def get_knowledge_state_by_KT(self, exer_ids, kc_log, scores, goalkc_id):#ques_log, kc_log, ans_log
        """使用AKT模型获取知识点状态"""
        
        # 准备输入数据
        q_data, qa_data, pid_data, target = self.prepare_input_for_akt(exer_ids, kc_log, scores)
        
      
        # 运行模型
        with torch.no_grad():
            # AKT模型的forward需要q_data, qa_data, target, pid_data
            # preds=d_output
            d_output = self.model(q_data, qa_data, target, pid_data)
        # print('d_output.shape',d_output.shape)
        # print('d_output',d_output)

        # d_output = d_output.mean(dim=-1, keepdim=True)  # keepdim=True 保持三维

        # # 步骤2：取最后一个时间步，形状从 [1, 21, 1] → [1, 1]
        learning_state = d_output[:, -1, :].item()  # 或 d_output_mean[0, -1, 0]

        # d_output = d_output[0, -1, :]
        # d_output = self.out(d_output)
        # print('d_output.shape',d_output.shape)
        # print('d_output',d_output)

        learning_state = torch.mean(d_output).item()  # .item() 将张量转换为 Python 标量
        # 取最后一个值作为最终认知状态
        #learning_state = d_output[-1].item()  # 最后一个值，不是平均值
        #print(f"平均知识状态: {learning_state}")
        return learning_state

def run_for_student(student_id, all_stu_lr_state4everygoal, E_all_e_small, E_all_s_small, E_all_sup_small, E_all_e_big, E_all_s_big, E_all_sup_big):
    all_stu_lr_state4everygoal[student_id] = []
    print(f'这是ID为{student_id}学习者的学习过程')
    MAX_STEP=20  ## 模拟学生思考步数
    COLD_NUM = 10  # 读取学习者历史数据来算初始认知状态
    llm = LlamaLLM()
    kt_Model= KTModel_AKT()

    all_logs = load_json(f"{DATA_PATH}/stu_logs.json")
    logs = None
    for student in all_logs:
        if student['user_id'] == student_id:
            logs = student['logs']
            break

    KCG = load_json(f"{DATA_PATH}/kcg.json")

    # 读取exercises.json文件
    with open('/data2/Chenzejun/RS_Agent_Code/Agent4Edu-main3/data/assist2009/exercises.json', 'r', encoding='utf-8') as f:
        exercises = json.load(f)
    know_course_list = load_json(f"{DATA_PATH}/know_course_list.json")
    know_name = load_json(f"{DATA_PATH}/know_name_list.json")
    profile = Profile(student_id)
    memory = Memory(KCG, know_course_list, know_name)
    action = AgentAction(profile, memory,llm)
    Teacher_RS = Teacher(llm)
    results = []
    
    logs_all = logs[:COLD_NUM]
    kc_log = [log['knowledge_code'] for log in logs_all]
    kc_old = set(kc_log)
    ques_log = [log['exer_id'] for log in logs_all]
    ans_log = [log['score'] for log in logs_all]
    print('kc_old', kc_old)

    for goalkc_id in kc_old:     

        
        print('goalkc_id', goalkc_id)
        learning_state_list = []
        print('ques_log', ques_log)
        # 获得学习者的初始认知状态
        learning_state= kt_Model.get_knowledge_state_by_KT(ques_log, kc_log, ans_log, goalkc_id)
        learning_state_list.append(learning_state)
        print('#################初始learning_state', learning_state)
        for step in range(MAX_STEP):

            advise= Teacher_RS.given_advise(step, logs_all, ques_log, profile, goalkc_id, learning_state)
            
            # 得到针对这道习题的意见和预测答案
            exer_id = advise.get('exer_id')
            #print('exer_id', exer_id)

            recommand_reason = advise.get('recommend_reason', '') or advise.get('recommand_reason', '')
            #print('recommand_reason', recommand_reason)

            teacher_predict_answer = advise.get('predict_answer', 0)

            #print('teacher_predict_answer', teacher_predict_answer)

            rec = exercises[exer_id]
            #print('rec:', rec)


            ## 利用学生自己给出的答案
            ans, raw, corr, summ = action.simulate_step(rec, step, teacher_predict_answer, similarity_fn=memory.reinforce)
            results.append({'ans':ans,'raw':raw,'corr':corr,'summ':summ})
            # print('ans:', ans)
            # print('raw:', raw)
            # print('corr:', corr)
            # print('summ:', summ)

            student_score = 0 if ans['task4'] == 'No' else 1
            print('student_score:', student_score)
            # 记录结果
            ques_log.append(exer_id)
            ans_log.append(student_score)
            kc_log.append(goalkc_id)

            # 教师要反思
            Teacher_RS.reflect_on_outcome(
                    question_id=exer_id,
                    predict_answer = teacher_predict_answer,
                    actual_outcome=student_score,
                    student_feedback=summ
                )
            
            
            learning_state= kt_Model.get_knowledge_state_by_KT(ques_log, kc_log, ans_log, goalkc_id)
            learning_state_list.append(learning_state)
            print(f'##################推荐{step + 1}次的learning_state:', learning_state)
            print(f'##################推荐learning_state_list[0]次的learning_state:', learning_state_list[0])
        if learning_state_list[0] < 0.5 and max(learning_state_list) < 0.7:
            E_p = abs( max(learning_state_list) -learning_state_list[0]) / (0.7-learning_state_list[0])
            if E_p >= 0.1:
                print(f'goal_id = {goalkc_id}, stu = {student_id}, E_p = {E_p}')
                all_stu_lr_state4everygoal[student_id].append(E_p)
                print('all_stu_lr_state4everygoal',all_stu_lr_state4everygoal)
                E_all_s_small += learning_state_list[0]
                E_all_e_small += max(learning_state_list)
                E_all_sup_small += 0.7
        else:
            E_p = abs( max(learning_state_list) -learning_state_list[0]) / (1-learning_state_list[0])
            if E_p >= 0.1:
                print(f'goal_id = {goalkc_id}, stu = {student_id}, E_p = {E_p}')
                all_stu_lr_state4everygoal[student_id].append(E_p)
                print('all_stu_lr_state4everygoal',all_stu_lr_state4everygoal)
                E_all_s_big += learning_state_list[0]
                E_all_e_big += max(learning_state_list)
                E_all_sup_big += 1

        
        
        
        print(f'##################推荐{step + 1}次的E_p:', E_p)
        




    return E_all_e_small, E_all_s_small, E_all_sup_small, E_all_e_big, E_all_s_big, E_all_sup_big

def main():
    # 加载学生ID列表
    try:
        agent_id_list = load_json(f"{DATA_PATH}/agent_id_list.json")
        print(f"找到 {len(agent_id_list)} 个学生")
    except Exception as e:
        print(f"无法加载学生列表: {e}")
        print("使用前5个学生作为测试...")
        # 如果没有列表，使用前5个学生
        all_logs = load_json(f"{DATA_PATH}/stu_logs.json")
        agent_id_list = list(range(min(5, len(all_logs))))
    
    # 选择要处理的学生数量（可以先测试少量）
    num_students_to_process = 5
    
    # 确保不超出学生总数
    num_students_to_process = min(num_students_to_process, len(agent_id_list))
    
    # 随机选择学生
    selected_students = random.sample(agent_id_list, num_students_to_process)
    E_all_e_small = 0
    E_all_s_small = 0
    E_all_sup_small = 0
    E_all_e_big = 0
    E_all_s_big = 0
    E_all_sup_big = 0
    print(f"随机选择的学生索引: {selected_students}")
    all_stu_lr_state4everygoal = {}
    for s in selected_students:
        E_all_e_small, E_all_s_small, E_all_sup_small, E_all_e_big, E_all_s_big, E_all_sup_big = run_for_student(s, all_stu_lr_state4everygoal, E_all_e_small, E_all_s_small, E_all_sup_small, E_all_e_big, E_all_s_big, E_all_sup_big)
        print('################E_all_sup###############', E_all_sup_small)
    E_all_p =  ((E_all_e_small+E_all_e_big) - (E_all_s_small+E_all_s_big)) / ((E_all_sup_small+ E_all_sup_big) - (E_all_s_small+E_all_s_big))
    print(f'E_all_p = {E_all_p}')

if __name__ == '__main__':
    main()
