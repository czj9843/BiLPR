import random
import torch
from torch import nn
from config import DATA_PATH
from stu_profile import Profile
from stu_memory import Memory
from stu_action import AgentAction
from utils import load_json
import os
import json
from agent_teacher import Teacher
from llm import LlamaLLM
from transformers import BertModel, BertTokenizer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class KTModel_AKT:
    def __init__(self, bert_path=".../bert-master"):
        self.bert = BertModel.from_pretrained(bert_path).to(device)
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)

        with open(".../keyid2idx.json", 'r') as f:
            self.keyid2idx = json.load(f)

        with open(".../k2txt.json", 'r') as f:
            self.k2text = json.load(f)

        with open(".../k2q.json", 'r') as f:
            self.k2q = json.load(f)

        with open(".../q2k.json", 'r') as f:
            self.q2k = json.load(f)

        with open(".../q2txt.json", 'r') as f:
            self.q2text = json.load(f)

        self.n_question = len(self.keyid2idx["questions"])
        print(f"q_num: {self.n_question}")

        with open(".../k2txt.json", 'r') as f:
            k2txt_data = json.load(f)
            self.n_knowledge = len(k2txt_data)
        print(f"kc_num: {self.n_knowledge}")

        self.model = self.load_akt_model()
        self.model.eval()

        self.q_diff = torch.load(
            ".../assist2009_exer_diff.pt").to(device)
        self.kc_diff = torch.load(
            ".../assist2009_kc_diff.pt").to(device)
        self.out = (nn.Sequential(
            nn.Linear(256, 1), nn.Sigmoid()
        ))


    def load_akt_model(self):
        model_params = {
            'n_question': self.n_knowledge,
            'n_pid': self.n_question,
            'd_model': 256,
            'n_blocks': 1,
            'kq_same': 1,
            'dropout': 0.3,
            'model_type': 'akt',
            'final_fc_dim': 512,
            'n_heads': 8,
            'd_ff': 2048,
            'l2': 1e-5,
            'separate_qa': False
        }

        try:
            from akt import AKT
            model = AKT(**model_params).to(device)
        except ImportError:
            model = AKT(**model_params).to(device)

        # 加载模型权重
        model_path = ".../assist2009_pid/_b24_nb1_gn-1_lr0.0002_s224_sl40_do0.3_dm256_ts1_kq1_l21e-05_8"


        if not os.path.exists(model_path):
            import glob
            model_files = glob.glob(".../assist2009_pid/*")
            if model_files:
                print(f"find:")
                for f in model_files:
                    print(f"  {f}")
                model_path = model_files[0]
                print(f"{model_path}")
            else:
                raise FileNotFoundError(f"can't find : {model_path}")

        try:
            checkpoint = torch.load(model_path, map_location=device)

            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                elif 'model' in checkpoint:
                    model.load_state_dict(checkpoint['model'])
                else:
                    model.load_state_dict(checkpoint)
            else:
                model = checkpoint.to(device)

        except Exception as e:
            print(f"{e}")


        return model

    def prepare_input_for_akt(self, exer_ids, kc_log, scores):  # (exer_ids, kc_log, scores)
        q_data = []
        qa_data = []
        pid_data = []

        if isinstance(kc_log[0], str):
            q_data = [int(kc) for kc in kc_log]
        else:
            q_data = kc_log

        pid_data = exer_ids

        for i, kc_id in enumerate(q_data):
            if i < len(scores) and scores[i] == 1:
                qa_data.append(kc_id + self.n_knowledge)
            else:
                qa_data.append(kc_id)

        q_data = torch.tensor([q_data], dtype=torch.long).to(device)
        qa_data = torch.tensor([qa_data], dtype=torch.long).to(device)
        pid_data = torch.tensor([pid_data], dtype=torch.long).to(device)
        target = torch.tensor([scores], dtype=torch.float).to(device)

        return q_data, qa_data, pid_data, target

    def get_knowledge_state_by_KT(self, exer_ids, kc_log, scores, goalkc_id):  # ques_log, kc_log, ans_log
        q_data, qa_data, pid_data, target = self.prepare_input_for_akt(exer_ids, kc_log, scores)
        with torch.no_grad():
            d_output = self.model(q_data, qa_data, target, pid_data)
        learning_state = torch.mean(d_output).item()

        return learning_state


def run_for_student(student_id, all_stu_lr_state4everygoal, E_all_e_small, E_all_s_small, E_all_sup_small, E_all_e_big,
                    E_all_s_big, E_all_sup_big):
    all_stu_lr_state4everygoal[student_id] = []

    MAX_STEP = 20
    COLD_NUM = 10
    llm = LlamaLLM()
    kt_Model = KTModel_AKT()

    all_logs = load_json(f"{DATA_PATH}/stu_logs.json")
    logs = None
    for student in all_logs:
        if student['user_id'] == student_id:
            logs = student['logs']
            break

    KCG = load_json(f"{DATA_PATH}/kcg.json")

    with open('.../exercises.json', 'r',
              encoding='utf-8') as f:
        exercises = json.load(f)
    know_course_list = load_json(f"{DATA_PATH}/know_course_list.json")
    know_name = load_json(f"{DATA_PATH}/know_name_list.json")
    profile = Profile(student_id)
    memory = Memory(KCG, know_course_list, know_name)
    action = AgentAction(profile, memory, llm)
    Teacher_RS = Teacher(llm)
    results = []

    logs_all = logs[:COLD_NUM]
    kc_log = [log['knowledge_code'] for log in logs_all]
    kc_old = set(kc_log)
    ques_log = [log['exer_id'] for log in logs_all]
    ans_log = [log['score'] for log in logs_all]
    for goalkc_id in kc_old:

        learning_state_list = []

        learning_state = kt_Model.get_knowledge_state_by_KT(ques_log, kc_log, ans_log, goalkc_id)
        learning_state_list.append(learning_state)

        for step in range(MAX_STEP):
            advise = Teacher_RS.given_advise(step, logs_all, ques_log, profile, goalkc_id, learning_state)

            exer_id = advise.get('exer_id')
            recommand_reason = advise.get('recommend_reason', '') or advise.get('recommand_reason', '')

            teacher_predict_answer = advise.get('predict_answer', 0)
            rec = exercises[exer_id]

            ans, raw, corr, summ = action.simulate_step(rec, step, teacher_predict_answer,
                                                        similarity_fn=memory.reinforce)
            results.append({'ans': ans, 'raw': raw, 'corr': corr, 'summ': summ})

            student_score = 0 if ans['task4'] == 'No' else 1
            print('student_score:', student_score)

            ques_log.append(exer_id)
            ans_log.append(student_score)
            kc_log.append(goalkc_id)


            Teacher_RS.reflect_on_outcome(
                question_id=exer_id,
                predict_answer=teacher_predict_answer,
                actual_outcome=student_score,
                student_feedback=summ
            )

            learning_state = kt_Model.get_knowledge_state_by_KT(ques_log, kc_log, ans_log, goalkc_id)
            learning_state_list.append(learning_state)
        if learning_state_list[0] < 0.5 and max(learning_state_list) < 0.7:
            E_p = abs(max(learning_state_list) - learning_state_list[0]) / (0.7 - learning_state_list[0])
            if E_p >= 0.1:
                all_stu_lr_state4everygoal[student_id].append(E_p)
                E_all_s_small += learning_state_list[0]
                E_all_e_small += max(learning_state_list)
                E_all_sup_small += 0.7
        else:
            E_p = abs(max(learning_state_list) - learning_state_list[0]) / (1 - learning_state_list[0])
            if E_p >= 0.1:
                print(f'goal_id = {goalkc_id}, stu = {student_id}, E_p = {E_p}')
                all_stu_lr_state4everygoal[student_id].append(E_p)
                print('all_stu_lr_state4everygoal', all_stu_lr_state4everygoal)
                E_all_s_big += learning_state_list[0]
                E_all_e_big += max(learning_state_list)
                E_all_sup_big += 1

    return E_all_e_small, E_all_s_small, E_all_sup_small, E_all_e_big, E_all_s_big, E_all_sup_big


def main():
    try:
        agent_id_list = load_json(f"{DATA_PATH}/agent_id_list.json")

    except Exception as e:


        all_logs = load_json(f"{DATA_PATH}/stu_logs.json")
        agent_id_list = list(range(min(5, len(all_logs))))

    num_students_to_process = 5
    num_students_to_process = min(num_students_to_process, len(agent_id_list))
    selected_students = random.sample(agent_id_list, num_students_to_process)
    E_all_e_small = 0
    E_all_s_small = 0
    E_all_sup_small = 0
    E_all_e_big = 0
    E_all_s_big = 0
    E_all_sup_big = 0
    all_stu_lr_state4everygoal = {}
    for s in selected_students:
        E_all_e_small, E_all_s_small, E_all_sup_small, E_all_e_big, E_all_s_big, E_all_sup_big = run_for_student\
            (s, all_stu_lr_state4everygoal,E_all_e_small,E_all_s_small,E_all_sup_small,E_all_e_big,E_all_s_big,E_all_sup_big)
        print('################E_all_sup###############', E_all_sup_small)
    E_all_p = ((E_all_e_small + E_all_e_big) - (E_all_s_small + E_all_s_big)) / (
                (E_all_sup_small + E_all_sup_big) - (E_all_s_small + E_all_s_big))
    print(f'E_all_p = {E_all_p}')


if __name__ == '__main__':
    main()
