

import json
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor

MODEL = "deepseek-r1-250120"
client = OpenAI(
    api_key="<your_api_key>",
    base_url="<url>",
)


def get_response(prompt, model=MODEL):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt},
        ]
    )
    return response.choices[0].message.content, response.choices[0].message.reasoning_content


def evaluate_answer(answers):
    prompt = f"""
        You are given 5 answers to the same question. Please evaluate whether the answers are consistent with each other.
        answer1: {answers[0]}
        answer2: {answers[1]}
        answer3: {answers[2]}
        answer4: {answers[3]}
        answer5: {answers[4]}

        If all answers are consistent, respond with true; otherwise, respond with false. Give the answer first(true or false), and then explain the reason.
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    result = response.choices[0].message.content.strip().lower()

    flag = "true" in result[:10]
    return flag, result



def eval_answer(answer,rfc_section):
    with open("rfc_index.json", "r", encoding="utf-8") as f:
        rfc_index = json.load(f)
  
    info = next((item["info"] for item in rfc_index if item.get("rfc_section") == rfc_section), None)


    pre_prompt = f"""
    The information of RFC {rfc_section} is as follows:{info}
    """
    # prompt1: 介绍rfc文档 work
    prompt = f"""
    According to the information: RFC{rfc_section}--{info}
    There is a introduction about RFC document{rfc_section}. Do you think it is factually correct? If it is factually correct, please answer "true". If it is not factually correct, please answer "false" and explain the reason.
    Description: {answer}
    """
    
    # # prompt2：obsoleted by
    # prompt = f"""
    #     {pre_prompt}
    #     According to the information,please check whether the answer of which RFC obsolete RFC{rfc_section} is correct. If it is correct, please answer "true". If it is not correct, please answer "false" and explain the reason.
    #     Answer: {answer}
    # """

    # # prompt3: obsoleted
    # prompt = f"""
    #     Please check whether the answer of which RFC was obsoleted by RFC{rfc_section} is correct. If it is correct, please answer "true". If it is not correct, please answer "false" and explain the reason.
    #     Answer: {answer}
    # """

    # # prompt4: updated by
    # prompt = f"""
    #     Please check whether the answer of which RFC was updated by RFC{rfc_section} is correct. If it is correct, please answer "true". If it is not correct, please answer "false" and explain the reason.
    #     Answer: {answer}
    # """
    
    
    # # prompt5：update
    # prompt = f"""
    #     Please check whether the answer of which RFC updates RFC {rfc_section} is correct. If it is correct, please answer "true". If it is not correct, please answer "false" and explain the reason.
    #     Answer: {answer}
    # """


    # # prompt6： publication date
    # prompt = f"""
    # Please check whether the answer of what is the publication date of RFC {rfc_section} is correct. If it is correct, please answer "true". If it is not correct, please answer "false" and explain the reason.
    # Answer: {answer}
    # """
    

    # # prompt7：current status
    # prompt = f""" 
    # Please check whether the answer of what is the current status of RFC {rfc_section} is correct. If it is correct, please answer "true". If it is not correct, please answer "false" and explain the reason.
    # Answer: {answer}
    # """


    # # prompt8: author
    # prompt = f""" 
    # Please check whether the answer of who is the author of RFC {rfc_section} is correct. If it is correct, please answer "true". If it is not correct, please answer "false" and explain the reason.
    # Answer: {answer}
    # """




    reasponse = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    result = reasponse.choices[0].message.content.strip().lower()
    
    flag = "true" in result[:10]
    return flag, result


# 处理单个 RFC 段落
def process_rfc_section(rfc_section, question_id):
    prompt = f"Please introduce me the RFC {rfc_section} in detail."

    # prompt2：osoleted by
    # prompt = f""" Do you know which RFC obsoleted RFC {rfc_section}?"""

    # # prompt3: obsoleted 
    # prompt = f"""Please tell me which RFC was obsoleted by RFC {rfc_section}."""

    # # prompt4: updated by
    # prompt = f"""Please tell me which RFC was updated by RFC {rfc_section}."""

    # # prompt5: update
    # prompt = f""" Do you know which RFC updates RFC {rfc_section}?"""

    # # prompt6： publication date
    # prompt = f"""Please tell me the publication date of RFC {rfc_section}."""
    
    # # prompt7：current status
    # prompt = f"""what is the current status of RFC {rfc_section} ?"""

    # # prompt8: author
    # prompt = f"""Please tell me the author of RFC {rfc_section}."""

    answers = []

    for i in range(5):
        answer, reasoning = get_response(prompt)
        double_check_answer_flag, double_check_result = eval_answer(answer,rfc_section)         

        answers.append({
            "id": question_id,
            "answer_id": i,
            "question": prompt,
            "answer": answer.strip(),
            "result": double_check_answer_flag,
            "eval_answer": double_check_result.strip(),
            "cot": reasoning.strip()
        })

    flag, result = evaluate_answer([a["answer"] for a in answers])


    return {
        "id": question_id,
        "RFC_section": rfc_section,
        "question": prompt,
        "question_type": "factually_correct",
        "answers": answers,
        "consistent": flag,
        "consistent_evaluation": result
    }


def main():
    rfc_sections = ['1364', '0755', '3023', '8920', '7053', '3786', '1251', '1655', '4718', '4843', '1379', '1105', '0811', '1120', '4327', '1384', '6304', '0835', '2407', '2821', '1603', '1777', '0360', '1072', '6485', '6046', '8540', '2200', '2002', '2023', '2240', '5575', '3392', '1020', '5680', '2401', '1131', '3567', '0362', '4282', '3300', '1425', '1050', '4893', '1573', '5204', '2338', '1323', '6859', '2851']
    results = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_rfc_section, rfc_section, idx + 1) for idx, rfc_section in enumerate(rfc_sections)]
        for future in futures:
            results.append(future.result())

    output_file = "./results/results.json"
    with open(output_file, "a", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()