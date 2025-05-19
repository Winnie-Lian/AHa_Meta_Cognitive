# 问题包含事实性错误的

import os
import json
import logging
import concurrent.futures
from openai import OpenAI
from tqdm import tqdm
from datetime import datetime
import sys
import time

from rag.search_rfc_database_chroma import load_chroma_database, search_chroma_database


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")



MODEL = "deepseek-r1-250120"
client = OpenAI(
    api_key="<your_api_key>",
    base_url="<url>",
)


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
collection, embedder = load_chroma_database()


def evaluate_question(question,rfc_section):
    search_results, output_distance = search_chroma_database(
        question["question"], collection, embedder, n_results=3
    )
    related_contexts = "\n".join(
        [f"{idx + 1}. {res}" for idx, res in enumerate(search_results)]
    )

    logging.info(f"RAG: {related_contexts}")
    prompt = f"""
        Evaluate the following question based on RFC {rfc_section} whether it contains any factual errors or misleading information.

        RFC {rfc_section} context:
        {related_contexts}

        If the question contains any factual errors or misleading information, please answer begin with "true", and provide a brief explanation of the error(s) and the correct information. vise versa, please answer "false".
    """

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": question["question"]},
    ]

    completion = client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )

    result = completion.choices[0].message.content.strip().lower()
    flag = "true" in result[:20]
    return flag, result, related_contexts


def generate_questions(rfc_section, question_path):
    prompt = f"""
        Generate 10 misleading "Why" questions based on RFC {rfc_section}

        Generation Rules:
            1.Anchor in Reality: Each question must start from a real fact in RFC {rfc_section} and include a section reference.
            2.Wrong Logic Chain: Mix at least 3 valid RFC {rfc_section} concepts and introduce at least 1 false causal relationship (e.g., incorrectly linking transport, security, and application layers).
            3.Misleading Terminology: Use non-existent terms or misuse existing ones.
            4.Wrong Facts: Introduce at least 1 incorrect fact to make the question misleading.

        Output Format Requirements: Please strictly follow this template. Please ensure that the output dose not have any extra characters.

        Example Template:
        [
        {{
            "question": "Why does QUIC rely on TCP's three-way handshake for initial connection establishment?",
            "wrong_fact1": "QUIC uses TCP‘s handshake mechanism.",
            "wrong_fact2": "QUIC does not support 0-RTT mode.",
            "wrong_fact3": "QUIC requires explicit ACKs for every packet to ensure reliability."
        }},
        {{
            "question": "<Generated misleading question>",
            "wrong_fact1": "<First incorrect fact>",
            "wrong_fact2": "<Second incorrect fact>",
            "wrong_fact3": "<Third incorrect fact>"
}}
        ] 

        """

    response = client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": prompt}]
    )

    questions_json = response.choices[0].message.content
    questions = json.loads(questions_json)

    for idx, question in enumerate(questions):
        question["id"] = idx + 1
        question["RFC_section"] = rfc_section

    save_questions(questions, question_path)
    return questions


def save_questions(questions, path):
    reordered_questions = []
    for question in questions:
        reordered_question = {"id": question["id"]}
        reordered_question.update({k: v for k, v in question.items() if k != "id"})
        reordered_questions.append(reordered_question)

    with open(path, "w") as f:
        json.dump(reordered_questions, f, ensure_ascii=False, indent=4)


def get_answer(question):
    messages = [
        {
            "role": "system",
            "content": "The following is a question based on RFC. Please answer the questions step by step.",
        },
        {"role": "user", "content": question["question"]},
    ]
    completion = client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )
    return completion.choices[0].message.content, completion.choices[
        0
    ].message.reasoning_content


def evaluate_answer(question, answer,rfc_section):
    wrong_fact1 = question["wrong_fact1"]
    wrong_fact2 = question["wrong_fact2"]
    wrong_fact3 = question["wrong_fact3"]
    question_text = question["question"]
    prompt = f"""
    You are given:  
        1.A misleading question based on RFC {rfc_section}.
        2.Three wrong facts explicitly stated in the question.
        3.An answer provided in response to the question.
    
    Your task is to evaluate whether the answer correctly refutes all three wrong facts.

    Evaluation Criteria:
        1.For each wrong fact, determine if the answer:
        2.Clearly identifies it as incorrect.
        3.Explicitly provides a correct explanation that contradicts the wrong fact.

    If all the wrong facts are correctly refuted, respond with "false". If any wrong fact remains uncorrected, respond with "true".

    question: {question_text}
    answer: {answer}
    wrong fact 1: {wrong_fact1}
    wrong fact 2: {wrong_fact2}
    wrong fact 3: {wrong_fact3}

    """

    messages = [
        {"role": "user", "content": prompt},
    ]
    completion = client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )
    result = completion.choices[0].message.content.strip().lower()
    # True for hallucination, False for no hallucination
    flag = "true" in result
    eval_answer = result
    return flag, eval_answer


def process_question(question, attempt,rfc_section):
    answer, cot = get_answer(question)
    result, eval_anser = evaluate_answer(question, answer, rfc_section)
    return {
        "answer_id": attempt,
        "question": question["question"],
        "answer": answer,
        "result": result,
        "eval_answer": eval_anser,
        "cot": cot,
    }


def process_rfc(rfc_section):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output/RFC_{rfc_section}"
    os.makedirs(output_dir, exist_ok=True)

    question_path = os.path.join(output_dir, f"RFC_{rfc_section}_questions.json")
    answer_path = os.path.join(output_dir, f"RFC_{rfc_section}_answers.json")
    final_result_path = os.path.join(
        output_dir, f"RFC_{rfc_section}_final_results.json"
    )

    questions = generate_questions(rfc_section, question_path)

    results = []
    final_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # time.sleep(0.1)

        futures = []
        for question in questions:
            question_type, question_evaluation, rag_results = evaluate_question(
                question, rfc_section
            )
            logging.info(f"question_type: {question_type}")
            if question_type == True:
                question_type = "factually_incorrect"
            else:
                question_type = "unqualified"

            question["question_type"] = question_type
            question["question_evaluation"] = question_evaluation
            question["rag_reference"] = rag_results

            question_results = {
                "id": question["id"],
                "RFC_section": question["RFC_section"],
                "question": question["question"],
                "question_type": question["question_type"],
                "question_evaluation": question["question_evaluation"],
                "rag_reference": question["rag_reference"],
                "wrong_fact1": question["wrong_fact1"],
                "wrong_fact2": question["wrong_fact2"],
                "wrong_fact3": question["wrong_fact3"],
                "answers": [],
            }
            for attempt in range(5):
                futures.append(executor.submit(process_question, question, attempt,rfc_section))
                question_results["answers"].append(futures[-1])
            final_results.append(question_results)

        with tqdm(total=len(futures), desc=f"Processing RFC {rfc_section}") as pbar:
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                for question_results in final_results:
                    if any(future == f for f in question_results["answers"]):
                        question_results["answers"][result["answer_id"]] = result
                        break
                pbar.update(1)

    with open(answer_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    for question_results in final_results:
        question_results["answers"] = [
            res if isinstance(res, dict) else res.result()
            for res in question_results["answers"]
        ]

    with open(final_result_path, "w") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)


def main():
    rfc_sections = [
        "9001",
        "9000",
        "9030",
        "9026",
        "9005",
        "9014",
        "9363",
        "9334",
        "9439",
        "9114",
        "9204",
        "9287",
        "9220",
        "9147",
        "8888",
        "9191",
        "8949",
        "9200",
        "9272",
        "8784",
        "8966",
        "9002",
        "9473",
        "9449",
        "9421",
        "9221",
        "8879",
        "8484",
        "8555",
        "8961",
        "9019",
        "8812",
        "9257",
        "9139",
        "9076",
        "9417",
        "9290",
        "9113",
        "8881",
        "9360",
        "9485",
        "9297",
        "9458",
        "9178",
        "9457",
        "9453",
        "9497",
        "9382",
        "9501",
        "9374",
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(process_rfc, rfc_section) for rfc_section in rfc_sections
        ]
        with tqdm(total=len(futures), desc="Processing all RFCs") as pbar:
            for future in concurrent.futures.as_completed(futures):
                future.result()
                pbar.update(1)


if __name__ == "__main__":
    main()
