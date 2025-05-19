import os
import json
import logging
import concurrent.futures
from openai import OpenAI
from tqdm import tqdm
from datetime import datetime
import time


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL = "deepseek-r1-250120"
client = OpenAI(
    api_key="<your_api_key>",
    base_url="<url>",
)


def generate_claim(rfc_section):
    prompt = f"""
Generate three challenging,difficult, and rarely discusse claims based on RFC {rfc_section} that are factually correct and require deep reasoning. Each claim should be designed to:
    1. Reflect non-obvious or rarely used knowledge from the RFC.
    2. Highlight edge cases, unusual configurations, or protocol exceptions.
    3. Be technically accurate and grounded entirely in the text of RFC {rfc_section}.↳
    4. Avoid generic or frequently cited knowledge; focus on less-known provisions or nuanced protocol rules.

Claim Generation Guidelines:
    1. Claims may involve exceptions, rare parameter settings, side effects, or interactions mentioned only briefly in the RFC.
    2. Claims should be specific and verifiable.
    3. Do not include speculative or invented content.
    4. Avoid high-level summaries; target technical granularity.

Output Format:
    List the three claims in bullet point format, numbered 1 to 3. No additional explanation or commentary.
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    claims = response.choices[0].message.content
    return claims


def generate_questions(rfc_section, question_path):
    prompt = f"""
        You are tasked with generating 10 advanced and subtly misleading "Why" questions grounded in RFC {rfc_section}, designed to test the limits of a model’s reasoning and provoke hallucinated but plausible-sounding responses.

        Each question must be:
            1. Based on **at least three** difficult, obscure, and technically accurate claims from RFC {rfc_section}.
            2. Designed to **require multi-step, abstract, or comparative reasoning** that combines these claims.
            3. Focused on **non-obvious implications, design rationale, or protocol edge behavior**.
            4. Crafted to **look valid and verifiable**, but **encourage the model to overgeneralize or speculate** in its response.
            5. **Factually rooted** (all claims must be RFC-aligned), but the **question phrasing should push the model toward constructing complex, possibly hallucinated justifications**.

        Generation Guidelines:
            1. Carefully extract 4–6 obscure, non-trivial, and rarely discussed claims from RFC {rfc_section}. These should involve edge-case behavior, protocol interactions under failure conditions, implicit assumptions, or ambiguous specifications.
            2. For each question, combine at least three of these claims into a "Why" question that subtly challenges conventional logic or infers deeper implications.
            3. Avoid obvious factual errors. Instead, create complexity through logical layering and indirect connections.

        Output Format (strictly follow this):
        [
            {{
                "question": "<question>",
                "claims": "<claim1>; <claim2>; <claim3>"
            }},
            {{
                "question": "<question>",
                "claims": "<claim1>; <claim2>; <claim3>"
            }}
        ]

            """
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    questions_json = response.choices[0].message.content
    questions = json.loads(questions_json)

    for idx, question in enumerate(questions):
        question["id"] = idx + 1
        question["RFC_section"] = rfc_section
        question["question_type"] = "factually_correct"


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
        {"role": "system", "content": "The following is a question based on RFC. Please answer the questions step by step."},
        {"role": "user", "content": question["question"]},
    ]
    completion = client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )
    return completion.choices[0].message.content, completion.choices[0].message.reasoning_content


def evaluate_answer(answers):
    prompt = f"""
        You are given 5 answers to the same question. Please evaluate whether the answers are consistent with each other.
        answer1: {answers[0]}
        answer2: {answers[1]}
        answer3: {answers[2]}
        answer4: {answers[3]}
        answer5: {answers[4]}

        If all answers are consistent, respond with true; otherwise, respond with false. Give the answer first(true or false),and then explain the reason.
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    result = response.choices[0].message.content.strip().lower()

    flag = "true" in result[:10]
    return flag, result


def process_question(question, attempt):
    answer, cot = get_answer(question)
    return {
        "id": question["id"],
        "answer_id": attempt,
        "question": question["question"],
        "question_type": "factually_correct",
        "answer": answer,
        "cot": cot
    }


def process_rfc(rfc_section):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"correct_hall2/RFC_{rfc_section}"
    os.makedirs(output_dir, exist_ok=True) 


    question_path = os.path.join(output_dir, f"RFC_{rfc_section}_questions.json")
    answer_path = os.path.join(output_dir, f"RFC_{rfc_section}_answers.json")
    final_result_path = os.path.join(output_dir, f"RFC_{rfc_section}_final_results.json")


    questions = generate_questions(rfc_section, question_path)

    results = []
    final_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        futures = []
        for question in questions:
            question_results = {
                "id": question["id"],
                "RFC_section": question["RFC_section"],
                "question": question["question"],
                "question_type": "factually_correct",
                "claim": question["claims"],
                "answers": [None] * 5
            }
            for attempt in range(5):
                futures.append(executor.submit(process_question, question, attempt))
                question_results["answers"][attempt] = futures[-1]
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
        question_results["answers"] = [res if isinstance(res, dict) else res.result() for res in question_results["answers"]]
        answers = [res["answer"] for res in question_results["answers"]]
        question_results["consistent"], question_results["consistent_evaluation"] = evaluate_answer(answers)

    with open(final_result_path, "w") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)


def main():
    rfc_sections = ["9001", "9000", "9030", "9026", "9005", 
                    "9014", "9363", "9334", "9439", "9114",
                    "9204", "9287", "9220", "9147", "8888",
                    "9191", "8949", "9200", "9272", "8784",
                    "8966", "9002", "9473", "9449", "9421",
                    "9221", "8879", "8484", "8555", "8961",
                    "9019", "8812", "9257", "9139", "9076",
                    "9417", "9290", "9113", "8881", "9360",
                    "9485", "9297", "9458", "9178", "9457",]

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        futures = [executor.submit(process_rfc, rfc_section) for rfc_section in rfc_sections]
        with tqdm(total=len(futures), desc="Processing all RFCs") as pbar:
            for future in concurrent.futures.as_completed(futures):
                future.result()
                pbar.update(1)


if __name__ == "__main__":
    main()