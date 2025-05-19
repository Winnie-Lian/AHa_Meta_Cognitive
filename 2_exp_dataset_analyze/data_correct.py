import openai
import httpx
import json
import logging


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
MODEL = "gpt-4o-mini"
base_url = '<url>'
api_key = '<your_api_key>'

client = openai.OpenAI(
    base_url=base_url,
    api_key=api_key,
    http_client=httpx.Client(
        base_url=base_url,
        follow_redirects=True
    )
)

def process_json_response(response):
    if "```json" in response and "```" in response:
        start_index = response.find("```json") + len("```json")
        end_index = response.rfind("```")
        response = response[start_index:end_index].strip()
    return response




def get_annotated_claims1(cot, eval_answer, result, rag_reference="\n", wrongfact=""):
    if wrongfact :
        prompt = f"""
        You are given a piece of text composed of multiple sentences, which is the chain of thougnt generated from a model. Your task is to carefully evaluate each sentence and determine whether it contains a hallucination (i.e., an unsupported or factually incorrect claim).
        For each sentence, output a JSON object with the following structure:
            {{
            "sentence_id": <sentence number starting from 1>,
            "claim": "<the original sentence>",
            "hallucination": true / false
            }}

        Mark "hallucination": true if the sentence includes fabricated information, unver
        Mark "hallucination": false if the sentence is factually correct, logically sound, or based on standard knowledge.

        And there is some reference you can use:
        - reference: user provided
        - eval_answer: the evaluation of answer
        - result: whether the answer contains hallucination or not(true:hallucinated;false:non-hallucinated)
        - wrongfact: Actually, the question is misleading, as it contains wrong facts

        reference:{rag_reference}
        eval_answer:{eval_answer}
        result:{result}
        worngfact:{wrongfact}


        Return the final result as a JSON array without any additional text or explanation.
        There is the text:
        {cot}

        """
        
    else:
        prompt = f"""
        You are given a piece of text composed of multiple sentences, which is the chain of thougnt generated from a model. Your task is to carefully evaluate each sentence and determine whether it contains a hallucination (i.e., an unsupported or factually incorrect claim).
        For each sentence, output a JSON object with the following structure:
            {{
            "sentence_id": <sentence number starting from 1>,
            "claim": "<the original sentence>",
            "hallucination": true / false
            }}

        Mark "hallucination": true if the sentence includes fabricated information, unver
        Mark "hallucination": false if the sentence is factually correct, logically sound, or based on standard knowledge.

        And there is some reference you can use:
        - reference: user provided
        - eval_answer: the evaluation of answer
        - result: whether the answer contains hallucination or not(true:hallucinated;false:non-hallucinated)

        reference:{rag_reference}
        eval_answer:{eval_answer}
        result:{result}


        Return the final result as a JSON array without any additional text or explanation.
        There is the text:
        {cot}

        """


    messages=  client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt   
            }
        ]
    )
    response = messages.choices[0].message.content
    return response


def get_annotated_claims2(cot,start_id,eval_answer,result,rag_reference="\n",wrongfact=""):
    if wrongfact :


        prompt = f"""
            You are given a piece of text composed of multiple sentences, which is the chain of thougnt generated from a model. Your task is to carefully evaluate each sentence and determine whether it contains a hallucination (i.e., an unsupported or factually incorrect claim).

            This is a continuation of an unfinished task. The previous sentences have already been evaluated and assigned sentence IDs starting from 1. Your task now is to continue evaluating the remaining sentences, starting from sentence ID {start_id}.

            For each sentence, output a JSON object with the following structure:
            {{
            "sentence_id": <sentence number starting from {start_id}>,
            "claim": "<the original sentence>",
            "hallucination": true / false
            }}

            Mark "hallucination": true if the sentence contains fabricated, unverifiable, or incorrect information.

            Mark "hallucination": false if the sentence is factually correct, logically sound, or based on common knowledge.

            Return the final result as a JSON array, without any extra text or explanation.


            And there is some reference you can use:
            - reference: user provided
            - eval_answer: the evaluation of answer
            - result: whether the answer contains hallucination or not(true:hallucinated;false:non-hallucinated)
            - wrongfact: Actually, the question is misleading, as it contains wrong facts

            reference:{rag_reference}
            eval_answer:{eval_answer}
            result:{result}
            wrongfact:{wrongfact}


            Here is the next portion of the text:
            {cot}    
            """
    else:
        logging.info("wrongfact is not exist")
        prompt = f"""
            You are given a piece of text composed of multiple sentences, which is the chain of thougnt generated from a model. Your task is to carefully evaluate each sentence and determine whether it contains a hallucination (i.e., an unsupported or factually incorrect claim).

            This is a continuation of an unfinished task. The previous sentences have already been evaluated and assigned sentence IDs starting from 1. Your task now is to continue evaluating the remaining sentences, starting from sentence ID {start_id}.

            For each sentence, output a JSON object with the following structure:
            {{
            "sentence_id": <sentence number starting from {start_id}>,
            "claim": "<the original sentence>",
            "hallucination": true / false
            }}

            Mark "hallucination": true if the sentence contains fabricated, unverifiable, or incorrect information.

            Mark "hallucination": false if the sentence is factually correct, logically sound, or based on common knowledge.

            Return the final result as a JSON array, without any extra text or explanation.


            And there is some reference you can use:
            - reference: user provided
            - eval_answer: the evaluation of answer
            - result: whether the answer contains hallucination or not(true:hallucinated;false:non-hallucinated)


            reference:{rag_reference}
            eval_answer:{eval_answer}
            result:{result}

            Here is the next portion of the text:
            {cot}    
        """
    messages=  client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt   
            }
        ]
    )
    response = messages.choices[0].message.content
    return response

def split_cot(cot):
    mid_index = len(cot) // 2  
    split_index = mid_index

    print("mid_index:", mid_index)
    
    while split_index > 0 and cot[split_index] not in ".!?":
        split_index -= 1
    print("split_index:", split_index)

    if split_index == 0:
        split_index = mid_index


    part1 = cot[:split_index + 1]  
    part2 = cot[split_index + 1:] 
    return part1, part2


def get_annotated_claims(cot,eval_answer,result,rag_reference="\n",wrongfact=""):
    part1, part2 = split_cot(cot)


    response1 = get_annotated_claims1(part1,eval_answer,result,rag_reference,wrongfact)
    response1 = process_json_response(response1)  
    json_response1 = json.loads(response1)  

    # the second part of cot
    start_id = len(json_response1) + 1  
    response2 = get_annotated_claims2(part2, start_id,eval_answer,result,rag_reference,wrongfact)
    response2 = process_json_response(response2)  
    json_response2 = json.loads(response2)  # 

    combined_response = json_response1 + json_response2
    return combined_response



def if_accepted(cot,claim):

    prompt = f"""
    Given a piece of discussion text and a few specific claims, your task is to determine how the claims are treated within the text:
    - Accepted: The text ultimately supports or agrees with the claim.
    - Corrected: The text first denies or questions the claim and then provides a new, corrected, or more accurate version of it.
    - Rejected: The text denies or refutes the claim, without providing a corrected or alternative answer.

    The output should be a JSON object with the following structure:
    Template
    {{
        "sentence_id": <original sentence id>,
        "claim": "<the claim>",
        "accepted": true / false,
        "corrected": true / false,
        "rejected": true / false
    }}
    Return the final result as a JSON array without any additional text or explanation.

    There is the text:{cot}
    There is the claim:{claim}
    """
    messages=  client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    response = messages.choices[0].message.content
    response = process_json_response(response)  
    response = json.loads(response) 
    return response



def get_important_hallucinated_claims(question,cot,answer,eval_answer):
    prompt = f"""
        You will be given a QA pair consisting of a question, a structured chain of thought (in JSON format), and an answer. And there is "eval_answer" which is the evaluation of the answer.
        Your task is:
            1.Analyze the chain of thought and identify the hallucinated sentences that are most critical to the final answer and reasoning process.
            2.Select up to 5 of the most important hallucinated sentences.
                If there are fewer than 5 hallucinated sentences, simply return all of them.
            3.For each selected sentence, count and report how many times its underlying idea or claim appears in the chain of thought.

    Key Clarifications
    - "Most critical" means the sentences that, if removed or corrected, would significantly affect the final answer or reasoning.
    - "Underlying idea" refers to the core viewpoint or factual claim conveyed by a sentence, regardless of slight wording differences.
    - If a hallucinated idea is paraphrased or reiterated elsewhere in the chain, all occurrences should be counted.

        The output should be a JSON object with the following structure:
        Template:
        
        [
        {{
            "effective_claim_id": < number from 1 to 5>,
            "claim": "<the hallucinated sentence>",
            "repetition_count": <number of times the underlying idea appears in the chain>,
            "hallucination": true
        }}


        There is the QA pair:
        question: {question}
        chain_of_thought: {cot}
        answer: {answer}
        eval_answer: {eval_answer}

    """
    messages=  client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    response = messages.choices[0].message.content
    response = process_json_response(response)  
    response = json.loads(response)  
    return response


def gpt_reflection_times(question,cot,answer):
    prompt = f"""
    You will be given a QA pair consisting of a question, a structured chain of thought (in JSON format), and an answer.
    Your task is to analyze the chain of thought and determine how many times the model reflects on its own reasoning process.
    A reflection is defined as a moment when the model evaluates or critiques its own reasoning, either positively or negatively.

    The output should be a JSON object with the following structure:
    Template:
    {{
        "reflection_times": <number of reflections in the chain_of_thought>
    }}

    There is the QA pair:
    question: {question}
    chain_of_thought: {cot}
    answer: {answer}
    """
    messages=  client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    response = messages.choices[0].message.content
    response = process_json_response(response)  
    response = json.loads(response)  
    return response


def transform_json(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    results = []


    for new_id,item in enumerate(data,start=1):
        if "answers" in item and len(item["answers"]) > 0:
            transformed_item = {
                "table_id": new_id,
                "original_id": item["id"],
                "RFC_section": item["RFC_section"],
                "question": item["question"],
                "question_type": item["question_type"],
                "answer": item["answers"][0].get("answer", ""), 
                "cot": item["answers"][0].get("cot", ""),
                "eval_answer": item["answers"][0].get("eval_answer", ""),
                "result": item["answers"][0].get("result", ""),
            }
            results.append(transformed_item)
    return results

def backup_data(item):
    json_data = json.dumps(item, ensure_ascii=False, indent=4)
    id = item["table_id"]
    with open(f"correct_data_result/correct_hallu_{id}.json", "w", encoding="utf-8") as f:
        f.write(json_data)


def get_internal_hall_claims(annotated_claims):
    result = []
    for item in annotated_claims:
        if item["hallucination"] == True:
            result.append(item)
    return result

def main():
    input_file = "input.json"
    data= transform_json(input_file)
    for item in data:
        try:
            question = item["question"]
            answer = item["answer"]
            cot = item["cot"]
            eval_answer = item["eval_answer"]
            result = item["result"]
            item["step"] =0
            rfc_section = item["RFC_section"]

            with open("rfc_index.json", "r", encoding="utf-8") as f:
                rfc_index = json.load(f)
    

            try:
                info = next((item["info"] for item in rfc_index if item.get("rfc_section") == rfc_section), None)
            except StopIteration:
                info = None

            if len(cot) > 0:
                annotated_claims = get_annotated_claims(cot,eval_answer,result,rag_reference=info)
                item["annotated_claims"] = annotated_claims
                item["step"] = 1
                backup_data(item)


                logging.info("Step 1: Annotated claims processed and backed up.")
                
                internal_hall_claims = get_internal_hall_claims(annotated_claims)
                if_accepted_results = if_accepted(cot,internal_hall_claims)
                item["internal_hallu_claims"] = if_accepted_results
                for if_accepted_result in if_accepted_results:
                    sentence_id = if_accepted_result["sentence_id"]
                    # logging.info("sentence_id: %s", sentence_id)
                    # logging.info("%s", type(sentence_id))
                    try:
                        sentence_id = int(sentence_id)
                    except Exception:
                        continue

                    if item["annotated_claims"][sentence_id-1]["claim"] == if_accepted_result["claim"]:
                        del if_accepted_result["claim"]
                        del if_accepted_result["sentence_id"]
                        item["annotated_claims"][sentence_id-1]["if_accepted"] = if_accepted_result
                item["step"] = 2
                backup_data(item)

                logging.info("Step 2: Internal hallucinated claims processed and backed up.")

                
                important_hallucinated_claims = get_important_hallucinated_claims(question,cot,answer,eval_answer)
                item["important_hallucinated_claims"] = important_hallucinated_claims
                item["step"] = 3
                backup_data(item)

                logging.info("Step 3: Important hallucinated claims processed and backed up.")

                reflection_times = gpt_reflection_times(question,cot,answer)
                item["reflection_times"] = reflection_times
                item["step"] = 4
                backup_data(item)
                logging.info("Step 4: Reflection times processed and backed up.")
            try:
                with open("correct_data_result/correct_hallu_439_without69_new_tag_all.json", "a", encoding="utf-8") as f:
                    json.dump(item, f, ensure_ascii=False, indent=4)
                    f.write(",\n")  
            except Exception as e:
                item["step"] = 5
                logging.info("backup_data:%s", json.dumps(item, ensure_ascii=False, indent=4))
                logging.error("Error writing to file: %s", str(e))
                
                backup_data(item)
                logging.error("Step 5: Error occurred, data backed up.")


        except Exception as e:
            logging.info("backup_data:%s", json.dumps(item, ensure_ascii=False, indent=4))

            logging.error("Error processing item with ID %s: %s", item["table_id"], str(e))
            item["step"] = 8
            backup_data(item)
            logging.error("Step 8: Error occurred, data backed up.")

    try:
        with open("correct_hallu_439_without69_new_tag_final.json", "a", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)  
    except Exception as e:
        logging.error("Error writing to file: %s", str(e))  
        logging.info("backup_data:%s", json.dumps(data, ensure_ascii=False, indent=4))





if __name__ == "__main__":
    main()