# Retrieval Augmented Generation (RAG)

The **R**etrieval **A**ugmented **G**eneration (RAG) [pipeline](https://github.ibm.com/conversational-ai/fms-dgt-internal/tree/develop/src/databuilders/rag) is used for generating synthetic data in the form of single turn question-answer pairs. 

> **INSTALLATION**
>
> RAG pipeline uses `magpie` based data quality assessment. To install `magpie` dependencies
> ```shell
> pip install -e ".[magpie]"
> ```

Let's try generating a small number of synthetic examples with RAG pipeline

```shell
$ python -m src.__main__ --data-path ./data/rag/direct --num-outputs-to-generate 3
```

> **NOTE**
>
> `--num-outputs-to-generate` is used to specify the number of examples to generate

The output will be written to `./output/rag_direct/data.jsonl`


```json
{"task_name": "rag_direct", "domain": "RAG", "task_description": "Generate single turn direct style grounded conversation over one or more documents.", "utterances": [{"speaker": "user", "text": "How far in advance should an employee submit a request for a day off, and what information should they include in the request?", "timestamp": 1736397160, "metadata": {"question type": "direct"}}, {"speaker": "agent", "text": "An employee should submit a request for a day off at least two weeks in advance and include the reason for the day off and the requested date in the request form.", "timestamp": 1736397160, "contexts": [{"id": "document 3", "text": "This is a sample document 3."}]}], "tags": {"metadata": {"label_model": ["mistralai/mixtral-8x7b-instruct-v01"]}, "input_quality": [{"mistralai/mixtral-8x7b-instruct-v01": "excellent"}], "input_quality_explanation": [{"mistralai/mixtral-8x7b-instruct-v01": "The user query is well-articulated with clear intent. It is specific and well-structured, and no additional context is needed for understanding the request."}], "judge_quality_score": [{"mistralai/mixtral-8x7b-instruct-v01": "4"}], "judge_quality_explanation": [{"mistralai/mixtral-8x7b-instruct-v01": "The assistant accurately addressed the instruction. The answer was logically structured and easy to understand. However, the presentation could be slightly improved with a more comprehensive response."}], "task_category": [{"mistralai/mixtral-8x7b-instruct-v01": "Advice seeking"}]}}
{"task_name": "rag_direct", "domain": "RAG", "task_description": "Generate single turn direct style grounded conversation over one or more documents.", "utterances": [{"speaker": "user", "text": "Why does the company have a policy requiring employees to wear identification badges during work hours?", "timestamp": 1736397160, "metadata": {"question type": "direct"}}, {"speaker": "agent", "text": "The company requires employees to wear identification badges during work hours to ensure the safety of all employees and to make it easier to identify who is an employee and who is not.", "timestamp": 1736397160, "contexts": [{"id": "document 2", "text": "This is a sample document 2."}]}], "tags": {"metadata": {"label_model": ["mistralai/mixtral-8x7b-instruct-v01"]}, "input_quality": [{"mistralai/mixtral-8x7b-instruct-v01": "good"}], "input_quality_explanation": [{"mistralai/mixtral-8x7b-instruct-v01": "The query is clear and specific, asking for a reason behind a company policy. It shows the user's intention very well."}], "intent": [{"mistralai/mixtral-8x7b-instruct-v01": "The user wants to understand the rationale behind the company policy on wearing identification badges."}], "knowledge": [{"mistralai/mixtral-8x7b-instruct-v01": "To solve this problem, the models need to know the purposes and impacts of wearing identification badges."}], "difficulty": [{"mistralai/mixtral-8x7b-instruct-v01": "medium"}], "task_category": [{"mistralai/mixtral-8x7b-instruct-v01": "Information seeking"}]}}
{"task_name": "rag_direct", "domain": "RAG", "task_description": "Generate single turn direct style grounded conversation over one or more documents.", "utterances": [{"speaker": "user", "text": "Can a visitor access the building without being monitored by the security team and not wearing a badge?", "timestamp": 1736397160, "metadata": {"question type": "direct"}}, {"speaker": "agent", "text": "Based on the company's visitor policy and security measures, it is unlikely that a visitor can access the building without being monitored by the security team and not wearing a badge.", "timestamp": 1736397160, "contexts": [{"id": "document 5", "text": "This is a sample document 5."}]}], "tags": {"metadata": {"label_model": ["mistralai/mixtral-8x7b-instruct-v01"]}, "input_quality": [{"mistralai/mixtral-8x7b-instruct-v01": "good"}], "input_quality_explanation": [{"mistralai/mixtral-8x7b-instruct-v01": "The query provides clear information regarding a visitor's access to a building under certain circumstances. It's specific and well-articulated."}], "judge_quality_score": [{"mistralai/mixtral-8x7b-instruct-v01": "4"}], "judge_quality_explanation": [{"mistralai/mixtral-8x7b-instruct-v01": "The response faithfully follows the instruction and provides an accurate answer that relates to the query. The answer might have been even better by clearly addressing all parts of the question or providing additional context to make the answer clearer and easier to understand."}], "intent": [{"mistralai/mixtral-8x7b-instruct-v01": "The user wants to identify if visitors can access the building without being monitored by the security team and not wearing a badge"}], "knowledge": [{"mistralai/mixtral-8x7b-instruct-v01": "To solve this problem, the models need to know the building access polices"}], "difficulty": [{"mistralai/mixtral-8x7b-instruct-v01": "medium"}], "task_category": [{"mistralai/mixtral-8x7b-instruct-v01": "Information seeking"}]}}
{"task_name": "rag_direct", "domain": "RAG", "task_description": "Generate single turn direct style grounded conversation over one or more documents.", "utterances": [{"speaker": "user", "text": "What is the process of creating an account on a website, from visiting the homepage to logging in for the first time?", "timestamp": 1736397160, "metadata": {"question type": "direct"}}, {"speaker": "agent", "text": "To create an account on a website, a user first visits the website's homepage and clicks on the \"Create Account\" button. This takes them to the account creation page, where they provide their personal information, such as name, email address, and password. After filling out the required fields, the user clicks on the \"Create Account\" button to submit their information. The website then sends a confirmation email to the user's email address, which the user must click on to verify their account. Once the account is verified, the user can log in to the website using their email address and password.", "timestamp": 1736397160, "contexts": [{"id": "document 4", "text": "This is a sample document 4."}]}], "tags": {"metadata": {"label_model": ["mistralai/mixtral-8x7b-instruct-v01"]}, "input_quality": [{"mistralai/mixtral-8x7b-instruct-v01": "good"}], "input_quality_explanation": [{"mistralai/mixtral-8x7b-instruct-v01": "The user query is quite detailed and covers the complete process of account creation on a website. It is clear, specific, and shows a good understanding of the required steps by the user."}], "judge_quality_score": [{"mistralai/mixtral-8x7b-instruct-v01": "4"}], "judge_quality_explanation": [{"mistralai/mixtral-8x7b-instruct-v01": "The AI Assistant's response addresses the question as it outlines the process of creating an account on a website, which includes visiting the homepage and logging in for the first time. The answer is accurate and easy to understand, although it could be improved with more detailed descriptions of each step. The response is logically structured, with each step following the previous step, which helps to enhance its ease of understanding."}], "intent": [{"mistralai/mixtral-8x7b-instruct-v01": "The user wants to understand the process of creating an account on a website, from visiting the homepage to logging in for the first time"}], "knowledge": [{"mistralai/mixtral-8x7b-instruct-v01": "To solve this problem, the models need to know about the steps of creating an account on a website, potentially including visiting the homepage, clicking on the 'sign-up' button, entering personal information, confirming account creation, and logging in"}], "difficulty": [{"mistralai/mixtral-8x7b-instruct-v01": "medium"}], "task_category": [{"mistralai/mixtral-8x7b-instruct-v01": "Information seeking"}]}}
{"task_name": "rag_direct", "domain": "RAG", "task_description": "Generate single turn direct style grounded conversation over one or more documents.", "utterances": [{"speaker": "user", "text": "How do the two documents discuss the process of achieving a common goal?", "timestamp": 1736397160, "metadata": {"question type": "direct"}}, {"speaker": "agent", "text": "Based on the documents, the first document discusses the process of setting a goal and breaking it down into smaller tasks, while the second document discusses the process of prioritizing tasks and completing them. Therefore, the common goal that can be inferred from the processes discussed in the documents is achieving a task through careful planning, prioritization, and execution.", "timestamp": 1736397160, "contexts": [{"id": "document 1", "text": "This is a sample document 1."}]}], "tags": {"metadata": {"label_model": ["mistralai/mixtral-8x7b-instruct-v01"]}, "input_quality": [{"mistralai/mixtral-8x7b-instruct-v01": "poor"}], "input_quality_explanation": [{"mistralai/mixtral-8x7b-instruct-v01": "The user query lacks coherence. The user mentions 'two documents' and 'common goal' but doesn't specify how they want the two documents and common goal to be analyzed or compared."}], "judge_quality_score": [{"mistralai/mixtral-8x7b-instruct-v01": "4"}], "judge_quality_explanation": [{"mistralai/mixtral-8x7b-instruct-v01": "The given instruction is diligently followed by the AI assistant as it thoroughly addresses the instruction by describing how both documents talk about the goal-achievement process through Step-by-Step Reaching and Prioritization. The answer highlights accuracy with direct-quote supporting evidences from the documents, and an easy-to-understand presentation put together in a concise structured response."}], "intent": [{"mistralai/mixtral-8x7b-instruct-v01": "The user wants to understand the commonality of the goal between the two given documents and how they describe this goal and the process of achieving it"}], "knowledge": [{"mistralai/mixtral-8x7b-instruct-v01": "To solve this problem, the model needs to understand the context of the two documents and the semantics of the user query"}], "difficulty": [{"mistralai/mixtral-8x7b-instruct-v01": "medium"}], "task_category": [{"mistralai/mixtral-8x7b-instruct-v01": "Information seeking"}]}}
```

> **IMPORTANT**
> known issues
>
> - By default, `magpie` uses `faiss-cpu` for identifying and marking duplicates. We observed performance and scaling issues for `faiss-cpu` library on Apple Silicons (M1 - M4). For large scale job, we recommend running on virutal machines with NVidia GPUs and `faiss-gpu` package.
>
> - The default value of `DEFAULT_MAX_GEN_REQUESTS` is set to 10 in the [base databuilder](). We recommend adding `--max-gen-requests 50` to command line arguments when `num-outputs-to-generate` is set to more than 50.


## How does RAG pipeline work?

The default RAG [pipeline](https://github.ibm.com/conversational-ai/fms-dgt-internal/tree/develop/src/databuilders/rag) operates in three stages viz. 1) Retrieval, 2) Generation, and 3) Validation.

### Retrieval 
In the first stage, it loads JSON file containing collection of documents to a simple in-memory retriever ([`JSONRetriever`](https://github.ibm.com/conversational-ai/fms-dgt-internal/blob/develop/src/blocks/retrievers/json_retriever.py)). 

The JSON file format
```json
[
    {
        "id": "document 1",
        "text": "This is a sample document 1."
    },
    {
        "id": "document 2",
        "text": "This is a sample document 2."
    }
]
```

Each JSON object represent a single document containing mandatory `id` and `text` fields.

### Generation
The RAG pipeline supports generating 5 different types of Question-Answer (QA) pair

- `Direct` - Simple questions that are easily answerable from documents. This includes question types such as factoid, binary, instructional, etc.
- `Keyword` - Underspecified short questions that are commonly asked by users who associate with chatbots.
- `Comparative` - Questions that ask the agent to compare/contrast two entities or events found in the document.
- `Unanswerable` - In-domain questions that are unanswerable from a given document.
- `Chit-chat` - Conversational, greeting kind of questions. For example, "How are you?", "How was your day today?" etc. 

Each of these different QA types are controlled by custom generation prompts available [here](https://github.ibm.com/conversational-ai/fms-dgt-internal/tree/develop/src/databuilders/rag/templates).


### Validation
The RAG pipeline uses quality assurance techniques proposed in this [publication](https://arxiv.org/abs/2406.08464) to validate generated QA pairs. First, it rates questions for clarity, specificity, coherence and difficulty and answer for correctness and presentation. Then it identifies semantically similar QA pairs. Lastly it preserves hight quality and distinct QA pairs.

For more information, look into [`MagpieTagger`](https://github.ibm.com/conversational-ai/fms-dgt-internal/tree/develop/src/blocks/generators/magpie/tag) and [`MagpieDistance`](https://github.ibm.com/conversational-ai/fms-dgt-internal/blob/develop/src/blocks/postprocessors/magpie/distance.py) blocks. 


## ❓ Frequently Asked Questions (FAQs)

### 1. How do I change Large Language Model (LLM) inference service used in `generation` and `validation` stages?

The `type` field for `generator` block and `lm_config -> type` for `tagger` block in the [`rag.yaml`](https://github.ibm.com/conversational-ai/fms-dgt-internal/blob/develop/src/databuilders/rag/rag.yaml) controls LLM inference service is used. 

### 2. How can I run RAG pipeline on a custom set of documents?

As explained in [`Retrieval`](https://pages.github.ibm.com/conversational-ai/dgt-website/examples/rag/#retrieval) section, you can create a custom collection of documents in the specified JSON format and point to it via `path` field for the `retriever` block in the [`rag.yaml`](https://github.ibm.com/conversational-ai/fms-dgt-internal/blob/develop/src/databuilders/rag/rag.yaml).

