INSTRUCTIONS = '''
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
'''

PROMPT_TEMPLATE = '''
QUESTION: {question}

CONTEXT:
{context}
'''.strip()


class RAGBase:

    def __init__(
        self,
        index,
        client,
        instructions=INSTRUCTIONS,
        prompt_template=PROMPT_TEMPLATE,
        model="llama3-8b-8192"
    ):
        self.index = index
        self.client = client
        self.instructions = instructions
        self.prompt_template = prompt_template
        self.model = model

    # -------------------------
    # 1) SEARCH (tu MyRAG lo sobrescribe)
    # -------------------------
    def search(self, query, num_results=5):
        return self.index.search(query, num_results=num_results)

    # -------------------------
    # 2) BUILD CONTEXT (adaptado a filename/content)
    # -------------------------
    def build_context(self, search_results):
        parts = []
        for doc in search_results:
            parts.append(f"Filename: {doc['filename']}\nContent:\n{doc['content']}\n")
        return "\n---\n".join(parts)

    # -------------------------
    # 3) BUILD PROMPT
    # -------------------------
    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)
        return self.prompt_template.format(
            question=query,
            context=context
        )

    # -------------------------
    # 4) LLM CALL (Groq)
    # -------------------------
    def llm(self, prompt: str):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.instructions},
                {"role": "user", "content": prompt}
            ]
        )
        return response  # devolvemos el objeto completo

    # -------------------------
    # 5) RAG PIPELINE
    # -------------------------
    def rag(self, query: str):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        response = self.llm(prompt)

        text = response.choices[0].message["content"]
        usage = response.usage  # prompt_tokens, completion_tokens, total_tokens

        return text, usage
