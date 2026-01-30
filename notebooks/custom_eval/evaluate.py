from .eval_prompts import context_compare
from .data_model_classes import BetterSearchType
from typing import List, Dict
from langchain_ollama import ChatOllama

class RetrievalSearchComparision:
    def __init__(self, queries:List[str], context_type_1:List[str], context_type_2:List[str]):
        """Creates dataset object for comparision of two retrieval techniques.
        Args:
            queries: List of queries or questions
            context_type_1: List of retrieved documents from the first type of retrieval technique
            context_type_2: List of retrieved documents from the second type of retrieval technique
        """
        self.queries = queries
        self.context_type_1 = context_type_1
        self.context_type_2 = context_type_2

        len_queries = len(self.queries)
        len_context_type_1 = len(self.context_type_1)
        len_context_type_2 = len(self.context_type_2)
        if (len_queries != len_context_type_1) or (len_queries != len_context_type_2):
            raise ValueError("Length of the queries, and contexts from both the retrieval techniques must be same!")
        
    def compare_contexts(self, model:ChatOllama) -> Dict[str, int]:
        """Compares the contexts given by two retrieval techniques
        Returns:
            results: a dictionary containing the percentage of relevancy of the contexts given by two retrieval techniques
        """
        struct_model = model.with_structured_output(BetterSearchType)
        chain = context_compare | struct_model 

        types = []
        for i in range(len(self.queries)):
            print(f"Evaluating question number - {i+1}")
            obj = chain.invoke({'query':self.queries[i],
                          'context_1':self.context_type_1[i], 'context_2':self.context_type_2[i]})
            types.append(obj.type)

        cnt_1 = types.count(1)
        cnt_2 = types.count(2)
        cnt_1_p = (cnt_1 / (cnt_1 + cnt_2)) * 100
        cnt_2_p = (cnt_2 / (cnt_1 + cnt_2)) * 100
        result = {
            'retrieval_type_1':cnt_1_p,
            'retrieval_type_2':cnt_2_p
        }
        return result

if __name__ == "__main__":
    print("Runs succesfully")
        

