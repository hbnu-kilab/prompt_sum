from tqdm import tqdm
from .data_loader_interface import DataLoaderInterface

class SummaryLoader(DataLoaderInterface):
    def __init__(self, 
                 file_system: DataLoaderInterface):
        self.file_system = file_system
    
    def load(self, json_lst, **kwargs):
        return self.file_system.load(json_lst, **kwargs)

class SummarySDSCLoader(DataLoaderInterface):
    def __init__():
        pass
    
    def load(json_file):
        pass

class SummaryAIHubNewsLoader(DataLoaderInterface):
    def __init__():
        pass
    
    def load(json_file):
        src_lst, sum_lst = [], []
        for doc in json_file["documents"]:
            title = doc["title"]
            src = []
            for text_el_lst in doc["text"]:
                for text_el in text_el_lst:
                    src.append(text_el["sentence"])
            
            src = ' '.join(src)
            ex_sum = doc["extractive"]
            sum = ' '.join(doc["abstractive"])
            
            src_lst.append(src)
            sum_lst.append(sum)

        return src_lst, sum_lst



class SummarySBSCLoader(DataLoaderInterface):
    def __init__():
        pass

    def load(json_lst):
        src_lst, sum_lst = [], []
        for json_doc in tqdm(json_lst, total=len(json_lst), desc="load json"):
            for doc in json_doc['document']:
                for issue_sum in doc['SC']['issue_summary']:
                    topic = issue_sum['issue']['topic']
                    ab_summary = issue_sum['summary']['abstract']['form']
                    ref_lst = issue_sum['summary']['abstract']['reference']

                    src_sents = []
                    for ref_id in ref_lst:
                        for sent in doc['sentence']:
                            if ref_id == sent['id']:
                                src_sents.append(sent['form'])
                                break

                    src_lst.append(' '.join(src_sents))
                    sum_lst.append(ab_summary)
                
        return src_lst, sum_lst


class SummaryETRILoader(DataLoaderInterface):
    def __init__(self) -> None:
        super().__init__()
    
    def load(json_lst):
        ex_sent_lst = []

        for json_doc in tqdm(json_lst, total=len(json_lst), desc="load json"):
            dial_dict = {dial["sentence_id"]: dial for dial in json_doc["dialogue"]}
            
            total_ex_dials_lst = []
            for total in json_doc["total_summary"]:
                extracted_dial_lst = [dial_dict[ids] for ids in total["total_sentence_ids"]]
                total_ex_dials_lst.append(extracted_dial_lst)
            
            ex_sent_lst.append(total_ex_dials_lst)
        
        return ex_sent_lst