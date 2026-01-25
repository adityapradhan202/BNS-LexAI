from datetime import date
import json

class Utility:

    @staticmethod
    def initial_json_file(path:str) -> None:
        """Initializes or clears an existing json file before logging anything. Make sure to create an empty json file at path same as the 'path' parmeter.
        Args:
            path(str): path of the json file
        Returns:
            status(str): Status message
        """
        user_inp = input("WARNING! If this path already contains valuable information, then this function will wipe out the data completely and re-initializes the json log file! Enter Y to proceed, enter N to cancel: ")
        if user_inp.lower() == "y":
            empty_dict = {}
            try:
                with open(path, 'w') as file:
                    json.dump(empty_dict, file, indent=4)
                    print(f"Succesfully initialized/emptied the json file at path - {path}")
            except Exception as e:
                print(f"Exception occured: {e}")
        elif user_inp.lower() == "n":
            print("Process cancelled!")
        else:
            print("Invalid input! Enter either Y or N!")

    @staticmethod
    def log_experiment(id:str, path:str, description:str, commit_message:str,
                       faithfulness:float, factual_correctness:float,
                       answer_relevance:float, context_relevance:float) -> None:
        """Adds or update Logs the outputs of the experiments performed using the custom rag evaluation pipeline - 'rag_badger'.
        Args:
            id(str): experiment id (either new or existing one incase of update)
            path(str): path or location where the logs will be stored
        Returns:
            status(str): Status message
        """
        try:
            with open(path, 'r') as file:
                data = json.load(file)
        except Exception as e:
            print(f"Exception occured: {e}")
            return
        
        day = date.today().day
        month = date.today().month
        year = date.today().year

        data[id] = {
            "faitfulness":faithfulness,
            "factual_correctness":factual_correctness,
            "answer_relevance":answer_relevance,
            "context_relevance":context_relevance,
            "log-metadata":{
                "log-description":description,
                "log-commit-mesage":commit_message,
                "date":{'day':day, 'month':month, 'year':year},
            }
        }

        try:
            with open(path, "w") as file:
                json.dump(data, file, indent=4)
                print("Added succesfully!")
                return
        except Exception as e:
            print(f"Exception occured - {e}")

        

    @staticmethod
    def delete_log(id:str, path:str) -> None:
        try:
            with open(path, 'r') as file:
                data = json.load(file)
        except Exception as e:
            print(f"Exception occured: {e}")
            return
        
        try:
            del data[id]
        except KeyError:
            raise KeyError("No such id in the logs! You might have already deleted it!")
        try:
            with open(path, "w") as file:
                json.dump(data, file, indent=4)
        except Exception as e:
            print(f"Exception occured: {e}")
            return

        print(f"Succesfully deleted id - {id}")


if __name__ == "__main__":
    print("Runs succesfully! No errors!")