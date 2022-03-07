import pandas as pd
import numpy as np


class BayesianAI:
    def __init__(self, data: pd.DataFrame, desc: pd.DataFrame, precaution: pd.DataFrame, severity: pd.DataFrame) -> None:
        self.data = data
        self.desc = desc
        self.precaution = precaution
        self.severity = severity

        self.found = False
        """ Whether we have found the disease. """

        self.candidates = []
        """ Track the possible candidates here. """
        
        self.cur_symptom = ""
        """ Track the current symptom that we are asking. """

    def build_model(self):
        """
        Build the model based on the data passed in constructor.
        """
        pass

    def give_first_symptom(self, symptom: str, days: int) -> str:
        """
        Give the first symptom to the bot so it can begin
        deducing possible candidates and symptom to ask.
        """
        # searching through the diseases and find candidates.
        # pick a symptom, save it, then ask the user whether
        # they have that symptom
        return self.cur_symptom


    def give_symptom_answer(self, answer: bool) -> str:
        # search through the candidate based on the answer.
        # pick a symptom, save it, then ask the user whether
        # they have that symptom
        return self.cur_symptom
