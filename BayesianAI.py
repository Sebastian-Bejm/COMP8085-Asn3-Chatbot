import pandas as pd
import numpy as np
from ProbDist import *


class BayesianAI:
    def __init__(self) -> None:
        self.disease_prob_dist = ProbDist("Disease")
        """
        The probability distribution of diseases.
        """

        self.disease_symptom_joint_prob_dist = JointProbDist(["disease", "symptom"])
        """
        The joint probability distribution (table) of diseases and symptoms.
        """

        self.finished = False
        """ Whether we have finished asking questions. """

        self.candidates = []
        """ Track the possible candidates here. """
        
        self.symp_to_ask = ""
        """ Track the current symptom that we are asking. """

    def build_model(self, data: pd.DataFrame, desc: pd.DataFrame, precaution: pd.DataFrame, severity: pd.DataFrame):
        """
        Build the model based on the data passed in constructor.
        """
        # data = prior info about diseases and the symptoms that they have, Columns = disease name, symptoms (variable len)
        # desc = description of the diseases. Columns = disease name, description
        # precaution = precaution of the diseases. Columns = disease name, precautions (variable len)
        # severity = severity of the symptoms. Columns = symptom name, severity as an int
        
        # steps:
        # 1. Find prob dist of diseases
        # 2. Create joint prob dist of diseases and symptom
        # 3. Keep track of count of diseases and total of all diseases
        pass

    def give_first_symptom(self, symptom: str, days: int):
        """
        Give the first symptom to the bot so it can begin
        deducing possible candidates and symptom to ask.
        """
        # search through the diseases and find candidates.
        # pick a symptom, save it
        pass

    def give_symptom_answer(self, answer: bool):
        # search through the candidate based on the answer.
        # pick a symptom, save it
        pass
