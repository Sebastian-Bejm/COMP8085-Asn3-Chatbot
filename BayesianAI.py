from typing import Union
import pandas as pd
import numpy as np
from ProbDist import *


class BayesianAI:
    def __init__(self) -> None:
        
        self.dataset: pd.DataFrame = None
        """ The full dataset. """

        self.disease_prob_dist: Union[ProbDist, None] = None
        """
        The probability distribution of diseases.
        """

        self.disease_symptom_joint_prob_dist = JointProbDist(["Symptom", "Disease"])
        """
        The joint probability distribution (table) of diseases and symptoms.
        """

        self.finished = False
        """ Whether we have finished asking questions. """

        self.candidates = []
        """ Track the possible candidates here. """

        self.disease_potential_symptoms = {}
        """ A dictionary tracking the next symptoms of a predicted disease """

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
        self.dataset = data

        # steps:
        # 1. Find prob dist of diseases
        # 2. Create joint prob dist of diseases and symptom
        # 3. Keep track of count of diseases and total of all diseases
        DISEASE_COL = 0

        # how many diseases are in the population
        disease_counter = {}

        # track how common is a symptom for a disease
        # key = disease, val = {} where key = symptom name (and "total") and val = count
        disease_symptom_counter = {}

        for row in data.iterrows():
            disease = row[1][DISEASE_COL]
            # use .get() so we can use default values
            count = disease_counter.get(disease, 0)
            disease_counter[disease] = count + 1

            # loop through rest of the columns to get the symptom
            for symptom in row[1][1:]:
                if pd.isna(symptom):
                    break

                symptom = symptom.strip()
                # symptom_count is a dict containing key = symptom, val = counter
                # and also tracks 'total' amount of symptoms
                symptom_count = disease_symptom_counter.get(disease, {})
                count = symptom_count.get(symptom, 0)
                symptom_count[symptom] = count + 1

                # also count the total symptom for each disease
                total = symptom_count.get("total", 0)
                symptom_count["total"] = total + 1

                # resave the symptom count
                disease_symptom_counter[disease] = symptom_count

        self.disease_prob_dist = ProbDist("Disease", disease_counter)

        for disease, symptom_count in disease_symptom_counter.items():
            total = symptom_count["total"]
            for symptom, count in symptom_count.items():
                if symptom == "total":
                    continue

                self.disease_symptom_joint_prob_dist[symptom, disease] = count / total


    def give_first_symptom(self, symptom: str):
        """
        Give the first symptom to the bot so it can begin
        deducing possible candidates and symptom to ask.
        """
        # search through the diseases and find candidates.
        # pick a symptom, save it

        prob_disease_dict = {}
        for disease in self.disease_prob_dist.values:
            prob_disease_dict[disease] = self.disease_prob_dist[disease]

            # multiply the probability into the disease prob. in the dict
            prob_disease_dict[disease] *= self.disease_symptom_joint_prob_dist[symptom, disease]

        # max_prob = max(prob_disease_dict.values())
        max_prob_key = max(prob_disease_dict, key=prob_disease_dict.get)

        # this list of symptoms excludes the initial symptom
        potential_symptoms = []

        # after getting the probability of the disease,
        # iterate through the all possible symptoms of that disease in the dataset
        for row in self.dataset.iterrows():
            if row[1][0] == max_prob_key:
                for next_symptom in row[1][1:]:
                    if not pd.isna(next_symptom):
                        if next_symptom not in potential_symptoms and next_symptom != symptom:
                            potential_symptoms.append(next_symptom)

        # create a dict with those symptoms and boolean values if they are experiencing them
        for s in potential_symptoms:
            self.disease_potential_symptoms[s] = False


    def calc_sickness_severity(self, severity: pd.DataFrame, days: int):
        # calculate the severity of the disease
        # ask if the first symptom needs to be included when calculating this

        size = len(self.disease_potential_symptoms)
        severity_sum = 0

        for key in self.disease_potential_symptoms:
            # check if the next symptom was experienced (True)
            if self.disease_potential_symptoms[key] is True:
                n_key = key.replace(" ", "") # NOTE: STRIP ALL WHITESPACES AT THE START
                row = severity.loc[(severity.iloc[:, 0] == n_key)]
                severity_sum += int(row[1])

        sickness_severity = (severity_sum * days) / size
        return sickness_severity
