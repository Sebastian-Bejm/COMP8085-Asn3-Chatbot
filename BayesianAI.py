from typing import Union, Tuple
import pandas as pd
import numpy as np
from ProbDist import *


class BayesianAI:
    def __init__(self) -> None:
        self.disease_symptom_counter = {}
        """
        Tracks the symptoms per disease.
        """

        self.disease_prob_dist: Union[ProbDist, None] = None
        """
        The probability distribution of diseases.
        """

        self.disease_symptom_joint_prob_dist = JointProbDist(["Symptom", "Disease"])
        """
        The joint probability distribution (table) of diseases and symptoms.
        """

        self.disease_potential_symptoms = {}
        """ A dictionary tracking the next symptoms of a predicted disease. """

        self.predicted_disease = ()
        """ A tuple containing the predicted disease and its confidence rate."""


    def build_model(self, data: pd.DataFrame):
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
        DISEASE_COL = 0

        # how many diseases are in the population
        disease_counter = {}

        for row in data.iterrows():
            disease = row[1][DISEASE_COL]
            # use .get() so we can use default values
            count = disease_counter.get(disease, 0)
            disease_counter[disease] = count + 1

            # loop through rest of the columns to get the symptom
            for symptom in row[1][1:]:
                if pd.isna(symptom):
                    break

                # symptom_count is a dict containing key = symptom, val = counter
                # and also tracks 'total' amount of symptoms
                symptom_count = self.disease_symptom_counter.get(disease, {})
                count = symptom_count.get(symptom, 0)
                symptom_count[symptom] = count + 1

                # also count the total symptom for each disease
                total = symptom_count.get("total", 0)
                symptom_count["total"] = total + 1

                # resave the symptom count
                self.disease_symptom_counter[disease] = symptom_count

        self.disease_prob_dist = ProbDist("Disease", disease_counter)

        for disease, symptom_count in self.disease_symptom_counter.items():
            total = symptom_count["total"]
            for symptom, count in symptom_count.items():
                if symptom == "total":
                    continue

                self.disease_symptom_joint_prob_dist[symptom, disease] = count / total


    def predict(self, input: pd.DataFrame) -> list:
        """
        Predict the output label based on the input.
        Each row should yield one matching output in the list.
        """
        predictions = []
        for row in input.iterrows():
            # copy the P(Disease) so we can multiply it
            prob_disease_dict = {}
            for disease in self.disease_prob_dist.values: # type: ignore
                prob_disease_dict[disease] = self.disease_prob_dist[disease] # type: ignore

            for symptom in row[1]:
                if pd.isna(symptom):
                    break

                for disease in prob_disease_dict:
                    # multiply the probability into the disease prob in the dict
                    prob_disease_dict[disease] *= self.disease_symptom_joint_prob_dist[symptom, disease]

            # all the symptoms are processed => pick the highest value
            most_likely_disease = max(prob_disease_dict, key=prob_disease_dict.get) # type: ignore
            predictions.append(most_likely_disease)
        return predictions

    def give_first_symptom(self, symptom: str):
        """
        Give the first symptom to the bot so it can begin
        deducing possible candidates and symptom to ask.
        """
        # search through the diseases and find candidates.
        # pick a symptom, save it

        prob_disease_dict = {}
        for disease in self.disease_prob_dist.values: # type: ignore
            prob_disease_dict[disease] = self.disease_prob_dist[disease] # type: ignore

            # multiply the probability into the disease prob. in the dict
            prob_disease_dict[disease] *= self.disease_symptom_joint_prob_dist[symptom, disease]

        most_likely_disease = max(prob_disease_dict, key=prob_disease_dict.get) # type: ignore
        confidence = max(prob_disease_dict.values())
        self.predicted_disease = (most_likely_disease, confidence)

        # this list of symptoms excludes the initial symptom and the "total"
        potential_symptoms = []
        for _symptom in self.disease_symptom_counter[most_likely_disease]:
            if _symptom != "total" and _symptom != symptom:
                potential_symptoms.append(_symptom)
        

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
                n_key = key.replace(" ", "") # must replace white space for this to work
                row = severity.loc[(severity.iloc[:, 0] == n_key)]
                severity_sum += int(row[1]) # type: ignore

        sickness_severity = (severity_sum * days) / (size + 1)
        return sickness_severity

    def get_disease_and_confidence(self, desc: pd.DataFrame) -> str:
        disease, confidence = self.predicted_disease  # type: ignore
        txt = "I'm {:.2f}% confident you are facing {}.\n".format(confidence * 100, disease)
        description = desc[desc[0] == disease].iloc[0][1] # type: ignore
        txt += f"{disease.capitalize()}: {description}"
        return txt

    def get_precautions(self, precautions: pd.DataFrame) -> list:
        disease = self.predicted_disease[0]  # type: ignore
        disease_precautions: pd.Series = precautions[precautions[0] == disease].iloc[0] # type: ignore
        precaution_strs = []
        # skip the first column cause that's the disease name
        for precaution_str in disease_precautions[1:]:
            precaution_strs.append(f"-{precaution_str.capitalize()}")
        return precaution_strs
