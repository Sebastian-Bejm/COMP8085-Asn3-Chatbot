from typing import Union, Tuple
import pandas as pd
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

        self.symptoms_to_ask = []
        """The symptoms we are asking the user."""

        self.finished = False
        """Whether we are finished asking questions."""

        self.possible_diseases = {}
        """Tracks the possible diseases and their probability."""


    def build_model(self, data: pd.DataFrame):
        """
        Build the model based on the data passed in constructor.
        """
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
        # fill out the possible diseases.
        for disease in self.disease_prob_dist.values: # type: ignore
            self.possible_diseases[disease] = self.disease_prob_dist[disease] # type: ignore

        # search through the diseases and find symptoms to ask for.
        self.symptoms_to_ask = []
        for symptom_counter in self.disease_symptom_counter.values():
            if symptom in symptom_counter.keys():
                for _symptom in symptom_counter:
                    if _symptom == "total" or _symptom == symptom or _symptom in self.symptoms_to_ask:
                        continue

                    self.symptoms_to_ask.append(_symptom)

        # sort it alphabetically
        self.symptoms_to_ask = sorted(self.symptoms_to_ask)
        self.symptoms_to_ask.reverse()

        self.update_possible_diseases(symptom)
        ###################################

        # # this list of symptoms excludes the initial symptom and the "total"
        # potential_symptoms = []
        # for _symptom in self.disease_symptom_counter[most_likely_disease]:
        #     if _symptom != "total" and _symptom != symptom:
        #         potential_symptoms.append(_symptom)
        

        # # create a dict with those symptoms and boolean values if they are experiencing them
        # for s in potential_symptoms:
        #     self.disease_potential_symptoms[s] = False

    def update_possible_diseases(self, symptom):
        for disease in self.possible_diseases: # type: ignore
            # multiply the probability into the disease prob. in the dict
            self.possible_diseases[disease] *= self.disease_symptom_joint_prob_dist[symptom, disease]

        # remove the zeroes
        self.possible_diseases = {k: v for k, v in self.possible_diseases.items() if v != 0}

        # check if we are done
        if len(self.possible_diseases.keys()) == 1:
            self.finished = True


    def get_symptom_to_ask(self):
        """
        Return the last symptom in the symptoms to ask list.
        This is so we can just pop it out.
        """
        if len(self.symptoms_to_ask) == 1:
            self.finished = True
        return self.symptoms_to_ask[-1]

    def give_symptom_answer(self, ans: bool):
        """
        Give the answer from the user.
        """
        # remove the symptom from the list
        symptom = self.symptoms_to_ask.pop()
        if ans:
            self.update_possible_diseases(symptom)

    def get_most_likely_disease(self, desc: pd.DataFrame, precautions: pd.DataFrame) -> str:
        # get the disease and confidence score
        most_likely_disease = max(self.possible_diseases, key=self.possible_diseases.get) # type: ignore
        prob = max(self.possible_diseases.values())
        sum_ = sum(self.possible_diseases.values())
        confidence = prob / sum_
        txt = "I'm {:.2f}% confident you are facing {}.\n\n".format(confidence * 100, most_likely_disease)

        # get description
        description = desc[desc[0] == most_likely_disease].iloc[0][1] # type: ignore
        txt += f"{most_likely_disease.capitalize()}: {description}\n\n"

        # get precautions
        disease_precautions: pd.Series = precautions[precautions[0] == most_likely_disease].iloc[0] # type: ignore
        precaution_strs = []
        # skip the first column cause that's the disease name
        for precaution_str in disease_precautions[1:]:
            precaution_strs.append(f"-{precaution_str.capitalize()}")

        txt += "Take the following precautions:\n" + "\n".join(precaution_strs)
        return txt


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

