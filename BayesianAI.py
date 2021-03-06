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

        self.predicted_disease = ()
        """ A tuple containing the predicted disease and its confidence rate."""

        self.symptoms_to_ask = []
        """ The symptoms we are asking the user. """

        self.finished = False
        """ Whether we are finished asking questions. """

        self.found_disease = False
        """ Whether we have narrowed down a likely disease. """

        self.symptoms_asked = {}
        """ A list tracking symptoms that have been asked and whether the user has it. 
        Key: Symptom Name. Value: Boolean for whether the user has it."""

        self.possible_diseases = {}
        """ Tracks the possible diseases and their probability. """

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
        # store the first symptom as confirmed symptoms
        self.symptoms_asked[symptom] = True

        # fill out the possible diseases.
        for disease in self.disease_prob_dist.values: # type: ignore
            self.possible_diseases[disease] = self.disease_prob_dist[disease] # type: ignore

        self.update_possible_diseases(symptom, True)

    def update_possible_diseases(self, symptom, hasSymptom: bool):
        for disease in self.possible_diseases: # type: ignore
            # multiply the probability into the disease prob. in the dict
            if hasSymptom:
                self.possible_diseases[disease] *= self.disease_symptom_joint_prob_dist[symptom, disease]
            else:
                # opposite => we want to get the diseases that DON'T have the symptom
                # thus, since the joint prob dist has the probs for having the symptom, we
                # need to manually 1 - that value
                self.possible_diseases[disease] *= (1 - self.disease_symptom_joint_prob_dist[symptom, disease])

        # remove the zeroes
        self.possible_diseases = {k: v for k, v in self.possible_diseases.items() if v != 0}

        most_likely_disease = max(self.possible_diseases, key=self.possible_diseases.get) # type: ignore
        symps_of_disease = self.disease_symptom_counter[most_likely_disease]
        self.symptoms_to_ask = [symp for symp in symps_of_disease if symp not in self.symptoms_asked and symp != "total"]

        if len(self.symptoms_to_ask) == 0:
            # no more questions to ask => we are truly done
            self.finished = True

        elif len(self.possible_diseases.keys()) == 0:
            self.finished = True

    def get_symptom_to_ask(self):
        """
        Return the last symptom in the symptoms to ask list.
        This is so we can just pop it out.
        """
        return self.symptoms_to_ask[-1]

    def give_symptom_answer(self, ans: bool):
        """
        Give the answer from the user.
        """
        # remove the symptom from the list
        symptom = self.symptoms_to_ask.pop()
        self.symptoms_asked[symptom] = ans
        self.update_possible_diseases(symptom, ans)

    def get_most_likely_disease(self, desc: pd.DataFrame, precautions: pd.DataFrame) -> str:
        # get the disease and confidence score
        # two cases for finishing: 0 or 1 disease found
        if len(self.possible_diseases) == 0:
            txt = "I'm not sure what disease you've gotten based on the symptoms you gave me.\nPlease retry and ensure your answers are correct."
            return txt

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
        """
        Calculate the severity of the disease given the confirmed symptoms.
        """
        severity_sum = 0
        confirmed_symptoms = [symptom for symptom, hasIt in self.symptoms_asked.items() if hasIt]
        size = len(confirmed_symptoms) + 1

        for symptom in confirmed_symptoms:
            row = severity.loc[(severity.iloc[:, 0] == symptom)]
            severity_sum += int(row[1]) # type: ignore

        sickness_severity = (severity_sum * days) / size
        return sickness_severity

