"""
Assignment 3 - Chatbot
"""

# pip install pandas
# pip install numpy

import pandas as pd
import numpy as np

from BayesianAI import BayesianAI


def load_dataset(filename):
    """ The first column contains the diseases, and each column after contains the symptoms associated with it """
    dataset = pd.read_csv(filename, low_memory=False)
    return dataset


def load_symptom_files():
    """ Read in the symptom csv files """
    # contains the description of each disease
    symptom_desc = pd.read_csv("disease_desc.csv", header=None)

    # contains the precautions for each disease
    symptom_precaution = pd.read_csv("disease_precaution.csv", header=None)

    # contains the severity of symptoms
    symptom_severity = pd.read_csv("symptom_severity.csv", header=None)
    return symptom_desc, symptom_precaution, symptom_severity


if __name__ == '__main__':
    # Load all our files
    dataset = load_dataset("dataset.csv")
    desc, precaution, severity = load_symptom_files()

    # build the bot
    bot = BayesianAI()
    bot.build_model(dataset)

    # start asking questions
    print("Hello,")
    print("Please tell me about the first symptom you are experiencing...")
    first_symptom = str(input("Enter symptom: "))

    # Look for any initial matches
    matches = severity.loc[(severity.iloc[:, 0].str.contains(first_symptom))]

    # if there were no matches then loop until we have one
    while matches.empty:
        print("I'm sorry, but I'm facing difficulty understanding the symptom.")
        print("Can you use another word to describe your symptom?")
        first_symptom = input("Enter symptom: ")
        matches = severity.loc[(severity.iloc[:, 0].str.contains(first_symptom))]

    result_count = len(matches)
    print(f"I found {result_count} symptom names matching your symptom")
    print("Please confirm your intended option:")

    # confirm the symptom by entering the number associated
    for idx, symptom in enumerate(matches[0]):
        print(f"\t{idx}) {symptom}")
    num_option = int(input())

    symptom_option_str = str(matches.iloc[num_option][0])
    # input the duration as an integer in days
    print(f"I see, for how many days have you been experiencing {symptom_option_str.replace('_', ' ')}?")
    symptom_duration_days = int(input())

    # pass the first symptom to the bot
    bot.give_first_symptom(symptom_option_str)

    print("I see. I have a hypothesis, let me test it further.")
    print("Please answer 'y' or 'n' to the following questions:")

    # check what symptoms have been experienced
    for symptom in bot.disease_potential_symptoms:
        ans = input(f"Are you experiencing {symptom.replace('_', ' ')}? ")
        bot.disease_potential_symptoms[symptom] = (ans == "y")

    print("")
    # calculate the severity of the sickness given the symptoms
    if bot.calc_sickness_severity(severity, symptom_duration_days) > 13:
        print("You should take consultation from the doctor.")
    else:
        print("It might not be that bad but you should take precautions.")

    # TODO: get confidence score of the predicted disease
    # don't know what to use for confidence score, see 
    # https://datascience.stackexchange.com/questions/93155/naives-bayes-text-classifier-confidence-score
    disease_txt = bot.get_disease_and_confidence(desc)
    print(disease_txt)

    # TODO: get the precautions
    precautions = bot.get_precautions(precaution)
    print("\nTake the following precautions:\n" + "\n".join(precautions))
