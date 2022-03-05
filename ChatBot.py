"""
Assignment 3 - Chatbot
"""

# pip install pandas
# pip install numpy

import pandas as pd
import numpy as np


def load_dataset(filename):
    """ The first column contains the diseases, and each column after contains the symptoms associated with it """
    dataset = pd.read_csv(filename, low_memory=False)
    return dataset


def load_symptom_files():
    """ Read in the symptom csv files """
    # contains the description of each disease
    symptom_desc = pd.read_csv("symptom_Description.csv", header=None)

    # contains the precautions for each disease
    symptom_precaution = pd.read_csv("symptom_precaution.csv", header=None)

    # contains the severity of symptoms
    symptom_severity = pd.read_csv("Symptom_severity.csv", header=None)
    return symptom_desc, symptom_precaution, symptom_severity


if __name__ == '__main__':
    # Load all our files
    dataset = load_dataset("dataset.csv")
    desc, precaution, severity = load_symptom_files()

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

    symptom_option_str = matches.iloc[num_option][0]
    # input the duration as an integer in days
    print(f"I see, how long have you been experiencing {symptom_option_str}?")
    symptom_duration_days = int(input())

    # TODO: find the most probable disease with the given symptom,
    #  check the rest of the symptoms related to that disease
