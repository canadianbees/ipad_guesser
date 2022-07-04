# Celina Alzenor
# ipad guesser game
# 2022

import os
import time

import graphviz
import pandas as pd  # load and manipulate data, used for one-hot encoding
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split  # to split data into training and testing sets
from sklearn.tree import DecisionTreeClassifier  # used to build classification tree

os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'


# 0 = no, 1 = yes
def read(ftype):
    dataset = pd.read_csv('dataset.csv')
    if ftype == "report":
        print(dataset.head())
    # one hot encoding
    dataset = pd.get_dummies(data=dataset, columns=['Age', 'Gender', 'Race', 'outside US', 'employment status',
                                                    'student status', 'major', 'stem status', 'other apple products',
                                                    'ipad'])
    dataset.drop('ipad_No', inplace=True, axis=1)
    return dataset


def split(dataset):
    # X represents all the other variables given, Y represents what will be predicted
    X = dataset.values[:, :-1]
    Y = dataset.values[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.70, random_state=100)

    return X, Y, X_train, X_test, y_train, y_test


def training(x_train, y_train, dataset, ftype):
    classify = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=4, min_samples_leaf=5)
    classify.fit(x_train, y_train)
    if ftype == "report": # if user selected to see the report, visualize the tree
        visualize(dataset, classify, "gini")
    return classify


def prediciton(X_test, obj, ftype):
    y_pred = obj.predict(X_test)
    predictions = []

    for entries in y_pred:
        if entries == 0:
            predictions.append("No")
        else:
            predictions.append("Yes")

    if ftype == "report":
        print("Predictions: ", predictions)

    return y_pred


def accuracy(y_test, y_pred):
    confusion = confusion_matrix(y_test, y_pred)
    print("Number of times I predicted yes and they do have an ipad:", confusion[0, 0])
    print("Number of times I predicted no and they do not have an ipad:", confusion[0, 1])
    print("Number of times I predicted yes and they do not have an ipad:", confusion[1, 0])
    print("Number of times I predicted no and they do have an ipad:", confusion[1, 1])

    print("Accuracy: ",
          accuracy_score(y_test, y_pred) * 100, "\n")
    print("Report: \n",
          classification_report(y_test, y_pred, target_names=['ipadless', 'ipadful']))


def visualize(data, obj, obj_name):
    features = data.columns
    features = features[:-1]
    class_names = list(set(data.iloc[:, -1]))

    for i in range(len(class_names)):
        if class_names[i] == 0:
            class_names[i] = "No"
        else:
            class_names[i] = "Yes"

    dot_data = tree.export_graphviz(obj, out_file=None, feature_names=features, class_names=class_names,
                                    filled=True, rounded=True, special_characters=True)
    graph = graphviz.Source(dot_data)
    print("Open pdf render of the decision tree? (Y/N)")
    if answer(input(">>"), "Y/N") == "Yes":
        graph.render('dtree_render_' + obj_name, view=True)
    else:
        return


def ask_input():
    print("What is your age? (numerical value)")
    age = answer(input(">> "), "age")

    print("What do you idenity as?")
    print("1. Man")
    print("2. Woman")
    print("3. Non-Binary")
    print("4. Transgender Man")
    print("5. Transgender Woman")
    print("6. Other")
    gen = answer(input(">> "), "gen")

    print("What is your race/ethnicity?")
    print("1. American Indian or Alaska Native")
    print("2. Asian")
    print("3. Black or African American")
    print("4. Hispanic or Latino or Spanish Origin of any race")
    print("5. Native Hawaiian or Other Pacific Islander")
    print("6. White")
    print("7. Two or more races")
    race = answer(input(">> "), "race")

    print("Have you ever been outside of the U.S")
    print_yn()
    travel = answer(input(">> "), "Y/N")

    print("Are you employed")
    print_yn()
    emp = answer(input(">> "), "Y/N")

    print("Are you a student? (University, High school, etc.)")
    print_yn()
    stu = answer(input(">> "), "Y/N")

    print("What is your major? (If you don't have one please put N/A, If you are a double major please put one)")
    major = answer(input(">> "), "N/A")

    print("Are you considered a STEM student?")
    print_yn()
    print("3. Not in University")
    stem = answer(input(">> "), "stem")

    print("Do you have any Apple products other than an IPad (Airpods, IPhone, Apple Watch, etc.)")
    print_yn()
    apple = answer(input(">> "), "Y/N")

    df = pd.DataFrame(data=[[age, gen, race, travel, emp, stu, major, stem, apple]], columns=['Age', 'Gender', 'Race',
                                                                                              'outside US',
                                                                                              'employment status',
                                                                                              'student status', 'major',
                                                                                              'stem status',
                                                                                              'other apple products'])
    df = pd.get_dummies(data=df,
                        columns=['Age', 'Gender', 'Race', 'outside US', 'employment status', 'student status', 'major',
                                 'stem status', 'other apple products'])
    return df


def answer(choice, qtype):
    # yes no question
    if qtype == "Y/N":
        if choice == "1" or choice == "Y":
            return "Yes"
        elif choice == "2" or choice == "N":
            return "No"
        else:
            return invalid("Y/N")

    # race question
    elif qtype == "race":
        if choice == "1":
            return "American Indian or Alaska Native"
        elif choice == "2":
            return "Asia"
        elif choice == "3":
            return "Black or African American"
        elif choice == "4":
            return "Hispanic or Latino or Spanish Origin of any race"
        elif choice == "5":
            return "Native Hawaiian or Other Pacific Islander"
        elif choice == "6":
            return "White"
        elif choice == "7":
            return "Two or more races"
        else:
            return invalid("race")

    # stem question
    elif qtype == "stem":
        if choice == "1":
            return "Yes"
        elif choice == "2":
            return "No"
        elif choice == "3":
            return "Not in University"
        else:
            return invalid("stem")

    # gender question
    elif qtype == "gen":
        if choice == "1":
            return "Man"
        elif choice == "2":
            return "Woman"
        elif choice == "3":
            return "Non-Binary"
        elif choice == "4":
            return "Transgender Man"
        elif choice == "5":
            return "Transgender Woman"
        elif choice == "6":
            return "Other"
        else:
            return invalid("gen")

    elif qtype == "age":
        if choice.strip().isdigit():
            return choice
        else:
            return invalid("age")
    elif qtype == "N/A":
        print("Your major is:", choice, "\nIs that correct? (Y/N)")
        if answer(input(">>"), "Y/N") == "Yes":
            return choice
        else:
            print("What is your major? (If you don't have one please put N/A, If you are a double major please put one)")
            return answer(input(">> "), "N/A")

def print_yn():
    print("1. Yes")
    print("2. No")


def invalid(qtype):
    print("Invalid choice. Please choose an option")

    if qtype == "Y/N":
        while 1:
            choice = input(">> ")
            if choice == "1" or choice == "2" or choice == "Y" or choice == "N":
                return choice
            else:
                print("Invalid choice. Please choose an option")

    elif qtype == "stem":
        while 1:
            choice = input(">> ")
            if choice == "1" or choice == "2" or choice == "3":
                return choice
            else:
                print("Invalid choice. Please choose an option")
    elif qtype == "gen":
        while 1:
            choice = input(">> ")
            if choice == "1" or choice == "2" or choice == "3" \
                    or choice == "4" or choice == "5" or choice == "6":
                return choice
            else:
                print("Invalid choice. Please choose an option")
    elif qtype == "race":
        while 1:
            choice = input(">> ")
            if choice == "1" or choice == "2" or choice == "3" or choice == "4" \
                    or choice == "5" or choice == "6" or choice == "7":
                return choice
            else:
                print("Invalid choice. Please choose an option")

    # age case
    else:
        while 1:
            choice = input(">> ")
            if choice.strip().isdigit():
                return choice
            else:
                print("Invalid choice. Please choose an option")


def menu():
    print("ipad guesser: main menu\n")
    print("1. Lets play!\n")
    print("2. See report.\n")
    print("3. Exit\n")
    choice = input(">>")

    while 1:

        if choice == "1":
            game()
            break

        elif choice == "2":
            report("report")
            break

        elif choice == "3":
            print("Exiting...")
            break

        else:
            print("Invalid choice")
            choice = input(">>")


def report(ftype):
    if ftype == "report":
        data = read("report")
        X, Y, X_train, X_test, y_train, y_test = split(data)
        clf_gini = training(X_train, y_train, data, "report")
        accuracy(y_test, prediciton(X_test, clf_gini, "report"))
        backtomenu()

    elif ftype == "game":
        data = read("")
        X, Y, X_train, X_test, y_train, y_test = split(data)
        clf_gini = training(X_train, y_train, data, "")
        info = ask_input()
        info = info.reindex(labels=data.columns, axis=1, fill_value=0)
        info.drop('ipad_Yes', inplace=True, axis=1)
        guess = clf_gini.predict(info.values)
        return guess


def game():
    guess = report("game")
    print("DRUMROLL PLEASE!")
    time.sleep(2)
    first = True
    confirm = ""

    while 1:
        if guess == 0:
            if first:
                print("You do not have an ipad!\nWas I correct? (Y/N)")
                confirm = answer(input(">>"), "Y/N")

            if confirm == "Yes":
                print("♪┏ ( ･o･) ┛♪ ┗ (･o･ ) ┓♪  ┏(･o･)┛ ♪ ᕕ(⌐■_■)ᕗ  ♪♬")
                break
            elif confirm == "No":
                print("(▰˘︹˘▰)")
                break
            else:
                confirm = invalid("Y/N")
                first = False
        else:
            if first:
                print("You do have an ipad!\nWas I correct? (Y/N)")
                confirm = answer(input(">>"), "Y/N")

            if confirm == "Yes":
                print("♪┏ ( ･o･) ┛♪ ┗ (･o･ ) ┓♪  ┏(･o･)┛ ♪ ᕕ(⌐■_■)ᕗ  ♪♬")
                break
            elif confirm == "No":
                print("(▰˘︹˘▰)")
                break
            else:
                confirm = invalid("Y/N")
                first = False

    backtomenu()


def backtomenu():
    print("Back to menu? (Y/N)")
    choice = answer(input(">>"), "Y/N")

    if choice == "Yes":
        menu()
    else:
        print("Exiting...")
        exit(0)


def main():
    menu()


main()
