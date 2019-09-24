import numpy as np
from utils import *
from solvers import *
import time
import warnings


class Evaluation(object):

    def __init__(self, train_path="public_set/train",
                 test_path="public_set/test",
                 score_path="data/evaluation/scoring.json"):
        self.train_path = train_path
        self.test_path = test_path
        self.score_path = score_path
        self.secondary_score = read_config(self.score_path)["secondary_score"]
        self.test_scores = []
        self.first_scores = []
        self.secondary_scores = []
        self.classifier = classifier.Solver()
        self.solvers = [
            solver1.Solver(),
            solver2.Solver(),
            solver3.Solver(),
            solver4.Solver(),
            solver5.Solver(),
            solver6.Solver(),
            solver7.Solver(),
            solver8.Solver(),
            solver9.Solver(),
            solver10.Solver(),
            solver10.Solver(),
            solver10.Solver(),
            solver13.Solver(),
            solver14.Solver(),
            solver15.Solver(),
            solver16.Solver(),
            solver17.Solver(train_size=0.9),
            solver17.Solver(train_size=0.85),
            solver17.Solver(train_size=0.85),
            solver17.Solver(train_size=0.85),
            solver21.Solver(),
            solver22.Solver(),
            solver23.Solver(),
            solver24.Solver(),
            solver25.Solver(),
            solver26.Solver()
        ]
        self.time_limit_is_ok = True
        time_limit_is_observed = self.solver_fitting()
        if time_limit_is_observed:
            print("Time limit of fitting is OK")
        else:
            self.time_limit_is_ok = False
            print("TIMEOUT: Some solvers fit longer than 10m!")
        self.clf_fitting()

    def solver_fitting(self):
        time_limit_is_observed = True
        for i, solver in enumerate(self.solvers):
            start = time.time()
            solver_index = i + 1
            train_tasks = load_tasks(self.train_path, task_num=solver_index)
            if hasattr(solver, "load"):
                print("Loading Solver {}".format(solver_index))
                solver.load("data/models/solver{}.pkl".format(solver_index))
            else:
                print("Fitting Solver {}...".format(solver_index))
                solver.fit(train_tasks)
            duration = time.time() - start
            if duration > 60:
                time_limit_is_observed = False
                print(
                    "Time limit is violated in solver {} which "
                    "has been fitting for {}m {:2}s".format(
                        solver_index, int(duration // 60), duration % 60))
            print("Solver {} is ready!\n".format(solver_index))
        return time_limit_is_observed

    def clf_fitting(self):
        tasks = []
        for filename in os.listdir(self.train_path):
            if filename.endswith(".json"):
                data = read_config(os.path.join(self.train_path, filename))
                tasks.append(data)
        print("Fitting Classifier...")
        self.classifier.fit(tasks)
        print("Classifier is ready!")
        return self

    def get_score(self, y_true, prediction):
        if y_true == prediction:
            return 1
        return 0

    def get_matching_score(self, y_true, pred):
        score = 0
        if len(y_true) != len(pred):
            return 0
        for key in y_true.keys():
            if y_true[key] == pred.get(key):
                score += 1
        return score

    def get_multiple_score(self, y_true, pred):
        score = 0
        for y in y_true:
            for p in pred:
                if y == p:
                    score += 1
        return score

    def variant_score(self, variant_scores):
        first_score = sum(variant_scores)
        mean_score = round(np.mean(variant_scores), 3)
        secondary_score = int(self.secondary_score[str(first_score)])
        scores = {"first_score": first_score, "mean_accuracy": mean_score, "secondary_score": secondary_score}
        self.first_scores.append(first_score)
        self.secondary_scores.append(secondary_score)
        return scores

    def get_overall_scores(self):
        overall_scores = {}
        for variant, variant_scores in enumerate(self.test_scores):
            scores = self.variant_score(variant_scores)
            print("***YOUR RESULTS***")
            print("Variant: {}".format(variant + 1))
            print("Scores: {}\n".format(scores))
            overall_scores[str(variant + 1)] = scores
        self.overall_scores = overall_scores
        return self

    def predict_from_baseline(self):
        time_limit_is_observed = True
        for filename in os.listdir(self.test_path):
            predictions = []
            print("Solving {}".format(filename))
            data = read_config(os.path.join(self.test_path, filename))[:-1]
            task_number = self.classifier.predict(data)
            for i, task in enumerate(data):
                start = time.time()
                task_index, task_type = i + 1, task["question"]["type"]
                print("Predicting task {}...".format(task_index))
                prediction = self.solvers[
                    task_number[i] - 1].predict_from_model(task)
                if task_type == "matching":
                    y_true = task['solution']['correct']
                    score = self.get_matching_score(y_true, prediction)
                elif task_index == 16:
                    y_true = task["solution"]["correct_variants"][
                        0] if "correct_variants" in task["solution"] \
                        else task["solution"]["correct"]
                    score = self.get_multiple_score(y_true, prediction)
                else:
                    y_true = task["solution"]["correct_variants"][
                        0] if "correct_variants" in task["solution"] \
                        else task["solution"]["correct"]
                    score = self.get_score(y_true, prediction)
                print("Score: {}\nCorrect: {}\nPrediction: {}\n".format(score,
                                                                        y_true,
                                                                        prediction))
                predictions.append(score)
                duration = time.time() - start
                if duration > 60:
                    time_limit_is_observed = False
                    self.time_limit_is_ok = False
                    print("Time limit is violated in solver {} "
                          "which has been predicting for {}m {:2}s".format(
                            i+1, int(duration // 60), duration % 60))
            self.test_scores.append(predictions)
        return time_limit_is_observed


def main():
    warnings.filterwarnings("ignore")
    evaluation = Evaluation()
    time_limit_is_observed = evaluation.predict_from_baseline()
    if not time_limit_is_observed:
        print('TIMEOUT: some solvers predict longer then 60s!')
    evaluation.get_overall_scores()

    mean_first_score = np.mean(evaluation.first_scores)
    mean_secondary_score = np.mean(evaluation.secondary_scores)
    print("Mean First Score: {}".format(mean_first_score))
    print("Mean Secondary Score: {}".format(mean_secondary_score))

    if evaluation.time_limit_is_ok:
        print("Time limit is not broken by any of the solvers.")
    else:
        print("TIMEOUT: Time limit by violated in some of the solvers.")


if __name__ == "__main__":
    main()
