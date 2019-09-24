import re
import random
from solvers.utils import standardize_task, AbstractSolver


class Solver(AbstractSolver):
    def __init__(self, **kwargs):
        self.known_examples = {
            "alternations": ["г..р", "з..р", "к..с", "кл..н",  "л..г", "л..ж", "м..к", "пл..в",
                             "р..вн", "р..с", "р..ст", "р..щ", "ск..к", "ск..ч", "тв..р", "б..р",
                             "д..р", "м..р", "п..р", "т..р", "бл..ст", "ж..г", "ст..л", "ч..т"],
            "verifiable": [],
            "unverifiable": []
        }

        self.exceptions = {
            "alternations": ["ст..лист", "прим..р", "г..рева", "г..рю", "г..рд", "алг..ритм", "г..ризонт", "г..рист"],
            "verifiable": [],
            "unverifiable": []
        }
        super().__init__()

    def predict_from_model(self, task):
        task = standardize_task(task)
        text, choices = task["text"], task["question"]["choices"]
        alt, unver = "чередующаяся", "непроверяемая"
        type_ = "alternations" if alt in text else "unverifiable" if unver in text else "verifiable"
        nice_option_ids = list()
        for option in choices:
            parsed_option = re.sub(r"^\d\)", "", option["text"]).split(", ")
            if all(self.is_of_type(word, type_) for word in parsed_option):
                nice_option_ids.append(option["id"])
        if choices[0]["text"].count(", ") == 0:
            if len(nice_option_ids) == 0:
                return [random.choice([str(i + 1) for i in range(5)])]
            elif len(nice_option_ids) == 1:
                return nice_option_ids
            else:
                return [random.choice(nice_option_ids)]
        else:
            if len(nice_option_ids) == 0:
                return sorted(random.sample([str(i + 1) for i in range(5)], 2))
            elif len(nice_option_ids) == 1:
                return sorted(nice_option_ids + [random.choice([str(i + 1) for i in range(5)
                                                                if str(i + 1) != nice_option_ids[0]])])
            elif len(nice_option_ids) in [2, 3]:
                return sorted(nice_option_ids)
            else:
                return sorted(random.sample(nice_option_ids, 2))

    def fit(self, tasks):
        alt, unver = "чередующаяся", "непроверяемая"
        for task in tasks:
            task = standardize_task(task)
            text = task["text"]

            if alt in text:
                type_ = "alternations"
            elif unver in text:
                type_ = "unverifiable"
            else:
                type_ = "verifiable"

            correct = task["solution"]["correct_variants"][0] if "correct_variants" in task["solution"] \
                else task["solution"]["correct"]
            for correct_id in correct:
                for word in task["choices"][int(correct_id) - 1]["parts"]:
                    word_sub = re.sub(r" *(?:^\d\)|\(.*?\)) *", "", word)
                    self.known_examples[type_].append(word_sub)
        return self

    def is_of_type(self, word, type_):
        if any(alternation in word for alternation in self.known_examples[type_]) \
                and not any(exception in word for exception in self.exceptions):
            return True
        else:
            return False
