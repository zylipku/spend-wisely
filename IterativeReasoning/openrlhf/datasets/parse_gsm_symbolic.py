"""
generate_random_problems is the main function to call. 
It generates random problems from the templates in the folder.
"""

import os
import json
import random
import re
from math import gcd
from fractions import Fraction
import pdb
from itertools import combinations
import numpy as np
import time
from tqdm import tqdm

# --------------------------------------------------------------------
# 1) Helper Functions
# --------------------------------------------------------------------

def is_int(x):
    """Check if x is very close to an integer."""
    return abs(x - round(x)) < 1e-6

def divides(a, b):
    """Check if integer b divides integer a."""
    if not (is_int(a) and is_int(b)):
        return False
    return is_int(a / b)

def custom_sample(*args):
    """
    Custom interpretation of sample(...):
    - sample(lst) => return lst (the entire set of possible values)
    - sample(lst, k) => return all k-combinations from lst
    """
    return args[0]

import random

def sample_sequential(lst, n):
    """从列表中顺序采样n个连续元素（仅正向）
    例如: sample_sequential(weekdays, 3) 可能返回 ['Monday', 'Tuesday', 'Wednesday']
    """
    if n > len(lst):
        raise ValueError("采样数量不能大于列表长度")
    
    # 计算可能的起始位置范围
    max_start = len(lst) - n
    
    # 随机选择起始位置
    start = random.randint(0, max_start)
    
    # 获取连续的n个元素
    return lst[start:start + n]

def numbers_within(start, end):
    """返回指定范围内的数字列表"""
    return list(range(start, end + 1))

# 添加倍数词映射
multiple = [2, 3, 4]
multiple_dict = {2: 'double', 3: 'triple', 4: 'quadruple'}

num_to_word = {
    0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 
    5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine',
    10: 'ten', 11: 'eleven', 12: 'twelve', 13: 'thirteen',
    14: 'fourteen', 15: 'fifteen', 16: 'sixteen',
    17: 'seventeen', 18: 'eighteen', 19: 'nineteen', 20: 'twenty',
    30: 'thirty', 40: 'forty', 50: 'fifty', 60: 'sixty', 
}

frac_to_word = {1/2: 'half', 1/3: 'one third', 1/4: 'one quarter', 1/5: 'one fifth', 2/3: 'two thirds', 
                1/6: 'one sixth', 1/7: 'one seventh', 1/8: 'one eighth', 1/9: 'one ninth', 1/10: 'one tenth'}

float_to_frac_list = {1/2, 1/3, 1/4, 1/5, 2/3}

def number_to_words(num):
    """将数字转换为英文单词"""
    return num_to_word.get(num, str(num))


def replace_placeholders(text, context, question=False):
    """
    Find { ... } placeholders in text, evaluate them as Python expressions
    using the context dict, and replace with the result. 
    If an expression fails, keep the original text.
    Also convert float results to Fractions if desired.
    """
    pattern = r"\{([^{}]+)\}"

    def replacer(match):
        expr = match.group(1).strip()
        try:
            val = eval(expr, {}, context)
            # Convert float to fraction if that’s preferable
            if isinstance(val, float) and val in float_to_frac_list:
                val = Fraction(val).limit_denominator()
                if val in frac_to_word and question:
                    # with 50% change convert fraction to word
                    if random.randint(0, 1):
                        val = frac_to_word[val]
                # when float is too long with numerical errors, truncate to 2 decimal places
            elif isinstance(val, float) and len(str(val)) > 5:
                    val = round(val, 2)
            if expr == 'mult' and val in multiple_dict and question:
                val = multiple_dict[val]
            if isinstance(val, int) and val in num_to_word and question and expr != 'mult':
                # with 50% change convert number to word
                if random.randint(0, 1):
                    val = num_to_word[val]
        except:
            val = expr  # If evaluation fails, just leave it as the expression
        return str(val)

    return re.sub(pattern, replacer, text)

# --------------------------------------------------------------------
# 2) Define Known Sample Sets / Globals
# --------------------------------------------------------------------
names = ["Alice", "Bob", "Charlie", "Diana", "Ethan", "Fiona", 
         "George", "Hannah", "Ian", "Julia"]
names_male = ["James", "John", "Robert", "Michael", "William", 
              "David", "Richard", "Joseph"] 
names_female = ["Mary", "Patricia", "Jennifer", "Linda", "Elizabeth",
                "Barbara", "Susan", "Jessica", "Sarah", "Karen"]
sports = ["basketball", "soccer", "tennis", "baseball", "volleyball"]
fruits = ["apple", "banana", "watermelon", 
            "honeydew", "kiwi", "lemon"]
length_lg = ["mile", "kilometer", "yard"]
weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
currencies_sym = ["$", "€", "£", "¥"]
colors = ["red", "blue", "green", "yellow", "purple", "orange", "black", "white"]
cities = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia", "San Antonio", "San Diego"]
weights_sm = ["gram", "ounce", "pound"]
weights_med = ["kilogram", "pound"]
fraction_decimals = [0.5, 0.25, 0.75, 0.2, 0.4, 0.6, 0.1, 0.15]
# teacher = ["Ms. Smith", "Mr. Johnson", "Mrs. Davis", "Mr. Brown", 
#             "Ms. Wilson", "Mr. Lee", "Mrs. Anderson", "Mr. Martinez"]
multiple_ice = [2, 3, 4]
multi_times = [2, 3, 4, 5]
fractions_list = [1/2, 1/3, 2/3, 1/4]
fraction_nums = [1/2, 1/4, 1/3, 1/5, 1/7, 1/8, 1/10, 1/16]

# This dictionary is used when we eval() the expressions in #init.
# Instead of actually picking a single element from sample(...),
# we want a list of possibilities. But for random sampling approach,
# we *will* pick a single random choice from those possibilities.
local_dict_for_eval = {
    "range": range,
    "np": np,
    "arange": np.arange, 
    "frange": np.arange,
    "names": names,
    "names_male": names_male,
    "names_female": names_female,
    "sports": sports,
    "fruits": fruits,
    "multiple_ice": multiple_ice,
    "multi_times": multi_times,
    "fractions": fractions_list,
    "fraction_alnum": fractions_list,
    "length_lg": length_lg,
    "currencies_sym": currencies_sym,
    "fraction_nums": fraction_nums,
    "weekdays": weekdays,
    "sample_sequential": sample_sequential,
    "numbers_within": numbers_within,
    "multiple": multiple,
    "colors": colors,
    "cities": cities,
    "fraction_alph": fractions_list,
    "weights_sm": weights_sm,
    "weights_med": weights_med,
    "fraction_decimals": fraction_decimals,
    # "multiple_reverse": multiple_reverse,

    # Use our custom_sample function instead of a simple lambda
    "sample": custom_sample
}

# --------------------------------------------------------------------
# 3) Utility to Parse a Single JSON and Return Problem Spec
# --------------------------------------------------------------------

def parse_json_template(json_path):
    """
    Reads a single JSON file and extracts:
      - question_annotated
      - answer_annotated
      - init_part (list of strings from #init:)
      - conditions_part (list of strings from #conditions:)
      - final question (with partial placeholders stripped)
      - other meta info like id_shuffled
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    question_annotated = data.get("question_annotated", "")
    answer_annotated = data.get("answer_annotated", "")
    id_shuffled = data.get("id_shuffled", "")

    init_part = []
    conditions_part = []
    answer_part = ""

    # We'll separate out #init:, #conditions:, #answer:
    current_section = None
    for line in question_annotated.split("\n"):
        line_stripped = line.strip()
        if line_stripped.startswith("#init:"):
            current_section = "init"
            continue
        elif line_stripped.startswith("#conditions:"):
            current_section = "conditions"
            continue
        elif line_stripped.startswith("#answer:"):
            current_section = "answer"
            continue
        else:
            if current_section == "init" and line_stripped.startswith("- "):
                init_part.append(line_stripped[2:])
            elif current_section == "conditions" and line_stripped.startswith("- "):
                conditions_part.append(line_stripped[2:])
            elif current_section == "answer":
                answer_part = line_stripped  # not always purely a Python expression, might be multiline

    # For convenience, let’s define "question" as the first line of question_annotated 
    # with the bracket format simplified if needed.
    lines = question_annotated.split("\n")
    if lines:
        # The first line of question_annotated
        question_raw = lines[0]
    else:
        question_raw = ""

    # Example: remove the second parameter in { ... , ...} placeholders:
    question_simplified = re.sub(r'{([^,]+),[^}]+}', r'{\1}', question_raw)

    spec = {
        "question_annotated": question_annotated,
        "answer_annotated": answer_annotated,
        "init_part": init_part,
        "conditions_part": conditions_part,
        "answer_part": answer_part,  # Might be multi-line, or partial
        "question_simplified": question_simplified,
        "id_shuffled": id_shuffled
    }
    return spec

# --------------------------------------------------------------------
# 4) Parse the #init lines to get a "generator" for random sampling
# --------------------------------------------------------------------

def parse_init_line(line):
    """
    Example line: 
        group2, group3 = sample(['dancers', 'choir members', 'debate team members', 'robotics club members'], 2)
    or a single variable assignment like:
        n = sample(names)
        x = range(30, 50)
    We return ( [var_names], generator ) where var_names is a list of 1+ variables.
    """
    left_side, expr = line.split("=", 1)
    left_side = left_side.strip()
    expr = expr.strip()

    # Detect if the left side has multiple comma-separated variables
    if "," in left_side:
        var_names = [v.strip().lstrip('$') for v in left_side.split(",")]
    else:
        var_names = [left_side.lstrip('$')]

    # pdb.set_trace()

    # Evaluate the expression in a context that collects *all* possible values 
    # (not yet randomly sampled).
    raw_result = eval(expr, {}, local_dict_for_eval)

    # Turn raw_result into a list if needed
    if isinstance(raw_result, np.ndarray):
        possible_values = raw_result.tolist()
    elif isinstance(raw_result, range):
        possible_values = list(raw_result)
    elif isinstance(raw_result, (list, tuple)):
        possible_values = list(raw_result)
    else:
        # Just a single item
        possible_values = [raw_result]
    
    # pdb.set_trace()

    def generator():
        """
        Each time we call generator(), we pick random assignments 
        for each variable in var_names from possible_values.
        - If there's only one var_name, pick one item from possible_values.
        - If multiple var_names, we pick that many distinct items 
          (like random.sample(...)) if you want them distinct.
        """
        out_dict = {}

        if len(var_names) == 1:
            # Single variable => pick one random item
            val = random.choice(possible_values)
            out_dict[var_names[0]] = val
        else:
            chosen = random.sample(possible_values, len(var_names))
            # pdb.set_trace()
            for name, val in zip(var_names, chosen):
                out_dict[name] = val

        return out_dict

    return var_names, generator


# --------------------------------------------------------------------
# 5) Randomly Sample Values, Check Conditions
# --------------------------------------------------------------------

def sample_solution_for_template(spec, max_tries=1000):
    init_part = spec["init_part"]
    conditions_part = spec["conditions_part"]

    # Build a list of (var_names_list, generator_function)
    generators = []
    for line in init_part:
        var_names, gen = parse_init_line(line)
        generators.append((var_names, gen))

    local_dict = {"is_int": is_int, "divides": divides}

    for _ in range(max_tries):
        vars_dict = {}
        
        # For each init line, call its generator => update vars_dict
        try:
            for (var_names, gen) in generators:
                # gen() returns a dict { varName : value, ... }
                assignment = gen()
                vars_dict.update(assignment)
        except ValueError:
            # This might happen if there's not enough items 
            # to do random.sample(...) for multiple variables
            # We'll treat it as a failed attempt and continue
            continue

        # pdb.set_trace()

        # Now check conditions
        all_good = True
        for cond in conditions_part:
            try:
                if not eval(cond, {}, {**local_dict, **vars_dict}):
                    all_good = False
                    break
            except:
                all_good = False
                break

        if all_good:
            # Found a valid assignment
            return vars_dict
    # print('Failed')
    return None


# --------------------------------------------------------------------
# 6) Putting It All Together
# --------------------------------------------------------------------

def generate_random_problems(
    templates_folder, 
    num_to_generate=1000, 
    max_tries_per_template=1000
):
    """
    1) Load all JSON templates from templates_folder (i.e. 250 files).
    2) For each problem to generate (total = num_to_generate):
       - Pick a random template file
       - Parse it
       - Randomly sample variable assignments up to max_tries_per_template times
       - If found a valid assignment, fill in the placeholders 
         in question/answer, store the result in a list.
    3) Return the list of generated problems.
    """
    # Gather all .json files from the folder
    import glob
    all_json_files = glob.glob(os.path.join(templates_folder, "*.json"))
    if not all_json_files:
        print("No JSON files found in", templates_folder)
        return []

    results = []
    for i in tqdm(range(num_to_generate), desc="Generating Problems"):
        while True:
            # 1) Pick a random template
            template_path = random.choice(all_json_files)

            # 2) Parse the template
            spec = parse_json_template(template_path)

            # 3) Randomly find a solution for it
            vars_dict = sample_solution_for_template(spec, max_tries=max_tries_per_template)
            if vars_dict is not None:
                break

        # 4) We have a valid assignment. Now build the final question and answer strings.
        local_context = {
            "is_int": is_int,
            "divides": divides
        }
        combined_context = {**vars_dict, **local_context}

        # If you want to use the full question_annotated with all lines,
        # you can do so. Or we can use question_simplified for just the first line.
        question_text = replace_placeholders(spec["question_simplified"], combined_context, question=True)
        answer_text = replace_placeholders(spec["answer_annotated"], combined_context)

        # (Optional) parse out final numeric answer if there's a #### expression
        pattern = r"####\s*\{([^{}]+)\}"
        match = re.search(pattern, spec["answer_annotated"])
        numeric_answer = None
        if match:
            numeric_expr = match.group(1).strip()
            try:
                numeric_answer = eval(numeric_expr, {}, combined_context)
            except:
                pass

        # pdb.set_trace()

        # Store the result
        result_item = {
            "template_path": template_path,
            "id_shuffled": spec["id_shuffled"],
            "vars": vars_dict,
            "question": question_text,
            "answer": answer_text,
            "numeric_answer": numeric_answer
        }
        results.append(result_item)

    return results

# --------------------------------------------------------------------
# 7) Usage Example
# --------------------------------------------------------------------
if __name__ == "__main__":
    # Suppose your templates are in folder "ml-gsm-symbolic/templates/p1"
    folder_path = "ml-gsm-symbolic/templates/symbolic"

    start_time = time.time()

    # Generate e.g. 5 random problems
    generated = generate_random_problems(
        templates_folder=folder_path, 
        num_to_generate=1000, 
        max_tries_per_template=10000
    )

    print("Time taken:", time.time() - start_time)
    print("Generated Fractions", len(generated)/1000)
    pdb.set_trace()

    # Print the results
    for i, item in enumerate(generated, 1):
        print(f"\nProblem {i}:")
        print("From template:", item["template_path"])
        print("id_shuffled:", item["id_shuffled"])
        print("Question:", item["question"])
        print("Answer:", item["answer"])
        print("Numeric answer (if any):", item["numeric_answer"])
        print("Vars used:", item["vars"])
