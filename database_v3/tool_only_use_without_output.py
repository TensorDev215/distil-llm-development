#!/usr/bin/env python3

import argparse
import json
import random
from math import gcd
from pathlib import Path


TOOL_USE_INSTRUCTION = (
    "You have access to a Python calculator. "
    "To solve the problem, write only a Python tool call using "
    "`<python>CODE</python>`. "
    "Inside the Python code, define clear variables with the values from the problem, "
    "compute the answer step by step, and print the final numeric answer. "
    "Do not include the tool output. "
    "Do not include a boxed final answer."
)


KINDS = [
    "shopping_budget",
    "recipe_scale",
    "travel_distance",
    "school_classroom",
    "garden_orchard",
    "bakery_orders",
    "library_books",
    "fundraiser",
    "trip_planning",
    "pets_animals",
    "sports_tournament",
    "construction",
    "modular_linear",
    "rate_distance",
    "mixture",
    "percentage",
    "gcd_lcm",
    "polynomial_eval",
    "arithmetic_series",
    "geometric_series",
    "digit_sum",
    "unit_conversion",
    "simultaneous",
    "factorial_mod",
    "set_intersect",
    "probability_count",
    "triangle_area",
    "coin_change",
    "time_arithmetic",
    "proportion",
]


CONSONANTS = [
    "b", "c", "d", "f", "g", "h", "j", "k", "l", "m",
    "n", "p", "r", "s", "t", "v", "w", "z",
    "br", "cl", "cr", "dr", "fl", "fr", "gl", "gr",
    "pl", "pr", "sk", "sl", "sm", "sn", "sp", "st", "sw", "tr",
    "th", "sh", "ch",
]

VOWELS = ["a", "e", "i", "o", "u", "ai", "ea", "ee", "ie", "oa", "ou", "ay"]

CODAS = ["", "n", "r", "l", "s", "t", "ck", "rd", "rt", "ng", "st", "ld"]


def make_tool_code(*lines: str) -> str:
    return "\n".join(lines)


def synthetic_syllable(r: random.Random) -> str:
    return r.choice(CONSONANTS) + r.choice(VOWELS) + r.choice(CODAS)


def synthetic_word(r: random.Random, n_syllables: int = 2) -> str:
    return "".join(synthetic_syllable(r) for _ in range(n_syllables))


def synthetic_name(r: random.Random) -> str:
    return synthetic_word(r, r.choice([2, 2, 3])).capitalize()


def synth_shop(r: random.Random) -> str:
    return synthetic_word(r, 2) + r.choice(["shop", "stand", "stall", "house"])


def synth_item(r: random.Random) -> str:
    word = synthetic_word(r, 2)

    if word.endswith(("s", "x", "z", "ch", "sh")):
        return word + "es"

    return word + "s"


def make_completion(tool_call: str) -> str:
    return (
        "<python>\n"
        f"{tool_call.strip()}\n"
        "</python>"
    )


def make_record(
    kind: str,
    question: str,
    gold: str,
    tool_call: str,
    seed: int | None = None,
) -> dict:
    prompt = f"{question}\n\n{TOOL_USE_INSTRUCTION}"
    completion = make_completion(tool_call)

    record = {
        "kind": kind,
        "question": question,
        "prompt": prompt,
        "completion": completion,
        "gold": gold,
        "used_tool": True,
        "tool_call": tool_call.strip(),
        "status": "tool_call_only",
    }

    if seed is not None:
        record["seed"] = seed

    return record


def generate_item(kind: str, r: random.Random, seed: int | None = None) -> dict:
    if kind == "shopping_budget":
        name = synthetic_name(r)
        friend = synthetic_name(r)
        start_money = r.choice([40, 50, 60, 80, 100, 120])
        n_items_a = r.randint(3, 8)
        price_a = r.choice([2, 3, 4, 5, 6, 7, 8])
        n_items_b = r.randint(2, 6)
        price_b = r.choice([3, 4, 5, 6, 8, 9, 10])
        distractor = r.choice([7, 11, 13, 14])
        shop = synth_shop(r)
        item_a = synth_item(r)
        item_b = synth_item(r)

        spent = n_items_a * price_a + n_items_b * price_b
        gold_n = start_money - spent

        question = (
            f"{name} goes to the {shop} with ${start_money}. "
            f"Their friend {friend}, who is {distractor} years old, "
            f"comes along but doesn't buy anything. "
            f"{name} buys {n_items_a} {item_a} at ${price_a} each "
            f"and {n_items_b} {item_b} at ${price_b} each. "
            f"How many dollars does {name} have left after the visit?"
        )

        tool_call = make_tool_code(
            f"start_money = {start_money}",
            f"n_items_a = {n_items_a}",
            f"price_a = {price_a}",
            f"n_items_b = {n_items_b}",
            f"price_b = {price_b}",
            "cost_a = n_items_a * price_a",
            "cost_b = n_items_b * price_b",
            "total_spent = cost_a + cost_b",
            "money_left = start_money - total_spent",
            "print(money_left)",
        )

        return make_record(kind, question, str(gold_n), tool_call, seed)

    elif kind == "recipe_scale":
        name = synthetic_name(r)
        base_servings = r.choice([4, 6, 8, 12])
        target_servings = base_servings * r.choice([2, 3, 4])
        cups_per_base = r.choice([2, 3, 4, 5])
        extra_topping = r.randint(2, 8)
        distractor = r.choice([45, 60, 90])
        recipe = r.choice(["banana bread", "pancakes", "cornbread", "biscuits"])

        gold_n = (target_servings // base_servings) * cups_per_base + extra_topping

        question = (
            f"A {recipe} recipe makes {base_servings} servings and uses "
            f"{cups_per_base} cups of flour. {name} wants to make "
            f"{target_servings} servings for a school bake sale. "
            f"They also need to add {extra_topping} extra cups of flour "
            f"for a dusting on top. The oven is preheated to "
            f"{distractor * 5} degrees, but that doesn't change the recipe. "
            f"How many total cups of flour does {name} need?"
        )

        tool_call = make_tool_code(
            f"base_servings = {base_servings}",
            f"target_servings = {target_servings}",
            f"cups_per_base = {cups_per_base}",
            f"extra_topping = {extra_topping}",
            "scale = target_servings // base_servings",
            "scaled_flour = scale * cups_per_base",
            "total_flour = scaled_flour + extra_topping",
            "print(total_flour)",
        )

        return make_record(kind, question, str(gold_n), tool_call, seed)

    elif kind == "travel_distance":
        name = synthetic_name(r)
        leg_a_speed = r.choice([40, 50, 60, 70])
        leg_a_hours = r.randint(2, 5)
        stop_distance = r.randint(15, 40)
        leg_b_speed = r.choice([45, 55, 65, 75])
        leg_b_hours = r.randint(2, 4)
        distractor = r.choice([8, 12, 24])

        gold_n = leg_a_speed * leg_a_hours + stop_distance + leg_b_speed * leg_b_hours

        question = (
            f"{name} drives east on the highway at {leg_a_speed} mph for "
            f"{leg_a_hours} hours. They stop at a rest area, then drive "
            f"another {stop_distance} miles north to pick up a friend. "
            f"From there, they continue at {leg_b_speed} mph for "
            f"{leg_b_hours} hours. The car holds {distractor} gallons of "
            f"fuel. How many miles total has {name} driven?"
        )

        tool_call = make_tool_code(
            f"leg_a_speed = {leg_a_speed}",
            f"leg_a_hours = {leg_a_hours}",
            f"stop_distance = {stop_distance}",
            f"leg_b_speed = {leg_b_speed}",
            f"leg_b_hours = {leg_b_hours}",
            "leg_a_distance = leg_a_speed * leg_a_hours",
            "leg_b_distance = leg_b_speed * leg_b_hours",
            "total_distance = leg_a_distance + stop_distance + leg_b_distance",
            "print(total_distance)",
        )

        return make_record(kind, question, str(gold_n), tool_call, seed)

    elif kind == "school_classroom":
        teacher = synthetic_name(r)
        n_classes = r.randint(3, 6)
        students_per_class = r.choice([18, 22, 24, 28, 30])
        absent_per_class = r.randint(1, 4)
        volunteer_helpers = r.randint(2, 5)
        distractor = r.choice([7, 9, 11])

        present = n_classes * (students_per_class - absent_per_class)
        gold_n = present + volunteer_helpers

        question = (
            f"Teacher {teacher} runs an after-school program with "
            f"{n_classes} classes. Each class has {students_per_class} "
            f"students enrolled, but on Monday {absent_per_class} students "
            f"in each class were absent. {volunteer_helpers} parent "
            f"volunteers also stayed to help. The school district has "
            f"{distractor} total grades, but that's not relevant here. "
            f"How many people, students plus volunteers, were present at "
            f"the program on Monday?"
        )

        tool_call = make_tool_code(
            f"n_classes = {n_classes}",
            f"students_per_class = {students_per_class}",
            f"absent_per_class = {absent_per_class}",
            f"volunteer_helpers = {volunteer_helpers}",
            "present_students_per_class = students_per_class - absent_per_class",
            "total_present_students = n_classes * present_students_per_class",
            "total_people_present = total_present_students + volunteer_helpers",
            "print(total_people_present)",
        )

        return make_record(kind, question, str(gold_n), tool_call, seed)

    elif kind == "garden_orchard":
        farmer = synthetic_name(r)
        n_rows = r.randint(4, 9)
        trees_per_row = r.randint(5, 12)
        apples_per_tree = r.choice([20, 25, 30, 40, 50])
        spoiled_pct = r.choice([10, 20, 25])
        saved_for_market = r.randint(50, 200)

        apples_total = n_rows * trees_per_row * apples_per_tree
        spoiled = apples_total * spoiled_pct // 100
        gold_n = apples_total - spoiled - saved_for_market

        question = (
            f"{farmer} runs an orchard with {n_rows} rows of apple trees. "
            f"Each row has {trees_per_row} trees, and each tree produces "
            f"{apples_per_tree} apples this season. {spoiled_pct}% of the "
            f"apples are spoiled by frost, and {farmer} saves "
            f"{saved_for_market} apples for the farmer's market next "
            f"weekend. How many apples are left for {farmer} to sell to "
            f"the local grocer?"
        )

        tool_call = make_tool_code(
            f"n_rows = {n_rows}",
            f"trees_per_row = {trees_per_row}",
            f"apples_per_tree = {apples_per_tree}",
            f"spoiled_pct = {spoiled_pct}",
            f"saved_for_market = {saved_for_market}",
            "total_trees = n_rows * trees_per_row",
            "apples_total = total_trees * apples_per_tree",
            "spoiled_apples = apples_total * spoiled_pct // 100",
            "apples_left_for_grocer = apples_total - spoiled_apples - saved_for_market",
            "print(apples_left_for_grocer)",
        )

        return make_record(kind, question, str(gold_n), tool_call, seed)

    elif kind == "bakery_orders":
        baker = synthetic_name(r)
        n_days = r.randint(3, 6)
        loaves_per_day = r.choice([24, 30, 36, 48, 60])
        wholesale_per_day = r.randint(8, 18)
        walkin_total = r.randint(10, 35)
        distractor = r.choice([5, 6, 7])

        produced = n_days * loaves_per_day
        sold = n_days * wholesale_per_day + walkin_total
        gold_n = produced - sold

        question = (
            f"Baker {baker} runs a small bakery with {distractor} staff. "
            f"They bake {loaves_per_day} loaves of sourdough every day "
            f"for {n_days} days straight. Each day they sell "
            f"{wholesale_per_day} loaves to a wholesale partner, and over "
            f"the {n_days} days they sell {walkin_total} loaves total to "
            f"walk-in customers. How many loaves are left in storage at "
            f"the end of the period?"
        )

        tool_call = make_tool_code(
            f"n_days = {n_days}",
            f"loaves_per_day = {loaves_per_day}",
            f"wholesale_per_day = {wholesale_per_day}",
            f"walkin_total = {walkin_total}",
            "produced_loaves = n_days * loaves_per_day",
            "wholesale_sold = n_days * wholesale_per_day",
            "total_sold = wholesale_sold + walkin_total",
            "loaves_left = produced_loaves - total_sold",
            "print(loaves_left)",
        )

        return make_record(kind, question, str(gold_n), tool_call, seed)

    elif kind == "library_books":
        librarian = synthetic_name(r)
        n_borrowers = r.randint(8, 20)
        books_per_borrower = r.randint(2, 5)
        late_returns = r.randint(3, 12)
        fine_per_late = r.choice([2, 3, 5])
        replacements_bought = r.randint(5, 15)
        distractor = r.choice([12, 15, 18])

        gold_n = late_returns * fine_per_late

        question = (
            f"Librarian {librarian} runs a reading club. {n_borrowers} "
            f"members each borrowed {books_per_borrower} books for the "
            f"month. The library opens at {distractor}:00 each day. By "
            f"the deadline, {late_returns} books were returned late. The "
            f"library charges ${fine_per_late} per late book. They also "
            f"used the fines to buy {replacements_bought} replacement "
            f"books later. How many dollars in late fees did the library "
            f"collect?"
        )

        tool_call = make_tool_code(
            f"late_returns = {late_returns}",
            f"fine_per_late = {fine_per_late}",
            "late_fee_revenue = late_returns * fine_per_late",
            "print(late_fee_revenue)",
        )

        return make_record(kind, question, str(gold_n), tool_call, seed)

    elif kind == "fundraiser":
        organizer = synthetic_name(r)
        silver_donors = r.randint(8, 25)
        silver_amount = r.choice([10, 15, 20, 25])
        gold_donors = r.randint(3, 9)
        gold_amount = r.choice([50, 75, 100, 150])
        corporate_match = r.choice([100, 200, 300, 500])
        distractor = r.choice([3, 5, 7])

        gold_n = silver_donors * silver_amount + gold_donors * gold_amount + corporate_match

        question = (
            f"{organizer} ran a {distractor}-hour charity fundraiser. "
            f"{silver_donors} silver-tier donors gave ${silver_amount} "
            f"each. {gold_donors} gold-tier donors gave ${gold_amount} "
            f"each. A local company contributed an additional "
            f"${corporate_match} as a flat corporate match. How many "
            f"dollars did the fundraiser raise in total?"
        )

        tool_call = make_tool_code(
            f"silver_donors = {silver_donors}",
            f"silver_amount = {silver_amount}",
            f"gold_donors = {gold_donors}",
            f"gold_amount = {gold_amount}",
            f"corporate_match = {corporate_match}",
            "silver_total = silver_donors * silver_amount",
            "gold_total = gold_donors * gold_amount",
            "fundraiser_total = silver_total + gold_total + corporate_match",
            "print(fundraiser_total)",
        )

        return make_record(kind, question, str(gold_n), tool_call, seed)

    elif kind == "trip_planning":
        traveler = synthetic_name(r)
        nights = r.randint(3, 8)
        lodging_per_night = r.choice([60, 80, 90, 120, 150])
        meals_per_day = r.choice([20, 30, 40])
        transport = r.choice([80, 120, 150, 200])
        distractor = r.choice([2, 4, 6])

        gold_n = nights * lodging_per_night + nights * meals_per_day + transport

        question = (
            f"{traveler} is planning a {nights}-night trip with "
            f"{distractor} other people, but they're each paying their "
            f"own way. {traveler}'s lodging costs ${lodging_per_night} "
            f"per night, meals cost ${meals_per_day} per day, and "
            f"round-trip transport costs ${transport}. How many dollars "
            f"will {traveler}'s share of the trip cost?"
        )

        tool_call = make_tool_code(
            f"nights = {nights}",
            f"lodging_per_night = {lodging_per_night}",
            f"meals_per_day = {meals_per_day}",
            f"transport = {transport}",
            "lodging_total = nights * lodging_per_night",
            "meals_total = nights * meals_per_day",
            "trip_cost = lodging_total + meals_total + transport",
            "print(trip_cost)",
        )

        return make_record(kind, question, str(gold_n), tool_call, seed)

    elif kind == "pets_animals":
        owner = synthetic_name(r)
        n_chickens = r.randint(8, 25)
        eggs_per_week_per_chicken = r.choice([4, 5, 6, 7])
        n_weeks = r.randint(2, 6)
        eggs_for_breakfast = r.randint(10, 25)
        eggs_donated = r.randint(5, 18)
        distractor = r.choice([12, 15, 18])

        total_eggs = n_chickens * eggs_per_week_per_chicken * n_weeks
        gold_n = total_eggs - eggs_for_breakfast - eggs_donated

        question = (
            f"{owner} keeps {n_chickens} chickens in a "
            f"{distractor}-meter coop. Each chicken lays "
            f"{eggs_per_week_per_chicken} eggs per week. Over "
            f"{n_weeks} weeks, the family ate {eggs_for_breakfast} eggs "
            f"for breakfast and donated {eggs_donated} eggs to a "
            f"neighbor. How many eggs are left at the end of the "
            f"{n_weeks} weeks?"
        )

        tool_call = make_tool_code(
            f"n_chickens = {n_chickens}",
            f"eggs_per_week_per_chicken = {eggs_per_week_per_chicken}",
            f"n_weeks = {n_weeks}",
            f"eggs_for_breakfast = {eggs_for_breakfast}",
            f"eggs_donated = {eggs_donated}",
            "total_eggs = n_chickens * eggs_per_week_per_chicken * n_weeks",
            "eggs_used = eggs_for_breakfast + eggs_donated",
            "eggs_left = total_eggs - eggs_used",
            "print(eggs_left)",
        )

        return make_record(kind, question, str(gold_n), tool_call, seed)

    elif kind == "sports_tournament":
        captain = synthetic_name(r)
        n_games = r.randint(8, 20)
        n_wins = r.randint(3, n_games - 2)
        n_losses = n_games - n_wins
        points_per_win = r.choice([2, 3])
        points_per_loss = r.choice([0, 1])
        bonus_pts = r.randint(2, 8)
        distractor = r.choice([5, 6, 7])
        sport = r.choice(["soccer", "hockey", "basketball", "rugby"])

        gold_n = n_wins * points_per_win + n_losses * points_per_loss + bonus_pts

        question = (
            f"Captain {captain}'s {sport} team played {n_games} games "
            f"with {distractor} players on the field at any time. They "
            f"won {n_wins} games and lost {n_losses} games. Each win is "
            f"worth {points_per_win} league points, each loss is worth "
            f"{points_per_loss} consolation points, and the team got an "
            f"extra {bonus_pts} bonus points for fair play. How many "
            f"total league points did {captain}'s team end the season "
            f"with?"
        )

        tool_call = make_tool_code(
            f"n_wins = {n_wins}",
            f"n_losses = {n_losses}",
            f"points_per_win = {points_per_win}",
            f"points_per_loss = {points_per_loss}",
            f"bonus_pts = {bonus_pts}",
            "win_points = n_wins * points_per_win",
            "loss_points = n_losses * points_per_loss",
            "total_points = win_points + loss_points + bonus_pts",
            "print(total_points)",
        )

        return make_record(kind, question, str(gold_n), tool_call, seed)

    elif kind == "construction":
        contractor = synthetic_name(r)
        n_walls = r.randint(3, 8)
        bricks_per_wall = r.choice([80, 100, 120, 150, 200])
        broken_bricks = r.randint(5, 25)
        extra_safety = r.choice([20, 30, 50])
        distractor = r.choice([8, 10, 12])

        bricks_total = n_walls * bricks_per_wall + extra_safety
        gold_n = bricks_total - broken_bricks

        question = (
            f"Contractor {contractor} is building {n_walls} brick walls "
            f"using a {distractor}-foot ladder. Each wall needs "
            f"{bricks_per_wall} bricks. {contractor} also orders "
            f"{extra_safety} extra bricks as a safety margin. During "
            f"delivery, {broken_bricks} bricks arrive broken and have "
            f"to be discarded. How many usable bricks does {contractor} "
            f"have to build the walls?"
        )

        tool_call = make_tool_code(
            f"n_walls = {n_walls}",
            f"bricks_per_wall = {bricks_per_wall}",
            f"extra_safety = {extra_safety}",
            f"broken_bricks = {broken_bricks}",
            "required_wall_bricks = n_walls * bricks_per_wall",
            "bricks_ordered = required_wall_bricks + extra_safety",
            "usable_bricks = bricks_ordered - broken_bricks",
            "print(usable_bricks)",
        )

        return make_record(kind, question, str(gold_n), tool_call, seed)

    elif kind == "modular_linear":
        a, b, c = r.randint(2, 11), r.randint(2, 11), r.randint(1, 49)
        x, y = r.randint(7, 39), r.randint(7, 39)
        m = r.choice([7, 11, 13, 17, 19, 23, 29, 31])

        gold_n = (a * x + b * y + c) % m

        question = (
            f"Compute (({a} * {x}) + ({b} * {y}) + {c}) mod {m}.\n"
            f"Give your final answer as the integer remainder."
        )

        tool_call = make_tool_code(
            f"a = {a}",
            f"b = {b}",
            f"c = {c}",
            f"x = {x}",
            f"y = {y}",
            f"modulus = {m}",
            "linear_value = a * x + b * y + c",
            "remainder = linear_value % modulus",
            "print(remainder)",
        )

        return make_record(kind, question, str(gold_n), tool_call, seed)

    elif kind == "rate_distance":
        v_a = r.randint(40, 80)
        v_b = v_a + r.randint(8, 28)
        t = r.randint(2, 9)
        head_start = r.randint(0, 30)

        gold_n = (v_b - v_a) * t - head_start

        question = (
            f"Two cars start traveling east on the same highway. "
            f"Car A travels at {v_a} kilometers per hour. "
            f"Car B starts {head_start} kilometers behind Car A "
            f"and travels at {v_b} kilometers per hour. "
            f"After {t} hours of driving, how many kilometers ahead of Car A is Car B?"
        )

        tool_call = make_tool_code(
            f"car_a_speed = {v_a}",
            f"car_b_speed = {v_b}",
            f"time_hours = {t}",
            f"car_b_start_behind = {head_start}",
            "relative_speed = car_b_speed - car_a_speed",
            "relative_gain = relative_speed * time_hours",
            "car_b_ahead_distance = relative_gain - car_b_start_behind",
            "print(car_b_ahead_distance)",
        )

        return make_record(kind, question, str(gold_n), tool_call, seed)

    elif kind == "mixture":
        v1, p1 = r.randint(20, 80), r.choice([10, 15, 20, 25, 30, 40])
        v2, p2 = r.randint(20, 80), r.choice([50, 60, 70, 80, 90])

        gold_n = v1 * p1 + v2 * p2

        question = (
            f"A chemist mixes {v1} grams of solution A, at {p1}% solute, "
            f"with {v2} grams of solution B, at {p2}% solute. "
            f"Compute the total mass of solute across both solutions in "
            f"centigrams. 1 centigram equals 0.01 grams. Give the integer "
            f"equal to v1*p1 + v2*p2."
        )

        tool_call = make_tool_code(
            f"solution_a_grams = {v1}",
            f"solution_a_percent = {p1}",
            f"solution_b_grams = {v2}",
            f"solution_b_percent = {p2}",
            "solute_a_centigrams = solution_a_grams * solution_a_percent",
            "solute_b_centigrams = solution_b_grams * solution_b_percent",
            "total_solute_centigrams = solute_a_centigrams + solute_b_centigrams",
            "print(total_solute_centigrams)",
        )

        return make_record(kind, question, str(gold_n), tool_call, seed)

    elif kind == "percentage":
        base = r.randint(200, 1500)
        pct = r.choice([5, 8, 10, 12, 15, 20, 25, 30, 40])
        extra = r.randint(7, 80)

        gold_n = (base * pct) // 100 + extra

        question = (
            f"A shop has {base} items in stock. "
            f"They sell {pct}% of them on Monday, then receive {extra} new items. "
            f"How many items have left the shop, plus the new arrivals, "
            f"that is, {pct}% of {base} plus {extra}?"
        )

        tool_call = make_tool_code(
            f"base_items = {base}",
            f"percent_sold = {pct}",
            f"new_arrivals = {extra}",
            "items_sold = base_items * percent_sold // 100",
            "sold_plus_new_arrivals = items_sold + new_arrivals",
            "print(sold_plus_new_arrivals)",
        )

        return make_record(kind, question, str(gold_n), tool_call, seed)

    elif kind == "gcd_lcm":
        x, y = r.randint(60, 999), r.randint(60, 999)
        choice = r.choice(["gcd", "lcm"])

        if choice == "gcd":
            gold_n = gcd(x, y)
            question = f"Compute the greatest common divisor of {x} and {y}."

            tool_call = make_tool_code(
                "import math",
                f"x = {x}",
                f"y = {y}",
                "greatest_common_divisor = math.gcd(x, y)",
                "print(greatest_common_divisor)",
            )
        else:
            gold_n = (x * y) // gcd(x, y)
            question = f"Compute the least common multiple of {x} and {y}."

            tool_call = make_tool_code(
                "import math",
                f"x = {x}",
                f"y = {y}",
                "greatest_common_divisor = math.gcd(x, y)",
                "least_common_multiple = x * y // greatest_common_divisor",
                "print(least_common_multiple)",
            )

        return make_record(kind, question, str(gold_n), tool_call, seed)

    elif kind == "polynomial_eval":
        a, b, c = r.randint(-7, 7), r.randint(-12, 12), r.randint(-25, 25)
        x = r.randint(-6, 6)

        gold_n = a * x * x + b * x + c

        question = (
            f"Evaluate the polynomial p(x) = {a}x^2 + {b}x + {c} at x = {x}. "
            f"Give the integer p({x})."
        )

        tool_call = make_tool_code(
            f"a = {a}",
            f"b = {b}",
            f"c = {c}",
            f"x = {x}",
            "x_squared = x * x",
            "polynomial_value = a * x_squared + b * x + c",
            "print(polynomial_value)",
        )

        return make_record(kind, question, str(gold_n), tool_call, seed)

    elif kind == "arithmetic_series":
        a1, d = r.randint(1, 25), r.randint(1, 9)
        n = r.randint(8, 30)

        gold_n = n * (2 * a1 + (n - 1) * d) // 2

        question = (
            f"Compute the sum of the first {n} terms of the arithmetic "
            f"sequence whose first term is {a1} and whose common difference is {d}."
        )

        tool_call = make_tool_code(
            f"first_term = {a1}",
            f"common_difference = {d}",
            f"n_terms = {n}",
            "last_term = first_term + (n_terms - 1) * common_difference",
            "series_sum = n_terms * (first_term + last_term) // 2",
            "print(series_sum)",
        )

        return make_record(kind, question, str(gold_n), tool_call, seed)

    elif kind == "geometric_series":
        a1 = r.randint(2, 9)
        ratio = r.choice([2, 3])
        n = r.randint(4, 8)

        gold_n = a1 * (ratio ** n - 1) // (ratio - 1)

        question = (
            f"Compute the sum of the first {n} terms of the geometric "
            f"sequence with first term {a1} and common ratio {ratio}."
        )

        tool_call = make_tool_code(
            f"first_term = {a1}",
            f"common_ratio = {ratio}",
            f"n_terms = {n}",
            "series_sum = first_term * (common_ratio ** n_terms - 1) // (common_ratio - 1)",
            "print(series_sum)",
        )

        return make_record(kind, question, str(gold_n), tool_call, seed)

    elif kind == "digit_sum":
        n_val = r.randint(10000, 999999)
        target = r.randint(2, 9)
        digit_sum = sum(int(d) for d in str(n_val))
        sign = r.choice(["minus", "plus"])
        offset = r.randint(1, 19)

        if sign == "minus":
            gold_n = digit_sum * target - offset
            op = "-"
        else:
            gold_n = digit_sum * target + offset
            op = "+"

        question = (
            f"Let S be the sum of the digits of {n_val}. "
            f"Compute S times {target}, {sign} {offset}."
        )

        tool_call = make_tool_code(
            f"number = {n_val}",
            f"multiplier = {target}",
            f"offset = {offset}",
            "digit_sum = sum(int(digit) for digit in str(number))",
            "base_value = digit_sum * multiplier",
            f"answer = base_value {op} offset",
            "print(answer)",
        )

        return make_record(kind, question, str(gold_n), tool_call, seed)

    elif kind == "unit_conversion":
        hours = r.randint(2, 14)
        extra_min = r.randint(0, 59)

        gold_n = hours * 60 + extra_min

        question = f"How many total minutes are in {hours} hours and {extra_min} minutes?"

        tool_call = make_tool_code(
            f"hours = {hours}",
            f"extra_minutes = {extra_min}",
            "minutes_per_hour = 60",
            "total_minutes = hours * minutes_per_hour + extra_minutes",
            "print(total_minutes)",
        )

        return make_record(kind, question, str(gold_n), tool_call, seed)

    elif kind == "simultaneous":
        x = r.randint(-9, 12)
        y = r.randint(-9, 12)

        a1, b1 = r.randint(1, 7), r.randint(1, 7)
        a2, b2 = r.randint(1, 7), r.randint(1, 7)

        while a1 * b2 - a2 * b1 == 0:
            a2 = r.randint(1, 7)
            b2 = r.randint(1, 7)

        c1 = a1 * x + b1 * y
        c2 = a2 * x + b2 * y

        ask = r.choice(["x_minus_y", "x_plus_y", "x_times_y"])

        if ask == "x_minus_y":
            gold_n = x - y
            tail = "x - y"
            expr = "x - y"
        elif ask == "x_plus_y":
            gold_n = x + y
            tail = "x + y"
            expr = "x + y"
        else:
            gold_n = x * y
            tail = "x * y"
            expr = "x * y"

        question = (
            f"Solve the system:\n"
            f"  {a1} x + {b1} y = {c1}\n"
            f"  {a2} x + {b2} y = {c2}\n"
            f"Then compute {tail} as an integer."
        )

        tool_call = make_tool_code(
            "from fractions import Fraction",
            f"a1 = {a1}",
            f"b1 = {b1}",
            f"c1 = {c1}",
            f"a2 = {a2}",
            f"b2 = {b2}",
            f"c2 = {c2}",
            "determinant = a1 * b2 - a2 * b1",
            "x = Fraction(c1 * b2 - c2 * b1, determinant)",
            "y = Fraction(a1 * c2 - a2 * c1, determinant)",
            f"answer = {expr}",
            "print(int(answer))",
        )

        return make_record(kind, question, str(gold_n), tool_call, seed)

    elif kind == "factorial_mod":
        p = r.choice([7, 11, 13, 17, 19, 23])
        n = r.randint(2, p - 1)

        fact = 1
        for k in range(1, n + 1):
            fact = (fact * k) % p

        question = (
            f"Compute {n}! mod {p}. That is, the remainder when "
            f"{n} factorial is divided by {p}."
        )

        tool_call = make_tool_code(
            f"n = {n}",
            f"modulus = {p}",
            "factorial_remainder = 1",
            "for k in range(1, n + 1):",
            "    factorial_remainder = (factorial_remainder * k) % modulus",
            "print(factorial_remainder)",
        )

        return make_record(kind, question, str(fact), tool_call, seed)

    elif kind == "set_intersect":
        a = r.randint(40, 80)
        b = r.randint(40, 80)
        ab = r.randint(5, min(a, b) - 5)

        gold_n = a + b - ab

        question = (
            f"In a class, {a} students study Spanish, {b} study French, "
            f"and {ab} study both. How many students study Spanish or "
            f"French, or both?"
        )

        tool_call = make_tool_code(
            f"spanish_students = {a}",
            f"french_students = {b}",
            f"both_students = {ab}",
            "students_in_either_language = spanish_students + french_students - both_students",
            "print(students_in_either_language)",
        )

        return make_record(kind, question, str(gold_n), tool_call, seed)

    elif kind == "probability_count":
        total = r.randint(8, 18)
        red = r.randint(2, total - 4)
        blue = r.randint(2, total - red - 1)
        green = total - red - blue
        colour = r.choice(["red", "blue", "neither red nor blue"])

        if colour == "red":
            numerator = red
            target_var_name = "red_marbles"
        elif colour == "blue":
            numerator = blue
            target_var_name = "blue_marbles"
        else:
            numerator = green
            target_var_name = "green_marbles"

        gold_n = numerator * 100 // total

        question = (
            f"A bag contains {red} red, {blue} blue, and {green} green marbles. "
            f"You draw one at random. What is the probability, in whole percent "
            f"rounded down, of drawing a {colour} marble?"
        )

        tool_call = make_tool_code(
            f"red_marbles = {red}",
            f"blue_marbles = {blue}",
            f"green_marbles = {green}",
            "total_marbles = red_marbles + blue_marbles + green_marbles",
            f"target_marbles = {target_var_name}",
            "probability_percent = target_marbles * 100 // total_marbles",
            "print(probability_percent)",
        )

        return make_record(kind, question, str(gold_n), tool_call, seed)

    elif kind == "triangle_area":
        base = r.choice([6, 8, 10, 12, 14, 16])
        height = r.choice([5, 7, 9, 11, 13, 15])

        gold_n = base * height // 2

        question = (
            f"A triangle has base {base} cm and corresponding height {height} cm. "
            f"What is its area in square centimeters? Use the formula "
            f"area = base * height / 2 and give the integer."
        )

        tool_call = make_tool_code(
            f"base_cm = {base}",
            f"height_cm = {height}",
            "area_square_cm = base_cm * height_cm // 2",
            "print(area_square_cm)",
        )

        return make_record(kind, question, str(gold_n), tool_call, seed)

    elif kind == "coin_change":
        target = r.choice([12, 15, 20, 25, 30, 35])
        gold_n = target // 5 + 1

        question = (
            f"How many ways can you make exactly {target} cents using only "
            f"5-cent and 1-cent coins? Each combination differing in the "
            f"number of 5-cent coins counts as distinct, including the "
            f"all-1-cent combination."
        )

        tool_call = make_tool_code(
            f"target_cents = {target}",
            "max_five_cent_coins = target_cents // 5",
            "ways = max_five_cent_coins + 1",
            "print(ways)",
        )

        return make_record(kind, question, str(gold_n), tool_call, seed)

    elif kind == "time_arithmetic":
        start_h = r.randint(1, 12)
        start_m = r.randint(0, 59)
        add_h = r.randint(2, 14)
        add_m = r.randint(5, 55)

        total_min = (start_h * 60 + start_m + add_h * 60 + add_m) % (12 * 60)
        end_h, end_m = divmod(total_min, 60)

        if end_h == 0:
            end_h = 12

        gold_n = end_h * 100 + end_m

        question = (
            f"A meeting starts at {start_h:02d}:{start_m:02d} on a 12-hour clock. "
            f"It runs for {add_h} hours and {add_m} minutes. At what time does "
            f"it end? Give the answer as the integer HHMM, for example 03:25 becomes 325."
        )

        tool_call = make_tool_code(
            f"start_hour = {start_h}",
            f"start_minute = {start_m}",
            f"add_hours = {add_h}",
            f"add_minutes = {add_m}",
            "minutes_per_hour = 60",
            "clock_cycle_minutes = 12 * minutes_per_hour",
            "start_total_minutes = start_hour * minutes_per_hour + start_minute",
            "duration_minutes = add_hours * minutes_per_hour + add_minutes",
            "end_total_minutes = (start_total_minutes + duration_minutes) % clock_cycle_minutes",
            "end_hour, end_minute = divmod(end_total_minutes, minutes_per_hour)",
            "if end_hour == 0:",
            "    end_hour = 12",
            "answer_hhmm = end_hour * 100 + end_minute",
            "print(answer_hhmm)",
        )

        return make_record(kind, question, str(gold_n), tool_call, seed)

    elif kind == "proportion":
        scale = r.choice([3, 4, 5, 6, 7, 8])
        base = r.randint(20, 90)
        other = r.randint(40, 200)

        gold_n = (base * other) // scale

        question = (
            f"If {scale} units of resource A produce {base} widgets, and we "
            f"have {other} units of resource A, how many widgets can be "
            f"produced, rounded down to a whole number?"
        )

        tool_call = make_tool_code(
            f"input_units = {scale}",
            f"output_widgets = {base}",
            f"available_units = {other}",
            "widgets_produced = output_widgets * available_units // input_units",
            "print(widgets_produced)",
        )

        return make_record(kind, question, str(gold_n), tool_call, seed)

    else:
        raise ValueError(f"Unknown kind: {kind}")


def build_records(seed: int, n_per_kind: int, shuffle: bool = True) -> list[dict]:
    main_rng = random.Random(seed)
    records = []

    for kind in KINDS:
        for _ in range(n_per_kind):
            item_seed = main_rng.randint(0, 2**31 - 1)
            item_rng = random.Random(item_seed)
            record = generate_item(kind, item_rng, seed=item_seed)
            records.append(record)

    if shuffle:
        main_rng.shuffle(records)

    return records


def write_jsonl(records: list[dict], output_path: Path, append: bool = False) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if append else "w"

    with output_path.open(mode, encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Build a tool-call-only JSONL Python tool-use math database."
    )

    parser.add_argument(
        "--output",
        type=str,
        default="dataset/tool_use_database_all_cases_without_output.jsonl",
        help="Output JSONL file path.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260501,
        help="Random seed.",
    )

    parser.add_argument(
        "--n-per-kind",
        type=int,
        default=10,
        help="Number of examples to generate for each problem kind.",
    )

    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Do not shuffle final records.",
    )

    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to output file instead of overwriting it.",
    )

    args = parser.parse_args()

    output_path = Path(args.output)

    records = build_records(
        seed=args.seed,
        n_per_kind=args.n_per_kind,
        shuffle=not args.no_shuffle,
    )

    write_jsonl(records, output_path, append=args.append)

    counts = {}
    for record in records:
        counts[record["kind"]] = counts.get(record["kind"], 0) + 1

    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Output: {output_path}")
    print(f"Total records: {len(records)}")
    print(f"Kinds: {len(KINDS)}")
    print(f"Records per kind: {args.n_per_kind}")
    print(f"Append mode: {args.append}")

    print("\nKind counts:")
    for kind in sorted(counts):
        print(f"  {kind}: {counts[kind]}")


if __name__ == "__main__":
    main()
