import { RNG } from "../engine/rng";
import { tokenizeWords, type CorpusStats } from "./corpusLarge";

/**
 * The Python specialist corpus: real, runnable python for the code-generation
 * presets.
 *
 * Design rules, chosen around the BPE tokenizer and the browser budget:
 *
 *  - Case is preserved. The dataset trains a case-SENSITIVE BPE tokenizer
 *    (see TokenizerSpec.caseSensitive), so `None`, `True`, `ValueError`,
 *    `Counter` and PEP8 CamelCase names survive tokenisation and the code a
 *    trained model emits is code that actually runs.
 *  - Four-space indentation, standard python style.
 *  - One complete idea per snippet, blank line between snippets, so context
 *    windows learn whole functions rather than fragments of half a class.
 *  - Snippets follow the pattern a code model actually needs: a comment
 *    stating the intent, then the code that satisfies it. That teaches
 *    comment -> code completion, the core of code generation.
 *  - Every generated variant is valid python by construction, and every
 *    computed table (squares, powers, fibonacci, times tables) is computed
 *    correctly at build time, so the model learns facts that are true. Most
 *    snippets end with an `assert`, so the examples verify themselves.
 */

export const PY_SEQ_LEN = 64;
export const PY_VOCAB_TARGET = 1024;
export const PY_MAX_WINDOWS = 100_000;

const SNIPPETS: string[][] = [
  ["# add two numbers", "def add(a, b):", '    return a + b', "", "assert add(2, 3) == 5"],
  ["# subtract one number from another", "def subtract(a, b):", "    return a - b", "", "assert subtract(10, 4) == 6"],
  ["# multiply two numbers", "def multiply(a, b):", "    return a * b", "", "assert multiply(3, 7) == 21"],
  ["# divide safely and return None on zero", "def safe_divide(a, b):", "    if b == 0:", "        return None", "    return a / b", "", "assert safe_divide(10, 2) == 5", "assert safe_divide(1, 0) is None"],
  ["# return the larger of two numbers", "def largest(a, b):", "    if a > b:", "        return a", "    return b", "", "assert largest(3, 9) == 9"],
  ["# return the smallest of three numbers", "def smallest_of_three(a, b, c):", "    return min(a, b, c)"],
  ["# clamp a value between a low and a high bound", "def clamp(x, low, high):", "    return max(low, min(x, high))", "", "assert clamp(15, 0, 10) == 10"],
  ["# check whether a number is even", "def is_even(n):", "    return n % 2 == 0", "", "assert is_even(4)", "assert not is_even(5)"],
  ["# check whether a number is odd", "def is_odd(n):", "    return n % 2 == 1"],
  ["# absolute value without the builtin", "def my_abs(x):", "    if x < 0:", "        return -x", "    return x", "", "assert my_abs(-3) == 3"],
  ["# sum every number in a list", "def total(numbers):", "    result = 0", "    for n in numbers:", "        result = result + n", "    return result", "", "assert total([1, 2, 3]) == 6"],
  ["# the average of a list of numbers", "def average(numbers):", "    if len(numbers) == 0:", "        return 0.0", "    return sum(numbers) / len(numbers)", "", "assert average([2, 4]) == 3.0"],
  ["# count how many items are in a list without len", "def count(items):", "    n = 0", "    for item in items:", "        n = n + 1", "    return n"],
  ["# reverse a list", "def reverse(items):", "    out = []", "    for item in items:", "        out.insert(0, item)", "    return out", "", "assert reverse([1, 2, 3]) == [3, 2, 1]"],
  ["# squares of the first n numbers with a comprehension", "def squares(n):", "    return [i * i for i in range(n)]", "", "assert squares(4) == [0, 1, 4, 9]"],
  ["# every even number below n", "def evens_below(n):", "    return [i for i in range(n) if i % 2 == 0]"],
  ["# double every number in a list", "def double_all(numbers):", "    return [n * 2 for n in numbers]"],
  ["# keep only the positive numbers", "def positives(numbers):", "    return [n for n in numbers if n > 0]", "", "assert positives([-1, 2, -3, 4]) == [2, 4]"],
  ["# find the maximum in a list by hand", "def find_max(numbers):", "    best = numbers[0]", "    for n in numbers:", "        if n > best:", "            best = n", "    return best"],
  ["# factorial with a loop", "def factorial(n):", "    result = 1", "    for i in range(2, n + 1):", "        result = result * i", "    return result", "", "assert factorial(5) == 120"],
  ["# factorial with recursion", "def factorial_rec(n):", "    if n <= 1:", "        return 1", "    return n * factorial_rec(n - 1)"],
  ["# the nth fibonacci number, iterative", "def fib(n):", "    a, b = 0, 1", "    for _ in range(n):", "        a, b = b, a + b", "    return a", "", "assert fib(10) == 55"],
  ["# the first n fibonacci numbers as a list", "def fib_list(n):", "    out = []", "    a, b = 0, 1", "    for _ in range(n):", "        out.append(a)", "        a, b = b, a + b", "    return out"],
  ["# greatest common divisor with the euclidean algorithm", "def gcd(a, b):", "    while b != 0:", "        a, b = b, a % b", "    return a", "", "assert gcd(12, 18) == 6"],
  ["# check whether a number is prime", "def is_prime(n):", "    if n < 2:", "        return False", "    for i in range(2, int(n ** 0.5) + 1):", "        if n % i == 0:", "            return False", "    return True", "", "assert is_prime(13)", "assert not is_prime(14)"],
  ["# fizzbuzz from one to n", "def fizzbuzz(n):", "    for i in range(1, n + 1):", "        if i % 15 == 0:", '            print("FizzBuzz")', "        elif i % 3 == 0:", '            print("Fizz")', "        elif i % 5 == 0:", '            print("Buzz")', "        else:", "            print(i)"],
  ["# count down from n to one", "def countdown(n):", "    while n > 0:", "        print(n)", "        n = n - 1", '    print("liftoff!")'],
  ["# loop with enumerate to get index and value", 'fruits = ["apple", "banana", "cherry"]', "for i, fruit in enumerate(fruits):", "    print(i, fruit)"],
  ["# loop over two lists at once with zip", 'names = ["ada", "grace", "alan"]', "scores = [98, 95, 91]", "for name, score in zip(names, scores):", "    print(name, score)"],
  ["# break out of a loop early", "for n in range(10):", "    if n == 5:", "        break", "    print(n)"],
  ["# skip values with continue", "for n in range(10):", "    if n % 2 == 0:", "        continue", "    print(n)"],
  ["# nested loops print a small multiplication table", "for a in range(1, 4):", "    for b in range(1, 4):", '        print(a * b, end=" ")', "    print()"],
  ["# a dictionary of letter counts", "counts = {}", 'for letter in "mississippi":', "    counts[letter] = counts.get(letter, 0) + 1", "print(counts)"],
  ["# loop over a dictionary", 'ages = {"ada": 36, "grace": 45, "alan": 41}', "for name, age in ages.items():", "    print(name, age)"],
  ["# check whether a key exists before using it", 'config = {"debug": True}', 'if "debug" in config:', '    print(config["debug"])'],
  ["# merge two dictionaries", 'defaults = {"color": "cyan", "size": 3}', 'overrides = {"size": 7}', "settings = {**defaults, **overrides}", 'assert settings["size"] == 7'],
  ["# build a dictionary with a comprehension", "square_map = {n: n * n for n in range(6)}"],
  ["# sort a list of words by length", 'words = ["pear", "fig", "watermelon", "plum"]', "words.sort(key=len)", 'assert words[0] == "fig"'],
  ["# sort pairs by the second value", 'pairs = [("a", 3), ("b", 1), ("c", 2)]', "pairs.sort(key=lambda p: p[1])", 'assert pairs[0] == ("b", 1)'],
  ["# a set removes duplicates", "unique = sorted(set([1, 2, 2, 3, 3, 3]))", "assert unique == [1, 2, 3]"],
  ["# reverse a string with a slice", 'text = "neuron"', "backwards = text[::-1]", 'assert backwards == "noruen"'],
  ["# check whether a word is a palindrome", "def is_palindrome(word):", "    return word == word[::-1]", "", 'assert is_palindrome("racecar")'],
  ["# count the vowels in a string", 'def count_vowels(text):', '    vowels = "aeiou"', "    return sum(1 for ch in text.lower() if ch in vowels)", "", 'assert count_vowels("hello") == 2'],
  ["# split a sentence into words", 'sentence = "the quick brown fox"', "words = sentence.split()", "assert len(words) == 4"],
  ["# join words back into a sentence", 'words = ["the", "quick", "brown", "fox"]', 'sentence = " ".join(words)'],
  ["# strings are immutable, so build a new one", 'greeting = "hello"', 'greeting = greeting + " world"'],
  ["# format a string with an f-string", 'name = "nova"', 'message = f"hello {name}, welcome back"', "print(message)"],
  ["# check prefixes and suffixes", 'filename = "report.txt"', 'assert filename.startswith("report")', 'assert filename.endswith(".txt")'],
  ["# strip whitespace from both ends", 'raw = "  padded  "', 'assert raw.strip() == "padded"'],
  ["# replace part of a string", 'text = "cat sat"', 'assert text.replace("cat", "dog") == "dog sat"'],
  ["# a simple counter class", "class Counter:", "    def __init__(self):", "        self.value = 0", "", "    def increment(self):", "        self.value = self.value + 1", "", "    def get(self):", "        return self.value"],
  ["# a point class with a method", "class Point:", "    def __init__(self, x, y):", "        self.x = x", "        self.y = y", "", "    def distance_to_origin(self):", "        return (self.x ** 2 + self.y ** 2) ** 0.5"],
  ["# a stack built on a list", "class Stack:", "    def __init__(self):", "        self.items = []", "", "    def push(self, item):", "        self.items.append(item)", "", "    def pop(self):", "        return self.items.pop()", "", "    def is_empty(self):", "        return len(self.items) == 0"],
  ["# a rectangle knows its own area", "class Rectangle:", "    def __init__(self, width, height):", "        self.width = width", "        self.height = height", "", "    def area(self):", "        return self.width * self.height", "", "    def __repr__(self):", '        return f"Rectangle({self.width}, {self.height})"'],
  ["# handle a division by zero", "try:", "    result = 10 / 0", "except ZeroDivisionError:", '    print("cannot divide by zero")', "else:", "    print(result)"],
  ["# raise a clear error for bad input", "def set_age(age):", "    if age < 0:", '        raise ValueError("age cannot be negative")', "    return age"],
  ["# clean up with finally, even on errors", "try:", '    data = open("data.txt").read()', "except OSError as exc:", "    print(exc)", "finally:", '    print("cleanup runs no matter what")'],
  ["# read a whole file with a context manager", 'with open("notes.txt") as f:', "    content = f.read()"],
  ["# read a file line by line", 'with open("lines.txt") as f:', "    for line in f:", "        print(line.strip())"],
  ["# write lines to a file", 'with open("out.txt", "w") as f:', "    for i in range(3):", '        f.write(f"line {i}\\n")'],
  ["# import and use the math module", "import math", "assert math.sqrt(16) == 4.0", "print(math.pi)"],
  ["# a random integer between one and six", "import random", "roll = random.randint(1, 6)", "assert 1 <= roll <= 6"],
  ["# pick a random item from a list", "import random", 'options = ["red", "green", "blue"]', "print(random.choice(options))"],
  ["# shuffle a deck in place", "import random", "deck = list(range(52))", "random.shuffle(deck)"],
  ["# today's date with datetime", "from datetime import date", "today = date.today()", "print(today.year, today.month, today.day)"],
  ["# count word frequencies with collections", "from collections import Counter", 'words = ["a", "b", "a", "c", "a", "b"]', 'assert Counter(words)["a"] == 3', "print(Counter(words).most_common(2))"],
  ["# group data with a defaultdict", "from collections import defaultdict", "groups = defaultdict(list)", 'groups["vowels"].append("a")', 'groups["consonants"].append("b")'],
  ["# chain iterators with itertools", "from itertools import chain", "all_items = list(chain([1, 2], [3, 4]))", "assert all_items == [1, 2, 3, 4]"],
  ["# every pair from two lists", "from itertools import product", 'pairs = list(product([1, 2], ["a", "b"]))', "assert len(pairs) == 4"],
  ["# serialise data to json", "import json", 'record = {"name": "nova", "version": 1}', "text = json.dumps(record)", "print(text)"],
  ["# parse json back into python", "import json", "record = json.loads('{\"name\": \"nova\"}')", 'assert record["name"] == "nova"'],
  ["# list, tuple, and dict are different shapes", "numbers = [1, 2, 3]", "pair = (1, 2)", 'lookup = {"a": 1, "b": 2}'],
  ["# unpack a tuple into names", "point = (3, 4)", "x, y = point", "assert x + y == 7"],
  ["# swap two variables the python way", "a, b = 1, 2", "a, b = b, a", "assert a == 2 and b == 1"],
  ["# slice a list", "nums = [0, 1, 2, 3, 4, 5]", "assert nums[1:4] == [1, 2, 3]", "assert nums[::-1] == [5, 4, 3, 2, 1, 0]"],
  ["# flatten a list of lists", "nested = [[1, 2], [3, 4]]", "flat = [x for pair in nested for x in pair]", "assert flat == [1, 2, 3, 4]"],
  ["# sum with a generator expression", "total = sum(n * n for n in range(10))", "assert total == 285"],
  ["# any and all answer yes-no questions", "numbers = [2, 4, 6]", "assert all(n % 2 == 0 for n in numbers)", "assert any(n > 5 for n in numbers)"],
  ["# find the first match with next", "numbers = [1, 8, 3]", "first_big = next((n for n in numbers if n > 5), None)", "assert first_big == 8"],
  ["# count with a while loop", "n = 0", "while n < 5:", "    print(n)", "    n += 1"],
  ["# a function with a default argument", 'def greet(name, greeting="hello"):', '    return f"{greeting}, {name}"', "", 'assert greet("ada") == "hello, ada"'],
  ["# a function that accepts any number of arguments", "def total(*args):", "    return sum(args)", "", "assert total(1, 2, 3) == 6"],
  ["# a function with keyword arguments", "def describe(name, **attrs):", "    for key, value in attrs.items():", "        print(name, key, value)"],
  ["# return several values as a tuple", "def min_max(numbers):", "    return min(numbers), max(numbers)", "", "assert min_max([3, 1, 2]) == (1, 3)"],
  ["# document a function with a docstring", "def celsius_to_fahrenheit(c):", '    """Convert celsius to fahrenheit."""', "    return c * 9 / 5 + 32", "", "assert celsius_to_fahrenheit(100) == 212"],
  ["# guard a script with a main check", "def main():", '    print("program started")', "", 'if __name__ == "__main__":', "    main()"],
  ["# test with plain asserts", "def add(a, b):", "    return a + b", "", "assert add(1, 1) == 2", "assert add(-1, 1) == 0", 'print("all tests passed")'],
  ["# binary search in a sorted list", "def binary_search(items, target):", "    low, high = 0, len(items) - 1", "    while low <= high:", "        mid = (low + high) // 2", "        if items[mid] == target:", "            return mid", "        if items[mid] < target:", "            low = mid + 1", "        else:", "            high = mid - 1", "    return -1", "", "assert binary_search([1, 3, 5, 7], 5) == 2"],
  ["# bubble sort, simple and slow", "def bubble_sort(items):", "    items = items[:]", "    for i in range(len(items)):", "        for j in range(len(items) - 1):", "            if items[j] > items[j + 1]:", "                items[j], items[j + 1] = items[j + 1], items[j]", "    return items"],
  ["# remove duplicates but keep the order", "def dedupe(items):", "    seen = set()", "    out = []", "    for item in items:", "        if item not in seen:", "            seen.add(item)", "            out.append(item)", "    return out"],
  ["# count words in a text", "def word_count(text):", "    counts = {}", "    for word in text.lower().split():", "        counts[word] = counts.get(word, 0) + 1", "    return counts"],
  ["# a cache that remembers past answers", "from functools import lru_cache", "", "@lru_cache(maxsize=None)", "def fib(n):", "    if n < 2:", "        return n", "    return fib(n - 1) + fib(n - 2)"],
  ["# read command line arguments", "import sys", "for arg in sys.argv[1:]:", "    print(arg)"],
  ["# time how long a block takes", "import time", "start = time.time()", "sum(range(100000))", 'print(time.time() - start, "seconds")'],
  ["# walk a directory tree", "import os", 'for root, dirs, files in os.walk("."):', "    for name in files:", "        print(os.path.join(root, name))"],
  ["# an enum makes states readable", "from enum import Enum, auto", "", "class State(Enum):", "    IDLE = auto()", "    RUNNING = auto()", "    DONE = auto()"],
  ["# a dataclass writes the boilerplate for you", "from dataclasses import dataclass", "", "@dataclass", "class Record:", "    name: str", "    value: int"],
  ["# exceptions can carry context", "class NotFoundError(Exception):", "    pass", "", "def fetch(key):", '    raise NotFoundError(f"no record for {key}")'],
  ["# try, except, else and finally together", "def read_number(text):", "    try:", "        value = int(text)", "    except ValueError:", "        return None", "    else:", "        return value", "    finally:", '        print("done parsing")'],
];

/* =============================== generated families =============================== */

const pick = <T,>(arr: T[], r: RNG): T => arr[r.int(arr.length)]!;

const WORDS = ["apple", "banana", "cherry", "delta", "echo", "fig", "grape", "harbor", "iris", "juno"];
const SMALL_NAMES = ["ada", "ben", "cleo", "dev", "elif", "farid", "grace", "hiro"];

function generateSnippets(r: RNG): string[][] {
  const out: string[][] = [];

  // scaled multipliers with correct examples
  for (let k = 2; k <= 12; k++) {
    const x = 1 + r.int(9);
    out.push([`# multiply a value by ${k}`, `def times_${k}(x):`, `    return x * ${k}`, "", `assert times_${k}(${x}) == ${k * x}`]);
  }

  // divisibility tests
  for (const d of [2, 3, 4, 5, 6, 7, 8, 9, 10]) {
    const yes = d * (2 + r.int(8));
    const no = yes + 1;
    out.push([`# check divisibility by ${d}`, `def divisible_by_${d}(n):`, `    return n % ${d} == 0`, "", `assert divisible_by_${d}(${yes})`, `assert not divisible_by_${d}(${no})`]);
  }

  // correct arithmetic facts as code
  for (let i = 0; i < 140; i++) {
    const a = 2 + r.int(18);
    const b = 2 + r.int(18);
    const op = r.int(3);
    if (op === 0) out.push([`# ${a} plus ${b} equals ${a + b}`, `result = ${a} + ${b}`, `assert result == ${a + b}`]);
    else if (op === 1) out.push([`# ${a} times ${b} equals ${a * b}`, `result = ${a} * ${b}`, `assert result == ${a * b}`]);
    else out.push([`# ${a * b} divided by ${b} equals ${a}`, `result = ${a * b} // ${b}`, `assert result == ${a}`]);
  }

  // squares table, computed correctly
  for (const n of [5, 8, 10, 12]) {
    const rows = Array.from({ length: n }, (_, i) => `    ${i}: ${i * i},`);
    out.push([`# squares of the numbers below ${n}`, "squares = {", ...rows, "}", `assert squares[${n - 1}] == ${(n - 1) * (n - 1)}`]);
  }

  // powers of two table, computed correctly
  for (const n of [8, 10, 16]) {
    const rows = Array.from({ length: n }, (_, i) => `    ${i}: ${2 ** i},`);
    out.push([`# powers of two below ${n}`, "powers = {", ...rows, "}", `assert powers[${n - 1}] == ${2 ** (n - 1)}`]);
  }

  // fibonacci list, computed correctly
  for (const n of [8, 10, 12]) {
    const fibs: number[] = [0, 1];
    for (let i = 2; i < n; i++) fibs.push(fibs[i - 1]! + fibs[i - 2]!);
    out.push([`# the first ${n} fibonacci numbers`, `fibs = [${fibs.join(", ")}]`, `assert len(fibs) == ${n}`, `assert fibs[-1] == ${fibs[fibs.length - 1]}`]);
  }

  // a times table as a dict, computed correctly
  for (let k = 2; k <= 9; k++) {
    const rows = Array.from({ length: 5 }, (_, i) => `    ${i + 1}: ${k * (i + 1)},`);
    out.push([`# the ${k} times table`, `table_${k} = {`, ...rows, "}", `assert table_${k}[5] == ${k * 5}`]);
  }

  // greeting functions with names
  for (let i = 0; i < 60; i++) {
    const name = pick(SMALL_NAMES, r);
    const word = pick(WORDS, r);
    out.push([`# greet ${name} by name`, `def greet_${name}():`, `    return "hello ${name}"`, "", `assert greet_${name}() == "hello ${name}"`]);
    if (r.int(3) === 0) out.push([`# echo the word ${word}`, `def echo_${word}(times):`, `    return " ".join(["${word}"] * times)`]);
  }

  // accumulate patterns with varying operators
  const ops = [
    { sym: "+", start: "0", name: "sum", verb: "add up" },
    { sym: "*", start: "1", name: "product", verb: "multiply together" },
  ];
  for (let i = 0; i < 60; i++) {
    const op = pick(ops, r);
    const n = 3 + r.int(6);
    const xs = Array.from({ length: n }, () => 1 + r.int(8));
    const total = xs.reduce((a, b) => (op.sym === "+" ? a + b : a * b), op.sym === "+" ? 0 : 1);
    out.push([
      `# ${op.verb} a list with a loop`,
      `def ${op.name}_loop(numbers):`,
      `    result = ${op.start}`,
      `    for n in numbers:`,
      `        result = result ${op.sym} n`,
      `    return result`,
      "",
      `assert ${op.name}_loop([${xs.join(", ")}]) == ${total}`,
    ]);
  }

  // list comprehension variations
  for (let i = 0; i < 80; i++) {
    const n = 4 + r.int(8);
    const kind = r.int(4);
    if (kind === 0) {
      const vals = Array.from({ length: n }, (_, k) => k * k);
      out.push([`# squares below ${n} with a comprehension`, `squares = [x * x for x in range(${n})]`, `assert squares == [${vals.join(", ")}]`]);
    } else if (kind === 1) {
      out.push([`# multiples of three below ${3 * n}`, `threes = [3 * x for x in range(${n})]`, `assert len(threes) == ${n}`]);
    } else if (kind === 2) {
      const limit = 4 + (n % 4);
      out.push([`# keep names shorter than ${limit} letters`, `names = [${WORDS.slice(0, n).map((w) => `"${w}"`).join(", ")}]`, `short = [w for w in names if len(w) < ${limit}]`]);
    } else {
      const vals = Array.from({ length: n }, (_, k) => (k % 2 === 0 ? 0 : 1));
      out.push([`# a parity mask for range(${n})`, `mask = [0 if x % 2 == 0 else 1 for x in range(${n})]`, `assert mask == [${vals.join(", ")}]`]);
    }
  }

  // string processing variations
  for (let i = 0; i < 60; i++) {
    const word = pick(WORDS, r);
    const kind = r.int(4);
    if (kind === 0) out.push([`# reverse the string ${word}`, `text = "${word}"`, `reversed_text = text[::-1]`, `assert reversed_text == "${[...word].reverse().join("")}"`]);
    else if (kind === 1) out.push([`# the length of ${word}`, `text = "${word}"`, `assert len(text) == ${word.length}`]);
    else if (kind === 2) out.push([`# repeat ${word} three times`, `text = "${word}" * 3`, `assert text == "${word.repeat(3)}"`]);
    else out.push([`# the first and last letter of ${word}`, `text = "${word}"`, `ends = text[0] + text[-1]`, `assert ends == "${word[0]}${word[word.length - 1]}"`]);
  }

  // if elif else ladders
  for (let i = 0; i < 45; i++) {
    const score = 40 + r.int(60);
    const grade = score >= 90 ? "a" : score >= 80 ? "b" : score >= 70 ? "c" : score >= 60 ? "d" : "f";
    out.push([
      "# map a score to a letter grade",
      `score = ${score}`,
      "if score >= 90:",
      '    grade = "a"',
      "elif score >= 80:",
      '    grade = "b"',
      "elif score >= 70:",
      '    grade = "c"',
      "elif score >= 60:",
      '    grade = "d"',
      "else:",
      '    grade = "f"',
      `assert grade == "${grade}"`,
    ]);
  }

  // while loops counting to a target
  for (let i = 0; i < 40; i++) {
    const target = 5 + r.int(15);
    out.push([
      `# count from zero up to ${target - 1}`,
      "n = 0",
      `while n < ${target}:`,
      "    print(n)",
      "    n += 1",
    ]);
  }

  // functions returning facts about lists
  for (let i = 0; i < 45; i++) {
    const n = 3 + r.int(5);
    const xs = Array.from({ length: n }, () => 1 + r.int(20));
    const kind = r.int(3);
    if (kind === 0) {
      const has = xs.includes(7);
      const list = has ? xs : xs.map((x) => (x === 7 ? 8 : x));
      out.push([`# does the list contain seven`, `numbers = [${list.join(", ")}]`, `has_seven = 7 in numbers`, `assert has_seven == ${list.includes(7) ? "True" : "False"}`]);
    } else if (kind === 1) {
      const mx = Math.max(...xs);
      out.push([`# the largest of [${xs.join(", ")}]`, `numbers = [${xs.join(", ")}]`, `assert max(numbers) == ${mx}`]);
    } else {
      const sm = xs.reduce((a, b) => a + b, 0);
      out.push([`# the sum of [${xs.join(", ")}]`, `numbers = [${xs.join(", ")}]`, `assert sum(numbers) == ${sm}`]);
    }
  }

  // tiny class variations
  for (let i = 0; i < 35; i++) {
    const name = pick(["Balance", "Temperature", "Score", "Timer", "Meter"], r);
    const start = 1 + r.int(20);
    const step = 1 + r.int(5);
    if (name === "Balance") {
      out.push([
        "# a small balance with deposit and withdraw",
        "class Balance:",
        `    def __init__(self, start=${start}):`,
        "        self.amount = start",
        "",
        "    def deposit(self, value):",
        "        self.amount += value",
        "",
        "    def withdraw(self, value):",
        "        if value > self.amount:",
        '            raise ValueError("not enough funds")',
        "        self.amount -= value",
        "",
        "    def total(self):",
        "        return self.amount",
      ]);
    } else {
      out.push([
        `# a ${name.toLowerCase()} that can be bumped by a step`,
        `class ${name}:`,
        `    def __init__(self, start=${start}, step=${step}):`,
        "        self.value = start",
        "        self.step = step",
        "",
        "    def bump(self):",
        "        self.value += self.step",
        "",
        "    def get(self):",
        "        return self.value",
        "",
        `x = ${name}()`,
        "x.bump()",
        `assert x.get() == ${start + step}`,
      ]);
    }
  }

  return out;
}

const pyCache = new Map<number, { text: string; stats: CorpusStats }>();

/**
 * Build the python corpus. Deterministic for a given seed, memoised per seed.
 */
export function buildPythonCorpus(seed = 99): { text: string; stats: CorpusStats } {
  const cached = pyCache.get(seed);
  if (cached) return cached;

  const r = new RNG(seed);
  const blocks: string[][] = [];

  for (const s of SNIPPETS) blocks.push(s);
  // hand written snippets are the backbone; the model should see each many times
  for (let pass = 0; pass < 4; pass++) for (const s of r.shuffle(SNIPPETS)) blocks.push(s);

  for (let pass = 0; pass < 4; pass++) for (const s of generateSnippets(r)) blocks.push(s);

  const mixed = r.shuffle(blocks);
  let lineCount = 0;
  for (const b of mixed) lineCount += b.length;
  // blank line between snippets keeps snippet boundaries visible to the model
  const text = mixed.map((b) => b.join("\n")).join("\n\n");

  const words = tokenizeWords(text);
  const freq = new Map<string, number>();
  for (const w of words) freq.set(w, (freq.get(w) ?? 0) + 1);
  const sorted = [...freq.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));

  const result = {
    text,
    stats: {
      chars: text.length,
      lines: lineCount,
      templateLines: 0,
      handwrittenLines: lineCount,
      tokens: words.length,
      uniqueTokens: freq.size,
      topTokens: sorted.slice(0, 25).map(([w, n]) => ({ token: w, count: n })),
    } satisfies CorpusStats,
  };
  if (pyCache.size > 8) pyCache.clear();
  pyCache.set(seed, result);
  return result;
}
