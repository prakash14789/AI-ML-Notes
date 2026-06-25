# 91. Python Data Structures: Lists and Dictionaries - Nishut Suman - 8 Jun 2026

# 📘 Post Lecture Note

# Python Data Structures: Lists and Dictionaries

---

# 🎯 Session Overview

This session introduced two of the most important built-in data structures in Python: **Lists** and **Dictionaries**. Data structures are fundamental to programming because they determine how data is stored, organized, accessed, and manipulated efficiently.

The lecture explored:

- The importance of data structures
- Python Lists and their operations
- Indexing and iteration
- Searching and membership checking
- Finding largest and second largest elements
- Python Dictionaries and key-value storage
- Frequency counting using dictionaries
- Choosing the right data structure for a problem

The session emphasized that understanding data structures is far more important than memorizing syntax because efficient data organization directly impacts application performance, scalability, and maintainability.

---

# Understanding Data Structures

## What are Data Structures?

A **data structure** is a way of organizing and storing data so that it can be accessed and modified efficiently.

Think of a data structure as a container designed for a specific purpose.

### Real-Life Analogy

Suppose you want to store water.

You would choose:

- A bucket
- A bottle
- A tank

You would not choose a basket with holes because it is unsuitable for storing water.

Similarly, in programming, choosing the wrong data structure can lead to:

- Slow programs
- Higher memory usage
- Difficult code maintenance

Choosing the correct data structure helps build efficient and scalable software systems.

---

# Why are Data Structures Important?

Modern applications handle enormous amounts of data.

Examples:

- Instagram stores and processes petabytes of images and videos.
- E-commerce platforms manage millions of products and transactions.
- Banking systems process millions of records daily.

Efficient data structures help:

- Reduce memory consumption
- Improve processing speed
- Simplify data retrieval
- Lower infrastructure costs
- Enhance application performance

---

# Python Lists

## What is a List?

A **list** is an ordered collection of items stored in a single variable.

Unlike arrays in many programming languages, Python lists can store different data types together.

### Example

```python
friends = ["Ved", "Sahil", "Shakshi", "Shubha", "Himakar"]

```

A list can also contain mixed data types:

```python
data = ["Masai", 25, True, 99.5]

```

---

# Characteristics of Lists

### Ordered

Elements maintain their insertion order.

```python
colors = ["Red", "Blue", "Green"]

```

Output order remains:

```
Red
Blue
Green

```

---

### Mutable

Lists can be modified after creation.

```python
students = ["Rahul", "Aman", "Priya"]

students[1] = "Rohit"

print(students)

```

Output:

```python
['Rahul', 'Rohit', 'Priya']

```

---

### Allows Duplicate Values

```python
numbers = [10, 20, 10, 30]

```

Duplicates are permitted.

---

# List Indexing

Python uses **zero-based indexing**.

### Example

```python
students = ["Rahul", "Priya", "Aman", "Neha"]

```

Index | Value
0 | Rahul
1 | Priya
2 | Aman
3 | Neha

Accessing elements:

```python
print(students[0])

```

Output:

```python
Rahul

```

---

# Negative Indexing

Python also supports negative indexing.

Index | Value
-1 | Neha
-2 | Aman
-3 | Priya
-4 | Rahul

Example:

```python
print(students[-1])

```

Output:

```python
Neha

```

This is useful when accessing elements from the end of a list.

---

# Common List Operations

## Accessing Elements

```python
students = ["Rahul", "Priya", "Aman"]

print(students[1])

```

Output:

```python
Priya

```

---

## Modifying Elements

```python
students[0] = "Rohan"

```

---

## Adding Elements

Using `append()`:

```python
students.append("Karan")

```

Before:

```python
['Rahul', 'Priya']

```

After:

```python
['Rahul', 'Priya', 'Karan']

```

---

## Finding Length

Using `len()`:

```python
students = ["Rahul", "Priya", "Aman"]

print(len(students))

```

Output:

```python
3

```

---

# Iterating Through Lists

Iteration means processing each element one by one.

### Example

```python
students = ["Rahul", "Priya", "Aman"]

for student in students:
    print(student)

```

Output:

```
Rahul
Priya
Aman

```

Iteration is one of the most important concepts in programming and is widely used for:

- Searching
- Sorting
- Calculations
- Data processing

---

# Finding the Largest Element in a List

## Problem Statement

Find the largest number without using Python's built-in `max()` function.

### Example

```python
numbers = [12, 45, 8, 99, 34]

```

Expected Output:

```
99

```

---

## Approach

1. Assume first element is largest.
2. Compare it with each remaining element.
3. Update largest whenever a bigger value is found.

---

## Solution

```python
numbers = [12, 45, 8, 99, 34]

largest = numbers[0]

for num in numbers:
    if num > largest:
        largest = num

print("Largest Number:", largest)

```

Output:

```
Largest Number: 99

```

---

# Built-in Function: max()

Python provides a shortcut:

```python
numbers = [12, 45, 8, 99, 34]

print(max(numbers))

```

Output:

```
99

```

However, understanding the iterative approach helps build problem-solving skills.

---

# Membership Checking in Lists

Checking whether an element exists inside a list is a common task.

### Using the `in` Operator

```python
students = ["Rahul", "Priya", "Aman"]

print("Rahul" in students)

```

Output:

```python
True

```

---

### Example

```python
print("Karan" in students)

```

Output:

```python
False

```

---

# Real-World Example: Vault Access

Suppose only authorized personnel can access a secure vault.

```python
authorized_users = ["Alice", "Bob", "Charlie"]

user = "Bob"

if user in authorized_users:
    print("Access Granted")
else:
    print("Access Denied")

```

This demonstrates practical use of membership checking.

---

# Searching for an Element in a List

### Using a Loop

```python
items = ["Laptop", "Phone", "Tablet"]

search_item = input("Enter item: ")

found = False

for item in items:
    if item == search_item:
        found = True
        break

if found:
    print("Item Found")
else:
    print("Item Not Found")

```

---

# Counting Occurrences in a List

### Example

```python
numbers = [1, 2, 3, 2, 4, 2, 5]

```

Count how many times 2 appears.

---

### Using count()

```python
print(numbers.count(2))

```

Output:

```
3

```

---

### Using a Loop

```python
count = 0

for num in numbers:
    if num == 2:
        count += 1

print(count)

```

Output:

```
3

```

---

# Finding the Second Largest Element

This was one of the important problem-solving exercises discussed during the session.

---

# Method 1: Two-Pass Approach

### Step 1

Find the largest element.

### Step 2

Find the largest value excluding the maximum.

---

### Example

```python
numbers = [10, 50, 30, 40]

```

Largest = 50

Second Largest = 40

---

# Method 2: One-Pass Approach

Maintain two variables:

```python
largest
second_largest

```

Update them during a single iteration.

---

### Solution

```python
numbers = [10, 50, 30, 40]

largest = float('-inf')
second_largest = float('-inf')

for num in numbers:

    if num > largest:
        second_largest = largest
        largest = num

    elif num > second_largest and num != largest:
        second_largest = num

print(second_largest)

```

Output:

```
40

```

---

# Understanding Dictionaries

## What is a Dictionary?

A dictionary stores data using **key-value pairs**.

Unlike lists, dictionary elements are accessed using keys rather than positions.

---

## Creating a Dictionary

```python
student = {
    "name": "Rohan",
    "age": 23,
    "address": "Bangalore"
}

```

---

# Key-Value Concept

Key | Value
name | Rohan
age | 23
address | Bangalore

The key acts as an identifier for retrieving information.

---

# Why Use Dictionaries?

Consider storing student information.

Using a list:

```python
student = ["Rohan", 23, "Bangalore"]

```

The meaning of each position must be remembered.

Using a dictionary:

```python
student = {
    "name": "Rohan",
    "age": 23,
    "address": "Bangalore"
}

```

The data becomes self-explanatory.

---

# Accessing Dictionary Values

Using keys:

```python
student = {
    "name": "Rohan",
    "age": 23
}

print(student["name"])

```

Output:

```
Rohan

```

---

# Modifying Dictionary Values

```python
student["age"] = 24

```

Updated dictionary:

```python
{
    "name": "Rohan",
    "age": 24
}

```

---

# Lists vs Dictionaries

Feature | List | Dictionary
Storage Style | Ordered elements | Key-value pairs
Access Method | Index | Key
Duplicate Values | Allowed | Keys must be unique
Best For | Sequential data | Associated data

---

# Nested Data Structures

Python allows combining lists and dictionaries.

---

## Dictionary Inside List

```python
students = [
    {"name": "Rahul", "age": 21},
    {"name": "Priya", "age": 22}
]

```

---

## Dictionary Inside Dictionary

```python
employee = {
    "name": "Rohan",
    "address": {
        "city": "Bangalore",
        "state": "Karnataka"
    }
}

```

Nested structures are widely used in APIs and real-world applications.

---

# Frequency Counting Using Dictionaries

## Problem

Count how many times each character appears in a string.

### Example

```python
word = "banana"

```

Expected Output:

```python
{
    'b': 1,
    'a': 3,
    'n': 2
}

```

---

## Solution

```python
word = "banana"

freq = {}

for char in word:

    if char in freq:
        freq[char] += 1
    else:
        freq[char] = 1

print(freq)

```

Output:

```python
{'b': 1, 'a': 3, 'n': 2}

```

This is one of the most common dictionary interview questions.

---

# Best Practices Discussed

## Focus on Concepts Before Syntax

Syntax can be learned with practice.

Understanding:

- Why lists exist
- Why dictionaries exist
- When to use each

is more important.

---

## Use Visualization Tools

Tools such as Python visualizers help understand:

- Variable changes
- Loop execution
- Data structure behavior

---

## Think About Data Requirements

Before selecting a data structure, ask:

- Is order important?
- Do I need fast lookup?
- Am I storing related information?

The answers help determine whether a list or dictionary is more appropriate.

---

## Test Edge Cases

Always consider:

- Empty lists
- Missing keys
- Duplicate values
- Case-sensitive string comparisons

Proper testing improves code reliability.

---

# Key Takeaways

- Data structures help organize and manage data efficiently.
- Lists store ordered collections of elements.
- Python lists support heterogeneous data types.
- Lists use zero-based and negative indexing.
- Iteration is essential for processing list data.
- Membership checking can be performed using the `in` operator.
- Dictionaries store information as key-value pairs.
- Keys provide meaningful access to data.
- Dictionaries are ideal for representing real-world entities and attributes.
- Nested data structures help model complex information.
- Understanding iteration and data structures is fundamental for solving programming problems efficiently.

---

# Conclusion

This session laid a strong foundation in Python data structures by introducing **Lists** and **Dictionaries**, two of the most frequently used tools in Python programming. Students learned how to store, access, search, and manipulate data efficiently while solving practical problems such as finding the largest and second-largest elements, counting occurrences, and building frequency maps.

These concepts form the backbone of software development, data analysis, machine learning, web development, and automation. A solid understanding of lists and dictionaries will make future topics such as functions, object-oriented programming, APIs, databases, and data science significantly easier to learn and apply.

            .markdown-preview table, 
            .markdown-preview th, 
            .markdown-preview td {
              background-color: white !important;
              color: black !important;
            }
            .markdown-preview pre, 
            .markdown-preview code {
              background-color: inherit !important;
              color: inherit !important;
              box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }