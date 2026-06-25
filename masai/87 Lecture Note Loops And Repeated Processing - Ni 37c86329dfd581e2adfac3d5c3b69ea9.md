# 87. Lecture Note: Loops And Repeated Processing - Nishut Suman - 2 Jun 2026

# 📘 Lecture Note: Python Fundamentals: Loops and Repeated Processing

# 🎯 Session Overview

This session introduced one of the most important concepts in programming: **Loops**. Loops allow programmers to execute a block of code repeatedly without writing the same instructions multiple times. The lecture built upon previously learned concepts such as variables, data types, operators, and conditional statements, and demonstrated how loops help automate repetitive tasks efficiently.

The session covered:

- Introduction to repetitive programming tasks
- For loops and while loops
- The `range()` function
- Loop control statements (`break` and `continue`)
- Boolean flags
- Prime number checking
- Multiplication table generation
- Real-world applications of loops
- Common mistakes such as infinite loops

By the end of the session, students understood how loops reduce code duplication and form the foundation of many programming and automation tasks.

---

# Why Do We Need Loops?

Imagine being asked to print the word **"Welcome"** 100 times.

Without loops:

```python
print("Welcome")
print("Welcome")
print("Welcome")
...

```

This approach becomes:

- Time-consuming
- Difficult to manage
- Error-prone

Loops solve this problem by allowing a set of instructions to execute repeatedly.

Example:

```python
for i in range(100):
    print("Welcome")

```

This single block performs the same task much more efficiently.

---

# Understanding Repetition in Programming

Many real-world activities involve repetition:

- Sending notifications to multiple users
- Processing thousands of records
- Checking passwords repeatedly
- Generating reports
- Displaying menu options until a user exits

Programming loops are designed specifically to handle such repetitive operations.

---

# Types of Loops in Python

Python provides two primary looping mechanisms:

1. **For Loop**
2. **While Loop**

Each serves a different purpose depending on the problem being solved.

---

# For Loop

## What is a For Loop?

A **for loop** is used when the number of iterations is known in advance.

It executes a block of code repeatedly for each value in a sequence.

### Basic Syntax

```python
for variable in sequence:
    statements

```

Example:

```python
for i in range(5):
    print(i)

```

Output:

```
0
1
2
3
4

```

---

# Understanding the Loop Variable

In the example:

```python
for i in range(5):

```

`i` is called the **loop variable**.

Its value changes automatically during each iteration.

Iteration | Value of i
1 | 0
2 | 1
3 | 2
4 | 3
5 | 4

---

# The range() Function

The `range()` function generates a sequence of numbers.

It is commonly used with for loops.

---

## Form 1: Single Parameter

```python
range(stop)

```

Example:

```python
for i in range(5):
    print(i)

```

Output:

```
0
1
2
3
4

```

Starts from 0 by default and stops before 5.

---

## Form 2: Two Parameters

```python
range(start, stop)

```

Example:

```python
for i in range(3, 10):
    print(i)

```

Output:

```
3
4
5
6
7
8
9

```

Starts from 3 and stops before 10.

---

## Form 3: Three Parameters

```python
range(start, stop, step)

```

Example:

```python
for i in range(1, 11, 2):
    print(i)

```

Output:

```
1
3
5
7
9

```

The step value determines how much the number increases after each iteration.

---

# Real-Life Analogy for Step Value

Imagine climbing stairs.

### Step = 1

You climb one stair at a time:

```
1 → 2 → 3 → 4 → 5

```

### Step = 2

You skip alternate stairs:

```
1 → 3 → 5 → 7 → 9

```

The range function works similarly.

---

# Practical Examples Using For Loops

---

## Printing Numbers from 0 to 10

```python
for i in range(11):
    print(i)

```

---

## Printing Numbers from 3 to 10

```python
for i in range(3, 11):
    print(i)

```

---

## Printing Even Numbers

```python
for i in range(2, 11, 2):
    print(i)

```

Output:

```
2
4
6
8
10

```

---

## Printing Odd Numbers

```python
for i in range(1, 11, 2):
    print(i)

```

Output:

```
1
3
5
7
9

```

---

# While Loop

## What is a While Loop?

A **while loop** executes as long as a specified condition remains true.

Unlike a for loop, the number of iterations is usually unknown beforehand.

---

## Basic Syntax

```python
while condition:
    statements

```

---

## Example

```python
count = 1

while count <= 5:
    print(count)
    count += 1

```

Output:

```
1
2
3
4
5

```

---

# Real-Life Analogy of While Loop

The lecture used the example of a strike.

Suppose workers continue their strike until 10 demands are fulfilled.

```
While demands are not fulfilled:
    Continue strike

```

The number of days required is unknown.

This is exactly how a while loop behaves.

---

# Difference Between For and While Loops

For Loop | While Loop
Used when iterations are known | Used when iterations are unknown
Uses range or sequence | Uses condition
Easier for counting | Better for condition-based repetition
Less prone to infinite loops | Requires careful condition management

---

# Infinite Loops

One common mistake is forgetting to update the loop variable.

Example:

```python
count = 1

while count <= 5:
    print(count)

```

Output:

```
1
1
1
1
...

```

The loop never stops because `count` is never increased.

This is called an **infinite loop**.

---

# Loop Control Statements

Python provides special statements to alter loop execution.

---

# Break Statement

The `break` statement immediately exits the loop.

---

## Example

```python
for i in range(10):
    if i == 5:
        break
    print(i)

```

Output:

```
0
1
2
3
4

```

When `i` becomes 5, the loop stops.

---

# Real-Life Example of Break

Imagine searching for a book in a shelf.

Once the book is found:

```
Stop searching.

```

This is similar to `break`.

---

# Continue Statement

The `continue` statement skips the current iteration and moves to the next one.

---

## Example

```python
for i in range(5):
    if i == 2:
        continue
    print(i)

```

Output:

```
0
1
3
4

```

The number 2 is skipped.

---

# Real-Life Example of Continue

Imagine calling names from a list.

If a student is absent:

```
Skip the student
Continue with the next one

```

This behavior resembles `continue`.

---

# Understanding Flags in Programming

A **flag** is a variable used to track a condition or status.

Typically, flags are Boolean values:

```python
True
False

```

---

## Why Use Flags?

Flags help remember information while a program runs.

Example:

```python
is_logged_in = False

```

Later:

```python
is_logged_in = True

```

---

# Prime Number Program Using Flags

The lecture demonstrated using flags to determine whether a number is prime.

---

## What is a Prime Number?

A prime number is divisible only by:

- 1
- Itself

Examples:

```
2, 3, 5, 7, 11

```

---

## Algorithm

1. 
Assume number is prime.

2. 
Check divisibility from 2 to n-1.

3. 
If divisible:

Set flag to False
Stop checking

4. 
Print result.

---

## Program

```python
num = int(input("Enter number: "))

is_prime = True

for i in range(2, num):
    if num % i == 0:
        is_prime = False
        break

if is_prime:
    print("Prime Number")
else:
    print("Not Prime Number")

```

---

# Understanding the Logic

Example:

```
Number = 15

```

Check:

```
15 % 2
15 % 3

```

Since:

```
15 % 3 = 0

```

The number is not prime.

Flag becomes:

```python
False

```

---

# Multiplication Table Program

Loops are excellent for generating multiplication tables.

---

## Example: Table of 5

```python
for i in range(1, 11):
    print("5 x", i, "=", 5 * i)

```

Output:

```
5 x 1 = 5
5 x 2 = 10
...
5 x 10 = 50

```

---

# Security Example: Brute Force Attack

The lecture connected loops to cybersecurity.

Imagine trying different passwords repeatedly:

```
1234
1235
1236
...

```

The loop continues until the correct password is found.

This concept forms the basis of:

- Password cracking attempts
- Brute force attacks
- Automated testing

Understanding loops therefore has applications beyond basic programming.

---

# Practical Applications of Loops

Loops are used in:

### Software Development

- Processing user data
- Generating reports
- Searching records

### Data Analysis

- Iterating through datasets
- Cleaning data

### Web Development

- Displaying products
- Processing forms

### Cybersecurity

- Log analysis
- Automated scanning
- Password testing

### Artificial Intelligence

- Model training iterations
- Data preprocessing

---

# Common Mistakes Beginners Make

## 1. Infinite Loops

Forgetting to update loop variables.

---

## 2. Wrong Range Values

Example:

```python
range(1,10)

```

Many beginners expect 10 to be included.

Actual output:

```
1 to 9

```

The stop value is excluded.

---

## 3. Misusing Break

Using break too early can terminate loops unexpectedly.

---

## 4. Misusing Continue

Using continue carelessly may skip important logic.

---

# Recommended Learning Tools

The instructor recommended using **Python Tutor** and similar visualizers.

Benefits:

- Execute code step by step
- Visualize variable values
- Understand loop execution flow
- Debug programs more effectively

---

# Homework and Practice Exercises

### Exercise 1

Write a program to print the multiplication table of 5.

---

### Exercise 2

Write a program that allows only three attempts to enter the correct ATM PIN.

---

### Exercise 3

Print all odd numbers between 1 and 50.

---

### Exercise 4

Modify the prime number program to check multiple numbers.

---

# Key Takeaways

- Loops automate repetitive tasks.
- Python provides two major loop types: `for` and `while`.
- The `range()` function controls iteration sequences.
- `break` exits loops immediately.
- `continue` skips the current iteration.
- Boolean flags help track conditions.
- Loops are fundamental to solving real-world programming problems.
- Understanding logic is more important than memorizing syntax.
- Proper loop design prevents infinite loops and improves efficiency.

---

# Conclusion

This session established a strong foundation in iterative programming through loops. Students learned how to automate repetitive tasks, control execution flow, and solve practical problems using both `for` and `while` loops. Through examples such as prime number checking, multiplication tables, login attempts, and brute force attack simulations, the lecture demonstrated how loops are one of the most powerful and frequently used constructs in programming.

Mastering loops is essential because they appear in nearly every area of software development, data science, automation, cybersecurity, and artificial intelligence. Continuous practice with loop-based problems will significantly strengthen programming logic and problem-solving skills.

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