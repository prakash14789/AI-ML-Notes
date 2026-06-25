# 84. Expressions, Operators, and Conditions - Nishut Suman - 27 May 2026

# 📘Lecture Note: Expressions, Operators, and Conditions

# 🎯 Session Overview

This lecture introduced the foundational concepts of Python programming with a focus on:

- Expressions and operators
- Variables and data types
- User input handling
- Arithmetic, comparison, and logical operations
- Conditional statements and decision making

The session combined theoretical understanding with practical coding examples such as:

- Login verification systems
- Billing systems with discounts and taxes
- Eligibility checking programs

The instructor emphasized that even in the age of AI and no-code platforms, understanding programming fundamentals is essential for building scalable, maintainable, and customizable software systems.

---

# 🐍 1. Introduction to Python

---

## 📌 What is Python?

Python is a high-level programming language designed to make coding simple, readable, and beginner-friendly.

It allows programmers to:

- Write instructions in a human-readable format
- Automate tasks
- Build applications
- Analyze data
- Develop AI and machine learning systems

Python acts as a bridge between human instructions and machine execution.

---

# 🚀 Why Python is Popular

The lecture discussed why Python has become one of the most widely used programming languages today.

---

## ✅ Beginner Friendly

Python syntax is simple and resembles English language structure.

Example:

```python
print("Welcome to Masai")

```

This simplicity helps beginners learn programming concepts quickly.

---

## ✅ Dynamically Typed Language

Python automatically detects data types.

Programmers do not need to declare variable types explicitly.

Example:

```python
age = 20
name = "Rahul"

```

Python automatically identifies:

- `20` as an integer
- `"Rahul"` as a string

---

## ✅ Extensive Libraries

Python provides powerful libraries for different domains.

Examples:

- Pandas → Data analysis
- NumPy → Numerical computation
- Flask/Django → Web development
- TensorFlow/PyTorch → Machine learning

---

## ✅ Wide Industry Usage

Python is used in:

- Artificial Intelligence
- Data Science
- Automation
- Web Development
- Backend Systems
- Cybersecurity
- Cloud Computing

---

# 📝 2. Writing Basic Python Programs

The lecture started with simple Python statements.

---

# 📌 The print() Function

The `print()` function displays output on the screen.

Example:

```python
print("Hello World")

```

Output:

```
Hello World

```

---

# 📌 Multiple Print Statements

```python
print("Hello World")
print("My name is ABC")

```

Each print statement outputs content on a new line.

---

# 📌 Comments in Python

Comments are notes written for programmers and are ignored during execution.

Comments begin with `#`.

Example:

```python
# This is a comment
print("Python")

```

Comments improve:

- Code readability
- Documentation
- Maintenance

---

# 📦 3. Variables in Python

---

# 📘 What are Variables?

Variables are containers used to store data values.

They help:

- Store information
- Reuse data
- Make programs dynamic

---

# 📌 Example

```python
name = "Masai"
age = 20

```

Here:

- `name` and `age` are variables
- `"Masai"` and `20` are stored values

---

# 📌 Assignment Operator (=)

The assignment operator stores values into variables.

```python
salary = 50000

```

Meaning:

- Left side → variable name
- Right side → value being assigned

---

# 📌 Variable Naming Rules

Variables:
✅ Can contain letters, numbers, underscores
✅ Are case-sensitive

Variables:
❌ Cannot contain spaces
❌ Cannot start with numbers

---

# 📌 Case Sensitivity

```python
age = 20
Age = 25

```

These are treated as two different variables.

---

# 🧩 4. Data Types in Python

---

# 📘 What are Data Types?

Data types define the kind of value stored in a variable.

Python uses data types to determine:

- Memory usage
- Valid operations
- Data handling behavior

---

# 📌 Main Data Types Covered

---

# 🔹 Integer (int)

Stores whole numbers.

Example:

```python
age = 25

```

---

# 🔹 Float (float)

Stores decimal values.

Example:

```python
price = 2500.50

```

---

# 🔹 String (str)

Stores text enclosed in quotes.

Example:

```python
name = "Rahul"

```

Strings can use:

- Single quotes `' '`
- Double quotes `" "`

---

# 🔹 Boolean (bool)

Stores logical values:

- `True`
- `False`

Example:

```python
is_enrolled = True

```

---

# 📌 Example Combining Data Types

```python
name = "Masai"
age = 20
is_enrolled = True
fee = 2500.50

print(name, age, is_enrolled, fee)

```

---

# 🔍 5. Taking User Input

---

# 📘 input() Function

The `input()` function allows users to enter data during program execution.

Example:

```python
name = input("Enter your name: ")
print("Welcome to Masai", name)

```

---

# 📌 Important Note

`input()` always returns data as a string.

---

# 📌 Type Conversion

To perform numerical operations, conversion is necessary.

Example:

```python
age = int(input("Enter your age: "))

```

This converts input from string to integer.

---

# ➕ 6. Arithmetic Operators

Arithmetic operators perform mathematical calculations.

---

# 📌 Common Arithmetic Operators

Operator | Purpose | Example
+ | Addition | a + b
- | Subtraction | a - b
* | Multiplication | a * b
/ | Division | a / b
% | Modulo (remainder) | a % b
** | Exponentiation | a ** b

---

# 📌 Example

```python
a = 24
b = 5

print("Addition:", a + b)
print("Subtraction:", a - b)
print("Multiplication:", a * b)
print("Division:", a / b)
print("Modulo:", a % b)
print("Power:", a ** b)

```

---

# 📌 Modulo Operator (%)

Returns the remainder after division.

Example:

```python
24 % 5

```

Output:

```
4

```

---

# 📌 Exponentiation Operator (**)

Used for powers.

Example:

```python
2 ** 3

```

Output:

```
8

```

---

# ⚖️ 7. Comparison Operators

Comparison operators compare values and return Boolean results.

---

# 📌 Comparison Operators Table

Operator | Meaning
== | Equal to
!= | Not equal to
> | Greater than
< | Less than
>= | Greater than or equal to
<= | Less than or equal to

---

# 📌 Example

```python
x = 5
y = 7

print(x == y)
print(x < y)
print(x != y)

```

Output:

```
False
True
True

```

---

# 🧠 8. Logical Operators

Logical operators combine multiple conditions.

---

# 📌 Types of Logical Operators

Operator | Meaning
and | Both conditions must be True
or | At least one condition must be True
not | Reverses the Boolean value

---

# 📌 Example

```python
age = 20
salary = 50000

print(age >= 20 and salary > 40000)
print(age > 30 or salary > 40000)
print(not(age > 30))

```

---

# 📌 Understanding AND

Returns `True` only if both conditions are true.

---

# 📌 Understanding OR

Returns `True` if at least one condition is true.

---

# 📌 Understanding NOT

Reverses the result.

Example:

```python
not(True)

```

Output:

```
False

```

---

# 🔀 9. Conditional Statements

Conditional statements allow programs to make decisions.

---

# 📘 Why Conditions Matter

Programs often need to:

- Validate inputs
- Check eligibility
- Apply business logic
- Handle different scenarios

Conditional statements make this possible.

---

# 📌 Python Conditional Keywords

Keyword | Purpose
if | Checks a condition
elif | Checks another condition
else | Executes when all conditions fail

---

# 📌 Basic Syntax

```python
if condition:
    statement
elif another_condition:
    statement
else:
    statement

```

---

# 📌 Importance of Indentation

Python uses indentation instead of braces `{}`.

Correct indentation is mandatory.

Example:

```python
if age >= 18:
    print("Eligible")

```

---

# 🗳️ 10. Voting Eligibility Example

```python
age = int(input("Enter your age: "))

if age >= 18:
    print("You are eligible to vote")
elif age == 17:
    print("Not eligible but close")
else:
    print("You are not eligible")

```

---

# 📌 Program Logic

- Age ≥ 18 → Eligible
- Age = 17 → Nearly eligible
- Otherwise → Not eligible

---

# 🔐 11. ATM Login Verification System

The lecture demonstrated a practical login verification example.

---

# 📌 Problem Statement

Validate:

- Username
- Password

---

# 📌 Example Logic

```python
username = input("Enter username: ")
password = input("Enter password: ")

if username == "admin" and password == "1234":
    print("Login successful")

elif username == "admin" or password == "1234":
    print("Forgot username or password")

else:
    print("Invalid credentials")

```

---

# 📌 Concepts Practiced

- Logical operators
- String comparison
- Conditional branching
- Authentication logic

---

# 🧾 12. Billing System Example

The lecture also demonstrated a discount and GST billing system.

---

# 📌 Problem Logic

- Spending > 6000 → 30% discount
- Spending > 2000 → 20% discount
- Add 18% GST after discount

---

# 📌 Code Example

```python
spending_amount = float(input("Enter spending amount: "))

if spending_amount > 6000:
    spending_amount *= 0.7

elif spending_amount > 2000:
    spending_amount *= 0.8

taxed_amount = spending_amount * 1.18

print("Final payable amount is", taxed_amount)

```

---

# 📌 Important Learning

The order of conditions matters.

Why?

If the lower condition is checked first, higher discounts may never execute.

---

# ⚠️ 13. Common Beginner Errors

The instructor highlighted common mistakes beginners make.

---

# 📌 Syntax Errors

Examples:

- Missing brackets
- Missing commas
- Incorrect indentation

Python usually provides line numbers indicating where the error occurred.

---

# 📌 Type Errors

Example:

```python
"10" + 5

```

This causes an error because:

- String and integer cannot be added directly

---

# 📌 Case Sensitivity Errors

```python
Name = "Rahul"
print(name)

```

This causes an error because:

- `Name` and `name` are different variables

---

# 🎯 14. Key Concepts Learned

---

# ✅ Python Fundamentals

- Python syntax
- Readability
- Comments
- Variables

---

# ✅ Data Types

- Integer
- Float
- String
- Boolean

---

# ✅ User Input Handling

- input()
- Type conversion

---

# ✅ Operators

- Arithmetic operators
- Comparison operators
- Logical operators

---

# ✅ Conditional Programming

- if
- elif
- else
- Decision making

---

# ✅ Real-World Problem Solving

- Login systems
- Billing systems
- Eligibility checking

---

# 📚 15. Summary

This session built a strong foundation in Python programming by introducing the core building blocks required for writing logical and interactive programs.

The lecture covered:

- Variables and data types
- Expressions and operators
- User input handling
- Conditional statements
- Real-world logic implementation

Students learned how Python combines simplicity and power, making it suitable for beginners while also being widely used in professional software development, AI, automation, and data science.

The instructor emphasized that mastering these foundational concepts through continuous practice is essential before progressing into advanced programming topics.

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