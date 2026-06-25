# 79. Lecture Note: Python Foundations, Variables and Data Types - Nishut Suman - 18 May 2026

# 📘 Post Lecture Note

# Python Fundamentals: Variables, Data Types, and Basic Programming

---

# 🎯 Session Overview

This lecture introduced the foundational concepts of Python programming and focused on understanding how Python is used to write instructions for computers in a simple and readable way.

The session covered:

- Introduction to Python
- Writing basic Python programs
- Variables
- Data types
- Print statements
- Type checking
- Basic arithmetic operations
- Practical coding examples

The lecture combined theory with hands-on examples to help beginners develop a strong understanding of programming fundamentals.

---

# 🐍 1. Introduction to Python

---

## 📌 What is Python?

Python is a high-level programming language used to communicate instructions to a computer.

It is designed to:

- Be simple
- Use English-like syntax
- Be easy to read and write

Python acts as a bridge between:

- Human instructions
- Machine-executable commands

---

# 🚀 Why Python is Popular

The lecture highlighted several reasons why Python is widely used:

### ✅ Easy to Learn

Python syntax is simple compared to languages like:

- Java
- C++
- C

---

### ✅ Used in Multiple Domains

Python is heavily used in:

- Software development
- Data analysis
- Machine learning
- Artificial intelligence
- Automation
- Web development
- Cybersecurity

---

### ✅ Rich Library Support

Python provides thousands of built-in and third-party libraries that simplify development.

Example:

- NumPy
- Pandas
- TensorFlow
- Flask

---

# 💻 2. Writing the First Python Program

Every programming language starts with a basic introductory program.

---

## Example

```python
print("Hello World")

```

### Output:

```python
Hello World

```

---

# 🖨️ Understanding the `print()` Function

The `print()` function is used to display output on the screen.

It can print:

- Text
- Numbers
- Variables
- Calculations

---

## More Examples

```python
print("My name is ABC")
print("Welcome to Python")

```

---

# 📝 Comments in Python

Comments are notes written inside code for explanation.

Python ignores comments during execution.

---

## Syntax

```python
# This is a comment

```

---

## Example

```python
print("Hello")
# This line prints greeting

```

Comments improve:

- Readability
- Documentation
- Collaboration

---

# 📦 3. Variables in Python

---

## 📌 What is a Variable?

A variable is a container used to store data.

It acts like a label attached to a value.

---

## Example

```python
name = "Rahul"

```

Here:

- `name` → variable
- `"Rahul"` → stored value

---

# 🔄 Assignment Operator (`=`)

The assignment operator stores a value inside a variable.

Structure:

```python
variable_name = value

```

---

# 🎯 Why Variables are Important

Variables help:

- Reuse values
- Avoid repetition
- Update values easily
- Make programs dynamic

---

## Example Without Variables

```python
print("Rahul")
print("Rahul")
print("Rahul")

```

---

## Example With Variables

```python
name = "Rahul"

print(name)
print(name)
print(name)

```

This is cleaner and easier to maintain.

---

# 📏 Rules for Naming Variables

---

## ✅ Valid Rules

- Use letters, digits, and underscores
- Variable names are case-sensitive

Example:

```python
age
Age
AGE

```

These are all different variables.

---

## ❌ Invalid Rules

### No spaces

```python
user name = "Rahul"   ❌

```

---

### Cannot start with numbers

```python
1name = "Rahul"   ❌

```

---

# 🧠 4. Data Types in Python

---

## 📌 What are Data Types?

Data types define:

- What kind of data is stored
- What operations can be performed on it

Python automatically detects the type of data.

---

# 🔢 Integer (`int`)

Stores whole numbers without decimals.

---

## Examples

```python
age = 25
marks = 100
temperature = -5

```

---

# 🔸 Float (`float`)

Stores decimal numbers.

---

## Examples

```python
price = 99.99
height = 5.8
temperature = 36.5

```

---

# 🔤 String (`str`)

Stores text data enclosed in quotes.

---

## Examples

```python
name = "Rahul"
message = "Hello World"

```

Strings can use:

- Single quotes
- Double quotes

---

# ✅ Boolean (`bool`)

Stores logical values:

- True
- False

---

## Examples

```python
is_admin = True
is_logged_in = False

```

Booleans are commonly used in:

- Conditions
- Decision-making
- Authentication systems

---

# 🧪 5. Type Checking with `type()`

---

## 📌 Purpose of `type()`

The `type()` function checks the data type of a variable.

---

## Example

```python
age = 25
name = "Rahul"

print(type(age))
print(type(name))

```

---

## Output

```python
<class 'int'>
<class 'str'>

```

---

# 🎯 Importance of Type Checking

It helps:

- Debug programs
- Understand variable behavior
- Prevent type-related errors

---

# 🖨️ 6. Combining Text and Variables

The lecture demonstrated how `print()` can combine text and variable values.

---

## Example

```python
name = "Rahul"
age = 25

print("My name is", name)
print("Age is:", age)

```

---

## Output

```python
My name is Rahul
Age is: 25

```

Python automatically adds spaces between arguments.

---

# ➕ 7. Basic Arithmetic Operations

Python supports standard mathematical operations.

---

## Example

```python
A = 10
B = 5

print("Add:", A + B)
print("Subtract:", A - B)
print("Multiply:", A * B)

```

---

## Output

```python
Add: 15
Subtract: 5
Multiply: 50

```

---

# 📦 Common Operators

Operator | Meaning
+ | Addition
- | Subtraction
* | Multiplication
/ | Division

---

# 🛒 8. Real-World Example: Product Purchase

The lecture connected Python concepts with practical examples.

---

## Example

```python
item = "Samsung Phone"
price = 20000
quantity = 2

print("Item:", item)
print("Total cost:", price * quantity)

```

---

## Output

```python
Item: Samsung Phone
Total cost: 40000

```

---

# 💼 9. Salary Calculation Example

---

## Example

```python
base_salary = 50000

bonus = 0.1 * base_salary

total_salary = base_salary + bonus

print("Bonus:", bonus)
print("Total Salary:", total_salary)

```

---

## Output

```python
Bonus: 5000
Total Salary: 55000

```

This demonstrates:

- Variable usage
- Arithmetic operations
- Dynamic calculations

---

# ⚠️ 10. Type Compatibility and Errors

The lecture explained that incompatible data types cannot always be combined directly.

---

## Example of Error

```python
name = "Rahul"
age = 25

print(name + age)

```

This causes an error because:

- String and integer types are incompatible for direct addition.

---

# ✅ Correct Approach

```python
print(name, age)

```

Or use type conversion.

---

# 🔄 11. Boolean and Integer Relationship

An interesting concept discussed was:

```python
True  → 1
False → 0

```

This is useful in:

- Conditions
- Logical operations
- Calculations

---

# 🧠 12. Important Programming Concepts Learned

---

## Variables

Store and reuse data dynamically.

---

## Data Types

Determine how data behaves.

---

## Print Statements

Display information to users.

---

## Type Checking

Helps identify variable types and debugging issues.

---

## Arithmetic Operations

Perform mathematical calculations.

---

# 🎓 13. Practical Importance of These Concepts

These foundational concepts are important because every advanced Python topic depends on them.

Future topics such as:

- Functions
- Loops
- Conditions
- Data structures
- Object-oriented programming
- Machine learning

all rely heavily on understanding:

- Variables
- Data types
- Basic syntax

---

# 🚀 14. Best Practices Introduced

---

## Use Meaningful Variable Names

Good:

```python
student_name

```

Bad:

```python
x

```

---

## Keep Code Readable

Use proper spacing and formatting.

---

## Add Comments

Explain important sections clearly.

---

## Practice Regularly

Programming improves through hands-on coding.

---

# 📝 Final Takeaways

- Python is beginner-friendly and powerful.
- Variables store information dynamically.
- Data types define the nature of stored data.
- `print()` helps display outputs.
- `type()` helps understand variable behavior.
- Arithmetic operations form the basis of logical programming.
- Strong fundamentals are essential before moving to advanced topics.

---

# 🎯 Conclusion

This lecture established the core building blocks of Python programming. By understanding variables, data types, printing outputs, and performing calculations, learners now have the foundation required to progress into more advanced programming concepts and real-world Python development.

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