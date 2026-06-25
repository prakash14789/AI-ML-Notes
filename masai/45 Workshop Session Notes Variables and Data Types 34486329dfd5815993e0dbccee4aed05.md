# 45. Workshop Session Notes: Variables and Data Types - Suman - 29 Jan 2026

# Workshop Session 1 Notes: Variables and Data Types

## Python for AI/ML Master Class

---

## Session Overview

In Session 1, we'll learn about:

- What variables are and how to use them
- Different types of data in Python
- How to convert between data types
- Python naming conventions
- Practical applications in data science

---

## 1. What are Variables?

### The Concept

Think of a variable as a labeled box that stores information. Just like you might have a box labeled "Books" that contains books, in Python you have variables with names that contain data.

```python
name = "Alice"
age = 25

```

In the above example:

- `name` is a variable that stores the text "Alice"
- `age` is a variable that stores the number 25

### Why Do We Need Variables?

- **Store information** for later use
- **Reuse values** without typing them repeatedly
- **Make code readable** by giving meaningful names to data
- **Perform calculations** and manipulations on data

### Real-World Analogy

Imagine you're organizing student information:

- Student's name: "John"
- Student's roll number: 101
- Student's grade: 'A'

Instead of writing these values multiple times, you store them once in variables and use them wherever needed.

---

## 2. Data Types in Python

Python automatically recognizes different types of data. Here are the main types you'll work with:

### 2.1 Numbers

### Integers (int)

Whole numbers without decimal points.

**Examples:**

- Age: 25
- Population: 1000000
- Temperature: -5
- Year: 2024

**Where used in AI/ML:**

- Counting samples in a dataset
- Storing epochs in model training
- Representing categorical labels (0, 1, 2)

### Floating-Point Numbers (float)

Numbers with decimal points.

**Examples:**

- Height: 5.9
- Weight: 68.5
- Price: 99.99
- Accuracy: 0.95

**Where used in AI/ML:**

- Storing measurements and continuous values
- Model accuracy and loss values
- Feature values in datasets
- Learning rates

### Complex Numbers (complex)

Numbers with real and imaginary parts (less common in basic programming).

**Examples:**

- 3 + 4j
- 2 + 5j

**Where used in AI/ML:**

- Signal processing
- Fourier transforms
- Advanced mathematical computations

### 2.2 Text (Strings)

Strings are sequences of characters enclosed in quotes.

**Examples:**

```python
name = "Alice"
city = 'New York'
message = '''This is a
multi-line message'''

```

**Key Points:**

- Can use single quotes ('') or double quotes ("")
- Triple quotes (''' ''' or """ """) for multi-line text
- Strings can contain letters, numbers, symbols, spaces

**Where used in AI/ML:**

- Storing text data for NLP tasks
- Feature names and labels
- File paths and dataset names
- Model descriptions

### 2.3 Boolean (True/False)

Boolean values represent truth: either True or False.

**Examples:**

```python
is_student = True
has_passed = False
is_valid = True

```

**Where used in AI/ML:**

- Checking conditions
- Representing binary outcomes (yes/no, pass/fail)
- Control flow in code
- Boolean indexing in data filtering

### 2.4 None

`None` represents the absence of a value.

**Example:**

```python
result = None

```

**Where used in AI/ML:**

- Initializing variables before assignment
- Representing missing or undefined values
- Default function parameters

---

## 3. Understanding Data Types Visually

```
Data Types in Python
│
├── Numeric Types
│   ├── int (integers)        → 1, 42, -5, 1000
│   ├── float (decimals)      → 3.14, 2.5, -0.5
│   └── complex               → 3+4j, 2+5j
│
├── Text Type
│   └── str (strings)         → "Hello", 'Python', """Text"""
│
├── Boolean Type
│   └── bool                  → True, False
│
└── None Type
    └── NoneType              → None

```

---

## 4. Variable Naming Rules

### Must Follow (Rules):

1. **Start with letter or underscore:** `name`, `_value` ✓
2. **Cannot start with number:** `2name` ✗
3. **Only letters, numbers, underscores:** `student_age` ✓, `student-age` ✗
4. **No spaces:** `student name` ✗, `student_name` ✓
5. **Case-sensitive:** `age` and `Age` are different
6. **No Python keywords:** `for`, `if`, `while` ✗

### Should Follow (Best Practices - PEP 8):

1. **Use lowercase with underscores:** `student_age` ✓
2. **Be descriptive:** `temperature` ✓, `t` ✗
3. **Avoid single letters** (except in loops): `count` ✓, `c` ✗
4. **Use meaningful names:** `total_marks` ✓, `tm` ✗

### Examples:

**Good Variable Names:**

```python
student_name = "John"
total_marks = 450
average_score = 89.5
is_passed = True

```

**Poor Variable Names:**

```python
n = "John"           # Not descriptive
TotalMarks = 450     # Should be lowercase with underscore
x = 89.5             # Not meaningful
p = True             # Not clear what it represents

```

---

## 5. Type Conversion (Type Casting)

Sometimes you need to convert data from one type to another.

### Common Conversions:

### String to Number

```python
age_text = "25"
age_number = int(age_text)     # Converts to 25 (integer)

price_text = "99.99"
price_number = float(price_text)  # Converts to 99.99 (float)

```

### Number to String

```python
age = 25
age_text = str(age)    # Converts to "25"

```

### Float to Integer

```python
height = 5.9
height_int = int(height)   # Converts to 5 (removes decimal)

```

**Important:** Converting float to int **removes** the decimal part (doesn't round it).

- `int(5.9)` → 5
- `int(5.1)` → 5

### Why Type Conversion Matters in Data Science:

- Data from CSV files is often read as strings
- Need to convert strings to numbers for calculations
- Converting between types for data preprocessing
- Formatting outputs for display

---

## 6. Checking Data Types

### Using type()

```python
age = 25
print(type(age))    # Output: <class 'int'>

height = 5.9
print(type(height))  # Output: <class 'float'>

name = "Alice"
print(type(name))    # Output: <class 'str'>

```

### Using isinstance()

```python
age = 25
print(isinstance(age, int))     # Output: True
print(isinstance(age, str))     # Output: False

```

---

## 7. Multiple Assignment

Python allows you to assign multiple variables at once.

### Same Value to Multiple Variables:

```python
x = y = z = 0
# Now x, y, and z all have the value 0

```

### Different Values to Multiple Variables:

```python
name, age, city = "Alice", 25, "New York"
# name = "Alice"
# age = 25
# city = "New York"

```

### Variable Swapping:

```python
a = 10
b = 20

# Python way (elegant!)
a, b = b, a

# Now a = 20 and b = 10

```

---

## 8. Practical Example: Student Information

Let's see how we might use variables to store student information:

```python
# Student details
student_name = "Alice Johnson"
roll_number = 101
grade = 'A'
marks_math = 95.5
marks_science = 88.0
marks_english = 92.5
is_passed = True

# Calculate total and average
total_marks = marks_math + marks_science + marks_english
average_marks = total_marks / 3

# Display information
print(f"Student: {student_name}")
print(f"Roll Number: {roll_number}")
print(f"Grade: {grade}")
print(f"Total Marks: {total_marks}")
print(f"Average: {average_marks}")
print(f"Passed: {is_passed}")

```

---

## 9. Why This Matters for AI/ML

Understanding variables and data types is crucial because:

### In Data Collection:

- Different features have different types (age: int, salary: float, name: str)
- Need to handle various data formats from different sources

### In Data Preprocessing:

- Converting string data to numeric for calculations
- Handling missing values (None)
- Boolean masks for filtering data

### In Model Building:

- Numeric features need to be float or int
- Labels might be strings or integers
- Parameters and hyperparameters have specific types

### Example Scenario:

```python
# Loading dataset information
dataset_name = "iris_dataset"
num_samples = 150
num_features = 4
accuracy = 96.7
has_missing_values = False

print(f"{dataset_name} has {num_samples} samples")
print(f"Number of features: {num_features}")
print(f"Model accuracy: {accuracy}%")
print(f"Missing values present: {has_missing_values}")

```

---

## 10. Common Beginner Mistakes to Avoid

### Mistake 1: Using keywords as variable names

```python
# WRONG
for = 10      # Error! 'for' is a Python keyword

# CORRECT
for_count = 10

```

### Mistake 2: Starting variable names with numbers

```python
# WRONG
2students = 50   # Error!

# CORRECT
students_2 = 50

```

### Mistake 3: Type mismatch in operations

```python
# WRONG
age = "25"
next_age = age + 1   # Error! Can't add string and number

# CORRECT
age = "25"
next_age = int(age) + 1   # Convert to int first

```

### Mistake 4: Confusing = (assignment) with == (comparison)

```python
# = is for assignment
age = 25

# == is for comparison (we'll cover this in Session 2)
if age == 25:
    print("Age is 25")

```

---

## 11. Quick Self-Check Questions

Before the session, test your understanding:

1. What is a variable?
2. Name three numeric data types in Python.
3. What's the difference between `"25"` and `25`?
4. How do you convert a string to an integer?
5. What will `int(5.9)` return?
6. Is `student_name` a valid variable name?
7. Is `2students` a valid variable name?
8. What does `type()` function do?
9. How do you swap two variables in Python?
10. What is the None type used for?

**Answers:**

1. A named storage location for data
2. int, float, complex
3. "25" is a string (text), 25 is an integer (number)
4. Use `int()` function: `int("25")`
5. 5 (truncates, doesn't round)
6. Yes ✓
7. No ✗ (starts with number)
8. Returns the data type of a variable
9. `a, b = b, a`
10. Represents absence of value or null

---

## 12. What to Prepare for the Session

### Before You Come:

1. ✅ Have Python installed (Jupyter Notebook or Python IDE)
2. ✅ Read through this pre-read material
3. ✅ Try answering the self-check questions
4. ✅ Think of examples from your own context
5. ✅ Prepare questions you might have

### During the Session:

- We'll code together (hands-on)
- You'll practice creating variables
- You'll work on type conversions
- You'll solve practical problems
- You'll ask questions and clarify doubts

### What You'll Need:

- Laptop with Python installed
- Notebook for taking notes (optional)
- Enthusiasm to learn!

---

## 13. Key Terminology to Remember

Term | Meaning
Variable | Named storage location for data
Data Type | Category of data (int, float, str, etc.)
Integer | Whole number without decimal
Float | Number with decimal point
String | Text data in quotes
Boolean | True or False value
Type Conversion | Changing data from one type to another
Assignment | Storing a value in a variable using =
PEP 8 | Python style guide for writing clean code
None | Represents absence of value

---

## 14. Visual Summary

```
Session 1 Learning Path
━━━━━━━━━━━━━━━━━━━━━━

1. Variables
   └─ What they are
   └─ How to create them
   └─ Naming rules

2. Data Types
   └─ Numbers (int, float, complex)
   └─ Text (strings)
   └─ Boolean (True/False)
   └─ None

3. Type Checking
   └─ type() function
   └─ isinstance() function

4. Type Conversion
   └─ Converting between types
   └─ When and why to convert

5. Best Practices
   └─ PEP 8 naming conventions
   └─ Writing readable code
   └─ Common mistakes to avoid

```

---

## 15. Additional Resources (Optional)

If you want to explore more before the session:

### Documentation:

- [Python Official Tutorial - Variables](https://docs.python.org/3/tutorial/introduction.html)
- [PEP 8 Style Guide](https://pep8.org/)
- [Python Built-in Types](https://docs.python.org/3/library/stdtypes.html)

### Videos (if you prefer visual learning):

- Search for "Python variables for beginners" on YouTube
- Look for "Python data types tutorial"

### Practice (optional):

- Try creating variables in Python
- Experiment with different data types
- Practice type conversions

**Note:** These are optional resources. The session will cover everything you need to know!

---

## 16. Tips for Success

### During Pre-Reading:

- Read at your own pace
- Take notes on concepts you find interesting
- Think about real-world applications
- Write down questions to ask during the session

### During the Session:

- 💻 Code along with the instructor
- 🙋 Ask questions when confused
- 🤝 Collaborate with peers during exercises
- 📝 Take notes on important points

### After the Session:

- Practice the concepts immediately
- Complete the assignment
- Experiment with different examples
- Review notes before next session

---

## 17. What to Expect in Session 1

### You Will:

- Create your first Python variables ✓
- Work with different data types ✓
- Convert between types ✓
- Practice naming conventions ✓
- Solve practical problems ✓
- Build a simple BMI calculator ✓

---

## 18. Questions to Think About

Reflect on these before the session:

1. Where in daily life do you use variables (even without realizing it)?
2. Why is it important to store data with meaningful names?
3. Can you think of situations where you'd need to convert data types?
4. What kind of data would you store for an AI/ML project?

---

## 19. Motivation

> **"The journey of a thousand miles begins with a single step."** - Lao Tzu
> 

Learning Python is your first step into the exciting world of AI and Machine Learning. Variables and data types are the foundation of everything you'll build. Take your time, practice regularly, and don't hesitate to ask questions.

**Remember:** Every expert was once a beginner. The fact that you're reading this pre-read shows you're serious about learning. Keep that momentum going!

---

## 20. Ready for the Session?

### Quick Checklist:

- Read through the pre-read material
- Understand what variables are
- Know the basic data types
- Familiar with type conversion concept
- Python environment ready
- Questions prepared (if any)
- Excited to learn!

---

## Final Note

This pre-read is designed to make your learning experience smoother and more effective. You don't need to memorize everything—just familiarize yourself with the concepts. During the session, everything will become clearer through hands-on practice.

**See you in Session 1!** 👋

---

## Contact

If you have any questions before the session:

- Note them down and ask during the session
- Review the examples in this pre-read
- Try experimenting with Python on your own

**Pro Tip:** The best way to learn programming is by doing. Don't just read—try writing some code before the session!

---

**Document Version:** 1.0

**Last Updated:** January 2025

**Course:** Python for AI/ML Master Class

**Session:** 1 - Variables and Data Types

---

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