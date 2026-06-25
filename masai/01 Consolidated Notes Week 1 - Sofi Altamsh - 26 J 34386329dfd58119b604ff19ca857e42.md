# 01. Consolidated Notes: Week 1 - Sofi Altamsh - 26 Jul 2025

# **Python Programming Basics – Simplified Consolidated Notes**

## **1. What is Programming?**

- Programming is giving instructions to a computer to solve problems automatically.
- It’s not about memorizing commands but about **problem-solving**:

**Identify problem** → **Plan solution** → **Write code**.
- Example: BMI Calculator → Take weight & height → Apply formula → Show result.

---

## **2. Why Python?**

- Python is:

**Beginner-friendly**: Simple and readable.
**Versatile**: Used in **websites, AI/ML, Data Science, Automation**.
Has a **large community** and many libraries (e.g., NumPy, Pandas).
- Created by **Guido van Rossum in 1991**.
- Supports **procedural** and **object-oriented** programming.

---

## **3. High-Level Language**

- Python is a **high-level language** (easy for humans to read and write).
- Computers understand **binary (0s and 1s)** → Python converts code into machine-readable form.

---

## **4. How Python Executes Code**

- **Python is an interpreted language**:

Executes code **line by line** using an **interpreter**.
- **Compiler vs Interpreter**:

**Compiler**: Converts whole code into machine code before running (e.g., C).
**Interpreter**: Converts and runs **line by line** (Python).
- **Advantages of interpretation**:

Easier debugging (errors caught line by line).
Works on multiple platforms.
- **Disadvantage**: Slightly slower than compiled languages.

---

## **5. Installing Python & Jupyter Notebook**

- Download from **python.org** for Windows/Mac/Linux.
- During installation → **Add Python to PATH**.
- Check installation:

`python --version` (for Python)
`pip --version` (for PIP)
- **PIP** = Python Package Installer (used to install libraries).
- **Jupyter Notebook**:

Install: `pip install notebook`
Open: `jupyter notebook` (runs in browser)
Files saved as **.ipynb**.
- Alternatives: **PyCharm**, **Google Colab** (no installation needed).

---

## **6. Writing Python Programs**

- **Program**: A set of instructions for a computer.
- Python program structure:

**Comments**: Start with `#` (ignored by interpreter).
**Statements**: Executable instructions.
**Execution**: Runs **line by line** in order.
- Python internally converts code into **bytecode** for execution.

---

## **7. Variables and Data in Python**

- **Variables** = names pointing to data in memory (RAM).
- Created using `=` (assignment).
- **Dynamic typing**: No need to declare type; Python detects automatically.
- Examples:

`age = 25
price = 19.99
name = "John"`
- **Rules for variable names**:

Can have letters, numbers, `_` (underscore).
**Cannot start with a number**.
**Case-sensitive** (`age` ≠ `Age`).
**Cannot use keywords** (e.g., `class`, `print`).

---

## **8. Data Types in Python**

- **int** → whole numbers (10, -5)
- **float** → decimal numbers (3.14, 19.99)
- **str** → text ("Hello")
- **bool** → True / False
- **None** → No value
- Everything in Python is an **object**.

---

## **9. Assigning Values**

- **Single assignment**:

`x = 10`
- **Multiple assignment**:

`x, y, z = 10, 20, 30`
- **Same value to many variables**:

`a = b = c = 100`

---

## **10. Operators in Python**

- **Arithmetic operators**:

`+` (add),  (subtract),  (multiply), `/` (divide)
- **Other operators**:

**Comparison**: `==`, `<`, `>`, `!=`
**Logical**: `and`, `or`, `not`
- Example:

`sum = num1 + num2`
- **Operator precedence**:

**PEMDAS** → Parentheses > Exponent > Multiply/Divide > Add/Subtract

---

## **11. Type Conversion**

- **Convert types** using:

`int(3.5)    # 3
float(10)   # 10.0`

---

## **12. Strings and Basic Operations**

- Strings are in quotes: `"Hello"`
- Can perform **concatenation**:

`name = "John"
greeting = "Hello " + name`

---

## **13. Memory Basics**

- **RAM**: Temporary memory (clears when power off).
- **Hard disk**: Permanent storage.
- **Cache**: Very fast small memory for quick access.

---

### ✅ **Key Takeaways**

- Python is **simple, versatile, and interpreted**.
- Variables store **different data types dynamically**.
- Use **operators** for computations.
- Use **Jupyter or IDEs** to write and run Python easily.

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