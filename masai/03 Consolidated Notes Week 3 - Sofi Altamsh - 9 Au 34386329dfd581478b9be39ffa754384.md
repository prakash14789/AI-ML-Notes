# 03. Consolidated Notes: Week 3 - Sofi Altamsh - 9 Aug 2025

### **1. Python Basics**

- **Data Types**:

String (`"text"`), Integer (`123`), Float (`3.14`), Boolean (`True/False`).
`type(variable)` → shows data type.
Convert types: `str()`, `int()`, `float()`.
Use **f-strings** → `f"Hello {name}"`.
- **Lists (Arrays)**:

Mutable, use `[]`.
Access by index (0-based).
Methods: `.append()`, `.insert()`, `.remove()`, `.pop()`, `.clear()`, `.reverse()`.
Can store mixed data types.
- **Dictionaries**:

`{key: value}` pairs.
Keys are unique, values can be any type.
Create from two lists with `dict(zip(keys, values))`.

---

### **2. Control Flow**

- **If / Elif / Else** → for decisions.

```python

if x > 5:
    print("Greater")
elif x == 5:
    print("Equal")
else:
    print("Smaller")

```

- **Logical Operators**: `and`, `or`, `not`.
- **Loops**:

`for item in sequence:` → fixed iterations.
`while condition:` → runs until false.
`break` (stop), `continue` (skip), `pass` (do nothing).
`enumerate()` → index + value.
`zip()` → combine lists element-wise.
- **Functions**:

```python

def greet(name):
    return f"Hello {name}"

```

- Reusable code, can have parameters & return values.

---

### **3. Git Essentials**

- **Basic Commands**:

```bash

git add file.py
git commit -m "message"
git push origin master

```

- Commit often to track changes.
- `origin` = remote repo, `master` = branch name.

---

### **4. Course Support & Processes**

- **Doubt Sessions**:

Official: Sat (5:00–6:30 PM).
Unofficial: Tue, Thu, Fri (6:50–7:50 PM).
- **Assignments**:

Practice only, not graded.
Submit via Jupyter Notebook.
- **Exams**:

2 online + 1 offline per trimester.
- **Kit/Certificates**:

Soft copy by email, hard copy in ~25–30 working days.
- **Study Material**:

Soft copies on LMS within 24 hrs of lectures.

---

✅ **Key Takeaway**: Master Python basics (types, lists, dicts, loops, functions), understand control flow, use Git for version control, and follow course processes for smooth learning.

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