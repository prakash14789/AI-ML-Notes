# 02. Consolidated Notes: Week 2 - Sofi Altamsh - 2 Aug 2025

# Consolidated Notes to Version Control (Git) + Python Basics

---

## ✅ Part 1: Introduction to Version Control and Git (Collaboration in Teams)

### 🎯 **Why Version Control?**

- Imagine working with friends on a project—how do you keep track of **who changed what**?
- Version control (like **Git**) helps you **track changes, work together, and avoid losing code**.

### 📖 **Real-Life Analogy:**

> Just like Google Docs lets friends comment and edit a document together, Git helps programmers collaborate safely.
> 

---

### 🧠 **Core Concepts**

Concept | What It Means | Real Example
Branching | Create your own copy of the code to work independently | Create afeature-loginbranch to add login without touching main code
Pull Request | Ask to merge your branch back into main | You click "Create Pull Request" on GitHub
Code Review | Team gives feedback before merging | Teammates suggest changes or catch bugs
Conflict Fix | When 2 people edit the same code, Git asks to resolve manually | Git shows conflict, and you choose which code to keep

---

### 📘 **Illustrative Example: Calculator Website**

1. **Main Branch** → working calculator code
2. You → make `feature-multiply` branch
3. Teammate → makes `feature-divide` branch
4. Both of you push changes separately
5. Conflicts? Git flags them, and you discuss + fix
6. Finally, merged to main!

---

### 🧪 Suggested Demo (Optional for Class)

- Use **VS Code** or **GitHub UI** to:

Init repo, create branch
Make a change, commit
Push and open pull request
Resolve merge conflict

---

### 💡 **Key Points to Emphasize**

- Work in branches → safer development
- Pull Requests → standard way to add code
- Conflicts → normal and fixable
- Code reviews = Learning + Better Code

---

### 📝 **Git Setup & Everyday Commands**

### 🔧 Setup

```bash
bash
 
git config --global user.name "Your Name"
git config --global user.email "your@email.com"

```

### 🚀 Starting a Project

```bash
bash
 
git init                     # Initialize repo
git clone <url>              # Copy remote repo

```

### 📌 Staging & Committing

```bash
bash
 
git add <file> or git add .  # Stage
git commit -m "Message"      # Commit

```

### 🌳 Branching

```bash
bash
 
git branch                   # List branches
git checkout -b new-branch   # Create + switch
git checkout main            # Switch branch

```

### 🔄 Syncing with GitHub

```bash
bash
 
git pull                     # Pull latest
git push                     # Push changes
git push -u origin branch    # Push new branch

```

### 🔁 Merge & Resolve Conflicts

```bash
bash
 
git merge branch-name        # Merge
# If conflict:
# Edit conflicted files, then:
git add .
git commit

```

### 🧹 Others

```bash
bash
 
git status
git log
git diff
git reset --soft HEAD~1      # Undo last commit (keep changes)
git checkout -- <file>       # Discard changes

```

---

### ✅ Git Tips

- Always pull before pushing
- Write clear commit messages
- Use `.gitignore` to skip unwanted files
- Don’t fear conflicts—talk to teammates

---

## 🧮 Part 2: Python Operators, Variables, and `print()` (from PDF)

### 🔤 Variables in Python

- Used to store values (like `x = 5`)
- Python is **dynamically typed** → no need to declare type
- **Rules:**

Start with a letter or `_`
Don’t start with a number
Don’t use special characters or keywords (like `class`, `if`)

```python
python
 
name = "Alice"
_temp = 42

```

---

### 🔁 Multiple Variable Assignment

```python
python
 
a, b, c = 1, 2, 3

```

---

### 🔣 Python Operators

### ➕ Arithmetic Operators

Operator | Meaning | Example | Output
+ | Addition | 5 + 3 | 8
- | Subtraction | 5 - 3 | 2
* | Multiplication | 5 * 3 | 15
/ | Division | 5 / 3 | 1.67
% | Modulus | 5 % 3 | 2
** | Exponentiation | 5 ** 3 | 125
// | Floor Division | 5 // 3 | 1

---

### 🔍 Relational (Comparison) Operators

Operator | Meaning | Example | Output
== | Equal to | 5 == 3 | False
!= | Not equal | 5 != 3 | True
> | Greater than | 5 > 3 | True
< | Less than | 5 < 3 | False

---

### 🔗 Logical Operators

---

### 📝 Assignment Operators

Operator | Example | Meaning
= | a = 10 | Assign 10
+= | a += 2 | a = a + 2
*= | a *= 3 | a = a * 3

---

### 💡 Bitwise Operators (for binary operations)

Operator | Example | Meaning
& | 5 & 3 | Bitwise AND → 1
` | ` | 5
^ | 5 ^ 3 | Bitwise XOR → 6
~ | ~5 | Bitwise NOT
<< | 5 << 1 | Shift left → 10
>> | 5 >> 1 | Shift right → 2

---

### 🔍 Membership Operators

Operator | Example | Result
in | 'a' in 'cat' | True
not in | 'x' not in 'cat' | True

---

### 🔎 Identity Operators

Operator | Example | Checks if...
is | a is b | Same object
is not | a is not b | Different object

---

### ⌨️ Getting Input from the User

```python
python
 
name = input("Enter your name: ")
age = int(input("Enter age: "))  # Convert string to int

```

---

### 🖨️ Python `print()` Function

- `print()` → shows output
- You can separate items using `sep`, and change the end using `end`

```python
python
 
print("Hello", "World", sep="-", end="!")

```

### ✅ f-Strings for formatting:

```python
python
 
name = "Alice"
print(f"Hello {name}")  # → Hello Alice

```

---

### 🔄 Control Flow Basics (Just Intro)

- Python uses `if`, `elif`, and `else` to control logic

```python
python
 
if score > 90:
    print("Excellent!")
elif score > 70:
    print("Good")
else:
    print("Try Again")

```

---

# 🧠 Final Summary: Git + Python Basics

Topic | What to Remember
Git & Version Control | Helps teams collaborate and avoid conflicts
GitHub | Share your code with others
Branches | Work on features without affecting others
Git Commands | init,add,commit,push,pull, etc.
Python Variables | No need to declare type
Operators | Perform math, comparisons, logic, etc.
Input & Output | input()andprint()
f-Strings | Easy way to format print output

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