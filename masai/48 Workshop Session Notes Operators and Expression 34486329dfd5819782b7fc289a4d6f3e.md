# 48. Workshop Session Notes: Operators and Expressions - Suman - 3 Feb 2026

# Lecture Notes: Operators and Expressions

## What You'll Learn

In this lesson, you'll learn to:

- Perform arithmetic operations using Python's mathematical operators
- Compare values and make decisions using comparison and logical operators
- Understand and apply operator precedence to write correct expressions
- Use assignment operators to modify variables efficiently
- Distinguish between value equality and identity using comparison and identity operators

---

## Introduction: What Are Operators and Expressions?

**Operators** are special symbols that perform operations on values (called **operands**).

**Expressions** are combinations of values, variables, and operators that Python evaluates to produce a result.

Think of operators as verbs in a sentence - they tell Python what action to perform:

```python
# Expression: 5 + 3
# Operator: +
# Operands: 5 and 3
# Result: 8

result = 5 + 3  # Python evaluates the expression and stores 8 in result

```

Just like in mathematics, Python expressions follow rules for how operations combine and in what order they execute.

---

## Part 1: Arithmetic Operators - Mathematical Operations

Arithmetic operators perform mathematical calculations on numbers.

### Basic Arithmetic Operators

**Addition (+)**

```python
# Add two numbers
total = 10 + 5
print(total)  # Output: 15

# Add floats
price = 19.99 + 5.50
print(price)  # Output: 25.49

# Add negative numbers
balance = 100 + (-50)
print(balance)  # Output: 50

# String concatenation (special case)
greeting = "Hello" + " " + "World"
print(greeting)  # Output: Hello World

```

**Subtraction (-)**

```python
# Subtract numbers
difference = 20 - 8
print(difference)  # Output: 12

# Create negative numbers
temp = 5 - 10
print(temp)  # Output: -5

# Calculate change
money_left = 100 - 73.25
print(money_left)  # Output: 26.75

```

**Multiplication (*)**

```python
# Multiply numbers
product = 7 * 6
print(product)  # Output: 42

# Calculate total cost
total_cost = 3 * 15.99
print(total_cost)  # Output: 47.97

# String repetition (special case)
stars = "*" * 10
print(stars)  # Output: **********

```

**Division (/)**

```python
# Regular division (always returns float)
result = 10 / 3
print(result)  # Output: 3.3333333333333335

result = 10 / 2
print(result)  # Output: 5.0 (float, not int!)

# Divide by float
half = 100 / 2.0
print(half)  # Output: 50.0

```

**Key Point:** Regular division (/) **always** returns a float, even if the result is a whole number.

### Advanced Arithmetic Operators

**Floor Division (//) - Division Rounded Down**

```python
# Floor division discards the decimal part
result = 10 // 3
print(result)  # Output: 3 (not 3.33...)

result = 17 // 5
print(result)  # Output: 3

# Works with floats too
result = 10.5 // 2
print(result)  # Output: 5.0

# Useful for distributing items
cookies = 23
kids = 4
cookies_per_kid = cookies // kids
print(f"Each kid gets {cookies_per_kid} cookies")
# Output: Each kid gets 5 cookies

```

**Modulo (%) - Remainder After Division**

```python
# Get the remainder
remainder = 10 % 3
print(remainder)  # Output: 1 (because 10 = 3*3 + 1)

remainder = 17 % 5
print(remainder)  # Output: 2 (because 17 = 5*3 + 2)

# Check if number is even or odd
number = 7
if number % 2 == 0:
    print("Even")
else:
    print("Odd")
# Output: Odd

# Leftover cookies
cookies = 23
kids = 4
leftover = cookies % kids
print(f"{leftover} cookies left over")
# Output: 3 cookies left over

```

**Practical Example: Floor Division and Modulo Together**

```python
# Convert minutes to hours and minutes
total_minutes = 187

hours = total_minutes // 60
minutes = total_minutes % 60

print(f"{total_minutes} minutes = {hours} hours and {minutes} minutes")
# Output: 187 minutes = 3 hours and 7 minutes

```

**Exponentiation (**) - Raise to Power**

```python
# Square a number
squared = 5 ** 2
print(squared)  # Output: 25

# Cube a number
cubed = 3 ** 3
print(cubed)  # Output: 27

# Calculate compound interest
principal = 1000
rate = 0.05  # 5%
years = 10
amount = principal * (1 + rate) ** years
print(f"Amount after {years} years: ${amount:.2f}")
# Output: Amount after 10 years: $1628.89

# Square root (power of 0.5)
sqrt = 16 ** 0.5
print(sqrt)  # Output: 4.0

# Cube root (power of 1/3)
cube_root = 27 ** (1/3)
print(cube_root)  # Output: 3.0

```

### Combining Arithmetic Operators

```python
# Complex calculation
result = 2 + 3 * 4
print(result)  # Output: 14 (not 20! Multiplication first)

# Use parentheses to control order
result = (2 + 3) * 4
print(result)  # Output: 20

# Real example: Calculate average
score1 = 85
score2 = 92
score3 = 78
average = (score1 + score2 + score3) / 3
print(f"Average score: {average:.2f}")
# Output: Average score: 85.00

```

### Common Mistakes and Best Practices

**Mistake 1: Integer Division in Python 2 vs Python 3**

```python
# Python 3 (current)
result = 5 / 2
print(result)  # Output: 2.5

# If you want integer division, use //
result = 5 // 2
print(result)  # Output: 2

```

**Mistake 2: Forgetting Parentheses**

```python
# Wrong
celsius = 30
fahrenheit = celsius * 9 / 5 + 32  # Unclear, but works
print(fahrenheit)  # Output: 86.0

# Better (clearer intent)
fahrenheit = (celsius * 9 / 5) + 32
print(fahrenheit)  # Output: 86.0

# Or even clearer
fahrenheit = ((celsius * 9) / 5) + 32
print(fahrenheit)  # Output: 86.0

```

**Mistake 3: Division by Zero**

```python
# This causes an error!
result = 10 / 0  # ZeroDivisionError: division by zero

# Safe approach
divisor = 0
if divisor != 0:
    result = 10 / divisor
else:
    print("Cannot divide by zero!")

```

---

## Part 2: Comparison Operators - Testing Relationships

Comparison operators compare two values and return **True** or **False** (Boolean values).

### Basic Comparison Operators

**Equal To (==)**

```python
# Compare numbers
print(5 == 5)   # Output: True
print(5 == 3)   # Output: False

# Compare strings
print("hello" == "hello")  # Output: True
print("Hello" == "hello")  # Output: False (case-sensitive!)

# Compare floats (be careful!)
print(0.1 + 0.2 == 0.3)  # Output: False (floating-point precision!)

# Practical use
password = "secret123"
user_input = input("Enter password: ")
if user_input == password:
    print("Access granted")
else:
    print("Access denied")

```

**Not Equal To (!=)**

```python
# Check inequality
print(5 != 3)   # Output: True
print(5 != 5)   # Output: False

# Practical use
status = "inactive"
if status != "active":
    print("User is not active")
# Output: User is not active

```

**Greater Than (>)**

```python
# Compare numbers
print(10 > 5)   # Output: True
print(5 > 10)   # Output: False
print(5 > 5)    # Output: False

# Practical use
age = 25
if age > 18:
    print("You are an adult")
# Output: You are an adult

```

**Less Than (<)**

```python
# Compare numbers
print(3 < 7)    # Output: True
print(7 < 3)    # Output: False

# Practical use
temperature = 15
if temperature < 20:
    print("It's cold outside")
# Output: It's cold outside

```

**Greater Than or Equal To (>=)**

```python
# Includes equality
print(10 >= 5)   # Output: True
print(5 >= 5)    # Output: True (equal counts!)
print(3 >= 5)    # Output: False

# Practical use: passing grade
score = 60
if score >= 60:
    print("Passed")
else:
    print("Failed")
# Output: Passed

```

**Less Than or Equal To (<=)**

```python
# Includes equality
print(3 <= 7)    # Output: True
print(7 <= 7)    # Output: True
print(10 <= 7)   # Output: False

# Practical use: speed limit
speed = 55
speed_limit = 60
if speed <= speed_limit:
    print("Driving safely")
else:
    print("Speeding!")
# Output: Driving safely

```

### Chaining Comparisons

Python allows chaining comparisons - very useful!

```python
# Traditional way
age = 25
if age >= 18 and age <= 65:
    print("Working age")

# Python way (chained)
if 18 <= age <= 65:
    print("Working age")
# Output: Working age

# Check if value in range
score = 85
if 0 <= score <= 100:
    print("Valid score")
# Output: Valid score

# Multiple chains
x = 5
if 0 < x < 10 < 20:
    print("x is between 0 and 10, and 10 is less than 20")
# Output: x is between 0 and 10, and 10 is less than 20

```

### Comparing Different Types

```python
# Numbers of different types
print(5 == 5.0)      # Output: True (values are equal)
print(5 is 5.0)      # Output: False (different types - we'll cover 'is' later)

# Strings and numbers
print(5 == "5")      # Output: False (different types)

# String comparison (lexicographic order)
print("apple" < "banana")  # Output: True (alphabetical)
print("Apple" < "apple")   # Output: True (uppercase comes first in ASCII)

```

---

## Part 3: Logical Operators - Combining Conditions

Logical operators combine multiple Boolean values (True/False) into a single result.

### The Three Logical Operators

**AND Operator (and)**

Returns True only if **both** conditions are True.

```python
# Basic usage
print(True and True)    # Output: True
print(True and False)   # Output: False
print(False and True)   # Output: False
print(False and False)  # Output: False

# Practical example: Login validation
username = "admin"
password = "secret123"

if username == "admin" and password == "secret123":
    print("Login successful")
else:
    print("Login failed")
# Output: Login successful

# Multiple conditions
age = 25
has_license = True
has_insurance = True

if age >= 18 and has_license and has_insurance:
    print("You can rent a car")
# Output: You can rent a car

# Range check
temperature = 22
if temperature >= 20 and temperature <= 30:
    print("Perfect weather!")
# Or better:
if 20 <= temperature <= 30:
    print("Perfect weather!")
# Output: Perfect weather!

```

**OR Operator (or)**

Returns True if **at least one** condition is True.

```python
# Basic usage
print(True or True)     # Output: True
print(True or False)    # Output: True
print(False or True)    # Output: True
print(False or False)   # Output: False

# Practical example: Weekend check
day = "Saturday"

if day == "Saturday" or day == "Sunday":
    print("It's the weekend!")
# Output: It's the weekend!

# Multiple payment methods
has_cash = False
has_card = True
has_mobile_payment = False

if has_cash or has_card or has_mobile_payment:
    print("You can make the purchase")
else:
    print("No payment method available")
# Output: You can make the purchase

```

**NOT Operator (not)**

Reverses the Boolean value - True becomes False, False becomes True.

```python
# Basic usage
print(not True)   # Output: False
print(not False)  # Output: True

# Practical example
is_raining = False

if not is_raining:
    print("You don't need an umbrella")
# Output: You don't need an umbrella

# Double negative (not recommended, but valid)
is_available = True
if not (not is_available):  # Same as: if is_available
    print("Item is available")
# Output: Item is available

# Checking if something is NOT in a range
age = 15
if not (18 <= age <= 65):
    print("Not in working age range")
# Output: Not in working age range

```

### Combining Logical Operators

```python
# Complex conditions
age = 25
is_student = True
has_id = True

# Discount eligibility: (student OR senior) AND has ID
if (age < 18 or age > 65 or is_student) and has_id:
    print("Eligible for discount")
# Output: Eligible for discount

# Website access control
is_logged_in = True
is_premium = False
is_trial = True

# Access if: logged in AND (premium OR in trial)
if is_logged_in and (is_premium or is_trial):
    print("Access granted")
# Output: Access granted

```

### Short-Circuit Evaluation

Python evaluates logical expressions efficiently by **stopping early** when the result is determined.

```python
# AND short-circuit
# If first is False, second is never checked
x = 5
y = 10

if x > 10 and y > 5:  # x > 10 is False, so y > 5 is never evaluated
    print("Both true")
else:
    print("At least one false")
# Output: At least one false

# OR short-circuit
# If first is True, second is never checked
if x == 5 or y > 100:  # x == 5 is True, so y > 100 is never evaluated
    print("At least one true")
# Output: At least one true

# Practical use: Safe division
divisor = 0
# Check divisor first to avoid division by zero error
if divisor != 0 and 10 / divisor > 2:
    print("Result is greater than 2")
else:
    print("Cannot divide or result too small")
# Output: Cannot divide or result too small
# If we used 'or' or checked division first, we'd get an error!

```

### Truth Values of Non-Boolean Types

In Python, many values are considered "truthy" or "falsy":

```python
# Falsy values (evaluate to False)
print(bool(0))        # Output: False
print(bool(0.0))      # Output: False
print(bool(""))       # Output: False (empty string)
print(bool([]))       # Output: False (empty list)
print(bool(None))     # Output: False

# Truthy values (evaluate to True)
print(bool(1))        # Output: True
print(bool(-1))       # Output: True
print(bool(3.14))     # Output: True
print(bool("hello"))  # Output: True
print(bool([1, 2]))   # Output: True

# Practical use
name = input("Enter your name: ")
if name:  # True if name is not empty
    print(f"Hello, {name}")
else:
    print("You didn't enter a name")

```

---

## Part 4: Assignment Operators - Updating Variables

Assignment operators assign values to variables, with optional arithmetic operations.

### Basic Assignment (=)

```python
# Simple assignment
x = 10
name = "Alice"
is_valid = True

# Chain assignment (all get same value)
a = b = c = 0
print(a, b, c)  # Output: 0 0 0

# Multiple assignment (unpack values)
x, y, z = 1, 2, 3
print(x, y, z)  # Output: 1 2 3

# Swap values (Python trick!)
a = 5
b = 10
a, b = b, a  # Swap without temp variable
print(a, b)  # Output: 10 5

```

### Compound Assignment Operators

These combine arithmetic with assignment for cleaner code.

**Addition Assignment (+=)**

```python
# Long way
count = 0
count = count + 1

# Short way
count = 0
count += 1
print(count)  # Output: 1

# Add to running total
total = 100
total += 50  # Same as: total = total + 50
print(total)  # Output: 150

# String concatenation
message = "Hello"
message += " World"
print(message)  # Output: Hello World

# Practical: Loop counter
score = 0
score += 10  # Earned 10 points
score += 5   # Earned 5 more
score += 20  # Earned 20 more
print(f"Total score: {score}")
# Output: Total score: 35

```

**Subtraction Assignment (-=)**

```python
# Decrease value
balance = 1000
balance -= 200  # Withdraw $200
print(balance)  # Output: 800

# Practical: Countdown
countdown = 10
countdown -= 1
print(countdown)  # Output: 9

```

**Multiplication Assignment (*=)**

```python
# Double a value
quantity = 5
quantity *= 2
print(quantity)  # Output: 10

# Compound interest
principal = 1000
rate = 1.05  # 5% interest
principal *= rate
print(principal)  # Output: 1050.0

```

**Division Assignment (/=)**

```python
# Divide value
total = 100
people = 4
total /= people  # Share equally
print(total)  # Output: 25.0 (always float!)

```

**Floor Division Assignment (//=)**

```python
# Integer division
cookies = 23
children = 4
cookies //= children  # Cookies per child
print(cookies)  # Output: 5

```

**Modulo Assignment (%=)**

```python
# Keep remainder
number = 17
number %= 5
print(number)  # Output: 2 (17 % 5 = 2)

# Practical: Cycle through values
index = 7
max_index = 5
index %= max_index  # Wrap around
print(index)  # Output: 2

```

**Exponentiation Assignment (**=)**

```python
# Square a number
base = 3
base **= 2  # Same as: base = base ** 2
print(base)  # Output: 9

# Compound growth
population = 1000
growth_rate = 1.02  # 2% growth
population = int(population * growth_rate ** 5)  # 5 years
print(population)  # Output: 1104

```

### Practical Example: Inventory System

```python
# Initialize inventory
apples = 100
oranges = 75
bananas = 50

# Receive shipment
apples += 50
oranges += 30
bananas += 40

# Sell items
apples -= 20
oranges -= 15
bananas -= 10

# Spoilage (10% loss)
apples = int(apples * 0.9)
oranges = int(oranges * 0.9)
bananas = int(bananas * 0.9)

print(f"Inventory:")
print(f"  Apples: {apples}")
print(f"  Oranges: {oranges}")
print(f"  Bananas: {bananas}")
# Output:
# Inventory:
#   Apples: 117
#   Oranges: 81
#   Bananas: 72

```

---

## Part 5: Operator Precedence and Associativity

### Understanding Precedence

Operator precedence determines which operations execute first in complex expressions.

**Precedence Hierarchy (Highest to Lowest):**

1. **Parentheses** `()`
2. **Exponentiation** `**`
3. **Unary** `+x`, `-x`, `not x`
4. **Multiplication/Division/Modulo** `*`, `/`, `//`, `%`
5. **Addition/Subtraction** `+`, `-`
6. **Comparison** `<`, `<=`, `>`, `>=`, `==`, `!=`
7. **Logical NOT** `not`
8. **Logical AND** `and`
9. **Logical OR** `or`

### Precedence Examples

```python
# Example 1: Arithmetic
result = 2 + 3 * 4
print(result)  # Output: 14
# Explanation: 3 * 4 = 12, then 2 + 12 = 14

# Example 2: Mixed operations
result = 10 - 2 ** 3
print(result)  # Output: 2
# Explanation: 2 ** 3 = 8, then 10 - 8 = 2

# Example 3: Division and multiplication
result = 20 / 4 * 2
print(result)  # Output: 10.0
# Explanation: Same precedence, left-to-right: 20/4=5.0, then 5.0*2=10.0

# Example 4: Comparison in expression
result = 5 + 3 > 7
print(result)  # Output: True
# Explanation: 5 + 3 = 8, then 8 > 7 = True

# Example 5: Logical operators
result = True or False and False
print(result)  # Output: True
# Explanation: 'and' before 'or': False and False = False, then True or False = True

```

### Using Parentheses for Clarity

**Always use parentheses when in doubt!** They make code more readable.

```python
# Confusing
result = 10 + 5 * 2 - 3
print(result)  # Output: 17

# Clear with parentheses
result = 10 + (5 * 2) - 3
print(result)  # Output: 17

# Change order with parentheses
result = (10 + 5) * (2 - 3)
print(result)  # Output: -15

# Temperature conversion (clear intent)
celsius = 30
fahrenheit = (celsius * 9 / 5) + 32
print(fahrenheit)  # Output: 86.0

# Complex logical expression
age = 25
is_student = True
has_id = True

# Confusing
if age < 18 or age > 65 or is_student and has_id:
    print("Discount")

# Clear
if (age < 18 or age > 65 or is_student) and has_id:
    print("Discount")

```

### Associativity

When operators have the **same precedence**, associativity determines evaluation order.

**Left-to-Right (Most operators):**

```python
# Arithmetic
result = 10 - 5 - 2
print(result)  # Output: 3
# Evaluation: (10 - 5) - 2 = 5 - 2 = 3

# Division and multiplication
result = 20 / 4 / 2
print(result)  # Output: 2.5
# Evaluation: (20 / 4) / 2 = 5.0 / 2 = 2.5

# NOT left-to-right would be: 20 / (4 / 2) = 20 / 2 = 10

```

**Right-to-Left (Exponentiation):**

```python
# Exponentiation is right-to-left
result = 2 ** 3 ** 2
print(result)  # Output: 512
# Evaluation: 2 ** (3 ** 2) = 2 ** 9 = 512

# NOT left-to-right: (2 ** 3) ** 2 = 8 ** 2 = 64

# Use parentheses for clarity!
result = (2 ** 3) ** 2
print(result)  # Output: 64

```

### Real-World Precedence Example

```python
# Calculate final price with tax and discount

base_price = 100
discount_percent = 20
tax_rate = 0.08

# Wrong way (confusing)
final_price = base_price - base_price * discount_percent / 100 + base_price * tax_rate

# Right way (clear with parentheses)
discount_amount = base_price * (discount_percent / 100)
price_after_discount = base_price - discount_amount
tax_amount = price_after_discount * tax_rate
final_price = price_after_discount + tax_amount

print(f"Final price: ${final_price:.2f}")
# Output: Final price: $86.40

# Or in one line (still clear)
final_price = (base_price * (1 - discount_percent/100)) * (1 + tax_rate)
print(f"Final price: ${final_price:.2f}")
# Output: Final price: $86.40

```

---

## Part 6: Identity Operators - Testing Object Identity

Identity operators check if two variables refer to the **same object in memory**, not just equal values.

### The Two Identity Operators

**IS Operator (is)**

```python
# Check if same object
x = [1, 2, 3]
y = x  # y points to same list as x
z = [1, 2, 3]  # z is a new list with same values

print(x is y)  # Output: True (same object)
print(x is z)  # Output: False (different objects)
print(x == z)  # Output: True (same values)

# Small integers are cached by Python
a = 5
b = 5
print(a is b)  # Output: True (Python caches small integers)

# Larger integers are not cached
a = 1000
b = 1000
print(a is b)  # Output: False (different objects)
print(a == b)  # Output: True (same value)

```

**IS NOT Operator (is not)**

```python
# Check if different objects
x = [1, 2, 3]
y = [1, 2, 3]

print(x is not y)  # Output: True (different objects)

# Checking for None (common use case)
value = None

if value is None:
    print("No value provided")
# Output: No value provided

# Use 'is' for None, not '=='
if value is not None:
    print(f"Value is {value}")
else:
    print("Value is None")
# Output: Value is None

```

### Identity vs. Equality

**Use == for value comparison, use is for identity comparison.**

```python
# Lists - same values, different objects
list1 = [1, 2, 3]
list2 = [1, 2, 3]

print(list1 == list2)  # Output: True (same values)
print(list1 is list2)  # Output: False (different objects)

# Strings - small strings are interned
str1 = "hello"
str2 = "hello"
print(str1 == str2)  # Output: True
print(str1 is str2)  # Output: True (Python interns small strings)

# None - always use 'is'
x = None
if x is None:  # Correct
    print("x is None")

# Don't use == with None
if x == None:  # Works but not recommended
    print("x equals None")

```

### When to Use Identity Operators

**Use `is` when:**

- Checking for `None`
- Checking for `True` or `False` (singleton objects)
- You specifically need to check object identity, not value

**Use `==` when:**

- Comparing values (numbers, strings, lists, etc.)
- General equality checks

```python
# Correct usage: Check for None
result = None
if result is None:
    print("No result yet")

# Correct usage: Check boolean singletons
flag = True
if flag is True:  # But usually just: if flag:
    print("Flag is set")

# Wrong usage: Check numeric equality
x = 100
y = 100
if x is y:  # Wrong! Use ==
    print("Same")

# Correct
if x == y:
    print("Equal values")

```

### Identity in Functions

```python
def modify_list(lst):
    lst.append(4)
    return lst

original = [1, 2, 3]
result = modify_list(original)

print(original)  # Output: [1, 2, 3, 4]
print(result)    # Output: [1, 2, 3, 4]
print(original is result)  # Output: True (same object!)

# Lists are mutable and passed by reference
# The function modifies the original list

```

---

## Part 7: Practical Applications and Examples

### Example 1: Simple Calculator

```python
def calculator(num1, num2, operation):
    """Perform arithmetic operations on two numbers."""
    
    if operation == '+':
        result = num1 + num2
    elif operation == '-':
        result = num1 - num2
    elif operation == '*':
        result = num1 * num2
    elif operation == '/':
        if num2 != 0:  # Check for division by zero
            result = num1 / num2
        else:
            return "Error: Division by zero"
    elif operation == '//':
        if num2 != 0:
            result = num1 // num2
        else:
            return "Error: Division by zero"
    elif operation == '%':
        if num2 != 0:
            result = num1 % num2
        else:
            return "Error: Division by zero"
    elif operation == '**':
        result = num1 ** num2
    else:
        return "Error: Invalid operation"
    
    return result

# Test the calculator
print(calculator(10, 5, '+'))   # Output: 15
print(calculator(10, 5, '-'))   # Output: 5
print(calculator(10, 5, '*'))   # Output: 50
print(calculator(10, 5, '/'))   # Output: 2.0
print(calculator(10, 3, '//'))  # Output: 3
print(calculator(10, 3, '%'))   # Output: 1
print(calculator(2, 3, '**'))   # Output: 8
print(calculator(10, 0, '/'))   # Output: Error: Division by zero

```

### Example 2: Temperature Converter

```python
def celsius_to_fahrenheit(celsius):
    """Convert Celsius to Fahrenheit."""
    return (celsius * 9/5) + 32

def fahrenheit_to_celsius(fahrenheit):
    """Convert Fahrenheit to Celsius."""
    return (fahrenheit - 32) * 5/9

def temperature_converter():
    """Interactive temperature converter."""
    print("Temperature Converter")
    print("1. Celsius to Fahrenheit")
    print("2. Fahrenheit to Celsius")
    
    choice = input("Enter choice (1 or 2): ")
    
    if choice == '1':
        celsius = float(input("Enter temperature in Celsius: "))
        fahrenheit = celsius_to_fahrenheit(celsius)
        print(f"{celsius}°C = {fahrenheit:.2f}°F")
    elif choice == '2':
        fahrenheit = float(input("Enter temperature in Fahrenheit: "))
        celsius = fahrenheit_to_celsius(fahrenheit)
        print(f"{fahrenheit}°F = {celsius:.2f}°C")
    else:
        print("Invalid choice!")

# Run converter
temperature_converter()

# Example output:
# Temperature Converter
# 1. Celsius to Fahrenheit
# 2. Fahrenheit to Celsius
# Enter choice (1 or 2): 1
# Enter temperature in Celsius: 30
# 30.0°C = 86.00°F

```

### Example 3: Grade Calculator

```python
def calculate_grade(score):
    """Determine letter grade based on numeric score."""
    
    # Input validation
    if not (0 <= score <= 100):
        return "Invalid score"
    
    # Determine grade using comparison and logical operators
    if score >= 90:
        grade = 'A'
    elif score >= 80:  # Implicitly: score < 90 and score >= 80
        grade = 'B'
    elif score >= 70:
        grade = 'C'
    elif score >= 60:
        grade = 'D'
    else:
        grade = 'F'
    
    # Determine pass/fail
    passed = score >= 60
    
    return grade, passed

# Test cases
scores = [95, 87, 72, 65, 58, 45]
for score in scores:
    grade, passed = calculate_grade(score)
    status = "Passed" if passed else "Failed"
    print(f"Score: {score} → Grade: {grade} ({status})")

# Output:
# Score: 95 → Grade: A (Passed)
# Score: 87 → Grade: B (Passed)
# Score: 72 → Grade: C (Passed)
# Score: 65 → Grade: D (Passed)
# Score: 58 → Grade: F (Failed)
# Score: 45 → Grade: F (Failed)

```

### Example 4: Discount Calculator

```python
def calculate_discount(price, quantity, is_member):
    """Calculate final price with quantity and membership discounts."""
    
    # Quantity discount
    if quantity >= 10:
        quantity_discount = 0.15  # 15% off
    elif quantity >= 5:
        quantity_discount = 0.10  # 10% off
    else:
        quantity_discount = 0  # No discount
    
    # Membership discount
    membership_discount = 0.05 if is_member else 0  # 5% for members
    
    # Calculate subtotal
    subtotal = price * quantity
    
    # Apply discounts
    total_discount = quantity_discount + membership_discount
    discount_amount = subtotal * total_discount
    final_price = subtotal - discount_amount
    
    return {
        'subtotal': subtotal,
        'quantity_discount': quantity_discount * 100,
        'membership_discount': membership_discount * 100,
        'total_discount': total_discount * 100,
        'discount_amount': discount_amount,
        'final_price': final_price
    }

# Example purchases
price_per_item = 20
purchases = [
    (3, False),   # 3 items, not a member
    (5, True),    # 5 items, member
    (12, True),   # 12 items, member
]

for quantity, is_member in purchases:
    result = calculate_discount(price_per_item, quantity, is_member)
    print(f"\nQuantity: {quantity}, Member: {is_member}")
    print(f"  Subtotal: ${result['subtotal']:.2f}")
    print(f"  Discounts: {result['total_discount']:.0f}%")
    print(f"  Savings: ${result['discount_amount']:.2f}")
    print(f"  Final Price: ${result['final_price']:.2f}")

# Output:
# Quantity: 3, Member: False
#   Subtotal: $60.00
#   Discounts: 0%
#   Savings: $0.00
#   Final Price: $60.00
#
# Quantity: 5, Member: True
#   Subtotal: $100.00
#   Discounts: 15%
#   Savings: $15.00
#   Final Price: $85.00
#
# Quantity: 12, Member: True
#   Subtotal: $240.00
#   Discounts: 20%
#   Savings: $48.00
#   Final Price: $192.00

```

### Example 5: BMI Calculator

```python
def calculate_bmi(weight_kg, height_m):
    """Calculate BMI and category."""
    
    # Calculate BMI
    bmi = weight_kg / (height_m ** 2)
    
    # Determine category using chained comparisons
    if bmi < 18.5:
        category = "Underweight"
    elif 18.5 <= bmi < 25:
        category = "Normal weight"
    elif 25 <= bmi < 30:
        category = "Overweight"
    else:  # bmi >= 30
        category = "Obese"
    
    return bmi, category

# Test cases
people = [
    (50, 1.75),  # Underweight
    (70, 1.75),  # Normal
    (85, 1.75),  # Overweight
    (100, 1.75), # Obese
]

print("BMI Calculator Results:")
print("-" * 50)
for weight, height in people:
    bmi, category = calculate_bmi(weight, height)
    print(f"Weight: {weight}kg, Height: {height}m")
    print(f"  BMI: {bmi:.1f} - {category}\n")

# Output:
# BMI Calculator Results:
# --------------------------------------------------
# Weight: 50kg, Height: 1.75m
#   BMI: 16.3 - Underweight
#
# Weight: 70kg, Height: 1.75m
#   BMI: 22.9 - Normal weight
#
# Weight: 85kg, Height: 1.75m
#   BMI: 27.8 - Overweight
#
# Weight: 100kg, Height: 1.75m
#   BMI: 32.7 - Obese

```

---

## Common Mistakes and How to Avoid Them

### Mistake 1: Confusing = with ==

```python
# Wrong - assignment instead of comparison
x = 10
if x = 5:  # SyntaxError!
    print("x is 5")

# Correct
if x == 5:
    print("x is 5")

```

### Mistake 2: Float Comparison with ==

```python
# Problematic
result = 0.1 + 0.2
if result == 0.3:  # False due to floating-point precision
    print("Equal")

# Better - use a tolerance
tolerance = 0.0001
if abs(result - 0.3) < tolerance:
    print("Approximately equal")

```

### Mistake 3: Using 'is' for Value Comparison

```python
# Wrong
x = 1000
y = 1000
if x is y:  # May be False!
    print("Same")

# Correct
if x == y:
    print("Equal values")

# Use 'is' only for None, True, False
if x is None:
    print("x is None")

```

### Mistake 4: Incorrect Operator Precedence

```python
# Misleading
result = 10 + 5 * 2  # Looks like (10 + 5) * 2 but isn't!
print(result)  # Output: 20, not 30

# Clear
result = (10 + 5) * 2
print(result)  # Output: 30

# Or
result = 10 + (5 * 2)
print(result)  # Output: 20

```

### Mistake 5: Not Checking for Division by Zero

```python
# Dangerous
divisor = 0
result = 10 / divisor  # ZeroDivisionError!

# Safe
if divisor != 0:
    result = 10 / divisor
    print(result)
else:
    print("Cannot divide by zero")

```

---

## Key Takeaways

- **Arithmetic operators** perform mathematical calculations. Remember: `/` always returns float, `//` for integer division, `%` for remainder, `**` for exponentiation.
- **Comparison operators** return Boolean values (True/False). Use `==` for equality, not `=` (which is assignment). You can chain comparisons: `18 <= age <= 65`.
- **Logical operators** combine conditions: `and` (both must be True), `or` (at least one True), `not` (reverse). They use short-circuit evaluation for efficiency.
- **Assignment operators** update variables efficiently. Use compound operators like `+=`, `-=`, `*=` instead of `x = x + 1`.
- **Operator precedence** determines evaluation order. From highest to lowest: parentheses, exponentiation, multiplication/division, addition/subtraction, comparison, logical. **Use parentheses for clarity!**
- **Identity operators** (`is`, `is not`) check if objects are the same in memory, not just equal values. Use `is` only for `None`, `True`, `False`. Use `==` for value comparison.
- **Mental model for expressions**: Think of expressions as questions Python answers. `5 + 3` asks "what is 5 plus 3?" and Python answers `8`. `age >= 18` asks "is age at least 18?" and Python answers `True` or `False`.
- **Best practices**:

Use parentheses when in doubt about precedence
Always check for division by zero
Use `==` for values, `is` for identity (especially with None)
Prefer compound assignment operators (`+=`) over verbose forms
Write clear, readable expressions over clever, compact ones

---

**You've Got This!** You now understand all major Python operators and how to combine them into expressions. Practice writing expressions for different scenarios, pay attention to precedence, and use parentheses liberally for clarity. These operators are the building blocks for all programming logic you'll write! 🎯

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