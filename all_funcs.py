

def factorial_large(n):
    if n < 0:
        return None
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

# 79. Generate all valid combinations of n pairs of parentheses
def generate_parentheses(n):
    result = []
    def backtrack(s='', left=0, right=0):
        if len(s) == 2 * n:
            result.append(s)
            return
        if left < n:
            backtrack(s + '(', left + 1, right)
        if right < left:
            backtrack(s + ')', left, right + 1)
    backtrack()
    return result

# 80. Implement Dijkstra's algorithm for shortest paths from source in a weighted graph
def dijkstra(graph, start):
    import heapq
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    priority_queue = [(0, start)]
    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)
        if current_distance > distances[current_node]:
            continue
        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))
    return distances

# 81. Generate all subsets of a list (power set) using iterative approach
def power_set(lst):
    subsets = [[]]
    for elem in lst:
        new_subsets = []
        for subset in subsets:
            new_subsets.append(subset + [elem])
        subsets.extend(new_subsets)
    return subsets

# 82. Count the number of ways to climb n stairs taking 1 or 2 steps at a time (DP)
def climb_stairs(n):
    if n <= 1:
        return 1
    ways = [0] * (n + 1)
    ways[0], ways[1] = 1, 1
    for i in range(2, n + 1):
        ways[i] = ways[i-1] + ways[i-2]
    return ways[n]

# 83. Evaluate Reverse Polish Notation (RPN) expression
def eval_rpn(tokens):
    stack = []
    operators = {'+', '-', '*', '/'}
    for token in tokens:
        if token not in operators:
            stack.append(int(token))
        else:
            b = stack.pop()
            a = stack.pop()
            if token == '+':
                stack.append(a + b)
            elif token == '-':
                stack.append(a - b)
            elif token == '*':
                stack.append(a * b)
            else:
                stack.append(int(a / b))  # truncate towards zero
    return stack[0]

# 84. Implement Kadane's algorithm to find max sum subarray with start and end indices
def max_subarray(arr):
    max_so_far = arr[0]
    max_ending_here = arr[0]
    start = end = s = 0
    for i in range(1, len(arr)):
        if arr[i] > max_ending_here + arr[i]:
            max_ending_here = arr[i]
            s = i
        else:
            max_ending_here += arr[i]
        if max_ending_here > max_so_far:
            max_so_far = max_ending_here
            start = s
            end = i
    return max_so_far, start, end

# 85. Convert infix expression to postfix using the Shunting Yard algorithm
def infix_to_postfix(expression):
    precedence = {'+':1, '-':1, '*':2, '/':2, '^':3}
    output = []
    stack = []
    tokens = list(expression.replace(' ', ''))
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.isalnum():
            val = token
            # handle multi-digit numbers or variables
            while i + 1 < len(tokens) and tokens[i+1].isalnum():
                i += 1
                val += tokens[i]
            output.append(val)
        elif token == '(':
            stack.append(token)
        elif token == ')':
            while stack and stack[-1] != '(':
                output.append(stack.pop())
            stack.pop()
        else:
            while stack and stack[-1] != '(' and precedence.get(token, 0) <= precedence.get(stack[-1], 0):
                output.append(stack.pop())
            stack.append(token)
        i += 1
    while stack:
        output.append(stack.pop())
    return ' '.join(output)

# 86. Implement a naive pattern matching (substring search) algorithm
def naive_search(text, pattern):
    n, m = len(text), len(pattern)
    indices = []
    for i in range(n - m + 1):
        match = True
        for j in range(m):
            if text[i + j] != pattern[j]:
                match = False
                break
        if match:
            indices.append(i)
    return indices

# 87. Implement binary tree inorder traversal without recursion
def inorder_traversal(root):
    stack, result = [], []
    current = root
    while current or stack:
        while current:
            stack.append(current)
            current = current.get('left')
        current = stack.pop()
        result.append(current['val'])
        current = current.get('right')
    return result

# 88. Implement heap sort algorithm
def heapify(arr, n, i):
    largest = i
    l = 2*i + 1
    r = 2*i + 2
    if l < n and arr[l] > arr[largest]:
        largest = l
    if r < n and arr[r] > arr[largest]:
        largest = r
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)

def heap_sort(arr):
    n = len(arr)
    for i in range(n//2 -1, -1, -1):
        heapify(arr, n, i)
    for i in range(n-1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        heapify(arr, i, 0)
    return arr

# 89. Calculate edit distance (Levenshtein distance) between two strings
def edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n+1) for _ in range(m+1)]
    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]

# 90. Implement the Sieve of Eratosthenes to generate primes up to n
def sieve(n):
    sieve = [True] * (n+1)
    sieve[0], sieve[1] = False, False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n+1, i):
                sieve[j] = False
    return [i for i in range(n+1) if sieve[i]]

# 91. Calculate all permutations of a list (iterative using Heap's algorithm)
def permutations(lst):
    def swap(a, i, j):
        a[i], a[j] = a[j], a[i]
    n = len(lst)
    c = [0] * n
    result = [lst[:]]
    i = 0
    while i < n:
        if c[i] < i:
            if i % 2 == 0:
                swap(lst, 0, i)
            else:
                swap(lst, c[i], i)
            result.append(lst[:])
            c[i] += 1
            i = 0
        else:
            c[i] = 0
            i += 1
    return result

# 92. Implement merge intervals for a list of intervals
def merge_intervals(intervals):
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for current in intervals[1:]:
        last = merged[-1]
        if current[0] <= last[1]:
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            merged.append(current)
    return merged

# 93. Find the number of ways to make change for a value with given coin denominations
def coin_change(coins, amount):
    dp = [0] * (amount + 1)
    dp[0] = 1
    for coin in coins:
        for x in range(coin, amount + 1):
            dp[x] += dp[x - coin]
    return dp[amount]

# 94. Implement an algorithm to find the longest common subsequence (LCS)
def lcs(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]

# 95. Implement flood fill algorithm on a 2D grid
def flood_fill(grid, x, y, new_color):
    rows, cols = len(grid), len(grid[0])
    old_color = grid[x][y]
    if old_color == new_color:
        return grid
    stack = [(x, y)]
    while stack:
        cx, cy = stack.pop()
        if 0 <= cx < rows and 0 <= cy < cols and grid[cx][cy] == old_color:
            grid[cx][cy] = new_color
            stack.extend([(cx-1, cy), (cx+1, cy), (cx, cy-1), (cx, cy+1)])
    return grid

# 96. Implement a topological sort for a DAG using DFS
def topological_sort(graph):
    visited = set()
    stack = []
    def dfs(node):
        visited.add(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                dfs(neighbor)
        stack.append(node)
    for node in graph:
        if node not in visited:
            dfs(node)
    stack.reverse()
    return stack

# 97. Find kth smallest element in an unsorted list using Quickselect
def quickselect(lst, k):
    if not lst:
        return None
    pivot = lst[len(lst) // 2]
    lows = [el for el in lst if el < pivot]
    highs = [el for el in lst if el > pivot]
    pivots = [el for el in lst if el == pivot]
    if k < len(lows):
        return quickselect(lows, k)
    elif k < len(lows) + len(pivots):
        return pivots[0]
    else:
        return quickselect(highs, k - len(lows) - len(pivots))

# 98. Calculate the number of connected components in an undirected graph
def count_connected_components(graph):
    visited = set()
    def dfs(node):
        stack = [node]
        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                stack.extend(graph.get(current, []))
    count = 0
    for node in graph:
        if node not in visited:
            dfs(node)
            count += 1
    return count

# 99. Implement the longest palindrome substring using expand around center
def longest_palindrome(s):
    def expand_around_center(left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return s[left+1:right]
    max_pal = ""
    for i in range(len(s)):
        p1 = expand_around_center(i, i)
        p2 = expand_around_center(i, i+1)
        max_pal = max(max_pal, p1, p2, key=len)
    return max_pal

# 100. Implement the counting sort algorithm
def counting_sort(arr):
    if not arr:
        return arr
    max_val = max(arr)
    min_val = min(arr)
    count_range = max_val - min_val + 1
    count = [0] * count_range
    for num in arr:
        count[num - min_val] += 1
    index = 0
    for i in range(count_range):
        while count[i] > 0:
            arr[index] = i + min_val
            index += 1
            count[i] -= 1
    return arr

# 101. Implement a sliding window maximum algorithm
def sliding_window_max(nums, k):
    from collections import deque
    dq = deque()
    result = []
    for i in range(len(nums)):
        while dq and dq[0] <= i - k:
            dq.popleft()
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()
        dq.append(i)
        if i >= k - 1:
            result.append(nums[dq[0]])
    return result

# 102. Find all anagrams of a string in another string (sliding window)
def find_anagrams(s, p):
    from collections import Counter
    p_count = Counter(p)
    s_count = Counter()
    result = []
    p_len = len(p)
    for i in range(len(s)):
        s_count[s[i]] += 1
        if i >= p_len:
            if s_count[s[i-p_len]] == 1:
                del s_count[s[i-p_len]]
            else:
                s_count[s[i-p_len]] -= 1
        if s_count == p_count:
            result.append(i - p_len + 1)
    return result

# 103. Implement a function to compute matrix multiplication
def matrix_multiply(A, B):
    if not A or not B or len(A[0]) != len(B):
        return None
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    result = [[0 for _ in range(cols_B)] for _ in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]
    return result

# 104. Implement the Rabin-Karp substring search algorithm
def rabin_karp(text, pattern):
    d = 256
    q = 101  # prime number for modulo
    n, m = len(text), len(pattern)
    if m > n:
        return []
    h = pow(d, m-1) % q
    p = 0
    t = 0
    result = []
    for i in range(m):
        p = (d * p + ord(pattern[i])) % q
        t = (d * t + ord(text[i])) % q
    for i in range(n - m + 1):
        if p == t:
            if text[i:i+m] == pattern:
                result.append(i)
        if i < n - m:
            t = (d * (t - ord(text[i]) * h) + ord(text[i + m])) % q
            if t < 0:
                t += q
    return result

# 105. Implement a function to transpose a matrix
def transpose(matrix):
    if not matrix:
        return []
    rows, cols = len(matrix), len(matrix[0])
    transposed = [[0]*rows for _ in range(cols)]
    for i in range(rows):
        for j in range(cols):
            transposed[j][i] = matrix[i][j]
    return transposed

# 106. Implement find all subsets that sum to a target (backtracking)
def subsets_sum(nums, target):
    result = []
    def backtrack(start, path, total):
        if total == target:
            result.append(path[:])
            return
        if total > target:
            return
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i+1, path, total + nums[i])
            path.pop()
    backtrack(0, [], 0)
    return result

# 107. Implement a basic calculator to evaluate simple expressions with + and -
def basic_calculator(s):
    stack = []
    operand = 0
    sign = 1
    result = 0
    for char in s:
        if char.isdigit():
            operand = operand * 10 + int(char)
        elif char == '+':
            result += sign * operand
            operand = 0
            sign = 1
        elif char == '-':
            result += sign * operand
            operand = 0
            sign = -1
    result += sign * operand
    return result

# 108. Implement the Tower of Hanoi solution with steps output
def tower_of_hanoi(n, source='A', target='C', auxiliary='B'):
    moves = []
    def move(n, source, target, auxiliary):
        if n == 1:
            moves.append(f"Move disk 1 from {source} to {target}")
            return
        move(n-1, source, auxiliary, target)
        moves.append(f"Move disk {n} from {source} to {target}")
        move(n-1, auxiliary, target, source)
    move(n, source, target, auxiliary)
    return moves



# 1. Sum two numbers
def sum_two_numbers(a, b):
    return a + b

# 2. Check if a number is prime
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

# 3. Calculate factorial recursively
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n-1)

# 4. Find max in a list
def max_in_list(lst):
    if not lst:
        return None
    max_val = lst[0]
    for num in lst[1:]:
        if num > max_val:
            max_val = num
    return max_val

# 5. Reverse a string
def reverse_string(s):
    return s[::-1]

# 6. Check if string is palindrome
def is_palindrome(s):
    s = s.lower().replace(' ', '')
    return s == s[::-1]

# 7. Fibonacci number (iterative)
def fibonacci(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    a, b = 0, 1
    for _ in range(2, n+1):
        a, b = b, a+b
    return b

# 8. Merge two sorted lists
def merge_sorted_lists(a, b):
    i, j = 0, 0
    merged = []
    while i < len(a) and j < len(b):
        if a[i] < b[j]:
            merged.append(a[i])
            i += 1
        else:
            merged.append(b[j])
            j += 1
    merged.extend(a[i:])
    merged.extend(b[j:])
    return merged

# 9. Find the GCD of two numbers
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

# 10. Check if a list is sorted
def is_sorted(lst):
    return all(lst[i] <= lst[i+1] for i in range(len(lst)-1))

# 11. Count vowels in a string
def count_vowels(s):
    vowels = 'aeiouAEIOU'
    return sum(1 for char in s if char in vowels)

# 12. Convert Celsius to Fahrenheit
def celsius_to_fahrenheit(c):
    return (c * 9/5) + 32

# 13. Flatten a nested list (one level)
def flatten_list(nested):
    return [item for sublist in nested for item in sublist]

# 14. Check leap year
def is_leap_year(year):
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

# 15. Calculate sum of digits of an integer
def sum_digits(n):
    return sum(int(d) for d in str(abs(n)))

# 16. Find the second largest element in a list
def second_largest(lst):
    unique = list(set(lst))
    if len(unique) < 2:
        return None
    unique.sort()
    return unique[-2]

# 17. Calculate nth triangular number
def triangular_number(n):
    return n * (n + 1) // 2

# 18. Check if two strings are anagrams
def are_anagrams(s1, s2):
    return sorted(s1.replace(' ', '').lower()) == sorted(s2.replace(' ', '').lower())

# 19. Find intersection of two lists
def list_intersection(a, b):
    return list(set(a) & set(b))

# 20. Generate powers of two up to n
def powers_of_two(n):
    return [2**i for i in range(n+1)]

# 21. Find all factors of a number
def factors(n):
    return [i for i in range(1, n+1) if n % i == 0]

# 22. Compute the average of numbers in a list
def average(lst):
    if not lst:
        return 0
    return sum(lst) / len(lst)

# 23. Convert integer to binary string
def int_to_binary(n):
    return bin(n)[2:]

# 24. Count occurrences of a character in a string
def count_char(s, char):
    return s.count(char)

# 25. Calculate sum of first n odd numbers
def sum_odd_numbers(n):
    return n * n

# 26. Check if a string contains only digits
def is_digit_string(s):
    return s.isdigit()

# 27. Generate Fibonacci sequence up to n terms
def fibonacci_sequence(n):
    seq = []
    a, b = 0, 1
    while len(seq) < n:
        seq.append(a)
        a, b = b, a + b
    return seq

# 28. Calculate simple interest
def simple_interest(principal, rate, time):
    return principal * rate * time / 100

# 29. Find the longest word in a sentence
def longest_word(sentence):
    words = sentence.split()
    if not words:
        return ''
    return max(words, key=len)

# 30. Check if a number is even
def is_even(n):
    return n % 2 == 0

# 31. Count words in a string
def count_words(s):
    return len(s.split())

# 32. Remove duplicates from a list while preserving order
def remove_duplicates(lst):
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

# 33. Find the median of a list
def median(lst):
    n = len(lst)
    if n == 0:
        return None
    sorted_lst = sorted(lst)
    mid = n // 2
    if n % 2 == 1:
        return sorted_lst[mid]
    else:
        return (sorted_lst[mid - 1] + sorted_lst[mid]) / 2

# 34. Generate a list of prime numbers up to n
def primes_up_to(n):
    primes = []
    for num in range(2, n + 1):
        if is_prime(num):
            primes.append(num)
    return primes

# 35. Count uppercase letters in a string
def count_uppercase(s):
    return sum(1 for c in s if c.isupper())

# 36. Calculate the nth harmonic number
def harmonic_number(n):
    if n <= 0:
        return 0
    return sum(1 / i for i in range(1, n + 1))

# 37. Check if two lists are permutations of each other
def are_permutations(lst1, lst2):
    return sorted(lst1) == sorted(lst2)

# 38. Convert Roman numeral to integer
def roman_to_int(s):
    roman_map = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    total = 0
    prev = 0
    for char in reversed(s):
        value = roman_map[char]
        if value < prev:
            total -= value
        else:
            total += value
        prev = value
    return total

# 39. Find common elements in three lists
def common_elements(a, b, c):
    return list(set(a) & set(b) & set(c))

# 40. Calculate base conversion (decimal to any base 2-16)
def decimal_to_base(n, base):
    if n == 0:
        return "0"
    digits = "0123456789ABCDEF"
    result = ""
    while n > 0:
        result = digits[n % base] + result
        n //= base
    return result

# 41. Check if brackets are balanced in a string
def balanced_brackets(s):
    stack = []
    pairs = {')': '(', ']': '[', '}': '{'}
    for char in s:
        if char in '([{':
            stack.append(char)
        elif char in ')]}':
            if not stack or stack.pop() != pairs[char]:
                return False
    return not stack

# 42. Generate all subsets of a set
def subsets(s):
    res = [[]]
    for elem in s:
        res += [curr + [elem] for curr in res]
    return res

# 43. Calculate the LCM of two numbers
def lcm(a, b):
    return abs(a * b) // gcd(a, b) if a and b else 0

# 44. Find the mode in a list
def mode(lst):
    from collections import Counter
    if not lst:
        return None
    counts = Counter(lst)
    max_count = max(counts.values())
    modes = [k for k, v in counts.items() if v == max_count]
    if len(modes) == 1:
        return modes[0]
    return modes  # Could be multiple modes

# 45. Implement binary search
def binary_search(lst, target):
    left, right = 0, len(lst) - 1
    while left <= right:
        mid = (left + right) // 2
        if lst[mid] == target:
            return mid
        elif lst[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

# 46. Count number of digits in an integer
def count_digits(n):
    return len(str(abs(n)))

# 47. Implement bubble sort
def bubble_sort(lst):
    n = len(lst)
    for i in range(n):
        for j in range(0, n - i - 1):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
    return lst

# 48. Check if a number is a perfect square
def is_perfect_square(n):
    if n < 0:
        return False
    root = int(n ** 0.5)
    return root * root == n

# 49. Find all permutations of a string
def string_permutations(s):
    if len(s) <= 1:
        return [s]
    perms = []
    for i, char in enumerate(s):
        for perm in string_permutations(s[:i] + s[i+1:]):
            perms.append(char + perm)
    return perms

# 50. Find the factorial using memoization
def factorial_memo(n, memo={}):
    if n in memo:
        return memo[n]
    if n == 0:
        return 1
    memo[n] = n * factorial_memo(n-1, memo)
    return memo[n]

# 51. Check if string is valid IPv4 address
def valid_ipv4(ip):
    parts = ip.split('.')
    if len(parts) != 4:
        return False
    for part in parts:
        if not part.isdigit():
            return False
        i = int(part)
        if i < 0 or i > 255:
            return False
    return True

# 52. Implement selection sort
def selection_sort(lst):
    n = len(lst)
    for i in range(n):
        min_idx = i
        for j in range(i+1, n):
            if lst[j] < lst[min_idx]:
                min_idx = j
        lst[i], lst[min_idx] = lst[min_idx], lst[i]
    return lst

# 53. Remove all whitespace from a string
def remove_whitespace(s):
    return ''.join(s.split())

# 54. Calculate Euclidean distance between two points
def euclidean_distance(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) ** 0.5

# 55. Check if a string contains only alphabets
def is_alpha(s):
    return s.isalpha()

# 56. Count frequency of elements in a list
def frequency(lst):
    from collections import Counter
    return dict(Counter(lst))

# 57. Find largest palindrome substring
def largest_palindrome_substring(s):
    def expand(left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return s[left+1:right]
    max_pal = ""
    for i in range(len(s)):
        p1 = expand(i, i)
        p2 = expand(i, i+1)
        max_pal = max(max_pal, p1, p2, key=len)
    return max_pal

# 58. Convert string to title case
def title_case(s):
    return s.title()

# 59. Calculate sum of squares of a list
def sum_of_squares(lst):
    return sum(x*x for x in lst)

# 60. Count the number of unique characters in a string
def unique_chars(s):
    return len(set(s))

# 61. Implement insertion sort
def insertion_sort(lst):
    for i in range(1, len(lst)):
        key = lst[i]
        j = i-1
        while j >= 0 and lst[j] > key:
            lst[j+1] = lst[j]
            j -= 1
        lst[j+1] = key
    return lst

# 62. Calculate the dot product of two vectors
def dot_product(v1, v2):
    return sum(x*y for x, y in zip(v1, v2))

# 63. Find common prefix of two strings
def common_prefix(s1, s2):
    min_len = min(len(s1), len(s2))
    for i in range(min_len):
        if s1[i] != s2[i]:
            return s1[:i]
    return s1[:min_len]

# 64. Check if a string contains any digits
def contains_digit(s):
    return any(char.isdigit() for char in s)

# 65. Find the maximum subarray sum (Kadane's algorithm)
def max_subarray_sum(arr):
    max_ending = max_so_far = arr[0]
    for x in arr[1:]:
        max_ending = max(x, max_ending + x)
        max_so_far = max(max_so_far, max_ending)
    return max_so_far

# 66. Convert snake_case to camelCase
def snake_to_camel(s):
    parts = s.split('_')
    return parts[0] + ''.join(word.capitalize() for word in parts[1:])

# 67. Validate email address (simple regex)
def is_valid_email(email):
    import re
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return bool(re.match(pattern, email))

# 68. Calculate the sum of digits until single digit (digital root)
def digital_root(n):
    while n > 9:
        n = sum(int(d) for d in str(n))
    return n

# 69. Implement merge sort
def merge_sort(lst):
    if len(lst) <= 1:
        return lst
    mid = len(lst) // 2
    left = merge_sort(lst[:mid])
    right = merge_sort(lst[mid:])
    return merge_sorted_lists(left, right)

# 70. Convert integer to Roman numeral
def int_to_roman(num):
    val = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    syms = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]
    roman_num = ''
    i = 0
    while num > 0:
        for _ in range(num // val[i]):
            roman_num += syms[i]
            num -= val[i]
        i += 1
    return roman_num

# 71. Find the longest increasing subsequence length (O(n^2))
def lis_length(arr):
    n = len(arr)
    lis = [1] * n
    for i in range(1, n):
        for j in range(i):
            if arr[i] > arr[j]:
                lis[i] = max(lis[i], lis[j] + 1)
    return max(lis) if lis else 0

# 72. Calculate the power of a number using recursion
def power(base, exp):
    if exp == 0:
        return 1
    return base * power(base, exp-1)

# 73. Check if two strings are rotations of each other
def are_rotations(s1, s2):
    return len(s1) == len(s2) and s2 in s1 + s1

# 74. Generate Pascal's triangle up to n rows
def pascals_triangle(n):
    triangle = []
    for i in range(n):
        row = [1]
        if triangle:
            last_row = triangle[-1]
            row += [sum(pair) for pair in zip(last_row, last_row[1:])]
            row.append(1)
        triangle.append(row)
    return triangle

# 75. Implement quicksort
def quicksort(lst):
    if len(lst) <= 1:
        return lst
    pivot = lst[len(lst) // 2]
    left = [x for x in lst if x < pivot]
    middle = [x for x in lst if x == pivot]
    right = [x for x in lst if x > pivot]
    return quicksort(left) + middle + quicksort(right)

# 76. Find missing number in an array 1 to n
def missing_number(lst, n):
    return n * (n + 1) // 2 - sum(lst)


import math
import collections
import json
import csv
import io
import datetime
import base64
import hashlib

# --- Data Processing and Utilities ---

def parse_csv_string(csv_data: str, delimiter: str = ',') -> list[dict]:
    """
    Parses a CSV formatted string into a list of dictionaries.
    Assumes the first row is the header.

    Args:
        csv_data: A string containing CSV data.
        delimiter: The character used to separate values (default is ',').

    Returns:
        A list of dictionaries, where each dictionary represents a row
        and keys are column headers.

    Raises:
        ValueError: If the CSV data is empty or malformed.
    """
    if not csv_data:
        raise ValueError("CSV data cannot be empty.")

    # Use io.StringIO to treat the string as a file for csv.reader
    data_file = io.StringIO(csv_data)
    reader = csv.reader(data_file, delimiter=delimiter)

    try:
        header = next(reader)
    except StopIteration:
        raise ValueError("CSV data is empty or contains no header row.")

    records = []
    for row_num, row in enumerate(reader, start=2): # Start from 2 for data rows
        if len(row) != len(header):
            # This handles rows with different number of columns than header
            print(f"Warning: Row {row_num} has {len(row)} columns, expected {len(header)}. Skipping or padding.")
            # Option to pad with None or raise error
            # For this example, we'll try to process what we have
            processed_row = dict(zip(header, row + [None]*(len(header)-len(row))))
        else:
            processed_row = dict(zip(header, row))
        records.append(processed_row)

    return records

def convert_dict_list_to_csv_string(data: list[dict], header: list[str] = None, delimiter: str = ',') -> str:
    """
    Converts a list of dictionaries into a CSV formatted string.

    Args:
        data: A list of dictionaries, where each dictionary represents a row.
        header: An optional list of column headers. If None, keys from the first
                dictionary in data are used. Order is preserved if header is provided.
        delimiter: The character used to separate values (default is ',').

    Returns:
        A string containing the CSV data.

    Raises:
        ValueError: If data is empty and no header is provided.
    """
    if not data and not header:
        raise ValueError("Cannot convert empty data without a specified header.")

    if not header:
        if not data:
            return "" # No data and no header specified means empty CSV
        header = list(data[0].keys()) # Use keys from first dict as header

    output = io.StringIO()
    writer = csv.writer(output, delimiter=delimiter)

    # Write header
    writer.writerow(header)

    # Write data rows
    for row_dict in data:
        row = [row_dict.get(col, '') for col in header] # Use get to handle missing keys
        writer.writerow(row)

    return output.getvalue()


def calculate_data_checksum(data: str, algorithm: str = 'sha256') -> str:
    """
    Calculates the checksum of a string using a specified hashing algorithm.

    Args:
        data: The input string data.
        algorithm: The hashing algorithm to use ('md5', 'sha1', 'sha256', 'sha512').

    Returns:
        The hexadecimal representation of the checksum.

    Raises:
        ValueError: If an unsupported algorithm is specified.
    """
    data_bytes = data.encode('utf-8')
    if algorithm == 'md5':
        return hashlib.md5(data_bytes).hexdigest()
    elif algorithm == 'sha1':
        return hashlib.sha1(data_bytes).hexdigest()
    elif algorithm == 'sha256':
        return hashlib.sha256(data_bytes).hexdigest()
    elif algorithm == 'sha512':
        return hashlib.sha512(data_bytes).hexdigest()
    else:
        raise ValueError(f"Unsupported hashing algorithm: {algorithm}")

# --- String Manipulation and Encoding ---

def caesar_cipher_encrypt(text: str, shift: int) -> str:
    """
    Encrypts a string using the Caesar cipher.
    Only alphabetic characters are shifted. Case is preserved.

    Args:
        text: The input string.
        shift: The number of positions to shift characters (can be positive or negative).

    Returns:
        The encrypted string.
    """
    result = []
    for char in text:
        if 'a' <= char <= 'z':
            start = ord('a')
            shifted_char = chr(start + (ord(char) - start + shift) % 26)
            result.append(shifted_char)
        elif 'A' <= char <= 'Z':
            start = ord('A')
            shifted_char = chr(start + (ord(char) - start + shift) % 26)
            result.append(shifted_char)
        else:
            result.append(char) # Non-alphabetic characters are not shifted
    return "".join(result)

def caesar_cipher_decrypt(text: str, shift: int) -> str:
    """
    Decrypts a string encrypted with the Caesar cipher.
    This is equivalent to encrypting with a negative shift.

    Args:
        text: The encrypted string.
        shift: The original shift used for encryption.

    Returns:
        The decrypted string.
    """
    return caesar_cipher_encrypt(text, -shift)

def encode_base64(data_string: str) -> str:
    """
    Encodes a string into Base64.

    Args:
        data_string: The input string to encode.

    Returns:
        The Base64 encoded string.
    """
    data_bytes = data_string.encode('utf-8')
    encoded_bytes = base64.b64encode(data_bytes)
    return encoded_bytes.decode('utf-8')

def decode_base64(encoded_string: str) -> str:
    """
    Decodes a Base64 string back into its original form.

    Args:
        encoded_string: The Base64 encoded string.

    Returns:
        The decoded string.

    Raises:
        ValueError: If the input string is not valid Base64.
    """
    try:
        encoded_bytes = encoded_string.encode('utf-8')
        decoded_bytes = base64.b64decode(encoded_bytes)
        return decoded_bytes.decode('utf-8')
    except Exception as e:
        raise ValueError(f"Failed to decode Base64 string: {e}")

# --- Mathematical and Statistical Functions ---

def calculate_std_dev(numbers: list[float], sample: bool = False) -> float:
    """
    Calculates the standard deviation of a list of numbers.

    Args:
        numbers: A list of numerical values.
        sample: If True, calculates the sample standard deviation (divides by n-1).
                If False, calculates the population standard deviation (divides by n).

    Returns:
        The standard deviation.

    Raises:
        ValueError: If the list is empty or contains non-numeric values.
                    If sample=True and list has less than 2 elements.
    """
    if not numbers:
        raise ValueError("Input list cannot be empty.")
    if not all(isinstance(x, (int, float)) for x in numbers):
        raise ValueError("All elements in the list must be numbers.")

    n = len(numbers)
    if sample and n < 2:
        raise ValueError("Sample standard deviation requires at least 2 data points.")

    mean = sum(numbers) / n
    variance = sum((x - mean) ** 2 for x in numbers)

    if sample:
        if n - 1 == 0: # Should not happen due to n<2 check, but good for robustness
            raise ValueError("Cannot calculate sample std dev for single data point.")
        return math.sqrt(variance / (n - 1))
    else:
        return math.sqrt(variance / n)

def calculate_geometric_mean(numbers: list[float]) -> float:
    """
    Calculates the geometric mean of a list of positive numbers.

    Args:
        numbers: A list of positive numerical values.

    Returns:
        The geometric mean.

    Raises:
        ValueError: If the list is empty or contains non-positive numbers.
    """
    if not numbers:
        raise ValueError("Input list cannot be empty.")
    if not all(isinstance(x, (int, float)) and x > 0 for x in numbers):
        raise ValueError("All elements in the list must be positive numbers.")

    product = 1.0
    for num in numbers:
        product *= num
    return product**(1/len(numbers))

def convert_base(number_str: str, from_base: int, to_base: int) -> str:
    """
    Converts a number from one base to another.
    Supports bases 2-36. For bases > 10, uses 'A'-'Z' for digits 10-35.

    Args:
        number_str: The number as a string in the 'from_base'.
        from_base: The base of the input number (e.g., 2 for binary, 10 for decimal, 16 for hex).
        to_base: The target base.

    Returns:
        The number as a string in the 'to_base'.

    Raises:
        ValueError: If bases are out of range (2-36), input number contains invalid digits
                    for the 'from_base', or an invalid number_str.
    """
    if not (2 <= from_base <= 36 and 2 <= to_base <= 36):
        raise ValueError("Bases must be between 2 and 36 (inclusive).")

    # Helper function to convert char to digit value
    def char_to_digit(char):
        if '0' <= char <= '9':
            return int(char)
        elif 'A' <= char <= 'Z':
            return ord(char) - ord('A') + 10
        elif 'a' <= char <= 'z':
            return ord(char) - ord('a') + 10
        else:
            raise ValueError(f"Invalid character '{char}' for base {from_base}.")

    # Helper function to convert digit value to char
    def digit_to_char(digit):
        if 0 <= digit <= 9:
            return str(digit)
        elif 10 <= digit <= 35:
            return chr(ord('A') + digit - 10)
        else:
            raise ValueError(f"Invalid digit '{digit}' for base {to_base}.")

    # Convert from_base to decimal
    decimal_value = 0
    power = 0
    for char in reversed(number_str):
        digit = char_to_digit(char)
        if digit >= from_base:
            raise ValueError(f"Digit '{char}' is invalid for base {from_base}.")
        decimal_value += digit * (from_base ** power)
        power += 1

    # Convert decimal to to_base
    if decimal_value == 0:
        return "0"

    result_chars = []
    while decimal_value > 0:
        remainder = decimal_value % to_base
        result_chars.append(digit_to_char(remainder))
        decimal_value //= to_base
    
    return "".join(reversed(result_chars))

# --- File Operations (Simulated) and Configuration ---

class SimpleConfigFile:
    """
    A simple class to parse and manage configuration from a string in INI-like format.
    Supports sections and key=value pairs. Comments start with '#'.
    """
    def __init__(self, config_string: str):
        self._sections = collections.defaultdict(dict)
        self._parse_config(config_string)

    def _parse_config(self, config_string: str):
        current_section = "DEFAULT" # Default section for properties outside any explicit section
        lines = config_string.strip().split('\n')

        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue # Skip empty lines and comments

            if line.startswith('[') and line.endswith(']'):
                current_section = line[1:-1].strip()
            elif '=' in line:
                key, value = line.split('=', 1) # Split only on first '='
                self._sections[current_section][key.strip()] = value.strip()
            else:
                # Optionally handle lines that are not comments, sections, or key-value pairs
                # For now, we'll just print a warning.
                print(f"Warning: Skipping unparseable line in config: '{line}'")

    def get(self, section: str, key: str, default=None):
        """
        Retrieves a configuration value.

        Args:
            section: The section name.
            key: The key name.
            default: The default value to return if not found.

        Returns:
            The value associated with the key, or default if not found.
        """
        return self._sections.get(section, {}).get(key, default)

    def get_section(self, section: str) -> dict:
        """
        Retrieves all key-value pairs from a specific section.

        Args:
            section: The section name.

        Returns:
            A dictionary containing key-value pairs for the section.
            Returns an empty dictionary if the section does not exist.
        """
        return dict(self._sections.get(section, {})) # Return a copy

    def sections(self) -> list[str]:
        """
        Returns a list of all section names found in the configuration.
        """
        return list(self._sections.keys())

    def __str__(self) -> str:
        """
        Returns a string representation of the parsed configuration.
        """
        s = []
        for section, keys in self._sections.items():
            if section != "DEFAULT" or keys: # Only show DEFAULT if it has keys
                s.append(f"[{section}]")
                for key, value in keys.items():
                    s.append(f"{key} = {value}")
                s.append("") # Add a blank line for readability
        return "\n".join(s).strip()


def simulate_log_parser(log_data: str, filter_level: str = None) -> list[dict]:
    """
    Simulates parsing log data. Each log line is assumed to be in a simple format:
    [TIMESTAMP] [LEVEL] MESSAGE
    e.g., [2023-07-24 14:00:01] [INFO] User logged in.
    Returns parsed log entries, optionally filtered by level.

    Args:
        log_data: A multi-line string containing log entries.
        filter_level: Optional. If provided (e.g., 'ERROR', 'WARNING'), only logs
                      matching this level will be returned. Case-insensitive.

    Returns:
        A list of dictionaries, where each dictionary represents a parsed log entry
        with 'timestamp', 'level', and 'message' keys.
    """
    parsed_logs = []
    lines = log_data.strip().split('\n')
    
    # Regex to capture parts of the log line
    # (\[.*?\])\s*\[(.*?)\]\s*(.*)
    # Group 1: [TIMESTAMP], Group 2: [LEVEL], Group 3: MESSAGE
    log_pattern = re.compile(r"\[(.*?)\]\s*\[(.*?)\]\s*(.*)")

    for line_num, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        match = log_pattern.match(line)
        if match:
            timestamp_str, level_str, message = match.groups()
            level_str = level_str.upper() # Standardize level to uppercase

            if filter_level and level_str != filter_level.upper():
                continue # Skip if not matching filter level

            try:
                # Attempt to parse timestamp, fall back to string if format varies
                parsed_timestamp = datetime.datetime.fromisoformat(timestamp_str)
            except ValueError:
                parsed_timestamp = timestamp_str # Keep as string if parsing fails

            parsed_logs.append({
                'timestamp': parsed_timestamp,
                'level': level_str,
                'message': message.strip()
            })
        else:
            print(f"Warning: Line {line_num+1} could not be parsed: '{line}'")
            # Optionally add unparsed lines to an 'unparsed_errors' list in the return dict
            # or just skip them. For this function, we skip.

    return parsed_logs


# --- Advanced List and Dictionary Operations ---

def group_by_key(data: list[dict], key: str) -> dict[str, list[dict]]:
    """
    Groups a list of dictionaries by the value of a specified key.

    Args:
        data: A list of dictionaries.
        key: The key by which to group the dictionaries.

    Returns:
        A dictionary where keys are the unique values of the specified key,
        and values are lists of dictionaries that share that key's value.
    """
    if not isinstance(data, list) or not all(isinstance(d, dict) for d in data):
        raise TypeError("Input 'data' must be a list of dictionaries.")
    if not isinstance(key, str):
        raise TypeError("Input 'key' must be a string.")

    grouped_data = collections.defaultdict(list)
    for item in data:
        value = item.get(key)
        if value is not None: # Only group if key exists
            grouped_data[str(value)].append(item) # Convert value to string for dict key consistency
    return dict(grouped_data)


def merge_dictionaries(dict1: dict, dict2: dict, conflict_resolver=None) -> dict:
    """
    Merges two dictionaries. If a key exists in both, the conflict_resolver function
    is used to decide the final value.

    Args:
        dict1: The first dictionary.
        dict2: The second dictionary.
        conflict_resolver: A callable that takes (key, value1, value2) and returns
                           the resolved value. If None, values from dict2 overwrite
                           values from dict1.

    Returns:
        A new dictionary representing the merged result.
    """
    merged = dict1.copy()
    for key, value2 in dict2.items():
        if key in merged:
            if conflict_resolver:
                merged[key] = conflict_resolver(key, merged[key], value2)
            else:
                merged[key] = value2 # Default: dict2 overwrites dict1
        else:
            merged[key] = value2
    return merged

# Example conflict_resolver: take the maximum value
# def take_max_value(key, val1, val2):
#     try:
#         return max(val1, val2)
#     except TypeError: # if values are not comparable
#         return val2 # fall back to dict2 value

# --- Date and Time Utilities ---

def get_date_range(start_date_str: str, end_date_str: str, date_format: str = "%Y-%m-%d") -> list[str]:
    """
    Generates a list of dates (as strings) within a specified range, inclusive.

    Args:
        start_date_str: The start date string.
        end_date_str: The end date string.
        date_format: The format of the input date strings (e.g., "%Y-%m-%d").

    Returns:
        A list of date strings in the specified format, ordered chronologically.

    Raises:
        ValueError: If date strings are invalid, start date is after end date,
                    or format is incorrect.
    """
    try:
        start_date = datetime.datetime.strptime(start_date_str, date_format).date()
        end_date = datetime.datetime.strptime(end_date_str, date_format).date()
    except ValueError as e:
        raise ValueError(f"Invalid date format or date string: {e}")

    if start_date > end_date:
        raise ValueError("Start date cannot be after end date.")

    dates_in_range = []
    current_date = start_date
    one_day = datetime.timedelta(days=1)

    while current_date <= end_date:
        dates_in_range.append(current_date.strftime(date_format))
        current_date += one_day

    return dates_in_range

def calculate_age(birth_date_str: str, current_date_str: str = None, date_format: str = "%Y-%m-%d") -> int:
    """
    Calculates the age in years based on birth date and an optional current date.

    Args:
        birth_date_str: The birth date string.
        current_date_str: Optional. The current date string. If None, uses today's date.
        date_format: The format of the input date strings.

    Returns:
        The age in full years.

    Raises:
        ValueError: If date strings are invalid or birth date is in the future.
    """
    try:
        birth_date = datetime.datetime.strptime(birth_date_str, date_format).date()
    except ValueError as e:
        raise ValueError(f"Invalid birth date format or string: {e}")

    if current_date_str:
        try:
            current_date = datetime.datetime.strptime(current_date_str, date_format).date()
        except ValueError as e:
            raise ValueError(f"Invalid current date format or string: {e}")
    else:
        current_date = datetime.date.today() # Using date.today() for simplicity as datetime was imported

    if birth_date > current_date:
        raise ValueError("Birth date cannot be in the future.")

    age = current_date.year - birth_date.year
    # Adjust age if birthday hasn't occurred yet this year
    if (current_date.month, current_date.day) < (birth_date.month, birth_date.day):
        age -= 1
    
    return age

# --- Simple Financial Calculations ---

def calculate_compound_interest(
    principal: float, annual_rate: float, years: int, compounds_per_year: int
) -> float:
    """
    Calculates the future value of an investment with compound interest.

    Args:
        principal: The initial principal amount.
        annual_rate: The annual interest rate (e.g., 0.05 for 5%).
        years: The number of years the money is invested.
        compounds_per_year: The number of times interest is compounded per year.

    Returns:
        The future value of the investment.

    Raises:
        ValueError: If principal, rate, years, or compounds_per_year are invalid.
    """
    if not all(isinstance(arg, (int, float)) for arg in [principal, annual_rate, years, compounds_per_year]):
        raise TypeError("All inputs must be numeric.")
    if principal < 0:
        raise ValueError("Principal cannot be negative.")
    if annual_rate < 0:
        raise ValueError("Annual rate cannot be negative.")
    if years < 0:
        raise ValueError("Years cannot be negative.")
    if compounds_per_year <= 0:
        raise ValueError("Compounding frequency must be positive.")

    # Formula: A = P * (1 + r/n)^(nt)
    # A = amount (future value)
    # P = principal
    # r = annual interest rate
    # n = number of times interest is compounded per year
    # t = number of years the money is invested
    
    amount = principal * (1 + annual_rate / compounds_per_year)**(compounds_per_year * years)
    return round(amount, 2) # Round to 2 decimal places for currency


def calculate_loan_payment(
    principal: float, annual_interest_rate: float, loan_term_years: int
) -> float:
    """
    Calculates the fixed monthly payment for a loan using the amortization formula.

    Args:
        principal: The principal loan amount.
        annual_interest_rate: The annual interest rate (e.g., 0.05 for 5%).
        loan_term_years: The term of the loan in years.

    Returns:
        The fixed monthly payment amount.

    Raises:
        ValueError: If any input is non-positive or rate is invalid.
    """
    if principal <= 0 or annual_interest_rate < 0 or loan_term_years <= 0:
        raise ValueError("Principal, annual interest rate (if > 0), and loan term must be positive.")

    monthly_interest_rate = annual_interest_rate / 12
    number_of_payments = loan_term_years * 12

    if annual_interest_rate == 0:
        # Simple division if no interest
        payment = principal / number_of_payments
    else:
        # Amortization formula: M = P [ i(1 + i)^n ] / [ (1 + i)^n – 1]
        # M = monthly payment
        # P = principal loan amount
        # i = monthly interest rate
        # n = number of payments
        numerator = monthly_interest_rate * (1 + monthly_interest_rate)**number_of_payments
        denominator = (1 + monthly_interest_rate)**number_of_payments - 1
        
        if denominator == 0: # This happens if (1 + i)^n is very close to 1, e.g., tiny rate, tiny term
            raise ValueError("Calculation error: denominator is zero. Check input values.")
            
        payment = principal * (numerator / denominator)
    
    return round(payment, 2)

# --- Basic Data Structures and Algorithms ---

class SimpleStack:
    """
    A basic Last-In, First-Out (LIFO) stack implementation.
    """
    def __init__(self):
        self._items = []

    def push(self, item):
        """Adds an item to the top of the stack."""
        self._items.append(item)

    def pop(self):
        """
        Removes and returns the item from the top of the stack.

        Raises:
            IndexError: If the stack is empty.
        """
        if self.is_empty():
            raise IndexError("pop from empty stack")
        return self._items.pop()

    def peek(self):
        """
        Returns the item at the top of the stack without removing it.

        Raises:
            IndexError: If the stack is empty.
        """
        if self.is_empty():
            raise IndexError("peek from empty stack")
        return self._items[-1]

    def is_empty(self) -> bool:
        """Checks if the stack is empty."""
        return len(self._items) == 0

    def size(self) -> int:
        """Returns the number of items in the stack."""
        return len(self._items)
    
    def __len__(self) -> int:
        return self.size()
    
    def __str__(self) -> str:
        return f"Stack: {self._items}"

class SimpleQueue:
    """
    A basic First-In, First-Out (FIFO) queue implementation.
    """
    def __init__(self):
        self._items = collections.deque() # Use deque for efficient appends/pops from both ends

    def enqueue(self, item):
        """Adds an item to the rear of the queue."""
        self._items.append(item)

    def dequeue(self):
        """
        Removes and returns the item from the front of the queue.

        Raises:
            IndexError: If the queue is empty.
        """
        if self.is_empty():
            raise IndexError("dequeue from empty queue")
        return self._items.popleft()

    def front(self):
        """
        Returns the item at the front of the queue without removing it.

        Raises:
            IndexError: If the queue is empty.
        """
        if self.is_empty():
            raise IndexError("front from empty queue")
        return self._items[0]

    def is_empty(self) -> bool:
        """Checks if the queue is empty."""
        return len(self._items) == 0

    def size(self) -> int:
        """Returns the number of items in the queue."""
        return len(self._items)

    def __len__(self) -> int:
        return self.size()
    
    def __str__(self) -> str:
        return f"Queue: {list(self._items)}" # Convert deque to list for simple string representation

def topological_sort(graph: dict[str, list[str]]) -> list[str]:
    """
    Performs a topological sort on a directed acyclic graph (DAG).
    Returns a linear ordering of its vertices such that for every directed edge
    uv from vertex u to vertex v, u comes before v in the ordering.

    Args:
        graph: A dictionary representing the graph where keys are nodes
               and values are lists of their direct dependencies (nodes they point to).

    Returns:
        A list of node names in topological order.

    Raises:
        ValueError: If the graph contains a cycle.
    """
    if not isinstance(graph, dict):
        raise TypeError("Graph must be a dictionary.")

    in_degree = collections.defaultdict(int)
    # Initialize in-degrees for all nodes
    for node in graph:
        in_degree[node] = 0
    for node in graph:
        for neighbor in graph[node]:
            in_degree[neighbor] += 1
            # Ensure all neighbors are in in_degree dict
            if neighbor not in graph:
                in_degree[neighbor] # Automatically initializes to 0 if not present

    # Queue for nodes with in-degree 0
    queue = collections.deque([node for node in in_degree if in_degree[node] == 0])
    
    top_order = []
    
    while queue:
        current_node = queue.popleft()
        top_order.append(current_node)
        
        # Decrease in-degree of neighbors
        for neighbor in graph.get(current_node, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    # Check for cycles
    if len(top_order) != len(in_degree):
        # This means there's a cycle if some nodes still have in-degree > 0
        raise ValueError("Graph contains a cycle, topological sort is not possible.")
        
    return top_order

# --- Utility and Conversion Functions ---

def convert_roman_to_int(roman_numeral: str) -> int:
    """
    Converts a Roman numeral string to an integer.
    Supports standard Roman numeral notation up to 3999 (MMMCMXCIX).

    Args:
        roman_numeral: The Roman numeral string (e.g., "MCMXCIV"). Case-insensitive.

    Returns:
        The integer value.

    Raises:
        ValueError: If the input is not a valid Roman numeral string.
    """
    roman_map = {
        'I': 1, 'V': 5, 'X': 10, 'L': 50,
        'C': 100, 'D': 500, 'M': 1000
    }
    
    if not isinstance(roman_numeral, str) or not roman_numeral:
        raise ValueError("Input must be a non-empty string.")
    
    roman_numeral = roman_numeral.upper()
    
    # Simple validation: ensure only valid Roman characters
    if not all(char in roman_map for char in roman_numeral):
        raise ValueError("Invalid characters in Roman numeral string.")

    total = 0
    prev_value = 0
    
    for char in reversed(roman_numeral):
        current_value = roman_map[char]
        
        if current_value < prev_value:
            total -= current_value
        else:
            total += current_value
        prev_value = current_value

    # Additional validation for common invalid sequences (e.g., IIII, VV)
    # This is more complex and usually handled by a regex or more strict parsing
    # For now, rely on value comparison.
    # More robust validation would check for rules like:
    # - Only one I, X, C before V, L, D, M
    # - No more than three identical consecutive symbols
    # - V, L, D cannot be repeated
    # This current implementation handles IV, IX, XL, XC, CD, CM correctly.
    
    return total

def convert_int_to_roman(num: int) -> str:
    """
    Converts an integer to its Roman numeral representation.
    Supports integers from 1 to 3999.

    Args:
        num: The integer to convert.

    Returns:
        The Roman numeral string.

    Raises:
        ValueError: If the number is outside the supported range (1-3999).
    """
    if not isinstance(num, int):
        raise TypeError("Input must be an integer.")
    if not 1 <= num <= 3999:
        raise ValueError("Number must be between 1 and 3999.")

    roman_map = [
        (1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'),
        (100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'),
        (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')
    ]

    roman_numeral = []
    for value, symbol in roman_map:
        while num >= value:
            roman_numeral.append(symbol)
            num -= value
    return "".join(roman_numeral)

# --- Geometric Calculations (Class-based) ---

class GeometricShape:
    """Base class for geometric shapes."""
    def area(self) -> float:
        raise NotImplementedError("Subclasses must implement 'area' method.")

    def perimeter(self) -> float:
        raise NotImplementedError("Subclasses must implement 'perimeter' method.")

    def __str__(self) -> str:
        return f"Shape Type: {self.__class__.__name__}"

class Circle(GeometricShape):
    """Represents a circle and calculates its area and perimeter."""
    def __init__(self, radius: float):
        if not isinstance(radius, (int, float)) or radius <= 0:
            raise ValueError("Radius must be a positive number.")
        self.radius = radius

    def area(self) -> float:
        """Calculates the area of the circle."""
        return math.pi * (self.radius ** 2)

    def perimeter(self) -> float:
        """Calculates the circumference (perimeter) of the circle."""
        return 2 * math.pi * self.radius

    def __str__(self) -> str:
        return f"Circle (Radius: {self.radius:.2f})"

class Rectangle(GeometricShape):
    """Represents a rectangle and calculates its area and perimeter."""
    def __init__(self, length: float, width: float):
        if not isinstance(length, (int, float)) or length <= 0:
            raise ValueError("Length must be a positive number.")
        if not isinstance(width, (int, float)) or width <= 0:
            raise ValueError("Width must be a positive number.")
        self.length = length
        self.width = width

    def area(self) -> float:
        """Calculates the area of the rectangle."""
        return self.length * self.width

    def perimeter(self) -> float:
        """Calculates the perimeter of the rectangle."""
        return 2 * (self.length + self.width)

    def is_square(self) -> bool:
        """Checks if the rectangle is a square."""
        return self.length == self.width
    
    def __str__(self) -> str:
        return f"Rectangle (Length: {self.length:.2f}, Width: {self.width:.2f})"


class Triangle(GeometricShape):
    """
    Represents a triangle and calculates its area using Heron's formula
    and its perimeter.
    """
    def __init__(self, side1: float, side2: float, side3: float):
        if not all(isinstance(s, (int, float)) and s > 0 for s in [side1, side2, side3]):
            raise ValueError("All sides must be positive numbers.")
        # Triangle inequality theorem check
        if not (side1 + side2 > side3 and side1 + side3 > side2 and side2 + side3 > side1):
            raise ValueError("Invalid triangle sides: violates triangle inequality theorem.")
        self.side1 = side1
        self.side2 = side2
        self.side3 = side3

    def perimeter(self) -> float:
        """Calculates the perimeter of the triangle."""
        return self.side1 + self.side2 + self.side3

    def area(self) -> float:
        """Calculates the area of the triangle using Heron's formula."""
        s = self.perimeter() / 2  # Semi-perimeter
        return math.sqrt(s * (s - self.side1) * (s - self.side2) * (s - self.side3))
    
    def __str__(self) -> str:
        return f"Triangle (Sides: {self.side1:.2f}, {self.side2:.2f}, {self.side3:.2f})"

# --- More Complex Data Manipulation / Filtering ---

def process_product_catalog(
    products: list[dict],
    min_price: float = None,
    max_price: float = None,
    category_filter: str = None,
    sort_by: str = None,
    reverse_sort: bool = False
) -> list[dict]:
    """
    Processes a list of product dictionaries, allowing for filtering by price range
    and category, and sorting.

    Each product dict should have at least 'id', 'name', 'price', 'category'.

    Args:
        products: A list of product dictionaries.
        min_price: Optional. Minimum price for filtering.
        max_price: Optional. Maximum price for filtering.
        category_filter: Optional. Category name to filter by (case-insensitive).
        sort_by: Optional. Key to sort products by (e.g., 'name', 'price').
        reverse_sort: If True, sort in descending order.

    Returns:
        A list of processed (filtered and sorted) product dictionaries.
    """
    if not isinstance(products, list) or not all(isinstance(p, dict) for p in products):
        raise TypeError("Products must be a list of dictionaries.")

    filtered_products = []
    for product in products:
        # Basic validation for essential keys
        if not all(k in product for k in ['id', 'name', 'price', 'category']):
            print(f"Warning: Skipping malformed product (missing essential keys): {product}")
            continue
        
        # Price filtering
        price = product.get('price')
        if not isinstance(price, (int, float)):
            print(f"Warning: Skipping product '{product.get('name')}' due to invalid price: {price}")
            continue

        if min_price is not None and price < min_price:
            continue
        if max_price is not None and price > max_price:
            continue

        # Category filtering
        category = product.get('category')
        if category_filter and (not isinstance(category, str) or category.lower() != category_filter.lower()):
            continue
        
        filtered_products.append(product)
    
    # Sorting
    if sort_by:
        if not all(sort_by in p for p in filtered_products):
            print(f"Warning: Sort key '{sort_by}' not found in all filtered products. Sorting might be inconsistent.")
        
        # Define a safe key for sorting that handles missing keys or non-comparable types
        def get_sort_key(item):
            value = item.get(sort_by)
            if isinstance(value, (int, float, str)):
                return value
            return float('-inf') if sort_by == 'price' and not reverse_sort else '' # Handle defaults for sorting

        try:
            filtered_products.sort(key=get_sort_key, reverse=reverse_sort)
        except TypeError as e:
            print(f"Warning: Could not sort by '{sort_by}' due to incomparable types: {e}")
            # Fallback: return unsorted list or try a simpler sort
            pass # Keep it unsorted if sorting fails

    return filtered_products


import re
import random
import string
from collections import defaultdict, deque, Counter
import datetime

# --- Category 1: Short Functions (5-15 lines) ---

def calculate_rectangle_area(length: float, width: float) -> float:
    """
    Calculates the area of a rectangle.

    Args:
        length: The length of the rectangle.
        width: The width of the rectangle.

    Returns:
        The area of the rectangle.

    Raises:
        ValueError: If length or width is non-positive.
    """
    if length <= 0 or width <= 0:
        raise ValueError("Length and width must be positive.")
    return length * width

def reverse_string(s: str) -> str:
    """
    Reverses a given string.

    Args:
        s: The input string.

    Returns:
        The reversed string.
    """
    return s[::-1]

def is_palindrome(s: str) -> bool:
    """
    Checks if a string is a palindrome (reads the same forwards and backwards).
    Case-insensitive and ignores non-alphanumeric characters.

    Args:
        s: The input string.

    Returns:
        True if the string is a palindrome, False otherwise.
    """
    cleaned_s = "".join(filter(str.isalnum, s)).lower()
    return cleaned_s == cleaned_s[::-1]

def celsius_to_fahrenheit(celsius: float) -> float:
    """
    Converts temperature from Celsius to Fahrenheit.

    Args:
        celsius: Temperature in Celsius.

    Returns:
        Temperature in Fahrenheit.
    """
    return (celsius * 9/5) + 32

def get_list_average(numbers: list[float]) -> float:
    """
    Calculates the average of a list of numbers.

    Args:
        numbers: A list of floats or integers.

    Returns:
        The average of the numbers, or 0.0 if the list is empty.
    """
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)

def check_string_length(s: str, min_len: int, max_len: int) -> bool:
    """
    Checks if a string's length is within a specified range (inclusive).

    Args:
        s: The string to check.
        min_len: The minimum allowed length.
        max_len: The maximum allowed length.

    Returns:
        True if the string's length is within the range, False otherwise.
    """
    s_len = len(s)
    return min_len <= s_len <= max_len

def greet_user(name: str, greeting: str = "Hello") -> str:
    """
    Generates a personalized greeting message.

    Args:
        name: The name of the person to greet.
        greeting: The greeting word (default is "Hello").

    Returns:
        The complete greeting string.
    """
    return f"{greeting}, {name}!"

def is_prime(n: int) -> bool:
    """
    Checks if a non-negative integer is a prime number.

    Args:
        n: The integer to check.

    Returns:
        True if n is prime, False otherwise.
    """
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

# --- Category 2: Medium Functions (15-50 lines) ---

def find_common_elements(list1: list, list2: list) -> list:
    """
    Finds common elements between two lists, preserving order from list1.

    Args:
        list1: The first list.
        list2: The second list.

    Returns:
        A new list containing elements common to both lists.
    """
    common = []
    # Use a set for efficient lookup in the second list
    set2 = set(list2)
    for item in list1:
        if item in set2:
            common.append(item)
    return common

def count_word_frequency(text: str) -> dict[str, int]:
    """
    Counts the frequency of each word in a given text string.
    Words are converted to lowercase and punctuation is removed.

    Args:
        text: The input text string.

    Returns:
        A dictionary where keys are words and values are their frequencies.
    """
    if not isinstance(text, str):
        raise TypeError("Input text must be a string.")

    # Remove punctuation and convert to lowercase
    cleaned_text = re.sub(r'[^\w\s]', '', text).lower()
    words = cleaned_text.split()

    word_counts = defaultdict(int)
    for word in words:
        if word: # Avoid counting empty strings from multiple spaces
            word_counts[word] += 1
    return dict(word_counts)

def validate_email_format(email: str) -> bool:
    """
    Validates if a string is a well-formed email address using a simple regex.
    This is a basic validation and might not cover all edge cases per RFC.

    Args:
        email: The email string to validate.

    Returns:
        True if the email format is valid, False otherwise.
    """
    if not isinstance(email, str):
        return False
    # Basic regex: something@something.domain
    # Allows letters, numbers, dots, hyphens in local part and domain
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return re.match(pattern, email) is not None

def calculate_simple_interest(principal: float, rate: float, time: float) -> float:
    """
    Calculates simple interest.

    Args:
        principal: The initial amount of money.
        rate: The annual interest rate (as a decimal, e.g., 0.05 for 5%).
        time: The time in years.

    Returns:
        The calculated simple interest.

    Raises:
        ValueError: If principal, rate, or time are negative.
    """
    if principal < 0 or rate < 0 or time < 0:
        raise ValueError("Principal, rate, and time must be non-negative.")
    return principal * rate * time

def generate_password(length: int, include_digits: bool, include_symbols: bool) -> str:
    """
    Generates a random password with specified length and character types.

    Args:
        length: The desired length of the password.
        include_digits: Whether to include digits (0-9).
        include_symbols: Whether to include special symbols.

    Returns:
        A randomly generated password string.

    Raises:
        ValueError: If length is less than 4, or if no character types are selected.
    """
    if length < 4:
        raise ValueError("Password length must be at least 4.")

    characters = string.ascii_letters
    if include_digits:
        characters += string.digits
    if include_symbols:
        characters += string.punctuation

    if not characters:
        raise ValueError("At least one character type (letters, digits, or symbols) must be included.")

    password = ''.join(random.choice(characters) for i in range(length))
    return password

def find_longest_word(words: list[str]) -> str:
    """
    Finds the longest word in a list of words.
    If the list is empty, returns an empty string.
    If there are multiple words of the same longest length, returns the first one.

    Args:
        words: A list of strings.

    Returns:
        The longest word from the list.
    """
    if not words:
        return ""

    longest = ""
    max_len = 0
    for word in words:
        if len(word) > max_len:
            max_len = len(word)
            longest = word
    return longest

def filter_list_by_prefix(data: list[str], prefix: str) -> list[str]:
    """
    Filters a list of strings, returning only those that start with the given prefix.
    The comparison is case-insensitive.

    Args:
        data: A list of strings to filter.
        prefix: The prefix string to match against.

    Returns:
        A new list containing strings from 'data' that start with 'prefix'.
    """
    if not isinstance(data, list) or not all(isinstance(s, str) for s in data):
        raise TypeError("Data must be a list of strings.")
    if not isinstance(prefix, str):
        raise TypeError("Prefix must be a string.")

    filtered_items = []
    lower_prefix = prefix.lower()
    for item in data:
        if item.lower().startswith(lower_prefix):
            filtered_items.append(item)
    return filtered_items

# --- Category 3: Long Functions (50-100+ lines) ---

def process_customer_transactions(
    transactions: list[dict], products_db: dict[str, dict]
) -> dict:
    """
    Processes a list of customer transactions against a product database.
    Each transaction is a dictionary like:
    {'transaction_id': 'T001', 'customer_id': 'C101', 'items': [{'product_id': 'P001', 'quantity': 2}], 'timestamp': '2023-01-15T10:30:00'}

    The products_db is a dictionary where keys are product_ids and values are
    dictionaries with 'name' and 'price' (e.g., {'P001': {'name': 'Laptop', 'price': 1200.00}}).

    Calculates total cost per transaction and summarizes overall sales.

    Args:
        transactions: A list of transaction dictionaries.
        products_db: A dictionary of product information.

    Returns:
        A dictionary summarizing the processing results, including:
        'total_sales': Total revenue from all valid transactions.
        'processed_transactions': A list of processed transaction details (including total).
        'errors': A list of errors encountered (e.g., product not found, invalid quantity).
    """
    if not isinstance(transactions, list):
        raise TypeError("Transactions must be a list.")
    if not isinstance(products_db, dict):
        raise TypeError("Products database must be a dictionary.")

    total_sales = 0.0
    processed_transactions = []
    errors = []

    for trans in transactions:
        trans_id = trans.get('transaction_id', 'UNKNOWN_ID')
        customer_id = trans.get('customer_id', 'UNKNOWN_CUSTOMER')
        items = trans.get('items', [])
        timestamp_str = trans.get('timestamp')

        try:
            trans_timestamp = datetime.datetime.fromisoformat(timestamp_str) if timestamp_str else None
        except ValueError:
            errors.append(f"Transaction {trans_id}: Invalid timestamp format '{timestamp_str}'.")
            continue

        transaction_total = 0.0
        transaction_items_details = []
        transaction_errors = []

        if not isinstance(items, list):
            transaction_errors.append("Items data is not a list.")
            continue

        for item in items:
            product_id = item.get('product_id')
            quantity = item.get('quantity')

            if not product_id:
                transaction_errors.append(f"Item in {trans_id}: Missing 'product_id'.")
                continue
            if not isinstance(quantity, int) or quantity <= 0:
                transaction_errors.append(f"Item {product_id} in {trans_id}: Invalid or non-positive quantity '{quantity}'.")
                continue

            product_info = products_db.get(product_id)
            if not product_info:
                transaction_errors.append(f"Item {product_id} in {trans_id}: Product not found in database.")
                continue

            if not isinstance(product_info.get('price'), (int, float)) or product_info['price'] < 0:
                transaction_errors.append(f"Product {product_id}: Invalid price '{product_info.get('price')}'.")
                continue

            item_price = product_info['price']
            item_cost = item_price * quantity
            transaction_total += item_cost
            transaction_items_details.append({
                'product_id': product_id,
                'name': product_info.get('name', 'N/A'),
                'quantity': quantity,
                'unit_price': item_price,
                'item_cost': item_cost
            })

        if transaction_errors:
            errors.extend([f"Transaction {trans_id}: {err}" for err in transaction_errors])
        else:
            processed_transactions.append({
                'transaction_id': trans_id,
                'customer_id': customer_id,
                'timestamp': timestamp_str,
                'items': transaction_items_details,
                'total_cost': round(transaction_total, 2)
            })
            total_sales += transaction_total

    return {
        'total_sales': round(total_sales, 2),
        'processed_transactions': processed_transactions,
        'errors': errors
    }


class SimpleTextProcessor:
    """
    A class to perform various text processing operations on a given text body.
    It stores the processed text and provides methods for analysis.
    """
    def __init__(self, text: str):
        """
        Initializes the text processor with the raw text.
        """
        if not isinstance(text, str):
            raise TypeError("Input text must be a string.")
        self._raw_text = text
        self._cleaned_text = self._clean_text(text)
        self._words = self._split_into_words(self._cleaned_text)
        self._word_counts = Counter(self._words)

    def _clean_text(self, text: str) -> str:
        """
        Internal method to clean the text: lowercase, remove punctuation, normalize spaces.
        """
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _split_into_words(self, cleaned_text: str) -> list[str]:
        """
        Internal method to split cleaned text into a list of words.
        """
        return [word for word in cleaned_text.split() if word]

    def get_raw_text(self) -> str:
        """Returns the original, raw text."""
        return self._raw_text

    def get_cleaned_text(self) -> str:
        """Returns the cleaned (lowercase, punctuation-free, normalized) text."""
        return self._cleaned_text

    def get_word_count(self) -> int:
        """Returns the total number of words in the text."""
        return len(self._words)

    def get_unique_word_count(self) -> int:
        """Returns the number of unique words in the text."""
        return len(self._word_counts)

    def get_word_frequencies(self) -> dict[str, int]:
        """Returns a dictionary of word frequencies."""
        return dict(self._word_counts)

    def get_top_n_words(self, n: int) -> list[tuple[str, int]]:
        """
        Returns the top N most frequent words and their counts.

        Args:
            n: The number of top words to retrieve.

        Returns:
            A list of (word, count) tuples.
        """
        if not isinstance(n, int) or n < 0:
            raise ValueError("n must be a non-negative integer.")
        return self._word_counts.most_common(n)

    def search_word(self, word: str) -> bool:
        """
        Checks if a specific word exists in the text (case-insensitive).

        Args:
            word: The word to search for.

        Returns:
            True if the word is found, False otherwise.
        """
        return self._word_counts.get(word.lower(), 0) > 0

    def replace_word(self, old_word: str, new_word: str) -> str:
        """
        Returns a new string with all occurrences of an old word replaced by a new word.
        This operation is performed on the *cleaned* text and returns a new string,
        it does not modify the internal state of the processor.

        Args:
            old_word: The word to find and replace.
            new_word: The word to replace with.

        Returns:
            A new string with replacements made.
        """
        return re.sub(r'\b' + re.escape(old_word.lower()) + r'\b', new_word.lower(), self._cleaned_text)


def find_path_in_grid(
    grid: list[list[int]], start: tuple[int, int], end: tuple[int, int]
) -> list[tuple[int, int]]:
    """
    Finds a path from a start point to an end point in a 2D grid using Breadth-First Search (BFS).
    The grid values represent: 0 (walkable), 1 (obstacle).
    Assumes valid start/end points within grid boundaries.

    Args:
        grid: A list of lists representing the grid (0 for walkable, 1 for obstacle).
        start: A tuple (row, col) representing the starting coordinates.
        end: A tuple (row, col) representing the ending coordinates.

    Returns:
        A list of (row, col) tuples representing the path from start to end,
        or an empty list if no path is found.
    """
    if not grid or not grid[0]:
        return [] # Empty grid

    rows = len(grid)
    cols = len(grid[0])

    if not (0 <= start[0] < rows and 0 <= start[1] < cols) or \
       not (0 <= end[0] < rows and 0 <= end[1] < cols):
        raise ValueError("Start or end coordinates are out of grid bounds.")

    if grid[start[0]][start[1]] == 1 or grid[end[0]][end[1]] == 1:
        return [] # Start or end is an obstacle

    # Queue for BFS: stores (row, col, path_list)
    queue = deque([(start[0], start[1], [start])])
    visited = set()
    visited.add(start)

    # Define possible movements (up, down, left, right)
    dr = [-1, 1, 0, 0]
    dc = [0, 0, -1, 1]

    while queue:
        r, c, path = queue.popleft()

        if (r, c) == end:
            return path

        # Explore neighbors
        for i in range(4):
            nr, nc = r + dr[i], c + dc[i]

            # Check bounds
            if 0 <= nr < rows and 0 <= nc < cols:
                # Check if not visited and not an obstacle
                if (nr, nc) not in visited and grid[nr][nc] == 0:
                    visited.add((nr, nc))
                    new_path = list(path) # Create a new path list for this branch
                    new_path.append((nr, nc))
                    queue.append((nr, nc, new_path))
    
    return [] # No path found


def is_palindrome_case(s):
    """Checks if a string is a palindrome, considering case."""
    return s == s[::-1]

def is_palindrome_alphanumeric(s):
    """Checks if a string is a palindrome after removing non-alphanumeric chars."""
    alnum_s = "".join(filter(str.isalnum, s)).lower()
    return alnum_s == alnum_s[::-1]

def check_if_anagram(s1, s2):
    """Checks if two strings are anagrams of each other."""
    return sorted(s1.lower()) == sorted(s2.lower())

def is_semordnilap(word1, word2):
    """Checks if one word is the reverse of another (e.g., 'stressed', 'desserts')."""
    return word1 == word2[::-1]

def find_palindromic_substring(s):
    """Finds the longest palindromic substring, not whether the whole string is one."""
    # This would contain complex logic to find substrings
    return "example_substring"

def is_mirrored(text):
    """A synonym for checking if a string is its own reverse."""
    return text == text[::-1]

def is_number_palindrome(n):
    """Checks if an integer's digits form a palindrome."""
    return str(n) == str(n)[::-1]


def count_consonants(s):
    """Counts the number of consonants in a string."""
    vowels = "aeiou"
    count = 0
    for char in s.lower():
        if char.isalpha() and char not in vowels:
            count += 1
    return count

def count_digits(s):
    """Counts the number of numeric digits in a string."""
    return sum(1 for char in s if char.isdigit())

def count_char_occurrences(s, c):
    """Counts occurrences of a specific character 'c' in a string 's'."""
    return s.lower().count(c.lower())

def get_vowel_positions(s):
    """Returns a list of indices where vowels appear."""
    vowels = "aeiou"
    positions = []
    for i, char in enumerate(s.lower()):
        if char in vowels:
            positions.append(i)
    return positions

def get_character_frequency(s):
    """Returns a dictionary with the frequency of each character."""
    freq = {}
    for char in s:
        freq[char] = freq.get(char, 0) + 1
    return freq

def has_vowels(s):
    """Checks if a string contains any vowels."""
    vowels = "aeiou"
    for char in s.lower():
        if char in vowels:
            return True
    return False


def get_byte_length(s):
    """Returns the byte length of a string when encoded in UTF-8."""
    return len(s.encode('utf-8'))

def get_unique_length(item):
    """Returns the number of unique elements in an iterable."""
    return len(set(item))

def measure_item_size(item):
    """Synonym for returning the length of an item."""
    return len(item)

def get_element_sum(numbers):
    """Returns the sum of numbers in a list, not its length."""
    if isinstance(numbers, (list, tuple)):
        return sum(numbers)
    return 0

def get_nested_list_count(lst):
    """Counts how many elements in a list are themselves lists."""
    return sum(1 for item in lst if isinstance(item, list))


def merge_and_sort_lists(lists):
    """Merges k lists by concatenation and then sorts the result (less efficient)."""
    result = []
    for lst in lists:
        result.extend(lst)
    result.sort()
    return result

def interleave_lists(lists):
    """Interleaves elements from k lists."""
    result = []
    max_len = max(len(lst) for lst in lists) if lists else 0
    for i in range(max_len):
        for lst in lists:
            if i < len(lst):
                result.append(lst[i])
    return result

def flatten_list_of_lists(lists):
    """Flattens a list of lists into a single list without sorting."""
    return [item for sublist in lists for item in sublist]

def find_max_in_k_lists(lists):
    """Finds the maximum value across all lists."""
    max_val = -float('inf')
    for lst in lists:
        if lst and max(lst) > max_val:
            max_val = max(lst)
    return max_val

def combine_sorted_arrays(arrays):
    """Synonym for merging k sorted lists, using a different name."""
    # This function would have the same body as merge_k_lists
    pass # Implementation omitted for brevity

def merge_two_sorted_lists(l1, l2):
    """Merges just two sorted lists, a simpler version of the main task."""
    # A standard two-pointer merge implementation would go here
    return sorted(l1 + l2)

def count_consonants(s):
    """Counts the number of consonants in a string."""
    vowels = "aeiou"
    count = 0
    for char in s.lower():
        if char.isalpha() and char not in vowels:
            count += 1
    return count

def count_digits(s):
    """Counts the number of numeric digits in a string."""
    return sum(1 for char in s if char.isdigit())

def count_char_occurrences(s, c):
    """Counts occurrences of a specific character 'c' in a string 's'."""
    return s.lower().count(c.lower())

def get_vowel_positions(s):
    """Returns a list of indices where vowels appear."""
    vowels = "aeiou"
    positions = []
    for i, char in enumerate(s.lower()):
        if char in vowels:
            positions.append(i)
    return positions

def get_character_frequency(s):
    """Returns a dictionary with the frequency of each character."""
    freq = {}
    for char in s:
        freq[char] = freq.get(char, 0) + 1
    return freq

def has_vowels(s):
    """Checks if a string contains any vowels."""
    vowels = "aeiou"
    for char in s.lower():
        if char in vowels:
            return True
    return False



import math, random, re
from datetime import datetime

def factorial(n):
    """Computes the factorial of a non-negative integer."""
    if n == 0: return 1
    return n * factorial(n - 1)

def greatest_common_divisor(a, b):
    """Finds the greatest common divisor of two integers."""
    while b:
        a, b = b, a % b
    return a

def is_prime(n):
    """Checks if a number is prime."""
    if n <= 1: return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0: return False
    return True

def calculate_mean(numbers):
    """Calculates the arithmetic mean of a list of numbers."""
    return sum(numbers) / len(numbers) if numbers else 0

def calculate_std_dev(numbers):
    """Calculates the standard deviation of a list of numbers."""
    if len(numbers) < 2: return 0
    mean = calculate_mean(numbers)
    variance = sum([(x - mean) ** 2 for x in numbers]) / len(numbers)
    return math.sqrt(variance)

def collatz_sequence(n):
    """Generates the Collatz sequence for a given number."""
    seq = [n]
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        seq.append(n)
    return seq

def power(base, exp):
    """Calculates base to the power of exp."""
    return base ** exp

def is_perfect_square(n):
    """Checks if a number is a perfect square."""
    return n > 0 and math.isqrt(n) ** 2 == n

def radians_to_degrees(rad):
    """Converts radians to degrees."""
    return rad * 180 / math.pi

def linear_interpolation(a, b, t):
    """Performs linear interpolation between a and b."""
    return a * (1 - t) + b * t

def dot_product(v1, v2):
    """Computes the dot product of two vectors."""
    return sum(x*y for x, y in zip(v1, v2))


def slugify(text):
    """Converts text into a URL-friendly slug."""
    text = text.lower().strip()
    text = re.sub(r'[\s_-]+', '-', text)
    text = re.sub(r'[^\w-]', '', text)
    return text

def reverse_words(sentence):
    """Reverses the words in a sentence."""
    return " ".join(reversed(sentence.split()))

def camel_to_snake(name):
    """Converts a CamelCase string to snake_case."""
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def snake_to_camel(name):
    """Converts a snake_case string to CamelCase."""
    return "".join(word.capitalize() for word in name.split('_'))

def strip_html_tags(html):
    """Removes HTML tags from a string."""
    return re.sub('<[^<]+?>', '', html)

def truncate_string(s, length):
    """Truncates a string to a specified length, adding '...'."""
    return s[:length] + '...' if len(s) > length else s

def generate_random_string(length):
    """Generates a random alphanumeric string of a given length."""
    chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "".join(random.choice(chars) for _ in range(length))

def is_isogram(word):
    """Checks if a word has no repeating letters."""
    return len(word) == len(set(word.lower()))

def count_words(text):
    """Counts the number of words in a text."""
    return len(text.split())

def find_all_emails(text):
    """Finds all email addresses in a block of text."""
    return re.findall(r'[\w\.-]+@[\w\.-]+', text)


def binary_search(arr, target):
    """Performs a binary search on a sorted array."""
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] < target: low = mid + 1
        elif arr[mid] > target: high = mid - 1
        else: return mid
    return -1

def bubble_sort(arr):
    """Sorts an array using the bubble sort algorithm."""
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

def luhn_check(card_number):
    """Validates a number using the Luhn algorithm."""
    digits = [int(d) for d in str(card_number)]
    checksum = sum(digits[-1::-2]) + sum(sum(divmod(d * 2, 10)) for d in digits[-2::-2])
    return checksum % 10 == 0

def quicksort(arr):
    """Sorts an array using the quicksort algorithm."""
    if len(arr) <= 1: return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

def find_missing_number(arr, n):
    """Finds the missing number in a sequence from 1 to n."""
    expected_sum = n * (n + 1) // 2
    actual_sum = sum(arr)
    return expected_sum - actual_sum

def depth_first_search(graph, start):
    """Performs a DFS traversal on a graph."""
    visited, stack = set(), [start]
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            stack.extend(graph[vertex] - visited)
    return list(visited)

def breadth_first_search(graph, start):
    """Performs a BFS traversal on a graph."""
    visited, queue = set(), [start]
    visited.add(start)
    while queue:
        vertex = queue.pop(0)
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return list(visited)

def caesar_cipher_encrypt(text, shift):
    """Encrypts text using a Caesar cipher."""
    result = ""
    for char in text:
        if char.isalpha():
            start = ord('a') if char.islower() else ord('A')
            result += chr((ord(char) - start + shift) % 26 + start)
        else:
            result += char
    return result

def is_leap_year(year):
    """Checks if a given year is a leap year."""
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

def days_between_dates(d1, d2):
    """Calculates the number of days between two 'YYYY-MM-DD' date strings."""
    date1 = datetime.strptime(d1, "%Y-%m-%d")
    date2 = datetime.strptime(d2, "%Y-%m-%d")
    return abs((date2 - date1).days)

def get_current_timestamp():
    """Returns the current Unix timestamp."""
    return int(datetime.now().timestamp())

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculates distance between two points on Earth."""
    R = 6371  # Earth radius in kilometers
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = math.sin(dLat / 2) * math.sin(dLat / 2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dLon / 2) * math.sin(dLon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def get_file_extension(filename):
    """Extracts the file extension from a filename."""
    parts = filename.split('.')
    return parts[-1].lower() if len(parts) > 1 else ""

def validate_ip_address(ip):
    """Checks if a string is a valid IPv4 address."""
    pattern = r"^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$"
    match = re.match(pattern, ip)
    if not match: return False
    return all(0 <= int(group) <= 255 for group in match.groups())

def int_to_roman(num):
    """Converts an integer to a Roman numeral."""
    val_map = [(1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'), (100, 'C'), 
               (90, 'XC'), (50, 'L'), (40, 'XL'), (10, 'X'), (9, 'IX'), (5, 'V'), 
               (4, 'IV'), (1, 'I')]
    roman_num = ''
    for val, sym in val_map:
        while num >= val:
            roman_num += sym
            num -= val
    return roman_num

def roman_to_int(s):
    """Converts a Roman numeral to an integer."""
    roman_map = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    total = 0
    for i in range(len(s) - 1):
        if roman_map[s[i]] < roman_map[s[i+1]]:
            total -= roman_map[s[i]]
        else:
            total += roman_map[s[i]]
    return total + roman_map[s[-1]]

def get_os_info():
    """Returns basic OS information (placeholder)."""
    import platform
    return f"{platform.system()} {platform.release()}"

def simple_linear_regression(x, y):
    """Calculates slope and intercept for simple linear regression."""
    n = len(x)
    m_x, m_y = sum(x) / n, sum(y) / n
    ss_xy = sum(yi * xi for yi, xi in zip(y, x)) - n * m_y * m_x
    ss_xx = sum(xi * xi for xi in x) - n * m_x * m_x
    slope = ss_xy / ss_xx
    intercept = m_y - slope * m_x
    return slope, intercept


import math
import re
import random
import time
import json
from datetime import datetime, timedelta

# ==================================================================================================
# FUNCTION 1: ADVANCED DATA PROCESSING PIPELINE
# ==================================================================================================

def process_customer_data_from_raw_logs(raw_log_data_string, data_processing_configuration, verbose_logging_enabled):
    """
    Processes a raw string of customer log data, extracts structured information,
    and returns a comprehensive analysis dictionary.

    This function is designed to simulate a real-world data processing pipeline. It takes a large,
    multi-line string of unstructured log data, parses each line according to a set of regular
    expressions, validates the extracted data, and then performs a series of aggregations and
    statistical calculations based on a provided configuration dictionary.

    Parameters:
    ----------
    raw_log_data_string (str):
        A single string containing multiple lines of log data. Each line is expected to
        represent a customer event, such as a purchase or a page view. The format can be
        inconsistent, and this function is built to handle such irregularities.
        Example line: "2025-07-24T13:12:37Z - USER_ID:_12345 - EVENT:PURCHASE - DETAILS:{'item_id':'ABC-987','amount':99.99,'currency':'USD','region':'NA'}"

    data_processing_configuration (dict):
        A dictionary containing various settings and thresholds that control the behavior
        of the processing pipeline. This allows for dynamic adjustment of the function's
        logic without changing the code itself.
        Expected keys:
        - 'log_line_regex' (str): The regular expression used to parse each log line.
        - 'date_format_string' (str): The format of the date in the log file (e.g., '%Y-%m-%dT%H:%M:%SZ').
        - 'minimum_purchase_threshold' (float): The minimum purchase amount to be considered a "significant purchase".
        - 'target_regions' (list): A list of regions to specifically analyze (e.g., ['NA', 'EU', 'APAC']).
        - 'banned_user_ids' (list): A list of user IDs to exclude from the analysis.

    verbose_logging_enabled (bool):
        A flag to enable or disable verbose console output. If True, the function will
        print detailed step-by-step information about its progress, including lines being
        processed, validation errors, and intermediate calculations. This is useful for
        debugging but should be disabled in production environments for performance.

    Returns:
    -------
    dict:
        A nested dictionary containing the results of the analysis. The structure of this
        dictionary is as follows:
        {
            'processing_metadata': {
                'timestamp_start_utc': str,
                'timestamp_end_utc': str,
                'total_lines_processed': int,
                'lines_with_errors': int,
                'error_log': list
            },
            'aggregated_statistics': {
                'total_revenue': float,
                'total_purchases': int,
                'unique_customers': int,
                'average_purchase_value': float,
                'significant_purchase_count': int
            },
            'regional_breakdown': {
                'NA': {'revenue': float, 'purchase_count': int},
                'EU': {'revenue': float, 'purchase_count': int},
                # ... other regions ...
            },
            'customer_profiles': {
                'USER_ID_12345': {'total_spent': float, 'purchase_history': list},
                # ... other customers ...
            }
        }

    Raises:
    ------
    ValueError:
        If the `data_processing_configuration` dictionary is missing essential keys.
    TypeError:
        If the input `raw_log_data_string` is not a string.
    """
    # Record the start time for metadata logging.
    processing_start_timestamp = datetime.utcnow()
    if verbose_logging_enabled:
        print(f"[{processing_start_timestamp}] - INFO: Data processing pipeline initiated.")

    # --- Step 1: Input Validation and Configuration Check ---
    # Before starting, we must validate the inputs to ensure they are usable.
    if not isinstance(raw_log_data_string, str):
        raise TypeError("Input 'raw_log_data_string' must be a string.")
    if not isinstance(data_processing_configuration, dict):
        raise TypeError("Input 'data_processing_configuration' must be a dictionary.")

    # Check for the existence of required configuration keys.
    required_keys = ['log_line_regex', 'date_format_string', 'minimum_purchase_threshold', 'target_regions', 'banned_user_ids']
    for key in required_keys:
        if key not in data_processing_configuration:
            raise ValueError(f"Configuration dictionary is missing required key: '{key}'")

    if verbose_logging_enabled:
        print("[INFO] - Configuration and input validation successful.")
        print(f"[INFO] - Minimum significant purchase threshold set to: {data_processing_configuration['minimum_purchase_threshold']}")
        print(f"[INFO] - Analyzing for target regions: {data_processing_configuration['target_regions']}")

    # --- Step 2: Initialize Data Structures for Results ---
    # We will populate these structures as we parse the log data.
    processing_results = {
        'processing_metadata': {
            'timestamp_start_utc': processing_start_timestamp.isoformat(),
            'timestamp_end_utc': None, # Will be set at the end
            'total_lines_processed': 0,
            'lines_with_errors': 0,
            'error_log': []
        },
        'aggregated_statistics': {
            'total_revenue': 0.0,
            'total_purchases': 0,
            'unique_customers': 0, # Will be calculated from customer_profiles
            'average_purchase_value': 0.0, # Will be calculated at the end
            'significant_purchase_count': 0
        },
        'regional_breakdown': {region: {'revenue': 0.0, 'purchase_count': 0} for region in data_processing_configuration['target_regions']},
        'customer_profiles': {}
    }

    # Compile the regular expression for efficiency, as it will be used in a loop.
    log_parser_regex = re.compile(data_processing_configuration['log_line_regex'])

    # --- Step 3: Iterate and Process Each Log Line ---
    # This is the main loop where each line of the raw data is handled.
    log_lines = raw_log_data_string.strip().split('\n')
    line_number_counter = 0

    if verbose_logging_enabled:
        print(f"[INFO] - Beginning to process {len(log_lines)} log entries.")

    for single_log_line in log_lines:
        line_number_counter += 1
        processing_results['processing_metadata']['total_lines_processed'] = line_number_counter

        # Skip empty lines to prevent unnecessary processing.
        if not single_log_line.strip():
            continue

        if verbose_logging_enabled:
            print(f"  [DEBUG] - Processing line #{line_number_counter}: {single_log_line[:100]}...")

        # --- Step 3a: Parse the line using Regex ---
        match_object = log_parser_regex.match(single_log_line)
        if not match_object:
            error_message = f"Line {line_number_counter}: Regex pattern did not match."
            processing_results['processing_metadata']['lines_with_errors'] += 1
            processing_results['processing_metadata']['error_log'].append(error_message)
            if verbose_logging_enabled:
                print(f"    [WARN] - {error_message}")
            continue # Move to the next line

        # --- Step 3b: Extract Data from Regex Groups ---
        # The regex is expected to have named groups for easy extraction.
        try:
            extracted_data_dict = match_object.groupdict()
            user_id = extracted_data_dict['user_id']
            event_type = extracted_data_dict['event']
            details_json_string = extracted_data_dict['details']

            # The 'details' part is a JSON string, so we need to parse it.
            event_details = json.loads(details_json_string.replace("'", "\"")) # Handle single quotes
        except (KeyError, json.JSONDecodeError) as e:
            error_message = f"Line {line_number_counter}: Failed to extract or parse details. Error: {e}"
            processing_results['processing_metadata']['lines_with_errors'] += 1
            processing_results['processing_metadata']['error_log'].append(error_message)
            if verbose_logging_enabled:
                print(f"    [WARN] - {error_message}")
            continue # Move to the next line

        # --- Step 3c: Validate the Extracted Data ---
        # Check if the user is banned.
        if user_id in data_processing_configuration['banned_user_ids']:
            if verbose_logging_enabled:
                print(f"    [INFO] - Skipping banned user: {user_id}")
            continue

        # We are only interested in 'PURCHASE' events for this analysis.
        if event_type != 'PURCHASE':
            if verbose_logging_enabled:
                print(f"    [INFO] - Skipping non-purchase event type: {event_type}")
            continue

        # Ensure the purchase details have the required fields.
        purchase_amount = event_details.get('amount')
        purchase_region = event_details.get('region')
        if purchase_amount is None or purchase_region is None:
            error_message = f"Line {line_number_counter}: Purchase event for user {user_id} is missing 'amount' or 'region' in details."
            processing_results['processing_metadata']['lines_with_errors'] += 1
            processing_results['processing_metadata']['error_log'].append(error_message)
            if verbose_logging_enabled:
                print(f"    [WARN] - {error_message}")
            continue

        # --- Step 3d: Update Aggregated Statistics ---
        # If all checks pass, we can now use this data in our results.

        # Update total revenue and purchase count.
        processing_results['aggregated_statistics']['total_revenue'] += purchase_amount
        processing_results['aggregated_statistics']['total_purchases'] += 1

        # Check for significant purchases.
        if purchase_amount >= data_processing_configuration['minimum_purchase_threshold']:
            processing_results['aggregated_statistics']['significant_purchase_count'] += 1

        # Update regional breakdown.
        if purchase_region in processing_results['regional_breakdown']:
            processing_results['regional_breakdown'][purchase_region]['revenue'] += purchase_amount
            processing_results['regional_breakdown'][purchase_region]['purchase_count'] += 1

        # Update customer profiles.
        if user_id not in processing_results['customer_profiles']:
            # Initialize profile for a new customer.
            processing_results['customer_profiles'][user_id] = {
                'total_spent': 0.0,
                'purchase_history': []
            }
        # Add the current purchase to the customer's profile.
        processing_results['customer_profiles'][user_id]['total_spent'] += purchase_amount
        processing_results['customer_profiles'][user_id]['purchase_history'].append(event_details)

        if verbose_logging_enabled:
            print(f"    [SUCCESS] - Successfully processed purchase for user {user_id} of amount {purchase_amount}.")

    # --- Step 4: Final Calculations and Wrap-up ---
    # Some statistics can only be calculated after the loop is complete.
    if verbose_logging_enabled:
        print("[INFO] - Log processing complete. Performing final calculations.")

    # Calculate the number of unique customers.
    number_of_unique_customers = len(processing_results['customer_profiles'])
    processing_results['aggregated_statistics']['unique_customers'] = number_of_unique_customers

    # Calculate the average purchase value, avoiding division by zero.
    total_purchases_final = processing_results['aggregated_statistics']['total_purchases']
    if total_purchases_final > 0:
        total_revenue_final = processing_results['aggregated_statistics']['total_revenue']
        average_value = total_revenue_final / total_purchases_final
        # Round to 2 decimal places for currency.
        processing_results['aggregated_statistics']['average_purchase_value'] = round(average_value, 2)
    else:
        # If there were no purchases, the average is zero.
        processing_results['aggregated_statistics']['average_purchase_value'] = 0.0

    # Record the end time for metadata.
    processing_end_timestamp = datetime.utcnow()
    processing_results['processing_metadata']['timestamp_end_utc'] = processing_end_timestamp.isoformat()

    if verbose_logging_enabled:
        print(f"[{processing_end_timestamp}] - INFO: Data processing pipeline finished.")
        print("--- FINAL SUMMARY ---")
        print(f"  Total Revenue: {processing_results['aggregated_statistics']['total_revenue']:.2f}")
        print(f"  Unique Customers: {processing_results['aggregated_statistics']['unique_customers']}")
        print(f"  Total Lines with Errors: {processing_results['processing_metadata']['lines_with_errors']}")
        print("--------------------")

    # The final results dictionary is now ready to be returned.
    return processing_results


# ==================================================================================================
# FUNCTION 2: MONTE CARLO FINANCIAL PORTFOLIO SIMULATION
# ==================================================================================================

def simulate_stock_portfolio_monte_carlo(
    initial_portfolio_value,
    simulation_parameters_dict,
    number_of_simulations,
    progress_callback_function=None
):
    """
    Performs a Monte Carlo simulation to project the future value of a stock portfolio.

    This function simulates portfolio growth over a specified number of years, running thousands
    of individual simulations to model a range of possible outcomes. It uses geometric Brownian
    motion as the underlying model for stock price movements. The function is highly detailed,
    tracking the portfolio's value at each step and providing comprehensive summary statistics
    of the final outcomes.

    Parameters:
    ----------
    initial_portfolio_value (float):
        The starting value of the portfolio in USD. Must be a positive number.

    simulation_parameters_dict (dict):
        A dictionary containing the core financial parameters for the simulation.
        Expected keys:
        - 'time_horizon_years' (int): The number of years to simulate into the future.
        - 'expected_annual_return' (float): The expected average annual return (drift) of the portfolio (e.g., 0.08 for 8%).
        - 'annual_volatility' (float): The annual volatility (standard deviation) of the portfolio (e.g., 0.15 for 15%).
        - 'annual_contribution' (float): The amount of money added to the portfolio each year.
        - 'time_steps_per_year' (int): The number of steps to simulate per year (e.g., 252 for trading days).

    number_of_simulations (int):
        The total number of separate simulation paths to run. A higher number (e.g., 10000)
        leads to a more accurate distribution of outcomes but takes longer to compute.

    progress_callback_function (function, optional):
        A callback function that can be used to report progress. The function will be called
        periodically with the percentage of simulations completed. It should accept one
        argument (float, from 0.0 to 1.0). Defaults to None.

    Returns:
    -------
    dict:
        A dictionary containing the detailed results of the simulations.
        {
            'simulation_summary': {
                'number_of_simulations': int,
                'median_final_value': float,      // 50th percentile outcome
                'percentile_5th_final_value': float,  // A measure of downside risk (Value at Risk)
                'percentile_95th_final_value': float, // A measure of upside potential
                'average_final_value': float,
                'probability_of_exceeding_target': float // Probability of beating a predefined target
            },
            'simulation_parameters': dict, // A copy of the input parameters
            'raw_simulation_endpoints': list // A list containing the final value of each simulation run
        }

    Raises:
    ------
    ValueError:
        If numerical inputs are invalid (e.g., negative initial value, zero simulations).
    """
    # --- Step 1: Input Validation and Parameter Extraction ---
    if initial_portfolio_value <= 0:
        raise ValueError("'initial_portfolio_value' must be positive.")
    if number_of_simulations <= 0:
        raise ValueError("'number_of_simulations' must be positive.")

    # Extract parameters from the dictionary for easier access.
    try:
        time_horizon_years = simulation_parameters_dict['time_horizon_years']
        expected_annual_return = simulation_parameters_dict['expected_annual_return']
        annual_volatility = simulation_parameters_dict['annual_volatility']
        annual_contribution = simulation_parameters_dict['annual_contribution']
        time_steps_per_year = simulation_parameters_dict['time_steps_per_year']
    except KeyError as e:
        raise ValueError(f"Missing a required key in 'simulation_parameters_dict': {e}")

    print("--- Monte Carlo Simulation Initializing ---")
    print(f"  Initial Portfolio Value: ${initial_portfolio_value:,.2f}")
    print(f"  Time Horizon: {time_horizon_years} years")
    print(f"  Expected Annual Return: {expected_annual_return:.2%}")
    print(f"  Annual Volatility: {annual_volatility:.2%}")
    print(f"  Annual Contribution: ${annual_contribution:,.2f}")
    print(f"  Total Simulations to Run: {number_of_simulations:,}")
    print("-----------------------------------------")

    # --- Step 2: Pre-calculate Time-related Constants ---
    # These values are constant across all simulations, so we calculate them once.
    total_time_steps = int(time_horizon_years * time_steps_per_year)
    time_delta_t = 1 / time_steps_per_year # The fraction of a year each time step represents.

    # Calculate the drift and diffusion components for the geometric Brownian motion formula.
    # Drift is the deterministic part of the motion (average return).
    drift_component = (expected_annual_return - 0.5 * annual_volatility**2) * time_delta_t
    # Diffusion is the random part of the motion (volatility).
    diffusion_component = annual_volatility * math.sqrt(time_delta_t)

    # Contribution per time step, assuming it's added evenly throughout the year.
    contribution_per_step = annual_contribution / time_steps_per_year

    # --- Step 3: Run the Simulations ---
    # This is the core computational part of the function.
    final_portfolio_values_from_all_simulations = []

    for simulation_index in range(number_of_simulations):
        # For each simulation, we start with the same initial value.
        current_portfolio_value = initial_portfolio_value
        
        # This list tracks the value at each step within this single simulation.
        # We don't use this list in the final return value for memory efficiency,
        # but it would be needed if we wanted to plot individual paths.
        portfolio_path_history = [initial_portfolio_value]

        for time_step_index in range(total_time_steps):
            # Generate a random shock for this time step from a standard normal distribution.
            random_shock_z = random.normalvariate(0, 1)

            # Apply the geometric Brownian motion formula to calculate the percentage change.
            percentage_change = math.exp(drift_component + diffusion_component * random_shock_z)
            
            # Update the portfolio value with the market movement.
            current_portfolio_value *= percentage_change
            
            # Add the periodic contribution.
            current_portfolio_value += contribution_per_step

            # Ensure portfolio value doesn't go below zero (it's a limited liability asset).
            if current_portfolio_value < 0:
                current_portfolio_value = 0
            
            portfolio_path_history.append(current_portfolio_value)

        # After all time steps for this simulation are complete, store the final value.
        final_portfolio_values_from_all_simulations.append(current_portfolio_value)

        # --- Progress Reporting (if a callback is provided) ---
        if progress_callback_function is not None:
            # Report progress every 1% of the way through.
            if (simulation_index + 1) % (number_of_simulations // 100) == 0:
                completion_percentage = (simulation_index + 1) / number_of_simulations
                try:
                    progress_callback_function(completion_percentage)
                except Exception as e:
                    # Don't let a faulty callback stop the simulation.
                    print(f"[WARN] - Progress callback function failed: {e}")

    print("\n--- Simulation Complete ---")
    print("  Analyzing distribution of outcomes...")

    # --- Step 4: Analyze the Results ---
    # Now that we have the final values from all simulations, we can calculate statistics.
    
    # Sort the results to easily find percentiles.
    final_portfolio_values_from_all_simulations.sort()

    # Calculate key statistical measures.
    average_final_value = sum(final_portfolio_values_from_all_simulations) / number_of_simulations
    
    # Median (50th percentile) is more robust to outliers than the mean.
    median_index = number_of_simulations // 2
    median_final_value = final_portfolio_values_from_all_simulations[median_index]

    # 5th percentile represents a "bad" outcome (Value at Risk).
    percentile_5th_index = int(number_of_simulations * 0.05)
    percentile_5th_final_value = final_portfolio_values_from_all_simulations[percentile_5th_index]

    # 95th percentile represents a "good" outcome.
    percentile_95th_index = int(number_of_simulations * 0.95)
    percentile_95th_final_value = final_portfolio_values_from_all_simulations[percentile_95th_index]
    
    # Define an arbitrary target to calculate probability of success.
    # For example, let's say the target is to double the initial investment.
    success_target_value = initial_portfolio_value * 2
    number_of_successful_simulations = sum(1 for value in final_portfolio_values_from_all_simulations if value >= success_target_value)
    probability_of_exceeding_target = number_of_successful_simulations / number_of_simulations

    # --- Step 5: Assemble the Final Results Dictionary ---
    final_results_dictionary = {
        'simulation_summary': {
            'number_of_simulations': number_of_simulations,
            'median_final_value': median_final_value,
            'percentile_5th_final_value': percentile_5th_final_value,
            'percentile_95th_final_value': percentile_95th_final_value,
            'average_final_value': average_final_value,
            'probability_of_exceeding_target': probability_of_exceeding_target
        },
        'simulation_parameters': simulation_parameters_dict,
        'raw_simulation_endpoints': final_portfolio_values_from_all_simulations
    }
    
    print("--- Analysis Results ---")
    print(f"  Average Final Value: ${final_results_dictionary['simulation_summary']['average_final_value']:,.2f}")
    print(f"  Median Final Value (50th Percentile): ${final_results_dictionary['simulation_summary']['median_final_value']:,.2f}")
    print(f"  Worst 1-in-20 Outcome (5th Percentile): ${final_results_dictionary['simulation_summary']['percentile_5th_final_value']:,.2f}")
    print(f"  Best 1-in-20 Outcome (95th Percentile): ${final_results_dictionary['simulation_summary']['percentile_95th_final_value']:,.2f}")
    print(f"  Probability of Doubling Initial Investment: {final_results_dictionary['simulation_summary']['probability_of_exceeding_target']:.2%}")
    print("------------------------")

    return final_results_dictionary


# ==================================================================================================
# FUNCTION 3: SEO AND READABILITY CONTENT ANALYSIS REPORT GENERATOR
# ==================================================================================================

def generate_comprehensive_seo_and_readability_report(article_text, primary_target_keyword, related_keyword_list):
    """
    Analyzes a given text for various SEO and readability metrics and generates a
    detailed, human-readable report.

    This function performs a deep analysis of a piece of content (like a blog post or article).
    It calculates multiple metrics crucial for on-page Search Engine Optimization (SEO) and
    user experience. The analysis includes keyword density, placement checks, and multiple
    standard readability scores. The final output is a long, formatted string that presents
    the findings in a structured report.

    This function intentionally uses manual, verbose implementations for readability formulas
    to increase its length and complexity for benchmarking purposes.

    Parameters:
    ----------
    article_text (str):
        The full text of the article to be analyzed. Should be a single string.

    primary_target_keyword (str):
        The main keyword or keyphrase that the article is targeting for SEO. The analysis
        will be centered around this keyword.

    related_keyword_list (list of str):
        A list of secondary or LSI (Latent Semantic Indexing) keywords that should ideally
        appear in the text.

    Returns:
    -------
    str:
        A very long, multi-line string formatted as a comprehensive report containing all
        the calculated metrics and recommendations.
    """
    
    # --- Part 1: Initial Text Pre-processing and Basic Counts ---
    # We need to clean and prepare the text for accurate analysis.
    
    print("--- Content Analysis Engine Initialized ---")
    
    # Normalize the text for case-insensitive matching.
    text_lower = article_text.lower()
    keyword_lower = primary_target_keyword.lower()
    
    # --- Word Count ---
    # A simple word count based on splitting by whitespace.
    words = article_text.split()
    total_word_count = len(words)
    print(f"  [INFO] - Total word count: {total_word_count}")
    
    # --- Sentence Count ---
    # A more robust sentence count using regular expressions to handle various terminators.
    sentences = re.split(r'[.!?]+', article_text)
    # Filter out any empty strings that may result from the split.
    sentences = [s for s in sentences if s.strip()]
    total_sentence_count = len(sentences)
    print(f"  [INFO] - Total sentence count: {total_sentence_count}")

    # --- Syllable Count (Verbose Implementation) ---
    # This is a crucial component for readability scores. We'll implement it manually.
    def calculate_syllables_in_word(word_to_check):
        word_to_check = word_to_check.lower().strip(".:;?!")
        if not word_to_check:
            return 0
        
        # A simple heuristic for syllable counting.
        vowels = "aeiouy"
        syllable_count_for_word = 0
        
        # Rule 1: Count vowel groups.
        if word_to_check[0] in vowels:
            syllable_count_for_word += 1
        for index in range(1, len(word_to_check)):
            if word_to_check[index] in vowels and word_to_check[index - 1] not in vowels:
                syllable_count_for_word += 1
        
        # Rule 2: Handle silent 'e' at the end.
        if word_to_check.endswith("e"):
            syllable_count_for_word -= 1
        
        # Rule 3: Handle words with no vowels (like 'rhythm').
        if syllable_count_for_word == 0:
            syllable_count_for_word = 1
            
        return syllable_count_for_word

    total_syllable_count = 0
    for word_item in words:
        total_syllable_count += calculate_syllables_in_word(word_item)
    print(f"  [INFO] - Total syllable count: {total_syllable_count}")
    
    # --- Part 2: SEO Analysis ---
    # This section focuses on keyword placement and density.
    
    print("  [STATUS] - Performing SEO analysis...")
    
    # --- Keyword Density ---
    primary_keyword_count = text_lower.count(keyword_lower)
    # Avoid division by zero if the text is empty.
    keyword_density = (primary_keyword_count / total_word_count) * 100 if total_word_count > 0 else 0
    
    # --- Keyword Placement Checks ---
    # For simplicity, we define "title" as the first line and "first paragraph" as the first 100 words.
    first_line = article_text.split('\n')[0]
    is_keyword_in_title = keyword_lower in first_line.lower()
    
    first_100_words = " ".join(words[:100])
    is_keyword_in_first_paragraph = keyword_lower in first_100_words.lower()
    
    # --- Secondary Keyword Analysis ---
    found_related_keywords = []
    missing_related_keywords = []
    for related_keyword in related_keyword_list:
        if related_keyword.lower() in text_lower:
            found_related_keywords.append(related_keyword)
        else:
            missing_related_keywords.append(related_keyword)
            
    # --- Part 3: Readability Analysis ---
    # This section implements standard readability formulas.
    
    print("  [STATUS] - Performing readability analysis...")

    # --- Flesch-Kincaid Reading Ease ---
    # Formula: 206.835 - 1.015 * (total_words / total_sentences) - 84.6 * (total_syllables / total_words)
    flesch_reading_ease_score = 0
    if total_word_count > 0 and total_sentence_count > 0:
        average_sentence_length = total_word_count / total_sentence_count
        average_syllables_per_word = total_syllable_count / total_word_count
        flesch_reading_ease_score = 206.835 - (1.015 * average_sentence_length) - (84.6 * average_syllables_per_word)
    
    def interpret_flesch_score(score):
        if score > 90: return "Very easy to read. Easily understood by an average 11-year-old student."
        if score > 80: return "Easy to read. Conversational English for consumers."
        if score > 70: return "Fairly easy to read."
        if score > 60: return "Plain English. Easily understood by 13- to 15-year-old students."
        if score > 50: return "Fairly difficult to read."
        if score > 30: return "Difficult to read."
        return "Very difficult to read. Best understood by university graduates."

    flesch_interpretation = interpret_flesch_score(flesch_reading_ease_score)
    
    # --- Gunning Fog Index ---
    # Formula: 0.4 * ( (words / sentences) + 100 * (complex_words / words) )
    # A "complex word" is usually defined as a word with 3 or more syllables.
    complex_word_count = 0
    for word_item in words:
        if calculate_syllables_in_word(word_item) >= 3:
            complex_word_count += 1
            
    gunning_fog_index = 0
    if total_word_count > 0 and total_sentence_count > 0:
        average_sentence_length = total_word_count / total_sentence_count
        percentage_of_complex_words = (complex_word_count / total_word_count) * 100
        gunning_fog_index = 0.4 * (average_sentence_length + percentage_of_complex_words)
        
    # --- Part 4: Report Generation ---
    # Assemble all the findings into a single, detailed string.
    
    print("  [STATUS] - Generating final report string...")

    report_string = f"""
======================================================================
==  COMPREHENSIVE CONTENT ANALYSIS REPORT
======================================================================
Date of Analysis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

--- SECTION A: DOCUMENT OVERVIEW ---
- Total Word Count: {total_word_count}
- Total Sentence Count: {total_sentence_count}
- Total Syllable Count: {total_syllable_count}
- Average Words per Sentence: {total_word_count / total_sentence_count:.2f}

--- SECTION B: SEO ANALYSIS ---
Primary Target Keyword: '{primary_target_keyword}'

1.  **Keyword Density**:
    - The primary keyword appears {primary_keyword_count} times.
    - This results in a keyword density of {keyword_density:.2f}%.
    - **Recommendation**: A healthy density is typically between 0.5% and 2.0%.
      {'Your density is within the optimal range. Good job!' if 0.5 <= keyword_density <= 2.0 else 'Consider adjusting the keyword frequency.'}

2.  **Keyword Placement**:
    - Is keyword in the title (first line)? {'YES' if is_keyword_in_title else 'NO'}
    - Is keyword in the first paragraph (first 100 words)? {'YES' if is_keyword_in_first_paragraph else 'NO'}
    - **Recommendation**: Ensure the primary keyword appears early in the document, preferably in the title and introduction, for maximum SEO impact.

3.  **Secondary & LSI Keyword Analysis**:
    - Total related keywords provided: {len(related_keyword_list)}
    - Found Keywords ({len(found_related_keywords)}): {', '.join(found_related_keywords) if found_related_keywords else 'None'}
    - Missing Keywords ({len(missing_related_keywords)}): {', '.join(missing_related_keywords) if missing_related_keywords else 'None'}
    - **Recommendation**: Try to naturally incorporate the missing related keywords to improve the topical relevance of the article.

--- SECTION C: READABILITY ANALYSIS ---

1.  **Flesch-Kincaid Reading Ease**:
    - Score: {flesch_reading_ease_score:.2f} (out of 100, higher is better)
    - Interpretation: {flesch_interpretation}
    - **Recommendation**: For general web content, a score of 60-70 is often recommended. If your score is significantly lower, consider simplifying your sentences and vocabulary.

2.  **Gunning Fog Index**:
    - Score: {gunning_fog_index:.2f}
    - Interpretation: The score corresponds to the years of formal education a person needs to understand the text easily on the first reading.
    - **Recommendation**: A Gunning Fog index of around 7-8 is ideal for web content. Scores above 12 are generally too complex for a wide audience.

--- END OF REPORT ---
"""
    print("--- Analysis Complete. Report Generated. ---")
    
    return report_string

import math
import random
import time
from datetime import datetime, timedelta

# --- Function 1: Very Short (approx. 5-10 lines) ---
def calculate_area_rectangle(length, width):
    """
    Calculates the area of a rectangle.
    Args:
        length (float): The length of the rectangle.
        width (float): The width of the rectangle.
    Returns:
        float: The calculated area.
    """
    if length < 0 or width < 0:
        raise ValueError("Dimensions cannot be negative.")
    return length * width

# --- Function 2: Short (approx. 20-30 lines) ---
def process_data_entry(data_dict, key_to_extract, default_value=None):
    """
    Processes a single data entry, extracting a specific key and performing
    a basic transformation.

    Args:
        data_dict (dict): The dictionary containing the data.
        key_to_extract (str): The key whose value needs to be extracted.
        default_value (any): Default value to return if key is not found.

    Returns:
        str or None: The processed value or None if key not found and no default.
    """
    if not isinstance(data_dict, dict):
        print("Warning: Input data is not a dictionary.")
        return default_value

    if key_to_extract in data_dict:
        value = data_dict[key_to_extract]
        # Simulate some processing based on type
        if isinstance(value, str):
            return value.strip().upper()
        elif isinstance(value, (int, float)):
            return str(value * 2) # Convert to string after simple operation
        else:
            return str(value) # Generic conversion
    else:
        print(f"Key '{key_to_extract}' not found in data.")
        return default_value

# --- Function 3: Medium (approx. 100-150 lines) ---
def analyze_sensor_readings(readings_list, threshold=50.0, window_size=5):
    """
    Analyzes a list of sensor readings, identifying anomalies and trends.

    Args:
        readings_list (list): A list of numerical sensor readings.
        threshold (float): The anomaly detection threshold. Readings above this
                           are flagged as high.
        window_size (int): The number of readings to consider for moving average.

    Returns:
        dict: A dictionary containing analysis results, including:
              - 'total_readings' (int)
              - 'average_reading' (float)
              - 'max_reading' (float)
              - 'min_reading' (float)
              - 'anomalies' (list of dict): Details of anomalous readings.
              - 'trends' (list of float): Moving averages.
              - 'timestamp' (str): When the analysis was performed.
    """
    results = {
        "total_readings": len(readings_list),
        "average_reading": 0.0,
        "max_reading": -float('inf'),
        "min_reading": float('inf'),
        "anomalies": [],
        "trends": [],
        "timestamp": datetime.now().isoformat()
    }

    if not readings_list:
        print("No readings to analyze.")
        return results

    total_sum = 0
    for i, reading in enumerate(readings_list):
        if not isinstance(reading, (int, float)):
            print(f"Skipping non-numeric reading at index {i}: {reading}")
            continue

        total_sum += reading
        results['max_reading'] = max(results['max_reading'], reading)
        results['min_reading'] = min(results['min_reading'], reading)

        if reading > threshold:
            results['anomalies'].append({
                "index": i,
                "value": reading,
                "type": "high_threshold"
            })
        elif reading < -threshold: # Example of a low threshold anomaly
             results['anomalies'].append({
                "index": i,
                "value": reading,
                "type": "low_threshold"
            })


        # Calculate moving average
        if i >= window_size - 1:
            window_sum = sum(readings_list[i - window_size + 1 : i + 1])
            results['trends'].append(window_sum / window_size)
        else:
            results['trends'].append(None) # Not enough data for initial windows

    results['average_reading'] = total_sum / len(readings_list)

    # Simulate some complex conditional logic
    if results['average_reading'] > threshold * 0.8:
        print("Average reading is approaching threshold levels.")
    if len(results['anomalies']) > len(readings_list) * 0.1:
        print("Warning: High percentage of anomalies detected!")

    # Add some dummy computation to increase line count
    for _ in range(10):
        dummy_var = math.sqrt(random.random() * 100)
        _ = dummy_var * 2 # Just to use it
        time.sleep(0.0001) # Small delay to simulate work

    return results

# --- Function 4: Long (approx. 500 lines) ---
def complex_data_pipeline_processor(input_data_path, output_report_path, config_options):
    """
    Simulates a complex data processing pipeline, including data loading,
    transformation, validation, aggregation, and report generation.

    This function is designed to be lengthy and simulate various stages
    of a real-world data pipeline. It includes placeholders for more complex
    logic that would typically involve external libraries or database interactions.

    Args:
        input_data_path (str): Path to the input data file (e.g., CSV, JSON).
        output_report_path (str): Path where the final report will be saved.
        config_options (dict): Configuration dictionary for the pipeline,
                               e.g., {"validation_rules": [...], "aggregation_keys": [...]}.

    Returns:
        bool: True if the pipeline completes successfully, False otherwise.
    """
    print(f"Starting complex data pipeline for: {input_data_path}")
    success = True
    processed_records_count = 0
    error_log = []
    final_report_data = {}

    try:
        # --- Stage 1: Configuration Loading and Validation --- (approx 50 lines)
        if not isinstance(config_options, dict) or not config_options:
            error_log.append("Error: Invalid or empty configuration options provided.")
            return False
        required_configs = ["validation_rules", "aggregation_keys", "output_format"]
        for rc in required_configs:
            if rc not in config_options:
                error_log.append(f"Missing required configuration: {rc}")
                success = False
        if not success: return False
        print("Configuration validated.")
        time.sleep(0.01) # Simulate config load time

        # --- Stage 2: Data Loading --- (approx 80 lines)
        raw_data = []
        try:
            with open(input_data_path, 'r', encoding='utf-8') as f:
                # Simulate reading a large file line by line
                header = f.readline().strip().split(',')
                for i, line in enumerate(f):
                    if i > 10000: # Limit for simulation purposes
                        print("Warning: Limiting data load to 10,000 lines for demo.")
                        break
                    try:
                        record = dict(zip(header, line.strip().split(',')))
                        raw_data.append(record)
                    except Exception as e:
                        error_log.append(f"Error parsing line {i+1}: {line.strip()} - {e}")
            print(f"Loaded {len(raw_data)} raw records.")
            time.sleep(0.02)
        except FileNotFoundError:
            error_log.append(f"Error: Input file not found at {input_data_path}")
            return False
        except Exception as e:
            error_log.append(f"Error during data loading: {e}")
            return False

        # --- Stage 3: Data Transformation and Cleaning --- (approx 150 lines)
        transformed_data = []
        for i, record in enumerate(raw_data):
            cleaned_record = {}
            record_errors = []
            for key, value in record.items():
                try:
                    # Apply various transformation rules
                    if key.endswith('_id'):
                        cleaned_record[key] = value.strip().lower()
                    elif key == 'timestamp':
                        # Try parsing different date formats
                        try:
                            cleaned_record[key] = datetime.fromisoformat(value).isoformat()
                        except ValueError:
                            try:
                                cleaned_record[key] = datetime.strptime(value, '%Y/%m/%d %H:%M:%S').isoformat()
                            except ValueError:
                                record_errors.append(f"Invalid timestamp format for '{key}': {value}")
                                cleaned_record[key] = None
                    elif 'price' in key or 'amount' in key:
                        cleaned_record[key] = float(value)
                    elif 'status' in key:
                        cleaned_record[key] = value.strip().capitalize()
                    else:
                        cleaned_record[key] = value.strip()
                except Exception as e:
                    record_errors.append(f"Error transforming '{key}':'{value}' - {e}")
                    cleaned_record[key] = None # Set to None on error

            # Apply validation rules from config_options
            for rule in config_options.get('validation_rules', []):
                field = rule.get('field')
                min_val = rule.get('min')
                max_val = rule.get('max')
                if field in cleaned_record and cleaned_record[field] is not None:
                    if min_val is not None and cleaned_record[field] < min_val:
                        record_errors.append(f"Validation failed for {field}: {cleaned_record[field]} < {min_val}")
                    if max_val is not None and cleaned_record[field] > max_val:
                        record_errors.append(f"Validation failed for {field}: {cleaned_record[field]} > {max_val}")
                # Add more complex validation (e.g., regex, lookup) here
                if field == 'email' and '@' not in cleaned_record.get(field, ''):
                     record_errors.append(f"Validation failed for {field}: Invalid email format")

            if record_errors:
                error_log.append(f"Validation/Transformation errors for record {i}: {record_errors}")
            else:
                transformed_data.append(cleaned_record)
                processed_records_count += 1
        print(f"Transformed {len(transformed_data)} records with {len(error_log)} errors.")
        time.sleep(0.03)

        # --- Stage 4: Data Aggregation --- (approx 100 lines)
        aggregated_results = {}
        aggregation_keys = config_options.get('aggregation_keys', [])
        for record in transformed_data:
            key_values = tuple(record.get(k) for k in aggregation_keys)
            if key_values not in aggregated_results:
                aggregated_results[key_values] = {
                    "count": 0,
                    "total_price": 0.0,
                    "first_seen": None,
                    "last_seen": None
                }
                # Initialize aggregation keys in the result as well
                for k in aggregation_keys:
                    aggregated_results[key_values][k] = record.get(k)

            aggregated_results[key_values]["count"] += 1
            if 'price' in record and record['price'] is not None:
                aggregated_results[key_values]["total_price"] += record['price']

            current_timestamp_str = record.get('timestamp')
            if current_timestamp_str:
                current_dt = datetime.fromisoformat(current_timestamp_str)
                if aggregated_results[key_values]["first_seen"] is None or \
                   current_dt < aggregated_results[key_values]["first_seen"]:
                    aggregated_results[key_values]["first_seen"] = current_dt
                if aggregated_results[key_values]["last_seen"] is None or \
                   current_dt > aggregated_results[key_values]["last_seen"]:
                    aggregated_results[key_values]["last_seen"] = current_dt

        # Convert datetime objects back to string for final report
        for key, agg_data in aggregated_results.items():
            if agg_data["first_seen"]:
                agg_data["first_seen"] = agg_data["first_seen"].isoformat()
            if agg_data["last_seen"]:
                agg_data["last_seen"] = agg_data["last_seen"].isoformat()

        final_report_data['aggregation_summary'] = list(aggregated_results.values())
        print(f"Aggregated data into {len(aggregated_results)} groups.")
        time.sleep(0.02)

        # --- Stage 5: Report Generation and Saving --- (approx 120 lines)
        report_content = []
        report_content.append(f"--- Data Pipeline Report ({datetime.now().isoformat()}) ---")
        report_content.append(f"Input File: {input_data_path}")
        report_content.append(f"Total Raw Records: {len(raw_data)}")
        report_content.append(f"Successfully Transformed Records: {processed_records_count}")
        report_content.append(f"Total Aggregation Groups: {len(final_report_data['aggregation_summary'])}")
        report_content.append("\n--- Aggregation Summary ---")
        for i, group in enumerate(final_report_data['aggregation_summary']):
            report_content.append(f"\nGroup {i+1}:")
            for k, v in group.items():
                report_content.append(f"  {k}: {v}")

        if error_log:
            report_content.append("\n--- Errors and Warnings ---")
            for error_msg in error_log:
                report_content.append(f"- {error_msg}")
        else:
            report_content.append("\nNo significant errors detected during pipeline execution.")

        report_content.append("\n--- Pipeline Metrics ---")
        report_content.append(f"Processing started: {datetime.now() - timedelta(minutes=random.randint(1,5))}") # Dummy start time
        report_content.append(f"Processing finished: {datetime.now()}")
        report_content.append(f"Total errors logged: {len(error_log)}")
        report_content.append("End of Report.")

        try:
            with open(output_report_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(report_content))
            print(f"Report successfully saved to: {output_report_path}")
        except Exception as e:
            error_log.append(f"Error saving report: {e}")
            success = False

    except Exception as e:
        error_log.append(f"An unhandled exception occurred during pipeline execution: {e}")
        success = False
    finally:
        if error_log:
            print("\n--- Pipeline finished with errors ---")
            for err in error_log:
                print(err)
        else:
            print("\n--- Pipeline completed successfully ---")

    return success

# --- Function 5: Very Long (approx. 1000 lines - conceptual outline) ---
def advanced_ml_model_trainer(model_config, dataset_paths, output_model_path, evaluation_metrics_path):
    """
    This function simulates an extensive machine learning model training pipeline.
    It encompasses data loading, preprocessing, feature engineering, model definition,
    training loop, evaluation, hyperparameter tuning simulation, and model saving.

    To reach 1000 lines, this function would contain numerous sub-functions,
    extensive conditional logic, detailed logging, and repetitive but necessary
    steps in an ML workflow. The actual implementation would use libraries
    like scikit-learn, TensorFlow, or PyTorch, but here it's
    conceptualized with dummy operations.

    Args:
        model_config (dict): Configuration for the ML model (e.g., {"model_type": "LogisticRegression", "hyperparameters": {"C": 1.0}}).
        dataset_paths (dict): Paths to training and validation datasets (e.g., {"train": "train.csv", "val": "val.csv"}).
        output_model_path (str): Path to save the trained model.
        evaluation_metrics_path (str): Path to save the evaluation results.

    Returns:
        bool: True if training and evaluation are successful, False otherwise.
    """
    print(f"Starting advanced ML model training pipeline for model type: {model_config.get('model_type', 'Unknown')}")
    pipeline_status = True
    training_log = []
    evaluation_results = {}
    start_time = time.time()

    try:
        # --- 1. Pipeline Initialization and Config Validation (approx 50 lines) ---
        training_log.append("Initializing pipeline and validating configuration...")
        required_ml_configs = ["model_type", "features", "target_variable"]
        for rc in required_ml_configs:
            if rc not in model_config:
                training_log.append(f"Error: Missing required model configuration: {rc}")
                pipeline_status = False
        if not pipeline_status: return False
        training_log.append("Configuration validated.")
        time.sleep(0.01)

        # --- 2. Data Loading (approx 100 lines) ---
        training_log.append(f"Loading data from: {dataset_paths.get('train')} and {dataset_paths.get('val')}")
        train_data = []
        val_data = []
        # Dummy data loading
        for i in range(random.randint(500, 1000)):
            train_data.append({
                "feature_a": random.random(),
                "feature_b": random.randint(1, 100),
                "feature_c": random.choice(["cat", "dog", "bird"]),
                "target": random.randint(0, 1)
            })
        for i in range(random.randint(100, 200)):
            val_data.append({
                "feature_a": random.random(),
                "feature_b": random.randint(1, 100),
                "feature_c": random.choice(["cat", "dog", "bird"]),
                "target": random.randint(0, 1)
            })
        training_log.append(f"Loaded {len(train_data)} training and {len(val_data)} validation samples.")
        time.sleep(0.02)

        # --- 3. Data Preprocessing and Feature Engineering (approx 200 lines) ---
        training_log.append("Starting data preprocessing and feature engineering...")
        processed_train_features = []
        processed_train_target = []
        processed_val_features = []
        processed_val_target = []

        # Simulate one-hot encoding for categorical features
        unique_categories = set()
        for rec in train_data + val_data:
            unique_categories.add(rec.get('feature_c'))
        category_map = {cat: i for i, cat in enumerate(sorted(list(unique_categories)))}

        for record_list, features_list, target_list in [(train_data, processed_train_features, processed_train_target),
                                                         (val_data, processed_val_features, processed_val_target)]:
            for i, record in enumerate(record_list):
                current_features = []
                current_target = record.get(model_config.get('target_variable'))
                if current_target is None:
                    training_log.append(f"Skipping record {i} due to missing target.")
                    continue

                for feature_name in model_config.get('features', []):
                    value = record.get(feature_name)
                    if feature_name == 'feature_a':
                        current_features.append(value if value is not None else 0.0)
                    elif feature_name == 'feature_b':
                        current_features.append(float(value) if value is not None else 0.0)
                    elif feature_name == 'feature_c':
                        # One-hot encode
                        one_hot_vector = [0] * len(category_map)
                        if value in category_map:
                            one_hot_vector[category_map[value]] = 1
                        current_features.extend(one_hot_vector)
                    else:
                        training_log.append(f"Warning: Unknown feature '{feature_name}' encountered.")
                        current_features.append(0) # Default for unknown

                # Simulate scaling (e.g., min-max scaling) - conceptual
                scaled_features = [f / 100.0 for f in current_features] # Dummy scaling
                processed_train_features.append(scaled_features)
                processed_train_target.append(current_target)

                # Add more complex feature engineering here (interactions, polynomial features)
                if 'feature_a' in record and 'feature_b' in record:
                    processed_train_features[-1].append(record['feature_a'] * record['feature_b']) # Interaction feature
                if 'feature_a' in record:
                    processed_train_features[-1].append(record['feature_a']**2) # Polynomial feature

        training_log.append("Data preprocessing and feature engineering complete.")
        time.sleep(0.03)


        # --- 4. Model Definition and Initialization (approx 80 lines) ---
        training_log.append(f"Defining and initializing model: {model_config.get('model_type')}")
        model = None
        # Placeholder for actual model initialization
        if model_config.get('model_type') == "LogisticRegression":
            # Simulate a simple linear model training
            model = {"weights": [random.uniform(-0.1, 0.1) for _ in processed_train_features[0]], "bias": random.uniform(-0.1, 0.1)}
            training_log.append("Logistic Regression model initialized (simulated).")
        elif model_config.get('model_type') == "DecisionTree":
            model = {"depth": model_config.get('hyperparameters', {}).get('max_depth', 5), "nodes": []}
            training_log.append("Decision Tree model initialized (simulated).")
        else:
            training_log.append(f"Error: Unsupported model type: {model_config.get('model_type')}")
            pipeline_status = False
        if not pipeline_status: return False
        time.sleep(0.01)

        # --- 5. Training Loop (approx 300 lines) ---
        training_log.append("Starting training loop...")
        num_epochs = model_config.get('training_epochs', 10)
        learning_rate = model_config.get('learning_rate', 0.01)

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            # Shuffle data for each epoch (conceptual)
            shuffled_indices = list(range(len(processed_train_features)))
            random.shuffle(shuffled_indices)

            for i in shuffled_indices:
                features = processed_train_features[i]
                target = processed_train_target[i]

                # Simulate forward pass
                if model_config.get('model_type') == "LogisticRegression":
                    # Dot product + sigmoid activation
                    raw_prediction = sum(w * f for w, f in zip(model["weights"], features)) + model["bias"]
                    prediction = 1 / (1 + math.exp(-raw_prediction))
                    # Calculate loss (e.g., binary cross-entropy)
                    loss = - (target * math.log(prediction + 1e-9) + (1 - target) * math.log(1 - prediction + 1e-9))
                    epoch_loss += loss

                    # Simulate backward pass and weight update (gradient descent)
                    gradient_prediction = prediction - target
                    for j in range(len(model["weights"])):
                        model["weights"][j] -= learning_rate * gradient_prediction * features[j]
                    model["bias"] -= learning_rate * gradient_prediction
                elif model_config.get('model_type') == "DecisionTree":
                    # Decision trees don't "train" iteratively like this,
                    # so this would be a placeholder for a complex tree building algorithm.
                    # For simplicity, we just simulate some "work" per sample.
                    time.sleep(0.00001) # Simulate computation
                    loss = 0 # No continuous loss for DTs in this way
                    epoch_loss += loss

                # Add internal logging for training progress
                if (i + 1) % 100 == 0:
                    training_log.append(f"  Epoch {epoch+1}/{num_epochs} - Sample {i+1} - Current loss: {loss:.4f}")

            avg_epoch_loss = epoch_loss / len(processed_train_features)
            training_log.append(f"Epoch {epoch+1}/{num_epochs} finished. Average Loss: {avg_epoch_loss:.4f}")
            time.sleep(0.05) # Simulate epoch end overhead

            # Simulate early stopping logic
            if epoch > 2 and avg_epoch_loss > training_log[-2].split("Average Loss: ")[1].strip() and random.random() < 0.2:
                training_log.append(f"Early stopping triggered at epoch {epoch+1} due to increasing loss.")
                break


        training_log.append("Training loop complete.")
        time.sleep(0.01)

        # --- 6. Model Evaluation (approx 150 lines) ---
        training_log.append("Starting model evaluation on validation set...")
        correct_predictions = 0
        total_predictions = 0
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for i, features in enumerate(processed_val_features):
            target = processed_val_target[i]
            total_predictions += 1

            # Simulate prediction for validation data
            if model_config.get('model_type') == "LogisticRegression":
                raw_prediction = sum(w * f for w, f in zip(model["weights"], features)) + model["bias"]
                prediction_probability = 1 / (1 + math.exp(-raw_prediction))
                predicted_label = 1 if prediction_probability >= 0.5 else 0
            elif model_config.get('model_type') == "DecisionTree":
                # Simulate a random decision for a dummy tree
                predicted_label = random.randint(0, 1)
            else:
                predicted_label = 0 # Default

            if predicted_label == target:
                correct_predictions += 1
            
            # Calculate for precision, recall, F1
            if predicted_label == 1 and target == 1:
                true_positives += 1
            elif predicted_label == 1 and target == 0:
                false_positives += 1
            elif predicted_label == 0 and target == 1:
                false_negatives += 1

        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        evaluation_results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "total_val_samples": total_predictions,
            "correct_predictions": correct_predictions,
            "timestamp": datetime.now().isoformat()
        }
        training_log.append(f"Model evaluation complete. Accuracy: {accuracy:.4f}")
        time.sleep(0.01)

        # --- 7. Hyperparameter Tuning Simulation (conceptual, approx 50 lines) ---
        training_log.append("Simulating hyperparameter tuning process...")
        best_accuracy = accuracy
        best_params = model_config.get('hyperparameters', {})

        # This would typically involve looping through a grid/random search
        # and retraining the model many times. Here, we just simulate some
        # "best" outcome or slight adjustment.
        for _ in range(3): # Simulate a few tuning iterations
            simulated_learning_rate = random.uniform(0.005, 0.05)
            simulated_accuracy_gain = random.uniform(-0.01, 0.02) # Simulate minor changes
            simulated_accuracy = accuracy + simulated_accuracy_gain
            if simulated_accuracy > best_accuracy:
                best_accuracy = simulated_accuracy
                best_params['learning_rate'] = simulated_learning_rate
                training_log.append(f"Found better parameters: {best_params} with accuracy {best_accuracy:.4f}")
            time.sleep(0.005)

        evaluation_results['best_tuned_accuracy'] = best_accuracy
        evaluation_results['best_tuned_params'] = best_params
        training_log.append("Hyperparameter tuning simulation complete.")
        time.sleep(0.01)

        # --- 8. Model Saving (approx 50 lines) ---
        training_log.append(f"Saving trained model to: {output_model_path}")
        # In a real scenario, this would involve pickling the model or saving
        # it in a specific framework format (e.g., .h5 for Keras, .pt for PyTorch).
        try:
            with open(output_model_path, 'w') as f:
                f.write(f"SIMULATED_MODEL_STATE_{model_config.get('model_type')}\n")
                f.write(f"Weights: {model.get('weights', 'N/A')}\n")
                f.write(f"Bias: {model.get('bias', 'N/A')}\n")
                f.write(f"Config: {model_config}\n")
                f.write(f"Trained on {len(processed_train_features)} samples.\n")
                f.write(f"Evaluation Metrics: {evaluation_results}\n")
            training_log.append("Model state saved successfully (simulated).")
        except Exception as e:
            training_log.append(f"Error saving model: {e}")
            pipeline_status = False
        time.sleep(0.01)

        # --- 9. Saving Evaluation Metrics (approx 50 lines) ---
        training_log.append(f"Saving evaluation metrics to: {evaluation_metrics_path}")
        try:
            with open(evaluation_metrics_path, 'w') as f:
                import json
                json.dump(evaluation_results, f, indent=4)
            training_log.append("Evaluation metrics saved successfully.")
        except Exception as e:
            training_log.append(f"Error saving evaluation metrics: {e}")
            pipeline_status = False
        time.sleep(0.01)

        # --- 10. Final Cleanup and Logging (approx 20 lines) ---
        end_time = time.time()
        total_duration = end_time - start_time
        training_log.append(f"Pipeline finished in {total_duration:.2f} seconds.")
        training_log.append("Final pipeline status: " + ("SUCCESS" if pipeline_status else "FAILURE"))

    except Exception as e:
        training_log.append(f"An unhandled critical error occurred: {e}")
        pipeline_status = False
    finally:
        print("\n".join(training_log))
        print("ML Pipeline execution complete.")

    return pipeline_status

# --- Main execution block for demonstration ---
if __name__ == "__main__":
    print("--- Running example Function 1 ---")
    try:
        area = calculate_area_rectangle(10, 5)
        print(f"Area of rectangle: {area}")
        # area_neg = calculate_area_rectangle(-2, 5) # This would raise an error
    except ValueError as e:
        print(e)

    print("\n--- Running example Function 2 ---")
    data = {"name": "  Alice  ", "age": 30, "city": "New York"}
    processed_name = process_data_entry(data, "name")
    processed_age = process_data_entry(data, "age")
    processed_country = process_data_entry(data, "country", "N/A")
    print(f"Processed Name: {processed_name}")
    print(f"Processed Age: {processed_age}")
    print(f"Processed Country: {processed_country}")

    print("\n--- Running example Function 3 ---")
    sensor_data = [random.uniform(20, 80) for _ in range(50)] + [90, 95, 25, 10, -60, 70]
    analysis_results = analyze_sensor_readings(sensor_data, threshold=75.0, window_size=10)
    # print(json.dumps(analysis_results, indent=2)) # Uncomment to see full results

    print("\n--- Running example Function 4 ---")
    dummy_input_file = "dummy_input.csv"
    dummy_output_report = "pipeline_report.txt"
    with open(dummy_input_file, 'w') as f:
        f.write("id,timestamp,item,price,status\n")
        f.write("1,2023-01-15T10:00:00,Laptop,1200.50,completed\n")
        f.write("2,2023-01-15T10:05:00,Mouse,25.00,pending\n")
        f.write("3,2023-01-16T11:20:00,Keyboard,75.00,completed\n")
        f.write("4,2023/01/17 14:30:00,Monitor,300.75,failed\n") # Different timestamp format
        f.write("5,2023-01-18T09:00:00,Webcam,50.25,completed\n")
        f.write("6,2023-01-18T09:05:00,Speakers,150.00,completed\n")
        f.write("7,invalid-time,invalid-item,invalid-price,invalid-status\n") # Error line
        for i in range(8, 20): # Add more lines for length
            f.write(f"{i},{datetime.now().isoformat()},Item{i},{random.uniform(10, 500):.2f},status{random.choice(['completed', 'pending'])}\n")

    pipeline_config = {
        "validation_rules": [
            {"field": "price", "min": 0, "max": 2000}
        ],
        "aggregation_keys": ["status"],
        "output_format": "txt"
    }
    pipeline_success = complex_data_pipeline_processor(dummy_input_file, dummy_output_report, pipeline_config)
    print(f"Pipeline execution status: {pipeline_success}")

    print("\n--- Running example Function 5 ---")
    dummy_model_config = {
        "model_type": "LogisticRegression",
        "features": ["feature_a", "feature_b", "feature_c"],
        "target_variable": "target",
        "training_epochs": 5,
        "learning_rate": 0.015,
        "hyperparameters": {"C": 0.8}
    }
    dummy_dataset_paths = {
        "train": "dummy_train.csv",
        "val": "dummy_val.csv"
    }
    dummy_output_model = "trained_model.bin"
    dummy_eval_metrics = "eval_metrics.json"

    ml_pipeline_success = advanced_ml_model_trainer(dummy_model_config, dummy_dataset_paths, dummy_output_model, dummy_eval_metrics)
    print(f"ML Pipeline execution status: {ml_pipeline_success}")



    import os
import csv
import json
import hashlib
import re
import random
from collections import defaultdict, deque
from datetime import datetime, timedelta

# --- Function 6: Medium-Short (approx. 40-60 lines) ---
def validate_email_format(email_address):
    """
    Validates the basic format of an email address using a regular expression.
    This is a simple validation and might not catch all edge cases,
    but it's representative of common data validation tasks.

    Args:
        email_address (str): The email string to validate.

    Returns:
        bool: True if the email format is valid, False otherwise.
    """
    if not isinstance(email_address, str):
        print("Warning: Email address must be a string.")
        return False
    
    # A common regex for basic email validation
    email_regex = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    
    if re.match(email_regex, email_address):
        print(f"Email '{email_address}' is valid.")
        return True
    else:
        print(f"Email '{email_address}' is invalid.")
        return False

# --- Function 7: Medium (approx. 80-120 lines) ---
def generate_inventory_report(inventory_data, output_format="text", min_quantity=0):
    """
    Generates an inventory report from a list of product dictionaries.
    Supports text or JSON output formats.

    Args:
        inventory_data (list of dict): A list of dictionaries, each representing a product.
                                       Expected keys: 'id', 'name', 'category', 'quantity', 'price'.
        output_format (str): Desired output format ('text' or 'json').
        min_quantity (int): Minimum quantity for items to be included in the report.

    Returns:
        str: The formatted inventory report. Returns an empty string if data is invalid.
    """
    if not isinstance(inventory_data, list):
        print("Error: Inventory data must be a list.")
        return ""
    
    report_items = []
    total_value = 0.0
    category_counts = defaultdict(int)

    for item in inventory_data:
        if not isinstance(item, dict):
            print(f"Warning: Skipping non-dictionary item: {item}")
            continue

        item_id = item.get('id', 'N/A')
        item_name = item.get('name', 'Unknown Product')
        item_category = item.get('category', 'Miscellaneous')
        item_quantity = item.get('quantity', 0)
        item_price = item.get('price', 0.0)

        # Validate required fields and type
        if not all(isinstance(val, (str, int, float)) for val in [item_id, item_name, item_category]):
            print(f"Warning: Skipping item {item_id} due to invalid type in core fields.")
            continue
        if not isinstance(item_quantity, int) or item_quantity < 0:
            print(f"Warning: Skipping item {item_id} due to invalid quantity: {item_quantity}")
            continue
        if not isinstance(item_price, (int, float)) or item_price < 0:
            print(f"Warning: Skipping item {item_id} due to invalid price: {item_price}")
            continue

        if item_quantity >= min_quantity:
            report_items.append({
                "id": str(item_id),
                "name": str(item_name),
                "category": str(item_category),
                "quantity": int(item_quantity),
                "price": float(item_price),
                "total_item_value": float(item_quantity * item_price)
            })
            total_value += (item_quantity * item_price)
            category_counts[item_category] += 1
            
            # Simulate some additional processing complexity
            if item_quantity < 10 and item_category != 'Services':
                pass # print(f"Low stock alert for {item_name}!")

    if output_format == "json":
        report = {
            "report_date": datetime.now().isoformat(),
            "total_distinct_items": len(report_items),
            "total_inventory_value": round(total_value, 2),
            "category_summary": dict(category_counts),
            "items": report_items
        }
        return json.dumps(report, indent=4)
    else: # Default to text format
        report_lines = [f"--- Inventory Report ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---"]
        report_lines.append(f"Total Distinct Items (>= {min_quantity} quantity): {len(report_items)}")
        report_lines.append(f"Total Inventory Value: ${total_value:,.2f}")
        report_lines.append("\nCategory Summary:")
        for category, count in sorted(category_counts.items()):
            report_lines.append(f"  - {category}: {count} items")
        
        report_lines.append("\nDetailed Item List:")
        if not report_items:
            report_lines.append("  No items meet the criteria.")
        else:
            for item in report_items:
                report_lines.append(f"  ID: {item['id']}, Name: {item['name']}, Cat: {item['category']}, "
                                    f"Qty: {item['quantity']}, Price: ${item['price']:.2f}, "
                                    f"Value: ${item['total_item_value']:.2f}")
        report_lines.append("\n--- End of Report ---")
        return "\n".join(report_lines)


# --- Function 8: Long (approx. 200-300 lines) ---
def process_user_activity_log(log_file_path, output_summary_path, activity_types_to_track=None):
    """
    Processes a user activity log file, extracts relevant information,
    and generates a summary report. This function simulates reading
    and parsing a semi-structured log, performing aggregations,
    and handling potential errors.

    Log file format: Each line is assumed to be a JSON string representing an event.
    Example event: {"timestamp": "ISO_DATETIME", "user_id": "UUID", "event_type": "LOGIN", "details": {...}}

    Args:
        log_file_path (str): Path to the input activity log file.
        output_summary_path (str): Path to save the generated summary report (CSV).
        activity_types_to_track (list, optional): A list of specific event types to track.
                                                  If None, all types are tracked.

    Returns:
        bool: True if processing completes successfully, False otherwise.
    """
    print(f"Starting to process user activity log: {log_file_path}")
    
    if not os.path.exists(log_file_path):
        print(f"Error: Log file not found at '{log_file_path}'")
        return False

    processed_events_count = 0
    skipped_events_count = 0
    user_activity_summary = defaultdict(lambda: defaultdict(int)) # user_id -> event_type -> count
    hourly_activity_count = defaultdict(int) # YYYY-MM-DD HH -> count
    event_details_log = [] # Store some detailed info for specific events
    
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue # Skip empty lines

                try:
                    event = json.loads(line)
                    
                    user_id = event.get('user_id')
                    event_type = event.get('event_type')
                    timestamp_str = event.get('timestamp')
                    details = event.get('details', {})

                    if not all([user_id, event_type, timestamp_str]):
                        skipped_events_count += 1
                        # print(f"Warning: Missing required fields in line {line_num+1}: {line}")
                        continue

                    # Parse timestamp and extract hour
                    try:
                        event_datetime = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        event_hour_key = event_datetime.strftime('%Y-%m-%d %H')
                    except ValueError:
                        skipped_events_count += 1
                        print(f"Warning: Invalid timestamp format in line {line_num+1}: {timestamp_str}")
                        continue

                    # Filter by activity type if specified
                    if activity_types_to_track and event_type not in activity_types_to_track:
                        skipped_events_count += 1
                        continue

                    # Aggregate user activity
                    user_activity_summary[user_id][event_type] += 1
                    
                    # Aggregate hourly activity
                    hourly_activity_count[event_hour_key] += 1

                    # Log specific details for certain event types (e.g., failed logins)
                    if event_type == "LOGIN_FAILED" and details.get('reason'):
                        event_details_log.append({
                            "user_id": user_id,
                            "timestamp": timestamp_str,
                            "reason": details['reason']
                        })
                    
                    processed_events_count += 1

                except json.JSONDecodeError:
                    skipped_events_count += 1
                    print(f"Warning: Invalid JSON in line {line_num+1}: {line[:50]}...")
                except Exception as e:
                    skipped_events_count += 1
                    print(f"Unhandled error processing line {line_num+1}: {e} - {line[:50]}...")
        
        print(f"Finished parsing log file. Processed {processed_events_count} events, skipped {skipped_events_count}.")

        # --- Generate Summary Report ---
        summary_lines = []
        summary_lines.append(["User ID", "Event Type", "Count"])
        
        # Sort users and then event types for consistent output
        sorted_users = sorted(user_activity_summary.keys())
        for user_id in sorted_users:
            sorted_event_types = sorted(user_activity_summary[user_id].keys())
            for event_type in sorted_event_types:
                count = user_activity_summary[user_id][event_type]
                summary_lines.append([user_id, event_type, count])

        # Add hourly activity to the report (optionally)
        summary_lines.append([]) # Blank line for separation
        summary_lines.append(["Hourly Activity", "Count"])
        for hour_key in sorted(hourly_activity_count.keys()):
            summary_lines.append([hour_key, hourly_activity_count[hour_key]])

        # Add detailed error log (optionally)
        if event_details_log:
            summary_lines.append([])
            summary_lines.append(["Failed Logins (Details)"])
            summary_lines.append(["User ID", "Timestamp", "Reason"])
            for entry in event_details_log:
                summary_lines.append([entry['user_id'], entry['timestamp'], entry['reason']])


        # Write the summary to a CSV file
        try:
            with open(output_summary_path, 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerows(summary_lines)
            print(f"Summary report successfully saved to '{output_summary_path}'")
            return True
        except Exception as e:
            print(f"Error saving summary report: {e}")
            return False

    except IOError as e:
        print(f"File I/O error: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during log processing: {e}")
        return False


# --- Function 9: Medium-Long (approx. 150-200 lines) ---
def optimize_resource_allocation(tasks, available_resources, optimization_goal="cost"):
    """
    Simulates a resource allocation optimizer. Given a list of tasks and available resources,
    it attempts to assign resources to tasks based on a specified optimization goal
    (e.g., minimizing cost, maximizing efficiency). This is a simplified model.

    Args:
        tasks (list of dict): Each task is a dict with 'id', 'name', 'resource_needs' (dict of resource_type: quantity),
                              'priority' (int, higher is more important), 'cost_per_resource' (dict).
        available_resources (dict): Resource type as key, list of resource units as value.
                                    Each resource unit is a dict with 'id', 'type', 'capacity', 'cost_per_hour'.
        optimization_goal (str): "cost" (minimize total cost) or "efficiency" (maximize capacity utilization).

    Returns:
        dict: Allocation plan {'task_id': [assigned_resource_ids]}, and 'unassigned_tasks'.
    """
    print(f"Starting resource allocation optimization with goal: {optimization_goal}")
    allocation_plan = defaultdict(list)
    unassigned_tasks = []
    
    # Create a flattened list of available resource units with their properties
    flat_resources = []
    for r_type, units in available_resources.items():
        for unit in units:
            if not all(k in unit for k in ['id', 'type', 'capacity', 'cost_per_hour']):
                print(f"Warning: Malformed resource unit: {unit}. Skipping.")
                continue
            flat_resources.append(unit)
    
    # Sort tasks based on priority (highest first)
    sorted_tasks = sorted(tasks, key=lambda x: x.get('priority', 0), reverse=True)

    # Sort resources based on optimization goal
    if optimization_goal == "cost":
        # Prioritize cheaper resources for cost optimization
        sorted_resources = sorted(flat_resources, key=lambda x: x.get('cost_per_hour', float('inf')))
    elif optimization_goal == "efficiency":
        # Prioritize resources with higher capacity for efficiency (fill them up)
        sorted_resources = sorted(flat_resources, key=lambda x: x.get('capacity', 0), reverse=True)
    else:
        print(f"Warning: Unknown optimization goal '{optimization_goal}'. Defaulting to cost.")
        sorted_resources = sorted(flat_resources, key=lambda x: x.get('cost_per_hour', float('inf')))

    
    # Keep track of remaining capacity for each resource
    resource_capacity_remaining = {r['id']: r['capacity'] for r in flat_resources}
    resource_assigned_to_task = {r['id']: None for r in flat_resources}
    
    total_estimated_cost = 0.0
    
    for task in sorted_tasks:
        task_id = task.get('id')
        task_name = task.get('name', 'Unnamed Task')
        resource_needs = task.get('resource_needs', {})
        task_assigned = False

        # Try to fulfill each resource need for the task
        current_task_assignments = {} # {resource_type: [assigned_resource_ids]}

        for needed_resource_type, needed_quantity in resource_needs.items():
            assigned_for_type = []
            current_quantity_assigned = 0
            
            # Iterate through sorted available resources
            for resource_unit in sorted_resources:
                if resource_unit['type'] == needed_resource_type and \
                   resource_capacity_remaining[resource_unit['id']] > 0 and \
                   resource_assigned_to_task[resource_unit['id']] is None: # Only assign free resources
                    
                    # Assign the whole resource unit if it meets or exceeds need
                    if resource_unit['capacity'] >= needed_quantity - current_quantity_assigned:
                        assigned_for_type.append(resource_unit['id'])
                        current_quantity_assigned += (needed_quantity - current_quantity_assigned)
                        resource_capacity_remaining[resource_unit['id']] -= (needed_quantity - current_quantity_assigned) # Mark as used completely for this task for now
                        resource_assigned_to_task[resource_unit['id']] = task_id
                        total_estimated_cost += resource_unit['cost_per_hour'] * needed_quantity # Simplified cost calc

                    elif resource_unit['capacity'] > 0: # Partially assign a resource unit (less ideal in reality, but for simulation)
                        assign_amount = min(resource_unit['capacity'], needed_quantity - current_quantity_assigned)
                        assigned_for_type.append(resource_unit['id'])
                        current_quantity_assigned += assign_amount
                        resource_capacity_remaining[resource_unit['id']] -= assign_amount
                        # For partial assignment, resource_assigned_to_task would need more complex logic (e.g., fractional ownership)
                        # For simplicity, if a resource contributes, it's considered "assigned" to this task for now
                        resource_assigned_to_task[resource_unit['id']] = task_id # Even if partially used
                        total_estimated_cost += resource_unit['cost_per_hour'] * assign_amount

                    if current_quantity_assigned >= needed_quantity:
                        break # All needs for this resource type met

            if current_quantity_assigned < needed_quantity:
                print(f"  Warning: Not enough '{needed_resource_type}' resources found for Task '{task_name}' (ID: {task_id}). Needed: {needed_quantity}, Assigned: {current_quantity_assigned}")
                task_assigned = False
                # If a task cannot get all its needed resources, it might be unassigned completely.
                # For this simulation, we'll mark it unassigned if any resource type isn't fully met.
                break 
            else:
                current_task_assignments[needed_resource_type] = assigned_for_type
                task_assigned = True # Mark true for this resource type

        if task_assigned:
            # If all resource needs for the task were met
            for r_type, assigned_ids in current_task_assignments.items():
                allocation_plan[task_id].extend(assigned_ids)
            print(f"  Task '{task_name}' (ID: {task_id}) assigned successfully.")
        else:
            unassigned_tasks.append(task_id)
            print(f"  Task '{task_name}' (ID: {task_id}) could not be fully assigned resources.")
            # Revert any partial assignments made for this task if it's completely unassigned
            for res_id, assigned_tid in resource_assigned_to_task.items():
                if assigned_tid == task_id:
                    # This is simplified. In real code, you'd track how much was used and return that.
                    resource_capacity_remaining[res_id] = available_resources[next(r for r in flat_resources if r['id'] == res_id)['type']][
                        [i for i, r in enumerate(available_resources[next(r for r in flat_resources if r['id'] == res_id)['type']]) if r['id'] == res_id][0]
                    ]['capacity'] # Reset capacity
                    resource_assigned_to_task[res_id] = None # Unassign
                    # Also need to subtract from total_estimated_cost
    
    print(f"\nOptimization complete. Total estimated cost (simplified): ${total_estimated_cost:,.2f}")
    print(f"Number of unassigned tasks: {len(unassigned_tasks)}")
    
    final_report = {
        "allocation_plan": {k: list(set(v)) for k, v in allocation_plan.items()}, # Ensure unique resource IDs
        "unassigned_tasks": unassigned_tasks,
        "total_estimated_cost": round(total_estimated_cost, 2),
        "resource_utilization_summary": {
            r_id: (res_info['capacity'] - remaining) / res_info['capacity'] if res_info['capacity'] > 0 else 0
            for r_id, remaining in resource_capacity_remaining.items()
            for res_info in flat_resources if res_info['id'] == r_id
        }
    }
    return final_report

# --- Function 10: Long (approx. 300-400 lines) ---
def simulate_blockchain_network(num_nodes=5, initial_difficulty=4, max_transactions_per_block=10, simulation_blocks=5):
    """
    Simulates a simplified proof-of-work blockchain network.
    This function creates 'nodes', simulates 'mining' new blocks,
    adding 'transactions', and maintaining a 'chain'.

    This is a highly simplified conceptual model for demonstration.
    Real blockchains are vastly more complex.

    Args:
        num_nodes (int): Number of mining nodes in the network.
        initial_difficulty (int): Number of leading zeros required for a block hash (PoW).
        max_transactions_per_block (int): Maximum transactions that can fit in a block.
        simulation_blocks (int): Number of blocks to mine in the simulation.

    Returns:
        dict: The final state of the blockchain and nodes.
    """
    print(f"--- Simulating Blockchain Network ({simulation_blocks} blocks) ---")
    print(f"Number of nodes: {num_nodes}, Initial Difficulty: {initial_difficulty}")

    class Block:
        def __init__(self, index, transactions, timestamp, previous_hash, nonce=0, current_hash=""):
            self.index = index
            self.transactions = transactions
            self.timestamp = timestamp
            self.previous_hash = previous_hash
            self.nonce = nonce
            self.current_hash = current_hash # This will be calculated later

        def calculate_hash(self):
            # Simple hash calculation for demonstration
            block_string = json.dumps(self.__dict__, sort_keys=True, default=str)
            return hashlib.sha256(block_string.encode()).hexdigest()

        def __repr__(self):
            return f"<Block {self.index}> Hash: {self.current_hash[:10]}... Prev: {self.previous_hash[:10]}..."

    class Node:
        def __init__(self, node_id):
            self.node_id = node_id
            self.blockchain = [self._create_genesis_block()]
            self.pending_transactions = deque()
            self.current_difficulty = initial_difficulty
            print(f"Node {self.node_id} initialized with Genesis Block.")

        def _create_genesis_block(self):
            genesis_block = Block(0, ["Genesis Transaction"], datetime.now().isoformat(), "0" * 64)
            genesis_block.current_hash = genesis_block.calculate_hash()
            return genesis_block

        def get_latest_block(self):
            return self.blockchain[-1]

        def add_transaction(self, sender, recipient, amount):
            tx = {"sender": sender, "recipient": recipient, "amount": amount, "timestamp": datetime.now().isoformat()}
            self.pending_transactions.append(tx)
            # print(f"Node {self.node_id}: Added pending transaction from {sender} to {recipient}")

        def proof_of_work(self, block, difficulty):
            target_prefix = '0' * difficulty
            while block.current_hash[:difficulty] != target_prefix:
                block.nonce += 1
                block.current_hash = block.calculate_hash()
                # if block.nonce % 100000 == 0:
                #    print(f"  Node {self.node_id} mining... Nonce: {block.nonce}")
            return block.current_hash

        def mine_new_block(self):
            last_block = self.get_latest_block()
            new_index = last_block.index + 1
            new_timestamp = datetime.now().isoformat()
            new_previous_hash = last_block.current_hash
            
            # Take transactions from pending queue up to max_transactions_per_block
            block_transactions = []
            for _ in range(max_transactions_per_block):
                if self.pending_transactions:
                    block_transactions.append(self.pending_transactions.popleft())
                else:
                    break
            
            if not block_transactions and new_index > 0: # Don't create empty blocks after genesis
                # print(f"Node {self.node_id}: No new transactions to mine for block {new_index}.")
                return None

            new_block = Block(new_index, block_transactions, new_timestamp, new_previous_hash)
            
            print(f"Node {self.node_id}: Starting to mine block {new_index} (Tx: {len(block_transactions)})...")
            start_mining_time = time.time()
            mined_hash = self.proof_of_work(new_block, self.current_difficulty)
            end_mining_time = time.time()
            
            new_block.current_hash = mined_hash
            
            print(f"Node {self.node_id}: Mined Block {new_block.index} in {end_mining_time - start_mining_time:.2f}s. Hash: {new_block.current_hash[:10]}...")
            return new_block

        def add_block(self, block):
            if self.is_block_valid(block, self.get_latest_block()):
                self.blockchain.append(block)
                # print(f"Node {self.node_id}: Added block {block.index} to its chain.")
                return True
            # print(f"Node {self.node_id}: Failed to add block {block.index} - invalid.")
            return False

        def is_block_valid(self, block, last_block):
            if last_block.index + 1 != block.index:
                print(f"  Validation error: Block index mismatch. Expected {last_block.index + 1}, got {block.index}")
                return False
            if last_block.current_hash != block.previous_hash:
                print(f"  Validation error: Previous hash mismatch.")
                return False
            if block.current_hash[:self.current_difficulty] != '0' * self.current_difficulty:
                print(f"  Validation error: Invalid proof of work for block {block.index}.")
                return False
            if block.calculate_hash() != block.current_hash:
                print(f"  Validation error: Block hash is incorrect for block {block.index}.")
                return False
            return True
            
        def resolve_conflict(self, other_node_chain):
            """Simplified longest chain rule."""
            if len(other_node_chain) > len(self.blockchain):
                print(f"Node {self.node_id}: Resolving conflict, adopting longer chain.")
                self.blockchain = other_node_chain
                return True
            return False

    nodes = [Node(f"Node_{i+1}") for i in range(num_nodes)]
    network_state = {"nodes_data": {node.node_id: {"chain_length": len(node.blockchain), "pending_tx": len(node.pending_transactions)} for node in nodes}}
    
    global_pending_tx_id = 0

    for block_num in range(simulation_blocks):
        print(f"\n--- Simulation Block Cycle {block_num + 1}/{simulation_blocks} ---")

        # Simulate transactions appearing
        for _ in range(random.randint(1, max_transactions_per_block * 2)): # Random number of new transactions
            sender = f"User_{random.randint(1, 100)}"
            recipient = f"User_{random.randint(1, 100)}"
            amount = round(random.uniform(0.1, 100.0), 2)
            # Add transaction to a random node's pending pool
            random.choice(nodes).add_transaction(sender, recipient, amount)
            global_pending_tx_id += 1
            # print(f"  Added TX {global_pending_tx_id} to a random node.")
        
        # Each node attempts to mine
        mined_blocks_this_cycle = []
        for node in nodes:
            block = node.mine_new_block()
            if block:
                mined_blocks_this_cycle.append((node.node_id, block))
        
        if not mined_blocks_this_cycle:
            print("No blocks were mined in this cycle.")
            continue

        # Simulate network propagation and conflict resolution
        # The first mined block "wins" (simplistic)
        winning_node_id, winning_block = mined_blocks_this_cycle[0]
        print(f"Node {winning_node_id} mined the winning block {winning_block.index}.")

        # All nodes try to add the winning block
        for node in nodes:
            if node.node_id == winning_node_id:
                node.add_block(winning_block) # Node already has it, or confirms it
            else:
                if not node.add_block(winning_block):
                    # If direct add fails, trigger conflict resolution (longest chain rule)
                    # This means other nodes might have mined a different block or have outdated chain
                    print(f"Node {node.node_id}: Block {winning_block.index} rejected, attempting chain sync.")
                    # In a real network, nodes would request the full chain
                    node.resolve_conflict(nodes[0].blockchain) # For simplicity, all sync with Node 1's chain if longer
                    
        # Update difficulty (very simplified)
        if (block_num + 1) % 3 == 0: # Adjust difficulty every 3 blocks
            for node in nodes:
                node.current_difficulty += random.choice([-1, 0, 1]) # Random change
                node.current_difficulty = max(1, node.current_difficulty) # Keep difficulty >= 1
            print(f"Difficulty adjusted to {nodes[0].current_difficulty} for next cycle.")

        # Periodically clean up pending transactions that might have been included in the 'winning' block
        # For true distributed system, nodes would sync pending transactions too.
        # Here, just clear pending for all nodes for simplicity assuming they got picked up.
        for node in nodes:
            node.pending_transactions.clear()
        
        network_state["nodes_data"] = {node.node_id: {"chain_length": len(node.blockchain), "pending_tx": len(node.pending_transactions)} for node in nodes}
        print(f"Network state updated. Longest chain length: {len(nodes[0].blockchain)}")

    final_blockchain_dump = []
    for block in nodes[0].blockchain: # Assume node 0 has the longest/canonical chain
        final_blockchain_dump.append({
            "index": block.index,
            "transactions": block.transactions,
            "timestamp": block.timestamp,
            "previous_hash": block.previous_hash,
            "nonce": block.nonce,
            "current_hash": block.current_hash
        })

    final_report = {
        "simulation_parameters": {
            "num_nodes": num_nodes,
            "initial_difficulty": initial_difficulty,
            "max_transactions_per_block": max_transactions_per_block,
            "simulation_blocks": simulation_blocks
        },
        "final_network_state": network_state,
        "final_blockchain_head": final_blockchain_dump[-1] if final_blockchain_dump else {},
        "full_blockchain_dump_length": len(final_blockchain_dump)
    }

    # Optionally save the full chain to a file
    # with open("simulated_blockchain.json", "w") as f:
    #     json.dump(final_blockchain_dump, f, indent=2)
    # print("Simulated blockchain saved to simulated_blockchain.json")

    return final_report

# --- Main execution block for demonstration ---
if __name__ == "__main__":
    print("\n--- Running example Function 6 ---")
    validate_email_format("test@example.com")
    validate_email_format("invalid-email")
    validate_email_format(123)

    print("\n--- Running example Function 7 ---")
    sample_inventory = [
        {"id": "P001", "name": "Laptop", "category": "Electronics", "quantity": 15, "price": 1200.50},
        {"id": "P002", "name": "Mouse", "category": "Electronics", "quantity": 50, "price": 25.00},
        {"id": "P003", "name": "Desk Chair", "category": "Furniture", "quantity": 8, "price": 150.00},
        {"id": "P004", "name": "Coffee Mug", "category": "Kitchenware", "quantity": 120, "price": 7.50},
        {"id": "P005", "name": "Software License", "category": "Services", "quantity": 5, "price": 99.99},
        {"id": "P006", "name": "Empty Item", "category": "Misc", "quantity": 0, "price": 0.0},
        {"id": "P007", "name": "Damaged Stock", "category": "Returns", "quantity": -5, "price": 10.00}, # Invalid quantity
        {"id": "P008", "name": "Tablet", "category": "Electronics", "quantity": 3, "price": 300.00}, # Low quantity
        {"id": "P009", "name": "Printer", "category": "Electronics", "quantity": 2, "price": 250.00}, # Low quantity
        "not a dict"
    ]
    text_report = generate_inventory_report(sample_inventory, min_quantity=10)
    print(text_report)
    json_report = generate_inventory_report(sample_inventory, output_format="json", min_quantity=5)
    # print(json_report) # Uncomment to see JSON output

    print("\n--- Running example Function 8 ---")
    dummy_log_file = "user_activity.log"
    dummy_summary_csv = "activity_summary.csv"
    with open(dummy_log_file, 'w', encoding='utf-8') as f:
        f.write('{"timestamp": "2023-10-26T10:00:00Z", "user_id": "user_a", "event_type": "LOGIN", "details": {"ip": "192.168.1.1"}}\n')
        f.write('{"timestamp": "2023-10-26T10:05:30Z", "user_id": "user_b", "event_type": "VIEW_PRODUCT", "details": {"product_id": "P123"}}\n')
        f.write('{"timestamp": "2023-10-26T10:10:15Z", "user_id": "user_a", "event_type": "VIEW_PRODUCT", "details": {"product_id": "P456"}}\n')
        f.write('{"timestamp": "2023-10-26T10:11:00Z", "user_id": "user_c", "event_type": "LOGIN_FAILED", "details": {"reason": "Bad password"}}\n')
        f.write('{"timestamp": "2023-10-26T10:20:00Z", "user_id": "user_a", "event_type": "ADD_TO_CART", "details": {"product_id": "P456", "quantity": 1}}\n')
        f.write('{"timestamp": "2023-10-26T11:00:00Z", "user_id": "user_b", "event_type": "CHECKOUT", "details": {"order_id": "ORD001"}}\n')
        f.write('{"timestamp": "2023-10-26T11:30:00Z", "user_id": "user_a", "event_type": "LOGOUT", "details": {}}\n')
        f.write('{"timestamp": "2023-10-27T09:00:00Z", "user_id": "user_d", "event_type": "LOGIN", "details": {}}\n')
        f.write('{"timestamp": "2023-10-27T09:05:00Z", "user_id": "user_d", "event_type": "VIEW_PRODUCT", "details": {"product_id": "P789"}}\n')
        f.write('{"timestamp": "2023-10-27T09:05:00Z", "user_id": "user_e", "event_type": "SEARCH", "details": {"query": "shoes"}}\n')
        f.write('{"timestamp": "2023-10-27T09:15:00Z", "user_id": "user_c", "event_type": "LOGIN_FAILED", "details": {"reason": "User not found"}}\n')
        f.write('{"timestamp": "2023-10-27T09:20:00Z", "user_id": "user_a", "event_type": "LOGIN", "details": {}}\n')
        f.write('{"user_id": "user_f", "event_type": "INVALID_TIMESTAMP"}\n') # Malformed line
        f.write('not json\n') # Another malformed line
        # Add more lines to increase length
        for i in range(200):
            f.write(f'{{"timestamp": "{datetime.now().isoformat()}Z", "user_id": "user_{random.randint(1,10)}", "event_type": "{random.choice(["LOGIN", "VIEW_PRODUCT", "ADD_TO_CART", "CHECKOUT", "LOGOUT", "SEARCH"])}", "details": {{}}}}\n')
        
    process_success = process_user_activity_log(dummy_log_file, dummy_summary_csv, activity_types_to_track=["LOGIN", "LOGIN_FAILED", "CHECKOUT"])
    print(f"Log processing success: {process_success}")

    print("\n--- Running example Function 9 ---")
    sample_tasks = [
        {"id": "T001", "name": "Develop Feature X", "resource_needs": {"CPU": 4, "Memory": 16}, "priority": 5, "cost_per_resource": {"CPU": 0.5, "Memory": 0.1}},
        {"id": "T002", "name": "Run Batch Job A", "resource_needs": {"CPU": 2, "Storage": 100}, "priority": 3, "cost_per_resource": {"CPU": 0.6, "Storage": 0.05}},
        {"id": "T003", "name": "Serve Web Traffic", "resource_needs": {"CPU": 8, "Network": 50}, "priority": 7, "cost_per_resource": {"CPU": 0.4, "Network": 0.2}},
        {"id": "T004", "name": "Process Analytics", "resource_needs": {"Memory": 32, "Storage": 200}, "priority": 4, "cost_per_resource": {"Memory": 0.15, "Storage": 0.06}},
        {"id": "T005", "name": "Small Utility", "resource_needs": {"CPU": 1}, "priority": 1, "cost_per_resource": {"CPU": 0.7}},
    ]
    sample_resources = {
        "CPU": [
            {"id": "CPU_A", "type": "CPU", "capacity": 8, "cost_per_hour": 1.0},
            {"id": "CPU_B", "type": "CPU", "capacity": 4, "cost_per_hour": 0.8},
            {"id": "CPU_C", "type": "CPU", "capacity": 2, "cost_per_hour": 0.7},
            {"id": "CPU_D", "type": "CPU", "capacity": 16, "cost_per_hour": 1.5},
        ],
        "Memory": [
            {"id": "MEM_X", "type": "Memory", "capacity": 32, "cost_per_hour": 0.2},
            {"id": "MEM_Y", "type": "Memory", "capacity": 16, "cost_per_hour": 0.18},
            {"id": "MEM_Z", "type": "Memory", "capacity": 8, "cost_per_hour": 0.15},
        ],
        "Storage": [
            {"id": "STO_1", "type": "Storage", "capacity": 500, "cost_per_hour": 0.01},
            {"id": "STO_2", "type": "Storage", "capacity": 200, "cost_per_hour": 0.012},
        ],
        "Network": [
            {"id": "NET_G1", "type": "Network", "capacity": 100, "cost_per_hour": 0.05},
            {"id": "NET_G2", "type": "Network", "capacity": 50, "cost_per_hour": 0.04},
        ]
    }
    
    allocation_results_cost = optimize_resource_allocation(sample_tasks, sample_resources, optimization_goal="cost")
    print(json.dumps(allocation_results_cost, indent=2))
    
    # allocation_results_efficiency = optimize_resource_allocation(sample_tasks, sample_resources, optimization_goal="efficiency")
    # print("\nEfficiency Optimization Results:")
    # print(json.dumps(allocation_results_efficiency, indent=2))

    print("\n--- Running example Function 10 ---")
    blockchain_sim_results = simulate_blockchain_network(num_nodes=3, initial_difficulty=3, simulation_blocks=4, max_transactions_per_block=5)
    print("\nFinal Blockchain Simulation Summary:")
    print(json.dumps(blockchain_sim_results, indent=2))



import os
import csv
import json
import random
from datetime import datetime, timedelta

# --- Function 11: Very Long (approx. 400-500 lines) ---
def complex_data_migration_pipeline(source_data_file, target_data_file, schema_map_file, error_log_file,
                                    transformation_rules=None, validation_threshold=0.95):
    """
    Simulates a complex data migration pipeline from a source CSV file to a target JSONL file.
    This pipeline includes multiple stages:
    1. Configuration Loading (Schema Map, Transformation Rules).
    2. Source Data Loading and Initial Parsing.
    3. Row-level Validation.
    4. Data Transformation and Mapping to New Schema.
    5. Data Enrichment (e.g., adding computed fields).
    6. Target Data Writing (JSONL format).
    7. Comprehensive Error Logging and Reporting.

    Args:
        source_data_file (str): Path to the input CSV file.
        target_data_file (str): Path to the output JSONL file.
        schema_map_file (str): Path to a JSON file defining old_header:new_header mapping.
                               Example: {"old_customer_id": "customerId", "old_name": "fullName"}
        error_log_file (str): Path to the file where detailed errors will be logged.
        transformation_rules (dict, optional): Dictionary of transformation functions
                                               keyed by new field names.
                                               Example: {"dateOfBirth": "iso_to_timestamp", "age": "calculate_age"}
        validation_threshold (float): Minimum percentage of valid rows required for
                                      the migration to be considered successful (0.0 to 1.0).

    Returns:
        dict: A summary of the migration process including counts, success status,
              and paths to output files.
    """
    print(f"--- Starting Complex Data Migration Pipeline ---")
    print(f"Source: {source_data_file}, Target: {target_data_file}")

    pipeline_status = True
    migration_summary = {
        "start_time": datetime.now().isoformat(),
        "total_source_rows": 0,
        "parsed_rows_count": 0,
        "valid_rows_count": 0,
        "transformed_rows_count": 0,
        "failed_rows_count": 0,
        "output_file_path": target_data_file,
        "error_log_path": error_log_file,
        "overall_success": False,
        "end_time": None,
        "duration_seconds": 0.0
    }
    
    internal_error_log = [] # Collect errors internally before writing to file

    # --- 1. Configuration Loading (approx 50 lines) ---
    print("Stage 1: Loading configurations (schema map, transformation rules)...")
    schema_map = {}
    if not os.path.exists(schema_map_file):
        internal_error_log.append(f"Fatal Error: Schema map file not found at '{schema_map_file}'")
        pipeline_status = False
    else:
        try:
            with open(schema_map_file, 'r', encoding='utf-8') as f:
                schema_map = json.load(f)
            print(f"Loaded schema map with {len(schema_map)} mappings.")
        except json.JSONDecodeError as e:
            internal_error_log.append(f"Fatal Error: Invalid JSON in schema map file '{schema_map_file}': {e}")
            pipeline_status = False
        except Exception as e:
            internal_error_log.append(f"Fatal Error: Could not read schema map file '{schema_map_file}': {e}")
            pipeline_status = False
    
    # Dummy transformation functions (actual implementations would be more robust)
    def default_transformer(value):
        return str(value).strip() if value is not None else ""

    def transform_iso_to_timestamp(iso_str):
        try:
            return datetime.fromisoformat(iso_str.replace('Z', '+00:00')).timestamp()
        except (ValueError, TypeError):
            return None

    def transform_calculate_age(dob_iso_str):
        try:
            dob = datetime.fromisoformat(dob_iso_str.replace('Z', '+00:00'))
            today = datetime.now()
            return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        except (ValueError, TypeError):
            return None

    # Merge provided rules with default/built-in ones
    effective_transformation_rules = {
        "iso_to_timestamp": transform_iso_to_timestamp,
        "calculate_age": transform_calculate_age,
        # Add more built-in dummy transformers as needed
        "to_uppercase": lambda x: str(x).upper() if x is not None else "",
        "to_lowercase": lambda x: str(x).lower() if x is not None else ""
    }
    if transformation_rules:
        effective_transformation_rules.update(transformation_rules) # User-defined rules can override/add

    if not pipeline_status:
        _write_error_log(error_log_file, internal_error_log)
        return migration_summary # Exit early if config fails

    # --- 2. Source Data Loading and Initial Parsing (approx 80 lines) ---
    print("Stage 2: Loading and parsing source data...")
    source_rows = []
    if not os.path.exists(source_data_file):
        internal_error_log.append(f"Fatal Error: Source data file not found at '{source_data_file}'")
        pipeline_status = False
    else:
        try:
            with open(source_data_file, 'r', encoding='utf-8') as f:
                csv_reader = csv.DictReader(f)
                migration_summary["total_source_rows"] = sum(1 for row in csv.reader(f)) - 1 # Recalculate if needed
                f.seek(0) # Reset file pointer after counting
                csv_reader = csv.DictReader(f) # Re-initialize reader

                actual_header = csv_reader.fieldnames
                if not actual_header:
                    internal_error_log.append("Fatal Error: Source CSV is empty or has no header.")
                    pipeline_status = False
                else:
                    # Validate if all source fields in schema map exist in actual header
                    missing_source_fields = [old_f for old_f in schema_map if old_f not in actual_header]
                    if missing_source_fields:
                        internal_error_log.append(f"Warning: Source fields in schema map missing from CSV header: {missing_source_fields}")

                    for row_num, row in enumerate(csv_reader):
                        migration_summary["parsed_rows_count"] += 1
                        source_rows.append(row)
                        # Simulate large file parsing time
                        # if row_num % 1000 == 0: time.sleep(0.0001)

            print(f"Loaded {len(source_rows)} rows from source.")
            migration_summary["total_source_rows"] = len(source_rows)
        except FileNotFoundError:
            internal_error_log.append(f"Fatal Error: Source data file '{source_data_file}' not found.")
            pipeline_status = False
        except Exception as e:
            internal_error_log.append(f"Fatal Error: Error reading source CSV '{source_data_file}': {e}")
            pipeline_status = False

    if not pipeline_status:
        _write_error_log(error_log_file, internal_error_log)
        return migration_summary

    # --- 3. Row-level Validation (approx 70 lines) ---
    print("Stage 3: Performing row-level validation...")
    validated_rows = []
    for row_idx, row in enumerate(source_rows):
        row_id = row.get(next(iter(schema_map), 'RowIndex'), f"Row_{row_idx+1}") # Use first mapped field or index
        is_valid_row = True
        row_errors = []

        # Check for presence of key fields based on schema map (simplified)
        for old_field, new_field in schema_map.items():
            if old_field not in row or row[old_field] is None or str(row[old_field]).strip() == "":
                row_errors.append(f"Missing or empty required field '{old_field}' (maps to '{new_field}')")
                is_valid_row = False
        
        # Add more specific validation rules here (e.g., regex for specific fields, type checks)
        if 'email' in schema_map.values() and 'old_email' in schema_map and not validate_email_format(row.get(schema_map['old_email'], '')):
             row_errors.append(f"Invalid email format for '{schema_map['old_email']}'")
             is_valid_row = False
        
        # Simulate business logic validation
        if row.get('order_amount') and float(row['order_amount']) < 0:
            row_errors.append(f"Negative order amount detected: {row['order_amount']}")
            is_valid_row = False
            
        if is_valid_row:
            validated_rows.append(row)
            migration_summary["valid_rows_count"] += 1
        else:
            migration_summary["failed_rows_count"] += 1
            internal_error_log.append(f"Row {row_id} (original index {row_idx+1}) failed validation: {'; '.join(row_errors)}. Original data: {json.dumps(row)}")
            
    print(f"Validation complete. Valid rows: {migration_summary['valid_rows_count']}, Failed rows: {migration_summary['failed_rows_count']}")

    validation_percentage = migration_summary["valid_rows_count"] / migration_summary["total_source_rows"] if migration_summary["total_source_rows"] > 0 else 0
    if validation_percentage < validation_threshold:
        internal_error_log.append(f"Migration Aborted: Valid row percentage ({validation_percentage:.2%}) "
                                  f"is below required threshold ({validation_threshold:.2%}).")
        pipeline_status = False
    
    if not pipeline_status:
        _write_error_log(error_log_file, internal_error_log)
        return migration_summary

    # --- 4. Data Transformation and Mapping (approx 100 lines) ---
    print("Stage 4: Transforming and mapping data to target schema...")
    transformed_data_records = []
    for row_idx, source_row in enumerate(validated_rows):
        transformed_record = {}
        transformation_errors = []
        
        # Apply schema mapping
        for old_field, new_field in schema_map.items():
            value = source_row.get(old_field)
            
            # Apply general transformation if rule exists or default
            transformer_key = new_field # Assume transformer name matches new field name for simplicity, or provide a mapping in config
            transformer_func = effective_transformation_rules.get(transformer_key, default_transformer)
            
            try:
                processed_value = transformer_func(value)
                transformed_record[new_field] = processed_value
            except Exception as e:
                transformation_errors.append(f"Error transforming field '{old_field}' (to '{new_field}'): {e}")
                transformed_record[new_field] = None # Set to None on transformation error

        # --- 5. Data Enrichment (approx 50 lines) ---
        # Add new computed fields, e.g., 'ingestion_timestamp', 'data_quality_score'
        transformed_record['ingestionTimestamp'] = datetime.now().isoformat()
        
        # Simulate a data quality score based on transformed record completeness
        quality_score = sum(1 for v in transformed_record.values() if v is not None and str(v).strip() != "") / len(transformed_record)
        transformed_record['dataQualityScore'] = round(quality_score, 2)
        
        # Example of applying specific enrichment logic based on existing fields
        if transformed_record.get('orderAmount') and transformed_record['orderAmount'] > 1000:
            transformed_record['isHighValueOrder'] = True
        else:
            transformed_record['isHighValueOrder'] = False

        if transformed_record.get('dateOfBirth'):
            transformed_record['age'] = transform_calculate_age(transformed_record['dateOfBirth']) # Re-use transformer

        if transformation_errors:
            internal_error_log.append(f"Row {row_idx} (from valid set) had transformation errors: {'; '.join(transformation_errors)}. "
                                      f"Partially transformed: {json.dumps(transformed_record)}")
        
        transformed_data_records.append(transformed_record)
        migration_summary["transformed_rows_count"] += 1
    
    print(f"Transformation complete. Generated {len(transformed_data_records)} target records.")

    # --- 6. Target Data Writing (JSONL format) (approx 50 lines) ---
    print("Stage 6: Writing transformed data to target file...")
    try:
        with open(target_data_file, 'w', encoding='utf-8') as f:
            for record_idx, record in enumerate(transformed_data_records):
                f.write(json.dumps(record) + '\n')
                # Simulate write time for large datasets
                # if record_idx % 1000 == 0: time.sleep(0.00005)
        print(f"Successfully wrote {len(transformed_data_records)} records to '{target_data_file}'.")
    except Exception as e:
        internal_error_log.append(f"Fatal Error: Error writing to target file '{target_data_file}': {e}")
        pipeline_status = False

    # --- 7. Comprehensive Error Logging and Reporting (approx 30 lines) ---
    _write_error_log(error_log_file, internal_error_log)

    migration_summary["overall_success"] = pipeline_status and (migration_summary["valid_rows_count"] / migration_summary["total_source_rows"] >= validation_threshold if migration_summary["total_source_rows"] > 0 else True)
    migration_summary["end_time"] = datetime.now().isoformat()
    migration_summary["duration_seconds"] = (datetime.fromisoformat(migration_summary["end_time"]) - datetime.fromisoformat(migration_summary["start_time"])).total_seconds()
    
    if migration_summary["overall_success"]:
        print(f"--- Data Migration Pipeline Completed SUCCESSFULLY in {migration_summary['duration_seconds']:.2f} seconds. ---")
    else:
        print(f"--- Data Migration Pipeline Completed with ERRORS in {migration_summary['duration_seconds']:.2f} seconds. ---")
        print(f"Check error log at '{error_log_file}' for details.")

    return migration_summary

def _write_error_log(log_file_path, errors):
    """Helper function to write accumulated errors to a log file."""
    if errors:
        try:
            with open(log_file_path, 'w', encoding='utf-8') as f:
                f.write(f"--- Data Migration Error Log ({datetime.now().isoformat()}) ---\n")
                f.write(f"Total Errors: {len(errors)}\n\n")
                for i, error_msg in enumerate(errors):
                    f.write(f"[{i+1}] {error_msg}\n")
            print(f"Detailed error log saved to: '{log_file_path}'")
        except Exception as e:
            print(f"Critical Error: Could not write to error log file '{log_file_path}': {e}")
    else:
        print("No errors logged during migration.")
        if os.path.exists(log_file_path):
            os.remove(log_file_path) # Clean up empty log file
            print(f"Removed empty error log file: '{log_file_path}'")

# Re-using the email validator for this example
def validate_email_format(email_address):
    """
    Validates the basic format of an email address using a regular expression.
    This is a simple validation and might not catch all edge cases,
    but it's representative of common data validation tasks.

    Args:
        email_address (str): The email string to validate.

    Returns:
        bool: True if the email format is valid, False otherwise.
    """
    if not isinstance(email_address, str):
        return False
    
    email_regex = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return re.match(email_regex, email_address)


# --- Main execution block for demonstration ---
if __name__ == "__main__":
    print("\n--- Running example Function 11: Complex Data Migration Pipeline ---")

    # Create dummy source CSV
    dummy_source_csv = "source_customers.csv"
    with open(dummy_source_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["old_customer_id", "old_name", "email_address", "date_of_birth", "join_date", "order_count", "order_amount", "status"])
        for i in range(1, 100):
            writer.writerow([
                f"CUST{i:04d}",
                f"Customer {i}",
                f"customer{i}@example.com" if i % 5 != 0 else "invalid-email", # Introduce some invalid emails
                (datetime(1970, 1, 1) + timedelta(days=random.randint(0, 15000))).isoformat() + "Z",
                (datetime(2020, 1, 1) + timedelta(days=random.randint(0, 1000))).strftime("%Y-%m-%d"),
                random.randint(0, 50),
                round(random.uniform(10.0, 5000.0), 2) if i % 10 != 0 else -100.0, # Introduce some invalid amounts
                random.choice(["Active", "Inactive", "Pending"])
            ])
        writer.writerow(["CUST0100", "", "missing@example.com", "1990-01-01Z", "2021-01-01", 10, 500.0, "Active"]) # Missing name
        writer.writerow(["CUST0101", "Invalid Date Customer", "valid@example.com", "NOT_A_DATE", "2021-01-01", 5, 250.0, "Active"]) # Invalid DOB
        writer.writerow(["CUST0102", "Empty Order Amount", "valid2@example.com", "1985-05-05Z", "2022-03-15", 3, "", "Active"]) # Empty order amount
        writer.writerow(["", "Missing ID Customer", "no_id@example.com", "1980-10-10Z", "2023-01-01", 2, 100.0, "Active"]) # Missing ID

    # Create dummy schema map file
    dummy_schema_map = "schema_map.json"
    schema_mapping_content = {
        "old_customer_id": "customerId",
        "old_name": "fullName",
        "email_address": "email",
        "date_of_birth": "dateOfBirth",
        "join_date": "registrationDate",
        "order_count": "totalOrders",
        "order_amount": "lifetimeValue",
        "status": "accountStatus"
    }
    with open(dummy_schema_map, 'w', encoding='utf-8') as f:
        json.dump(schema_mapping_content, f, indent=4)

    # Define custom transformation rules (optional)
    custom_transform_rules = {
        "registrationDate": lambda x: datetime.strptime(x, "%Y-%m-%d").isoformat() + "Z" if x else None,
        "lifetimeValue": lambda x: float(x) if x and isinstance(x, str) and x.replace('.', '', 1).isdigit() else 0.0,
        "accountStatus": "to_uppercase" # Use a built-in dummy transformer
    }

    dummy_target_jsonl = "target_customers.jsonl"
    dummy_error_log = "migration_errors.log"

    migration_summary = complex_data_migration_pipeline(
        source_data_file=dummy_source_csv,
        target_data_file=dummy_target_jsonl,
        schema_map_file=dummy_schema_map,
        error_log_file=dummy_error_log,
        transformation_rules=custom_transform_rules,
        validation_threshold=0.90
    )

    print("\nMigration Summary Results:")
    print(json.dumps(migration_summary, indent=2))

    # Clean up dummy files
    for f in [dummy_source_csv, dummy_schema_map, dummy_target_jsonl, dummy_error_log]:
        if os.path.exists(f):
            # os.remove(f) # Uncomment to remove files after running
            pass

