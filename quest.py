from collections import deque

# def closestToZero(arr):
#     closest = arr[0]

#     for x in arr:
#         if abs(x) < abs(closest):
#             closest = x
        
#     if closest < 0 and abs(closest) in arr:
#         return abs(closest)
#     else:
#         return closest

# arr1 = [-2, 3]
# print(closestToZero(arr1))


# def mergeStringsAlt(word1, word2):
#     A = len(word1)
#     B = len(word2)
#     a, b = 0, 0
#     s = []

#     word = 1

#     while a < A and b < B:
#         if word == 1:
#             s.append(word1[a])
#             a += 1
#             word = 2
#         else:
#             s.append(word2[b])
#             b += 1
#             word = 1
    
#     while a < A:
#         s.append(word1[a])
#         a += 1

#     while b < B:
#         s.append(word2[b])
#         b += 1

#     return ''.join(s)

# word1 = "abc"
# word2 = "pqrs"

# print(mergeStringsAlt(word1, word2))


# def romanToInt(str):
#     d = {'I' : 1, 'V' : 5, 'X' : 10, 'L' : 50, 'C' : 100, 'D' : 500, 'M' : 1000}

# def getConcatenation(nums):
#     return nums * 2

# nums = [1, 2, 3]

# print(getConcatenation(nums))

# def shuffle(nums, n):
#     N = len(nums)
#     a = 0
#     b = n
#     s = []
 
#     place = 1
    
#     while a < N and b < N:
#         if place == 1:
#             s.append(nums[a])
#             a += 1
#             place = 2

#         else:
#             s.append(nums[b])
#             b += 1
#             place = 1
    
#     return s

# nums = [2,5,1,3,4,7]
# n = 3

# print(shuffle(nums, n))

# def findMaxConsecutiveOnes(nums):
#     n = len(nums)
#     a = 0
#     maxCount = 0
#     count = 0

#     while a < n:
#         if nums[a] == 1:
#             count += 1
#             a += 1
#             if count > maxCount:
#                 maxCount = count

#         else:
#             count = 0
#             a += 1

#     return maxCount


# nums = [1,0,1,1,0,1]
# print(findMaxConsecutiveOnes(nums)) 

# def findErrorNums(nums):
#     N = len(nums)
#     s = set()
#     dup = []
#     i = 1
#     for n in nums:
#         if n not in s:
#            s.add(n)
#         else:
#            dup.append(n)
        
#     while i <= N:
#         if i in s:
#             i += 1
#             continue
            
#         else:
#             dup.append(i)
#             i += 1
#     return dup

# nums = [2, 2]
# print(findErrorNums(nums))

# def smallerNumbersThanCurrent(nums):
#     n = len(nums)
#     T = []
#     a, b = 0, 0
#     count = 0

#     while a <= n - 1:
#         while b <= n - 1:
#             if nums[a] > nums[b]:
#                 count += 1
#                 b += 1
#             else:
#                 b += 1
                
#         T.append(count)
#         a += 1
#         b = 0
#         count = 0
#     return T



# nums = [8,1,2,2,3]


# def smallerNumbersThanCurrent(nums):
#     sorted_nums = sorted(nums)

#     rank = {}

#     for i, val in enumerate(sorted_nums):
#         if val not in rank:
#             rank[val] = i

#     return [rank[val] for val in nums]

# print(smallerNumbersThanCurrent(nums))

# def findDisappearedNumbers(nums):
#     N = len(nums)
#     S = set(nums)
#     T = []

#     for n in range(1, N + 1):
#         if n not in S:
#             T.append(n)
    
#     return T

# nums = [4,3,2,7,8,2,3,1]
# print(findDisappearedNumbers(nums))

# def buildArray(target, n):
#         T = []
#         i = 0

#         for s in range(1, n + 1):
#             if i == len(target):
#                 break
            
#             T.append("Push")
#             if s == target[i]:
#                 i += 1
#             else:
#                 T.append("Pop")
        
#         return T

# target = [1,3]
# n = 4
# print(buildArray(target, n))

# def evalRPN(tokens):
#     stack = []
#     op = {'+', '-', '/', '*'}

#     for t in tokens:
#         if t not in op:
#             stack.append(int(t))
#         else:
#             b = stack.pop()
#             a = stack.pop()

#             if t == '+':
#                 res = a + b
#             elif t == '-':
#                 res = a - b
#             elif t == '*':
#                 res = a * b
#             else:
#                 res = int(a / b)

#             stack.append(res)

#     return stack[-1]

# tokens = ["2","1","+","3","*"]
# print(evalRPN(tokens))


# def exclusiveTime(n, logs):
#     L = len(logs)
#     stack = []
#     T = []
#     i = 0

#     for log in logs:
#         parts = log.split(':')
#         stack.append(parts)

#     for i in range(L):
#         T.append(int(stack[i][-3]))
#         if i != int(stack[0][-1]):
#             T.append(int(stack[0][-3]))
            

    
#     return stack[0][-1]

# logs = ["0:start:0","1:start:2","1:end:5","0:end:6"]
# n = 2

# print(exclusiveTime(n, logs))



# def finalPrices(prices):
#     n = len(prices)
#     stack = []
#     result = []

#     for price in prices:
#         result.append(price)

#     for i in range(n):
#         while stack and prices[stack[-1]] >= prices[i]:
#             idx = stack.pop()
#             result[idx] = prices[idx] - prices[i]

#         stack.append(i)
    
#     return result

# prices = [8,4,6,2,3]
# print(finalPrices(prices))

# def dailyTemperatures(temperatures):
#     n = len(temperatures)
#     result = [0] * n
#     stack = []

#     for i in range(n):
#         while stack and temperatures[i] > temperatures[stack[-1]]:
#             idx = stack.pop()
#             result[idx] = i - idx
        
#         stack.append(i)

#     return result



# temperatures = [73,74,75,71,69,72,76,73]
# print(dailyTemperatures(temperatures))

# def largestRectangleArea(heights):
#     n = len(heights)
#     stack = []
#     max_area = 0

#     for i, height in enumerate(heights):
#         start = i
#         while stack and stack[-1][0] > height:
#             h, j = stack.pop()
#             w = i - j
#             a = h * w
#             max_area = max(max_area, a)
#             start = j
#         stack.append((height, start))

#     while stack:
#         h, j = stack.pop()
#         w = n - j
#         a = h * w
#         max_area = max(max_area, a)

#     return max_area


# heights = [2,1,5,6,2,3]
# print(largestRectangleArea(heights))

def countStudents(students, sandwiches):
    queue = deque(students)
    sw = deque(sandwiches)
    n = len(queue)
    attempts = 0

    while queue and attempts < n:
        if queue[0] == sw[0]:
            queue.popleft()
            sw.popleft()
            attempts = 0
        else:
            a = queue.popleft()
            queue.append(a)
            attempts += 1
        
    return len(queue)

