from collections import deque
import heapq

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

# def countStudents(students, sandwiches):
#     queue = deque(students)
#     sw = deque(sandwiches)
#     attempts = 0
#     n = len(queue)

#     while queue and attempts < n:
#         if queue[0] == sw[0]:
#             queue.popleft()
#             sw.popleft()
#             attempts = 0
#         else:
#             a = queue.popleft()
#             queue.append(a)
#             attempts += 1

#     return len(queue)

# students = [1,1,1,0,0,1] 
# sandwiches = [1,0,0,0,1,1]

# print(countStudents(students, sandwiches))


# def timeRequiredToBuy(tickets, k):
#     res = 0

#     for i in range(len(tickets)):
#         if i <= k:
#             res += min(tickets[i], tickets[k])
#         else:
#             res +=min(tickets[i], tickets[k] - 1)
    
#     return res

# tickets = [5,1,1,1]
# k = 0
# print(timeRequiredToBuy(tickets, k))

# class MyQueue:

#     def __init__(self):
#         self.s1 = []
#         self.s2 = []

#     def push(self, x: int) -> None:
#         self.s1.append(x)

#     def pop(self) -> int:
#         if not self.s2:
#             while self.s1:
#                 self.s2.append(self.s1.pop())
#         return self.s2.pop()
#     def peek(self) -> int:
#         if not self.s2:
#             while self.s1:
#                 self.s2.append(self.s1.pop())
#         return self.s2[-1]

#     def empty(self) -> bool:
#         return max(len(self.s1), len(self.s2)) == 0
    
# def lastStoneWeight(stones):
#     for i in range(len(stones)):
#         stones[i] = -stones[i]

#     heapq.heapify(stones)

#     while len(stones) > 1:
#         largest = heapq.heappop(stones)
#         next_largest = heapq.heappop(stones)

#         if largest != next_largest:
#             heapq.heappush(stones, largest - next_largest)

#     if len(stones) == 1:
#         return -heapq.heappop(stones)
#     else:
#         return 0
    
# stones = [2,7,4,1,8,1]
# print(lastStoneWeight(stones))

# def kSmallestPairs(nums1, nums2, k):
#     if not nums1 or not nums2 or k == 0:
#         return []
    
#     res = []
#     heap = []

#     for i in range(min(k, len(nums1))):
#         heapq.heappush(heap, (nums1[i] + nums2[0], i, 0))

#     while heap and k > 0:
#         _, i, j = heapq.heappop(heap)
#         res.append([nums1[i], nums2[j]])
#         k -= 1

#         if j + 1 < len(nums2):
#             heapq.heappush(heap, (nums1[i] + nums2[j + 1], i, j + 1))
    
#     return res

# nums1 = [1,7,11] 
# nums2 = [2,4,6] 
# k = 3
# print(kSmallestPairs(nums1, nums2, k))

# def isPossible(target):
#     if len(target) == 1:
#         return target[0] == 1

#     total = sum(target)
#     target = [-x for x in target]
#     heapq.heapify(target)

#     while True:
#         largest = -heapq.heappop(target)
#         rest = total - largest

#         if largest == 1 or rest == 1:
#             return True
#         if rest == 0 or largest <= rest:
#             return False

#         prev = largest % rest
#         if prev == 0:
#             return False

#         heapq.heappush(target, -prev)
#         total = rest + prev

# target = [9,3,5]
# print(isPossible(target))

# def detectCapitalUse(word):
#     return (
#         word.isupper() or word.islower() or (word[0].isupper() and word[1:].islower())
#     )

# word = "GooGle"
# print(detectCapitalUse(word))

def licenseKeyFormatting(s, k):
    s = s.upper()
    s = s.replace("-", "")
    
    res = []
    count = 0
    
    for ch in reversed(s):
        if count == k:
            res.append("-")
            count = 0
        res.append(ch)
        count += 1

    return "".join(reversed(res))

s = "5F3Z-2e-9-w"
k = 4
print(licenseKeyFormatting(s, k))