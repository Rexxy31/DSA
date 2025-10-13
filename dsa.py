# def binarySearch(arr, target):
#     L = 0
#     R = len(arr) - 1

#     while L <= R:
#         mid = L + (R - L) // 2
        
#         if arr[mid] == target:
#             return mid
        
#         if arr[mid] < target:
#             L = mid + 1
#         else:
#             R = mid - 1
        
#     return -1


# A = [1, 2, 3, 4, 5]

# print(binarySearch(A, 1))

# def closestToZero(arr):
#     closest = arr[0]

#     for x in arr:
#         if abs(x) < abs(closest):
#             closest = x

#     if closest < 0 and abs(closest) in arr:
#         return abs(closest)
#     else:
#         return closest
        
# B = [-1, 2, 3, -4, -2]

# print(closestToZero(B))


# def mergeAlternately(word1, word2):
#     A = len(word1)
#     B = len(word2)

#     a, b = 0,0
#     s = []

#     word = 1

#     while a < A  and b < B:
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


# print(mergeAlternately('abcde', 'pqr'))

# def romanToInt(s):
#     d = {'I': 1, 'V' : 5, 'X' : 10, 'L' : 50, 'C' : 100, 'D' : 500, 'M' : 1000}

#     summ = 0
#     n = len(s)
#     i = 0

#     while i < n:
#         if i < n - 1 and d[s[i]] < d[s[i+1]]:
#             summ += d[s[i+1]] - d[s[i]]
#             i += 2
#         else:
#             summ += d[s[i]]
#             i += 1
    
#     return summ

# print(romanToInt('MCMXCIV'))


# def isSubsequence(s, t):
#     S = len(s)
#     T = len(t)

#     if s == '':
#         return True
    
#     if S > T:
#         return False
    
#     i = 0

#     for j in range(T):
#         if s[i] == t[j]:
#             if i == S - 1:
#                 return True
            
#             i += 1
        
#     return False

# print(isSubsequence('abc', 'apbqcr'))

# def maxProfit(prices):
#     min_price = float('inf')
#     max_profit = 0

#     for price in prices:
#         if price < min_price:
#             min_price = price

#         profit = price - min_price

#         if profit > max_profit:
#             max_profit = profit
    
#     return max_profit

# C = [7,1,5,3,6,4]

# print(maxProfit(C))

# def longestCommonPrefix(strs):
#     min_length = float('inf')

#     for s in strs:
#         if len(s) < min_length:
#             min_length = len(s)

#     i = 0
#     while i < min_length:
#         for s in strs:
#             if s[i] != strs[0][i]:
#                 return s[:i]
            
#         i += 1

#     return s[:i]

# D = ["flower","flow","flight"]
# print(longestCommonPrefix(D))


# def summaryRanges(nums):
#     ans = []
#     i = 0

#     while i < len(nums):
#         start = nums[i]

#         while i < len(nums) - 1 and nums[i] + 1 == nums[i+1]:
#             i += 1
        
#         if start != nums[i]:
#             ans.append(str(start) + " -> " + str(nums[i]))
#         else:
#             ans.append(str(nums[i]))

#         i += 1
    
#     return ans

# E = [1, 2, 3, 5, 7, 8, 9]

# print(summaryRanges(E))

# def productExceptSelf(nums):
#     l_mult = 1
#     r_mult = 1

#     n = len(nums)

#     l_arr = [0] * n
#     r_arr = [0] * n

#     for i in range(n):
#         j = -i -1
#         l_arr[i] = l_mult
#         r_arr[j] = r_mult
#         l_mult *= nums[i]
#         r_mult *= nums[j]

#     return [l*r for l, r in zip(l_arr, r_arr)]


# F = [1, 2, 3, 4]

# print(productExceptSelf(F))

# E = [[1,3],[2,6],[8,10],[15,18]]
# [[4,7],[1,4]] -> [1, 4], [4,7]


# def mergeIntervals(intervals):
#     intervals.sort(key=lambda interval: interval[0])
#     merged = []

#     for interval in intervals:
#         if not merged or merged[-1][1] < interval[0]:                                         #not merged          4<4 false
#             merged.append(interval)                                                           #[1,4]  
#         else:
#             merged[-1] = [merged[-1][0], max(merged[-1][1], interval[1])]                                           #[1,4] = [1, max(4, 7) ] -> [1, 7]                                          
    
#     return merged



# # print(mergeIntervals(E))

# def numJewelsInStones(jewels, stones):
#     s = set(jewels)
#     count = 0

#     for stone in stones:
#         if stone in s:
#             count += 1

#     return count


# print(numJewelsInStones('aA', 'aAAbbbb'))

# def containsDuplicate(nums):
#     s = set()

#     for num in nums:
#         if num in s:
#             return True
#         else:
#             s.add(num)
    
#     return False

#or one line->  return len(nums) != len(set(nums))


# G = [1, 2, 3, 2]

# print(containsDuplicate(G))

from collections import Counter, defaultdict

# def canConstruct(ransomNote, magazine):
#     ransom_counter = Counter(ransomNote)
#     magazine_counter = Counter(magazine)

#     return all(ransom_counter[c] <= magazine_counter[c] for c in ransom_counter)
    
# print(canConstruct('aab', 'baa'))


# def isAnagram(s, t):
#     if len(s) != len(t):
#         return False

#     s_counter = Counter(s)
#     t_counter = Counter(t)

#     return s_counter == t_counter
    

# print(isAnagram('anagram', 'bagaram'))


# def twoSum(nums, target):
#     seen = {}

#     for i, num in enumerate(nums):
#         comp = target - num
#         if comp in seen:
#             return [seen[comp], i]
#         else:
#             seen[num] = i

# nums = [2, 7, 4, 6]

# print(twoSum(nums, 9))

# def spiralOrder(matrix):
#     m = len(matrix)
#     n = len(matrix[0])
#     ans = []
#     i = 0
#     j = 0                                                                                   # 1 2 3   
#     UP = 0                                                                                  # 4 5 6
#     RIGHT = 1                                                                               # 7 8 9
#     DOWN = 2
#     LEFT = 3
#     direction = RIGHT

#     UP_WALL = 0
#     RIGHT_WALL = n
#     DOWN_WALL = m
#     LEFT_WALL = -1

#     while len(ans) != m*n:
#         if direction == RIGHT:
#             while j < RIGHT_WALL:
#                 ans.append(matrix[i][j])
#                 j += 1
#             i, j = i+1, j-1
#             RIGHT_WALL -=1
#             direction = DOWN
#         elif direction == DOWN:
#             while i < DOWN_WALL:
#                 ans.append(matrix[i][j])
#                 i += 1
#             i, j = i-1, j-1
#             DOWN_WALL -= 1
#             direction = LEFT
#         elif direction == LEFT:
#             while j > LEFT_WALL:
#                 ans.append(matrix[i][j])
#                 j -= 1
#             i, j = i-1, j+1
#             LEFT_WALL += 1
#             direction = UP
#         else:
#             while i > UP_WALL:
#                 ans.append(matrix[i][j])
#                 i -= 1
#             i, j = i+1, j+1
#             UP_WALL += 1
#             direction = RIGHT

#     return ans

# mx = [[1,2,3],[4,5,6],[7,8,9]]

# print(spiralOrder(mx))

# def rotateImage(matrix):
#     n = len(matrix)

#     for i in range(n):
#         for j in range(i+1, n):
#             matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    
#     for i in range(n):
#         for j in range(n // 2):
#             matrix[i][j], matrix[i][n-j-1] = matrix[i][n-j-1], matrix[i][j]

#     return matrix

# rI = [[1,2,3],[4,5,6],[7,8,9]]
# print(rotateImage(rI))

# def majorityElement(nums):
#     # n = len(nums)

#     # c_element = Counter(nums)

#     # for i in c_element:
#     #     if c_element[i] > n // 2:
#     #         return i

#     candidate = None
#     count = 0

#     for num in nums:
#         if count == 0:
#             candidate = num
#         if candidate == num:
#             count += 1
#         else:
#             count -= 1
            
#     return candidate
        
    

# mE = [1, 2, 3]
# print(majorityElement(mE))

# def maxNumberOfBalloons(text):
#     counter = defaultdict(int)
#     target = "ballon"

#     for c in text:
#         if c in target:
#             counter[c] += 1

#     if any(c not in counter for c in target):
#         return 0
#     else:
#         return min(counter["b"], counter["a"], counter["l"] // 2, counter["o"] // 2, counter["n"])
    

# text = "loonbalxballpoon"

# print(maxNumberOfBalloons(text))

# def sortedSquares(nums):
#     n = len(nums)
#     ans = [0] * n
#     left, right = 0, n - 1
#     pos = n - 1

#     while left <= right:
#         if abs(nums[left]) > abs(nums[right]):
#             ans[pos] = nums[left] * nums[left]
#             left += 1
#         else:
#             ans[pos] = nums[right] * nums[right]
#             right -= 1
#         pos -= 1

#     return ans

# nums = [-4,-1,0,3,10]

# print(sortedSquares(nums))

# def reverseString(s):
#     n = len(s)
#     l, r = 0, n - 1

#     while l < r:
#         s[l], s[r] = s[r], s[l]
#         l +=1
#         r -= 1
#     return s

# rS = ["h","e","l","l","o"]
# print(reverseString(rS))

# def isPalindrome(s):
#     n = len(s)
#     L = 0
#     R = n - 1

#     while L < R:
#         if not s[L].isalnum():
#             L += 1
#             continue

#         if not s[R].isalnum():
#             R -= 1
#             continue

#         if s[L].lower() != s[R].lower():
#             return False
        
#         L += 1
#         R -= 1
#     return True

# Pal = "race  car"

# print(isPalindrome(Pal))

# def twoSum2(nums, target):
#     n = len(nums)
#     l = 0
#     r = n - 1

#     while l < r:
#         summ = nums[l] + nums[r]
#         if summ == target:
#             return [l+1, r+1]
#         elif summ < target:
#             l += 1
#         else:
#             r -= 1

# ts2 = [2,11,15, 7]
# print(twoSum2(ts2, 9))


# def calPoints(operations):
#     stk = []

#     for op in operations:
#         if op == "+":
#             stk.append(stk[-1] + stk[-2])
#         elif op == "D":
#             stk.append(stk[-1] * 2)
#         elif op == "C":
#             stk.pop()
#         else:
#             stk.append(int(op))
    
#     return sum(stk)

# ops = ["5","2","C","D","+"]
# print(calPoints(ops))

# def validParenthese(s):
#     hashmap = {")": "(", "}": "{", "]": "["}
#     stk = []

#     for c in s:
#         if c not in hashmap:
#             stk.append(c)
#         else:
#             if not stk:
#                 return False
#             else:
#                 popped = stk.pop()
#                 if popped != hashmap[c]:
#                     return False
    
#     return not stk

# vP = '([])'
# print(validParenthese(vP))

# def dailyTemperatures(temperatures):
#     temps = temperatures
#     n = len(temps)
#     answer = [0] * n
#     stk = []


#     for i, t in enumerate(temps):
#         while stk and stk[-1][0] < t:
#             stk_t, stk_i = stk.pop()
#             answer[stk_i] = i - stk_i

#         stk.append((t, i))

#     return answer

# dT = [73,74,75,71,69,72,76,73]

# print(dailyTemperatures(dT))

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


# def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
#     cur = head

#     while cur and cur.next:
#         if cur.val == cur.next.val:
#             cur.next = cur.next.next
#         else:
#             cur = cur.next
    
#     return head


# def searchInsert(nums, target):
#     L, R = 0, len(nums) - 1

#     while L <= R:
#         mid = L + (R - L) // 2
#         if target == nums[mid]:
#             return mid
#         elif target > nums[mid]:
#             L = mid + 1
#         else:
#             R = mid - 1
        
#     if nums[mid] < target:
#         return mid + 1
#     else:
#         return mid

# sI = [1,3,5,6]
# print(searchInsert(sI, 0))


# Example implementation of isBadVersion for testing purposes
# def isBadVersion(version):
#     # Replace 4 with the first bad version for your test case
#     first_bad = 5
#     return version >= first_bad

# def firstBadVersion(n):
#     L = 1
#     R = n

#     while L < R:
#         M = L + (R - L) // 2
#         if isBadVersion(M):
#             R = M
#         else:
#             L = M + 1

#     return L


# print(firstBadVersion(4))


# def isPerfectSquare(n):
#     if n == 1:
#         return True
    
#     L = 1
#     R = n - 1

#     while L <= R:
#         M = (L + R) // 2
#         r = M * M
#         if n == (r):
#             return True
#         elif n > (r):
#             L = M + 1
#         else:
#             R = M - 1
#     return False

# print(isPerfectSquare(16))


# def reverseList(head):
#     cur = head
#     prev = None


#     while cur:
#         temp = cur.next
#         cur.next = prev
#         prev = cur
#         cur = temp

#     return prev


def mergeTwoLists(list1, list2):
    d = ListNode()
    cur = d

    while list1 and list2:
        if list1.val < list2.val:
            cur.next = list1
            cur = list1
            list1 = list1.next
        else:
            cur.next = list2
            cur = list2
            list2 = list2.next

    cur.next = list1 if list1 else list2

    return d.next


def hasCycle(self, head: Optional[ListNode]) -> bool:
    slow = fast = head

    while fast and fast.next:
        fast = fast.next.next
        slow = slow.next

        if fast == slow:
            return True
    
    return False


def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:
    slow = fast = head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next

    return slow