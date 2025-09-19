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

def maxProfit(prices):
    min_price = float('inf')
    max_profit = 0

    for price in prices:
        if price < min_price:
            min_price = price

        profit = price - min_price

        if profit > max_profit:
            max_profit = profit
    
    return max_profit

C = [7,1,5,3,6,4]

print(maxProfit(C))