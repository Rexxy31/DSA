import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class dsa {
    public static void main(String[] args) {

        // int[] A = {1, 2, 3, 4, 5};
        // int[] B = {-4, -3, 2, 3, -2};
        // int[] C = {7,5,3,6,4};
        // int[] D = {1, 2, 3, 5, 7, 8, 9};
        // String[] F = {"flower","flow","flight"};
        // int[] pES = {1, 2, 3, 4};
        // int[][] mI = {{1, 3}, {2, 6}, {8, 10}, {15, 18}};
        // String jewels = "aA";
        // String stones = "aAAbbbb";
        // int[] cD = {1, 2, 3, 4};
        // String ransomeNote = "aab";
        // String magazine = "baa";
        // String s = "anagram";
        // String t = "nagaram";
        // int[] twoSum = {2, 7, 4, 6};
        // int[][] mX = {{1,2,3},{4,5,6},{7,8,9}};
        // int[] mE = {1, 2, 4};
        // String text = "loonbalxballpoon";
        // int[] sS = {-4,-1,0,3,10};
        // char [] rS = {'h', 'e', 'l', 'l', 'o'};
        String pal = "A man, a plan, a canal: Panama";
        // int[][] mX2 = {{1,2,3,4},{5,6,7,8},{9,10,11,12}};
        // System.out.println(binarySearch(A, 6));
        // System.out.println(closestToZero(B));
        // System.out.println(mergeAlternately("abc", "pqrst"));
        // System.out.println(romanToInt("MCMXCIV"));
        // System.out.println(isSubsequence("abc", "apqcrs"));
        // System.out.println(maxProfit(C));
        // System.out.println(longestCommonPrefix(D));
        // System.out.println(summaryRanges(F));
        // System.out.println(productExceptSelf(pES));
        // System.out.println(Arrays.deepToString(mergeIntervals(mI)));
        // System.out.println(numJewelsInStones(jewels, stones));
        // System.out.println(containsDuplicates(cD));
        // System.out.println(canConstruct(ransomeNote, magazine));
        // System.out.println(isAnagram(s, t));
        // System.out.println(Arrays.toString(twoSum(twoSum, 9)));
        // System.out.println((spiralOrder(mX2)));
        // System.out.println(Arrays.deepToString(rotateImage(mX)));
        // System.out.println(majorityElement(mE));
        // System.out.println(maxNumberOfBalloons(text));
        // System.out.println(Arrays.toString(sortedSquares(sS)));
        // System.out.println(Arrays.toString(reverseString(rS)));
        System.out.println(isPalindrome(pal));


    }

    public static int binarySearch(int[] arr, int target){
        int L = 0;
        int R = arr.length - 1;

        while (L <= R){
            int mid = L + (R - L) / 2;

            if (arr[mid] == target) {
                return mid;
            } else if (arr[mid] < target) { 
                    L = mid + 1; 
            } else {
                R = mid - 1;
            }
        }

        return - 1;
    }


    public static int closestToZero(int[] arr) {
        int closest = arr[0];

        for (int x : arr) {
            if (Math.abs(x) < Math.abs(closest)) {
                closest = x;
            }
        }

            if (closest < 0 && contains(arr, Math.abs(closest))) {
                return Math.abs(closest);
            } else {
                return closest;
            }

    }

    private static boolean contains(int[] nums, int value) {
        for (int num : nums) {
            if (num == value) {
                return true;
            }
        }
        return false;
    }

    public static String mergeAlternately(String word1, String word2) {
        int A = word1.length();
        int B = word2.length();

        int a = 0;
        int b = 0;

        StringBuilder s = new StringBuilder();

        int word = 1;

        while (a < A && b < B) {
            if (word == 1) {
                s.append(word1.charAt(a++));
                word = 2;
            } else {
                s.append(word2.charAt(b++));
                word = 1;
            }
        }

        while (a < A) {
            s.append(word1.charAt(a++));
        }

        while (b < B) {
            s.append(word2.charAt(b++));
        }

        return s.toString();

    }

    public static int romanToInt(String s) {

        HashMap<Character, Integer> d = new HashMap<>();
        d.put('I', 1);
        d.put('V', 5);
        d.put('X', 10);
        d.put('L', 50);
        d.put('C', 100);
        d.put('D', 500);
        d.put('M', 1000);

        int summ = 0;
        int n = s.length();
        int i = 0;

        while (i < n) {
            if (i < n - 1 && d.get(s.charAt(i)) < d.get(s.charAt(i+1)) ) {
                summ += d.get(s.charAt(i+1)) - d.get(s.charAt(i));
                i += 2;
            } else {
                summ += d.get(s.charAt(i));
                i += 1;
            }
        }
        return summ;
    }


    public static boolean isSubsequence(String s, String t) {
        int S = s.length();
        int T = t.length();

        int sp = 0;
        int tp = 0;

        while (sp < S && tp < T) {
            if (s.charAt(sp) == t.charAt(tp)) {
                sp++;
            }
            tp++;
        }

        return sp == S;
    }

    public static int maxProfit(int[] prices) {
        float min_price = Integer.MAX_VALUE;
        int max_profit = 0;

        for (int price : prices) {
            if (price < min_price) {
                min_price = price;
            }

            int profit = (int) (price - min_price);

            if (profit > max_profit) {
                max_profit = profit;
            }
        }

        return max_profit;
    }

    public static String longestCommonPrefix(String[] strs) {

        if (strs == null || strs.length == 0) {
            return "";
        }
        int min_length = Integer.MAX_VALUE;

        for (String s : strs) {
            if (s.length() < min_length) {
                min_length = s.length();
            }
        }

        int i = 0;

        while (i < min_length) {
            for (String s : strs) {
                if (s.charAt(i) != strs[0].charAt(i)) {
                    return strs[0].substring(0, i);
                }
            }

            i += 1;
        }

        return strs[0].substring(0,i);
    }

    public static String summaryRanges(int[] nums) {
        List<String> ans = new ArrayList<>();
        int i = 0;

        while (i < nums.length) {
            int start = nums[i];

            while (i < nums.length - 1 && nums[i] + 1 == nums[i+1]) {
                i += 1;
            }

            if (start != nums[i]) {
                ans.add(start + " -> " + nums[i]);
            } else {
                ans.add(String.valueOf(nums[i]));
            }

            i += 1;
        }

        return ans.toString();
    }

    public static String productExceptSelf(int[] nums) {
        int l_mult = 1;
        int r_mult = 1;
        int n = nums.length;

        int[] result = new int[n];

        for ( int i = 0; i < n; i++) {
            result[i] = l_mult;
            l_mult *= nums[i];        
        }

        for (int i = n - 1; i >= 0; i--) {
            result[i] *= r_mult;
            r_mult *= nums[i];
        }

        return Arrays.toString(result);
    }

    public static int[][] mergeIntervals(int[][] intervals) {
        if (intervals.length == 0) {
            return new int[0][0];
        }

        Arrays.sort(intervals, (a, b) -> Integer.compare(a[0], b[0]));

        List<int[]> merged = new ArrayList<>();

        for (int[] interval : intervals) {
            if (merged.isEmpty() || merged.get(merged.size() - 1)[1] < interval[0]) {
                merged.add(interval);
            } else {
                merged.get(merged.size() - 1)[1] = Math.max(merged.get(merged.size() - 1)[1], interval[1]);
            }
        }

        return merged.toArray(new int[merged.size()][]);
    }

    public static int numJewelsInStones(String jewels, String stones) {
        Set<Character> jSet = new HashSet<>();
        int count = 0;

        for (char j : jewels.toCharArray()) {
            jSet.add(j);
        }

        for (char s : stones.toCharArray()){
            if (jSet.contains(s)) {
                count += 1;
            }
        }

        return count;
    }

    public static boolean containsDuplicates(int[] nums) {
        Set<Integer> s = new HashSet<>();

        for ( int num : nums) {
            if (s.contains(num)){
                return true;
            } else {
                s.add(num);
            }
        }
        return false;
    }

    public static boolean canConstruct(String ransomeNote, String magazine) {
        HashMap<Character, Integer> map = new HashMap<>();

        for (char c : magazine.toCharArray()) {
            map.put(c, map.getOrDefault(c, 0) + 1);
        }

        for (char c : ransomeNote.toCharArray()) {
            if (map.getOrDefault(c, 0) > 0) {
                map.put(c, map.get(c) - 1);
            } else {
                return false;
            }
        }

        return true;
    }

    public static boolean isAnagram(String s, String t) {
        if (s.length() != t.length()) {
            return false;
        }

        HashMap<Character, Integer> map = new HashMap<>();
        // HashMap<Character, Integer> mmap = new HashMap<>();

        for (char c : s.toCharArray()) {
            map.put(c, map.getOrDefault(c, 0) + 1);
        }

        for (char c : t.toCharArray()) {
            map.put(c, map.getOrDefault(c, 0) - 1);
        }

        for (int val : map.values()) {
            if (val != 0) {
                return false;
            }
        }

        return true;
    }

    public static int[] twoSum(int[] nums, int target) {
        HashMap<Integer, Integer> map = new HashMap<>();

        for (int i = 0; i < nums.length; i++) {
            map.put(nums[i], i);
        }

        for (int i = 0; i < nums.length; i++) {
            int y = target - nums[i];
            if (map.containsKey(y) && map.get(y) != i) {
                return new int[] {i, map.get(y)};
            }
        }

        throw new IllegalArgumentException("No twosum solution");
    }

    public static List<Integer> spiralOrder(int[][] matrix) {
        List<Integer> ans = new ArrayList<>();

        if (matrix.length == 0) {
            return ans;
        }

        int top = 0;
        int bottom = matrix.length - 1;
        int left = 0;
        int right = matrix[0].length - 1;

        while (top <= bottom && left <= right) {
            for (int j = left; j <= right; j++) {
                ans.add(matrix[top][j]);
            }
            top++;

            for (int i = top; i <= bottom; i++) {
                ans.add(matrix[i][right]);
            }
            right--;

            if (top <= bottom){
                for (int j = right; j >= left; j--) {
                    ans.add(matrix[bottom][j]);
                }
                bottom--;
            }

            if (left <= right) {
                for (int i = bottom; i >= top; i--) {
                    ans.add(matrix[i][left]);
                }
                left++;
            }
        }
        return ans;
    }

    public static int[][] rotateImage(int[][] matrix) {
        int n = matrix.length;

        for (int i = 0; i < n ; i++) {
            for (int j = i + 1; j < n; j++) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[j][i];
                matrix[j][i] = temp;
            }
        }

            for (int i = 0; i < n; i++) {
                int left = 0;
                int right = n - 1;

                while (left < right) {
                    int temp = matrix[i][left];
                    matrix[i][left] = matrix[i][right];
                    matrix[i][right] = temp;
                    left++;
                    right--;
                }
            }
        return matrix;
    }

    public static Integer majorityElement(int[] nums) {
        int candidate = nums[0];
        int count = 0;

        for (int num : nums) {
            if (count == 0) {
                candidate = num;
            }
            if (candidate == num) {
                count += 1;
            } else {
                count -= 1;
            }
        }
        
        count = 0;
        for (int num : nums) {
            if (num == candidate) {
                count++;
            }
        }

        if (count > nums.length / 2) {
            return candidate;
        } else {
            return null;
        }
    }

    public static int maxNumberOfBalloons(String text) {
        HashMap<Character, Integer> map = new HashMap<>();
        String target = "balloon";

        for (char c : text.toCharArray()) {
            if (target.indexOf(c) != -1) {
                map.put(c, map.getOrDefault(c, 0) + 1);
            }
        }

        if(!map.containsKey('b') ||
        !map.containsKey('a') ||
        !map.containsKey('l') ||
        !map.containsKey('o') ||
        !map.containsKey('n')) {
            return 0;
        } else {
            return Math.min(Math.min(map.get('b'), map.get('a')),
                Math.min(map.get('l') / 2, 
                    Math.min(map.get('o') / 2, map.get('n'))));
        }
    }

    public static int[] sortedSquares(int[] nums) {
        int n = nums.length;
        int left = 0;
        int right = n - 1;
        int pos = n - 1;
        int[] ans = new int[n];

        while (left <= right) {
            if (Math.abs(nums[left]) > Math.abs(nums[right])) {
                ans[pos] = (int) Math.pow(nums[left], 2);
                left++;
            } else {
                ans[pos] = (int) Math.pow(nums[right], 2);
                right--;
            }
            pos--;
        }
        return ans;
    }

    public static char[] reverseString(char[] s) {
        int n = s.length;
        int l = 0;
        int r = n -1;

        while (l < r) {
           char temp = s[l];
           s[l] = s[r];
           s[r] = temp;
           l++;
           r--;
        }
        return s;
    }

    public static boolean isPalindrome(String s) {
        int n = s.length();
        int L = 0;
        int R = n - 1;

        while (L < R) {
            while (L < R && !Character.isLetterOrDigit(s.charAt(L))) {
            L++;
        }
        while (L < R && !Character.isLetterOrDigit(s.charAt(R))) {
            R--;
        }
        if (L < R && Character.toLowerCase(s.charAt(L)) != Character.toLowerCase(s.charAt(R))) {
            return false;
        }
        L++;
        R--;
        }
        return true;
    }
}
