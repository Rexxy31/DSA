import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

public class dsa {
    public static void main(String[] args) {

        // int[] A = {1, 2, 3, 4, 5};
        // int[] B = {-4, -3, 2, 3, -2};
        // int[] C = {7,5,3,6,4};
        // int[] D = {1, 2, 3, 5, 7, 8, 9};
        // String[] F = {"flower","flow","flight"};
        // int[] pES = {1, 2, 3, 4};
        int[][] mI = {{1, 3}, {2, 6}, {8, 10}, {15, 18}};
        // System.out.println(binarySearch(A, 6));
        // System.out.println(closestToZero(B));
        // System.out.println(mergeAlternately("abc", "pqrst"));
        // System.out.println(romanToInt("MCMXCIV"));
        // System.out.println(isSubsequence("abc", "apqcrs"));
        // System.out.println(maxProfit(C));
        // System.out.println(longestCommonPrefix(D));
        // System.out.println(summaryRanges(F));
        // System.out.println(productExceptSelf(pES));
        System.out.println(Arrays.deepToString(mergeIntervals(mI)));


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

}
