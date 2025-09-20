import java.util.HashMap;

public class dsa {
    public static void main(String[] args) {

        // int[] A = {1, 2, 3, 4, 5};
        // int[] B = {-4, -3, 2, 3, -2};
        // int[] C = {7,1,5,3,6,4};
        // System.out.println(binarySearch(A, 6));
        // System.out.println(closestToZero(B));
        // System.out.println(mergeAlternately("abc", "pqrst"));
        // System.out.println(romanToInt("MCMXCIV"));
        // System.out.println(isSubsequence("abc", "apqcrs"));
        // System.out.println(maxProfit(C));


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

    

}
