import java.util.HashSet;
import java.util.Set;

public class DSA {
    public static void main(String[] args) {

        int[] A = {1, 2, 3, 4, 5};
        int[] B = {-4, -3, 2, 3, -2};
        // System.out.println(binarySearch(A, 6));
        // System.out.println(closestToZero(B));
        System.out.println(mergeAlternately("abc", "pqrst"));


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
}
