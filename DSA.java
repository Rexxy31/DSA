public class DSA {
    public void main(String[] args) {

        int[] A = {1, 2, 3, 4, 5};


        System.out.println(binarySearch(A, 5));

    }

    public int binarySearch(int[] arr, int target){
        int L = 0;
        int R = arr.length - 1;

        while (L <= R){
            int mid = L + (R - L) / 2;

            if (arr[mid] == target){
                return mid;
            } else if (arr[mid] < target){
                    L = mid + 1; 
            } else {
                R = mid - 1;
            }
        }

        return - 1;
    }
}
