import java.util.*;

public class quest {
    public static void main(String[] args) {
        int[] students = {1,1,1,0,0,1};
        int[] sandwiches = {1,0,0,0,1,1};
        int[] tickets = {2, 3, 2};
        int k = 2;
        // System.out.println(countStudents(students, sandwiches));
        System.out.println(timeRequiredToBuy(tickets, k));
    }

    private static int countStudents(int[] students, int[] sandwiches) {
        Deque<Integer> queue = new ArrayDeque<>();
        Deque<Integer> sw = new ArrayDeque<>();

        for (int s : students) queue.addLast(s);
        for (int s : sandwiches) sw.addLast(s);

        int attempts = 0;
        int n = queue.size();

        while (!queue.isEmpty() && attempts < n) {
            if (queue.peekFirst().equals(sw.peekFirst())) {
                queue.pollFirst();
                sw.pollFirst();
                attempts = 0;
            } else {
                int a = queue.pollFirst();
                queue.addLast(a);
                attempts += 1;
            }
        }

        return queue.size();
        
    }

    private static int timeRequiredToBuy(int[] tickets, int k) {
        int res = 0;

        for (int i = 0; i < tickets.length; i++) {
            if (i <= k) {
                res += Math.min(tickets[i], tickets[k]);
            } else {
                res += Math.min(tickets[i], tickets[k] - 1);
            }
        }

        return res;
    } 
}