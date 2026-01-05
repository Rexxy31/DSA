import java.util.*;

public class quest {
    public static void main(String[] args) {
        int[] students = {1,1,1,0,0,1};
        int[] sandwiches = {1,0,0,0,1,1};
        System.out.println(countStudents(students, sandwiches));
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
}