import java.io.*;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;

/**
 * Created by eshulga on 1/9/17.
 */
public class TemplateCodeForces {
    public static void main(String[] args) {
        InputStream inputStream = System.in;
        OutputStream outputStream = System.out;
        InputReader in = new InputReader(inputStream);
        PrintWriter out = new PrintWriter(outputStream);
        TaskA solver = new TaskA();
        solver.solve(in, out);
        out.close();
    }

    static class TaskA {

        public void solve(InputReader in, PrintWriter out) {
            int n = 5;
            int k = 3;
            int[] ar = new int[]{3, 4 ,1 ,2,5};
//            int n = in.nextInt();
//            int k = in.nextInt();
//            int[] ar = new int[n];
            for (int i = 0; i < n; i++) {

            }

            out.print("ss");
        }


        public void swapChar(char[] ar, int i, int j) {
            char tmp = ar[i];
            ar[i] = ar[j];
            ar[j] = tmp;
        }

        public void swap(int[] ar, int i, int j) {
            int tmp = ar[i];
            ar[i] = ar[j];
            ar[j] = tmp;
        }
    }


    static class InputReader {
        public BufferedReader reader;
        public StringTokenizer tokenizer;

        public InputReader(InputStream stream) {
            reader = new BufferedReader(new InputStreamReader(stream), 32768);
            tokenizer = null;
        }

        public String next() {
            while (tokenizer == null || !tokenizer.hasMoreTokens()) {
                try {
                    tokenizer = new StringTokenizer(reader.readLine());
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            }
            return tokenizer.nextToken();
        }

        public int nextInt() {
            return Integer.parseInt(next());
        }
    }
}
