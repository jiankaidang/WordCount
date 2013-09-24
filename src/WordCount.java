import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

import java.io.IOException;
import java.util.TreeSet;

/**
 * Author: Jiankai Dang
 * Date: 9/22/13
 * Time: 8:20 PM
 */
public class WordCount {
    private static int wordLength;
    private static String prefix;

    public static void main(String[] args) throws Exception {
        boolean hasCombiner = false;
        for (int i = 4; i < args.length; i++) {
            if (args[i].equals("-combiner")) {
                hasCombiner = true;
                continue;
            }
            if (args[i].equals("-word-length")) {
                wordLength = Integer.parseInt(args[++i]);
                continue;
            }
            prefix = args[++i];
        }

        long startTime = System.currentTimeMillis();
        Configuration conf = new Configuration();

        Job job = new Job(conf, "WordCount");

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);

        job.setMapperClass(Map.class);
        job.setReducerClass(Reduce.class);

        if (hasCombiner) {
            job.setCombinerClass(Combine.class);
        }

        job.setInputFormatClass(TextInputFormat.class);
        job.setOutputFormatClass(TextOutputFormat.class);

        FileInputFormat.addInputPath(job, new Path(args[1]));
        FileOutputFormat.setOutputPath(job, new Path(args[3]));

        job.waitForCompletion(true);
        long endTime = System.currentTimeMillis();
        long seconds = (endTime - startTime) / 1000;
        System.out.println(seconds);
    }

    public static class Map extends Mapper<LongWritable, Text, Text, IntWritable> {
        private final static IntWritable one = new IntWritable(1);
        private Text word = new Text();

        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString().trim();
            if (line.equals("")) {
                return;
            }
            String[] result = line.split("\\s+");
            for (String aResult : result) {
                word.set(aResult);
                context.write(word, one);
            }
        }
    }

    public static class Reduce extends Reducer<Text, IntWritable, Text, IntWritable> {

        private TreeSet<WordCountNode> mostFrequentWords = new TreeSet<WordCountNode>();

        public void reduce(Text key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }
            mostFrequentWords.add(new WordCountNode(new Text(key), sum));
            if (mostFrequentWords.size() > 100) {
                mostFrequentWords.pollLast();
            }
        }

        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {
            for (WordCountNode wordCount : mostFrequentWords) {
                context.write(wordCount.word, new IntWritable(wordCount.count));
            }
        }

        private class WordCountNode implements Comparable<WordCountNode> {
            Text word;
            Integer count;

            WordCountNode(Text word, Integer count) {
                this.word = word;
                this.count = count;
            }

            @Override
            public int compareTo(WordCountNode wordCountNode) {
                if (this.count > wordCountNode.count) {
                    return -1;
                }
                return 1;
            }
        }
    }

    private static class Combine extends Reducer<Text, IntWritable, Text, IntWritable> {
        public void reduce(Text key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }
            context.write(key, new IntWritable(sum));
        }
    }
}
