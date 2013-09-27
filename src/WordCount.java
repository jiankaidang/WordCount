import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
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

    public static void main(String[] args) throws Exception {
        boolean hasCombiner = false;

        Configuration conf = new Configuration();

        for (int i = 4; i < args.length; i++) {
            if (args[i].equals("-combiner")) {
                hasCombiner = true;
                continue;
            }
            if (args[i].equals("-word-length")) {
                conf.setInt("wordLength", Integer.parseInt(args[++i]));
                continue;
            }
            conf.set("prefix", args[++i]);
        }

        Job job = new Job(conf, "WordCount");

        job.setNumReduceTasks(1);

        job.setJarByClass(WordCount.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);

        job.setMapperClass(Map.class);

        job.setReducerClass(Reduce.class);

        job.setInputFormatClass(TextInputFormat.class);
        job.setOutputFormatClass(TextOutputFormat.class);

        FileInputFormat.addInputPath(job, new Path(args[1]));

        String outputPath = args[3];
        FileOutputFormat.setOutputPath(job, new Path(outputPath));
        String outputFilesDirectory = outputPath + "Txt";

        long startTime = System.currentTimeMillis();
        job.waitForCompletion(true);
        long endTime = System.currentTimeMillis();
        long seconds = (endTime - startTime) / 1000;

        if (hasCombiner) {
            Job combinerJob = new Job(conf, "WordCountCombiner");

            combinerJob.setNumReduceTasks(1);

            combinerJob.setJarByClass(WordCount.class);

            combinerJob.setOutputKeyClass(Text.class);
            combinerJob.setOutputValueClass(IntWritable.class);

            combinerJob.setMapperClass(Map.class);
            combinerJob.setReducerClass(Reduce.class);
            combinerJob.setCombinerClass(Combine.class);

            combinerJob.setInputFormatClass(TextInputFormat.class);
            combinerJob.setOutputFormatClass(TextOutputFormat.class);

            FileInputFormat.addInputPath(combinerJob, new Path(args[1]));

            FileOutputFormat.setOutputPath(combinerJob, new Path(outputPath + "Combiner"));

            long startTimeCombiner = System.currentTimeMillis();
            combinerJob.waitForCompletion(true);
            long endTimeCombiner = System.currentTimeMillis();
            long secondsCombiner = (endTimeCombiner - startTimeCombiner) / 1000;

            FileSystem fs = FileSystem.get(new Path(outputFilesDirectory).toUri(), conf);

            FSDataOutputStream out = fs.create(new Path(outputFilesDirectory + "/b_output"));
            out.writeChars(seconds + "\t" + secondsCombiner);
            out.close();
        }
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
            for (String str : result) {
                if (str.matches("^[a-zA-Z\\d']+$")) {
                    word.set(str);
                    context.write(word, one);
                }
            }
        }
    }

    public static class Reduce extends Reducer<Text, IntWritable, Text, IntWritable> {

        private TreeSet<WordCountNode> mostFrequentWords = new TreeSet<WordCountNode>();
        private TreeSet<WordCountNode> mostFrequentWordsWordLength = new TreeSet<WordCountNode>();
        private TreeSet<WordCountNode> mostFrequentWordsPrefix = new TreeSet<WordCountNode>();
        private Path outputFilesDirectory;
        private FileSystem fs;
        private int wordLength;
        private String prefix;

        public void reduce(Text key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }
            String word = key.toString();
            getMostFrequentWords(mostFrequentWords, word, sum);

            Configuration configuration = context.getConfiguration();

            wordLength = configuration.getInt("wordLength", -1);
            prefix = configuration.get("prefix");

            if (wordLength != -1) {
                reduceWordLength(word, sum);
            }

            if (prefix != null) {
                reducePrefix(word, sum);
            }
        }

        private void reduceWordLength(String word, int sum) {
            if (word.length() == wordLength) {
                getMostFrequentWords(mostFrequentWordsWordLength, word, sum);
            }
        }

        private void reducePrefix(String word, int sum) {
            if (word.startsWith(prefix)) {
                getMostFrequentWords(mostFrequentWordsPrefix, word, sum);
            }
        }

        private void getMostFrequentWords(TreeSet<WordCountNode> priorityQueue, String word, int sum) {
            priorityQueue.add(new WordCountNode(word, sum));
            if (priorityQueue.size() > 100) {
                priorityQueue.pollFirst();
            }
        }

        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {
            outputFilesDirectory = new Path(FileOutputFormat.getOutputPath(context).getParent(), "outputTxt");
            fs = FileSystem.get(outputFilesDirectory.toUri(), context.getConfiguration());

            write("a_output", mostFrequentWords);

            if (wordLength != -1) {
                cleanupWordLength();
            }

            if (prefix != null) {
                cleanupPrefix();
            }
        }

        private void cleanupWordLength() throws IOException {
            write("c_output", mostFrequentWordsWordLength);
        }

        private void cleanupPrefix() throws IOException {
            write("d_output", mostFrequentWordsPrefix);
        }

        private void write(String outputFile, TreeSet<WordCountNode> priorityQueue) throws IOException {
            FSDataOutputStream out = fs.create(new Path(outputFilesDirectory, outputFile));
            while (priorityQueue.size() > 0) {
                WordCountNode wordCount = priorityQueue.pollLast();
                out.writeChars(wordCount.word + '\t' + wordCount.count +
                        (priorityQueue.isEmpty() ? "" : '\n'));
            }
            out.close();
        }

        private class WordCountNode implements Comparable<WordCountNode> {
            String word;
            Integer count;

            WordCountNode(String word, Integer count) {
                this.word = word;
                this.count = count;
            }

            @Override
            public int compareTo(WordCountNode wordCountNode) {
                if (this.count > wordCountNode.count) {
                    return 1;
                }
                return -1;
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
