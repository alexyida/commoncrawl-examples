package org.commoncrawl.examples;

// Java classes
import java.lang.Math;
import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileReader;
import java.io.InputStreamReader;
import java.io.IOException;
import java.net.URI;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.TreeMap;

// Apache Project classes
import org.apache.log4j.Logger;

// Hadoop classes
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.PathFilter;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.SequenceFileInputFormat;
import org.apache.hadoop.mapred.TextOutputFormat;
import org.apache.hadoop.mapred.lib.LongSumReducer;
import org.apache.hadoop.util.Progressable;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

/**
 * work with Common Crawl corpus text content.
 * 
 * @author Yida Wang <yida.wang@mail.utoronto.ca>
 */
public class SoccerPlayerAnalyzer extends Configured implements Tool {

  public static class CountingLong {
    long val = 1;

    public void CountingLong(){
      this.val = 1;
    }

    public long getVal(){
      return this.val;
    }

    public void incr(){
      this.val += 1;
    }
  }

  private static final Logger LOG = Logger.getLogger(SoccerPlayerAnalyzer.class);

  private static ArrayList<String> playerNames = new ArrayList<String>(Arrays.asList(
              "Lionel Messi",
              "Cristiano Ronaldo",
              "Radamel Falcao",
              "Gareth Bale",
              "Andres Iniesta",
              "Philipp Lahm",
              "Neymar",
              "Luis Suarez",
              "Franck Ribery",
              "Sergio Busquets",
              "Thiago Silva",
              "Gianluigi Buffon",
              "Bastian Schweinsteiger",
              "Thomas Muller",
              "Edinson Cavani",
              "Robert Lewandowski",
              "David Silva",
              "Zlatan Ibrahimovic",
              "Xavi",
              "Robin van Persie",
              "Gerard Pique",
              "Mesut Ozil",
              "David Luiz",
              "Mario Gotze",
              "Wayne Rooney",
              "Juan Mata",
              "Toni Kroos",
              "Marco Reus",
              "Andrea Pirlo",
              "Karim Benzema",
              "Luka Modric",
              "Sergio Aguero",
              "Cesc Fabregas",
              "Sergio Ramos",
              "Manuel Neuer",
              "Arturo Vidal",
              "Hulk",
              "Santi Cazorla",
              "Angel di Maria",
              "Iker Casillas",
              "Xabi Alonso",
              "Mario Balotelli",
              "Vincent Kompany",
              "Jordi Alba",
              "Daniele De Rossi",
              "Javi Martinez",
              "Gonzalo Higuain",
              "Yaya Toure",
              "Ilkay Gundogan",
              "Dante"));

  private static ArrayList<String> commonWords = new ArrayList<String>(Arrays.asList("the","be","to","of","and","a","in","that","have","I","it","for","not","on","with","he","as","you","do","at","this","but","his","by","from","they","we","say","her","she","or","an","will","my","one","all","would","there","their","what","so","up","out","if","about","who","get","which","go","me","when","make","can","like","time","no","just","him","know","take","person","into","year","your","good","some","could","them","see","other","than","then","now","look","only","come","its","over","think","also","back","after","use","two","how","our","work","first","well","way","even","new","want","because","any","these","give","day","most","us","is","i","are","has"));

  /**
   * Perform a simple word count mapping on text data from the Common Crawl corpus.
   */
  public static class SoccerPlayerAnalyzerMapper
      extends    MapReduceBase 
      implements Mapper<Text, Text, IntWritable, Text> {

    // create a counter group for Mapper-specific statistics
    private final String _counterGroup = "Custom Mapper Counters";

    public void map(Text key, Text value, OutputCollector<IntWritable, Text> output, Reporter reporter)
        throws IOException {

      reporter.incrCounter(this._counterGroup, "Records In", 1);

      try {

        // Get the text content as a string.
        String pageText = value.toString();

        // Removes all punctuation.
        pageText = pageText.replaceAll("[^a-zA-Z0-9 ]", "");

        // Normalizes whitespace to single spaces.
        pageText = pageText.replaceAll("\\s+", " ");

        if (pageText == null || pageText == "") {
          reporter.incrCounter(this._counterGroup, "Skipped - Empty Page Text", 1);
        }

        // Extract the domain
	URI uri = new URI(key.toString());
        String domain = uri.getHost();
        domain = domain.startsWith("www.") ? domain.substring(4) : domain;

        // Look for occurrence of players by name
        for (int i = 0; i < playerNames.size(); i++) {
          if (pageText.indexOf(playerNames.get(i)) >= 0) {
            // emit domain
            output.collect(new IntWritable(i), new Text("d" + domain));

            // emit uncommon words from this page and associate them with player
            for (String word : pageText.split(" ")) {
              word = word.toLowerCase().trim();
              if (commonWords.indexOf(word) == -1)
                output.collect(new IntWritable(i), new Text("w" + word));
            }
          }
        }
      }
      catch (Exception ex) {
        LOG.error("Caught Exception", ex);
        reporter.incrCounter(this._counterGroup, "Exceptions", 1);
      }
    }
  }

  public static class SoccerPlayerAnalyzerReducer extends MapReduceBase implements Reducer<IntWritable, Text, Text, Text> {
    public void reduce(IntWritable key, Iterator<Text> values, OutputCollector<Text, Text> output, Reporter reporter) throws IOException {
      try {
        // player name lookup
        String player = playerNames.get(key.get());

        HashMap<String, CountingLong> wordFreq = new HashMap<String, CountingLong>();
        HashMap<String, CountingLong> domainFreq = new HashMap<String, CountingLong>();
        long totalPlayerCount = 0;

        // assemble freq map
        while (values.hasNext()) {
          String fullVal = values.next().toString();
          String type = fullVal.substring(0, 1);
          String content = fullVal.substring(1);

          // find out if we have a domain key or a word key
          if (type.equals("d")) {
            if (domainFreq.get(content) == null)
              domainFreq.put(content, new CountingLong());
            else
              domainFreq.get(content).incr();

            totalPlayerCount++;
          } else {
            if (wordFreq.get(content) == null)
              wordFreq.put(content, new CountingLong());
            else
              wordFreq.get(content).incr();
          }
        }

        // WORDS
        // sort by key and emit one result with the top 50 words sorted in descending order

        // assemble reverse mapping: count -> word
        TreeMap<Long, ArrayList<String>> revMap = new TreeMap<Long, ArrayList<String>>();
        for (String word : wordFreq.keySet()) {
          Long freq = new Long(wordFreq.get(word).getVal());

          if(revMap.get(freq) == null)
            revMap.put(freq, new ArrayList<String>());
          revMap.get(freq).add(word);
        }

        ArrayList<String> wordsToEmit = new ArrayList<String>();
        for (Long count : revMap.descendingKeySet()) {
          if(wordsToEmit.size() >= 50)
            break;
          ArrayList<String> words = revMap.get(count);

          for (String word : words) {
            if(wordsToEmit.size() >= 50)
              break;
            else
              wordsToEmit.add(word + ":" + count.toString());
          }
        }

        String outputWords = "";
        for (String word : wordsToEmit) {
          outputWords += word + ", ";
        }

        // emit word association information
        output.collect(new Text("WRD | " + player + " | "), new Text(outputWords));

        // DOMAINS
        // sort by key and emit one result with the top 50 domains sorted in descending order
        // assemble reverse mapping: count -> domain
          
        TreeMap<Long, ArrayList<String>> revDomainMap = new TreeMap<Long, ArrayList<String>>();
        for (String domain : domainFreq.keySet()) {
          Long freq = new Long(domainFreq.get(domain).getVal());

          if (revDomainMap.get(freq) == null)
            revDomainMap.put(freq, new ArrayList<String>());

          revDomainMap.get(freq).add(domain);
        }

        ArrayList<String> domainsToEmit = new ArrayList<String>();
        for (Long count : revDomainMap.descendingKeySet()) {
          if (domainsToEmit.size() >= 50)
            break;

          ArrayList<String> domains = revDomainMap.get(count);
          for (String domain : domains) {
            if (domainsToEmit.size() >= 50)
              break;
            else
              domainsToEmit.add(domain + ":" + count.toString());           
          }
        }

        String outputDomains = "";
        for (String domain : domainsToEmit) {
          outputDomains += domain+", ";
        }

        // emit domain association information
        output.collect(new Text("DMN | " + player + " | "), new Text(outputDomains));

        // emit count information
        output.collect(new Text("CNT | " + player + " | "), new Text(Long.toString(totalPlayerCount)));
      } catch (Exception ex) {
        LOG.error("Caught Exception", ex);
        reporter.incrCounter("SoccerPlayerAnalyzerReducer", "Exceptions", 1);
      }
    }
  }

  /**
   * Hadoop FileSystem PathFilter for ARC files, allowing users to limit the
   * number of files processed.
   *
   * @author Chris Stephens <chris@commoncrawl.org>
   */
  public static class SampleFilter
      implements PathFilter {

    private static int count =         0;
    private static int max   = 999999999;

    public boolean accept(Path path) {

      if (!path.getName().startsWith("textData-"))
        return false;

      SampleFilter.count++;

      if (SampleFilter.count > SampleFilter.max)
        return false;

      return true;
    }
  }

  /**
   * Implmentation of Tool.run() method, which builds and runs the Hadoop job.
   *
   * @param  args command line parameters, less common Hadoop job parameters stripped
   *              out and interpreted by the Tool class.  
   * @return      0 if the Hadoop job completes successfully, 1 if not. 
   */
  @Override
  public int run(String[] args)
      throws Exception {

    String outputPath = null;
    String configFile = null;

    // Read the command line arguments.
    if (args.length <  1)
      throw new IllegalArgumentException("Example JAR must be passed an output path.");

    outputPath = args[0];

    if (args.length >= 2)
      configFile = args[1];

    // For this example, only look at a single text file.
    // String inputPath = "s3n://aws-publicdatasets/common-crawl/parse-output/segment/1341690166822/textData-01666";
 
    // Switch to this if you'd like to look at all text files.  May take many minutes just to read the file listing.
    // String inputPath = "s3n://aws-publicdatasets/common-crawl/parse-output/segment/*/textData-*";


    // Creates a new job configuration for this Hadoop job.
    JobConf job = new JobConf(this.getConf());

    job.setJarByClass(SoccerPlayerAnalyzer.class);

    // =+=+ find files from the list 
    String segmentListFile = "s3n://aws-publicdatasets/common-crawl/parse-output/valid_segments.txt";

    FileSystem fsInput = FileSystem.get(new URI(segmentListFile), job);
    BufferedReader reader = new BufferedReader(new InputStreamReader(fsInput.open(new Path(segmentListFile))));

    String segmentId;

    int i = 0;
    while ((segmentId = reader.readLine()) != null) {
        if (i == 0) { // FIXME: only processing the first segment to save time
            String inputPath = "s3n://aws-publicdatasets/common-crawl/parse-output/segment/"+segmentId+"/textData-*";
            FileInputFormat.addInputPath(job, new Path(inputPath));
        }
        i++;
    }
    // =+=+

/*
    // =+=+ single file
    String inputPath = "s3n://aws-publicdatasets/common-crawl/parse-output/segment/1341690166822/textData-01666";
    FileInputFormat.addInputPath(job, new Path(inputPath));
    FileInputFormat.setInputPathFilter(job, SampleFilter.class);
    // =+=+
*/

    // Read in any additional config parameters.
    if (configFile != null) {
      LOG.info("adding config parameters from '"+ configFile + "'");
      this.getConf().addResource(configFile);
    }

    // Scan the provided input path for ARC files.
    // LOG.info("setting input path to '"+ inputPath + "'");
    // FileInputFormat.addInputPath(job, new Path(inputPath));
    // FileInputFormat.setInputPathFilter(job, SampleFilter.class);

    // Delete the output path directory if it already exists.
    LOG.info("clearing the output path at '" + outputPath + "'");

    FileSystem fs = FileSystem.get(new URI(outputPath), job);

    if (fs.exists(new Path(outputPath)))
      fs.delete(new Path(outputPath), true);

    // Set the path where final output 'part' files will be saved.
    LOG.info("setting output path to '" + outputPath + "'");
    FileOutputFormat.setOutputPath(job, new Path(outputPath));
    FileOutputFormat.setCompressOutput(job, false);

    // Set which InputFormat class to use.
    job.setInputFormat(SequenceFileInputFormat.class);

    job.setMapOutputKeyClass(IntWritable.class);
    job.setMapOutputValueClass(Text.class);

    // Set which OutputFormat class to use.
    job.setOutputFormat(TextOutputFormat.class);

    // Set the output data types.
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(Text.class);

    // Set which Mapper and Reducer classes to use.
    job.setMapperClass(SoccerPlayerAnalyzer.SoccerPlayerAnalyzerMapper.class);
    job.setReducerClass(SoccerPlayerAnalyzer.SoccerPlayerAnalyzerReducer.class);

    if (JobClient.runJob(job).isSuccessful())
      return 0;
    else
      return 1;
  }

  /**
   * Main entry point that uses the {@link ToolRunner} class to run the SoccerPlayerAnalyzer
   * Hadoop job.
   */
  public static void main(String[] args)
      throws Exception {
    int res = ToolRunner.run(new Configuration(), new SoccerPlayerAnalyzer(), args);
    System.exit(res);
  }
}

