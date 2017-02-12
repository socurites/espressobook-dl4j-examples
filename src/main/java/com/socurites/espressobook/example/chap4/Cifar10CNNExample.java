package com.socurites.espressobook.example.chap4;


import java.io.File;
import java.io.IOException;
import java.util.Random;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * MNIST classifier example using DenseLayer
 * 
 * @author socurites
 *
 */
public class Cifar10CNNExample {
    private static Logger log = LoggerFactory.getLogger(Cifar10CNNExample.class);
    
    protected static final String [] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
    
    protected static final long seed = 12345;
    public static final Random randNumGen = new Random(seed);
    
    protected static final int width = 32;
    protected static final int height = 32;
    protected static final int channels = 3;
    
    
    
    
    
    
    /*
     * batch size for each epoch.
     * uses all train items for batch gradient learning
     */
    public static final int BATCH_SIZE = 50;
    
    /* 
     * # of output classes
     * Iris-setosa, Iris-versicolor, Iris-virginica
     */
    public static final int OUTPUT_NUM = 10;
    
    /*
     * labels of output classes
     */
//    public static final String[] LABELS = {"Iris-setosa", "Iris-versicolor", "Iris-virginica"};

    public static void main(String[] args) throws Exception {
    	Cifar10CNNExample example = new Cifar10CNNExample();
    	
    	// 1. Loads datasets for training and evaluation
    	ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
    	BalancedPathFilter pathFilter = new BalancedPathFilter(randNumGen, allowedExtensions, labelMaker);
    	
    	InputSplit[] filesInDirSplit = example.loadDataSets(pathFilter);
    	InputSplit trainData 					= filesInDirSplit[0];
    	InputSplit testData 					= filesInDirSplit[1];
    	
    	ImageRecordReader trainRecordReader = new ImageRecordReader(height, width, channels, labelMaker);
    	trainRecordReader.initialize(trainData);
    	DataSetIterator trainDataSetIterator = new RecordReaderDataSetIterator(trainRecordReader, BATCH_SIZE, 1, OUTPUT_NUM);
    	DataNormalization trainScaler = new ImagePreProcessingScaler(0,1);
    	trainScaler.fit(trainDataSetIterator);
        trainDataSetIterator.setPreProcessor(trainScaler);
    	
    	ImageRecordReader testRecordReader = new ImageRecordReader(height, width, channels, labelMaker);
    	testRecordReader.initialize(testData);
    	DataSetIterator testDataSetIterator = new RecordReaderDataSetIterator(testRecordReader, 10, 1, OUTPUT_NUM);
    	DataNormalization testScaler = new ImagePreProcessingScaler(0,1);
    	testScaler.fit(trainDataSetIterator);
        trainDataSetIterator.setPreProcessor(testScaler);
    	
    	// 2. Configures network
    	MultiLayerConfiguration conf 			= example.configureNetwork();
    	
    	// 3. Trains network using training datasets
    	MultiLayerNetwork model					= example.train(conf, trainDataSetIterator);
    	
    	// 4. Evaluates network using test datasets
    											  example.evaluate(model, testDataSetIterator);
    }
    
    
    
    private InputSplit[] loadDataSets(BalancedPathFilter pathFilter) throws IOException, InterruptedException {
    	File parentDir = new File(System.getProperty("user.home"), "Projects/write-dl4j-book-espresso/datasets/cifar_10_png/");
    	
    	System.out.println(parentDir);
    	
    	FileSplit filesInDir = new FileSplit(parentDir, allowedExtensions, randNumGen);
    	
    	InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, 80, 20);
        
    	return filesInDirSplit;
    }
    
    
    
    
    private MultiLayerConfiguration configureNetwork() {
    	log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .regularization(true).l2(0.0005)
                .weightInit(WeightInit.XAVIER)
                .iterations(1)
                .learningRate(0.015)
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
                        .nIn(channels)
                        .stride(1, 1)
                        .nOut(6)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(2, new ConvolutionLayer.Builder(5, 5)
                        //Note that nIn need not be specified in later layers
                        .stride(1, 1)
                        .nOut(16)
                        .activation(Activation.RELU)
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(4, new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(400).build())
                .layer(4, new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(120).build())
                .layer(4, new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(84).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(OUTPUT_NUM)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(32,32,3))
                .backprop(true).pretrain(false).build();
        
        return conf;
    }
    
    private MultiLayerNetwork train(MultiLayerConfiguration conf, DataSetIterator trainSetIterator) throws IOException {
        int numEpochs = 25;


        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(1));

        log.info("Train model....");
        for( int i=0; i<numEpochs; i++ ){
            model.fit(trainSetIterator);
        }

        return model;
    }

    private void evaluate(MultiLayerNetwork model, DataSetIterator testSetIterator) {
    	log.info("Evaluate model....");
        Evaluation eval = new Evaluation(OUTPUT_NUM);
        while(testSetIterator.hasNext()){
            DataSet t = testSetIterator.next();
            INDArray features = t.getFeatureMatrix();
            INDArray lables = t.getLabels();
            INDArray predicted = model.output(features,false);
            
            eval.eval(lables, predicted);
        }

        log.info(eval.stats());
    }
}
