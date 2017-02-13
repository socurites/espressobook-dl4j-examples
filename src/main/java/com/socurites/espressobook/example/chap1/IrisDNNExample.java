package com.socurites.espressobook.example.chap1;


import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import org.datavec.api.io.converters.LabelWriterConverter;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.socurites.espressobook.example.util.Constants;

/**
 * IRIS classifier example using DenseLayer
 * 
 * @author socurites
 *
 */
public class IrisDNNExample {
    private static Logger log = LoggerFactory.getLogger(IrisDNNExample.class);
    
    /*
     * Batch size for each epoch.
     * uses all train items for batch gradient learning
     */
    protected static final int batchSize = 120;
    
    /*
     * Batch size for evaluation. 
     */
    protected static final int batchSizeEval = 1;
    
    /* 
     * # of input features. 
     */
    protected static final int numInput = 4;
    
    /* 
     * # of output classes.
     * Iris-setosa, Iris-versicolor, Iris-virginica
     */
    protected static final int numOutput = 3;
    
    /*
     * # of iterations.
     */
    protected static final int numIterations = 1; 
    
    /*
     * Learning rate.
     */
    protected static final double learningRate = 0.15;
    
    /*
     * # of epohcs.
     */
    protected static final int numEpochs = 25;
    
    /*
     * labels of output classes
     */
    protected static final List<String> labels = Arrays.asList(new String[] {"Iris-setosa", "Iris-versicolor", "Iris-virginica"});

    public static void main(String[] args) throws Exception {
    	IrisDNNExample example = new IrisDNNExample();
    	
    	// 1. Loads datasets for training and evaluation
    	DataSetIterator trainDataSetIterator = example.loadDataSet("/iris/iris_train.data", batchSize);
    	DataSetIterator testDataSetIterator = example.loadDataSet("/iris/iris_test.data", batchSizeEval);
    	
    	// 2. Configures network
    	MultiLayerConfiguration conf = example.configureNetwork();
    	
    	// 3. Trains network using training datasets
    	MultiLayerNetwork model = example.train(conf, trainDataSetIterator);
    	
    	// 4. Evaluates network using test datasets
    	example.evaluate(model, testDataSetIterator);
    }
    
    /**
     * Loads datasets for training and evaluation
     * 
     * @param dataFilePath file path for datasets
     * @param batchSize size of batch
     * @return Instance of DatasetIterator
     * @throws IOException
     * @throws InterruptedException
     */
    private DataSetIterator loadDataSet(String dataFilePath, int batchSize) throws IOException, InterruptedException {
    	log.info("Load datasets....");
    	CSVRecordReader recordReader = new CSVRecordReader();
    	recordReader.initialize(new FileSplit(new File(Constants.DATASET_DIR + dataFilePath)));
    	
    	LabelWriterConverter labelConverter = new LabelWriterConverter(labels);
    	
    	RecordReaderDataSetIterator dataSetIterator = new RecordReaderDataSetIterator(recordReader, labelConverter, batchSize, 4, 3);
    	
    	return dataSetIterator;
    }
    
    /**
     * Configures neural network
     * 
     * @return
     */
    private MultiLayerConfiguration configureNetwork() {
    	log.info("Build model....");
    	int layerIndex = 0;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(numIterations)
                .learningRate(learningRate)
                .list()
                .layer(layerIndex++, new DenseLayer.Builder()
                        .nIn(numInput)
                        .nOut(10)
                        .activation(Activation.TANH)
                        .build())
                .layer(layerIndex++, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(10)
                        .nOut(numOutput)
                        .activation(Activation.SOFTMAX)
                        .build())
                .pretrain(false).backprop(true) 
                .build();
        
        return conf;
    }
    
    private MultiLayerNetwork train(MultiLayerConfiguration conf, DataSetIterator trainSetIterator) throws IOException {
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
        Evaluation eval = new Evaluation(numOutput);
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
