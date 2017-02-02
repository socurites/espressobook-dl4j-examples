package com.socurites.espressobook.example.chap1;


import java.io.IOException;
import java.util.Arrays;

import org.datavec.api.io.converters.LabelWriterConverter;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * IRIS classifier example using DenseLayer
 * 
 * @author socurites
 *
 */
public class IrisDNNExample {
    private static Logger log = LoggerFactory.getLogger(IrisDNNExample.class);
    
    /*
     * batch size for each epoch.
     * uses all train items for batch gradient learning
     */
    public static final int BATCH_SIZE = 120;
    
    /* 
     * # of input features. 
     */
    public static final int INPUT_NUM = 4;
    
    /* 
     * # of output classes
     * Iris-setosa, Iris-versicolor, Iris-virginica
     */
    public static final int OUTPUT_NUM = 3;
    
    /*
     * labels of output classes
     */
    public static final String[] LABELS = {"Iris-setosa", "Iris-versicolor", "Iris-virginica"};

    public static void main(String[] args) throws Exception {
    	IrisDNNExample example = new IrisDNNExample();
    	
    	// 1. Loads datasets for training and evaluation
    	DataSetIterator trainDataSetIterator 	= example.loadDataSet("data/iris/iris_train.data", BATCH_SIZE);
    	DataSetIterator testDataSetIterator 	= example.loadDataSet("data/iris/iris_test.data", 1);
    	
    	// 2. Configures network
    	MultiLayerConfiguration conf 			= example.configureNetwork();
    	
    	// 3. Trains network using training datasets
    	MultiLayerNetwork model					= example.train(conf, trainDataSetIterator);
    	
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
    	recordReader.initialize(new FileSplit(new ClassPathResource(dataFilePath).getFile()));
    	
    	LabelWriterConverter labelConverter = new LabelWriterConverter(Arrays.asList(LABELS));
    	
    	RecordReaderDataSetIterator dataSetIterator = new RecordReaderDataSetIterator(recordReader, labelConverter, batchSize, 4, 3);
    	
    	return dataSetIterator;
    }
    
    private MultiLayerConfiguration configureNetwork() {
    	log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(1)
                .learningRate(0.15)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(INPUT_NUM)
                        .nOut(10)
                        .activation(Activation.TANH)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(10)
                        .nOut(OUTPUT_NUM)
                        .activation(Activation.SOFTMAX)
                        .build())
                .pretrain(false).backprop(true) 
                .build();
        
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
